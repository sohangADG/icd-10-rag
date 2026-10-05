"""Runtime end-to-end verification against real PostgreSQL + a real uvicorn server.

    docker compose run --rm app python -m scripts.runtime_e2e

Flow: fresh database -> alembic upgrade -> synthetic sources -> CLI validate/ingest (+embed)
-> start uvicorn -> HTTP requests (health, datasets, search, codes, suggest, errors)
-> direct SQL integrity checks -> JSON report. Exits non-zero on any failed check.

Uses its own database (DATABASE_NAME + "_e2e"); the development database is never touched.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from alembic import command
from alembic.config import Config
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.indexing.embeddings import get_embedding_provider
from app.synthetic.cli import build
from app.synthetic.paraphrase import PARAPHRASE_PAIRS

ROOT = Path(__file__).resolve().parents[1]
PORT = 8100
CHECKS: list[dict[str, Any]] = []
EXPECTED: dict[str, Any] = {}


def check(name: str, condition: bool, detail: Any = None) -> None:
    CHECKS.append({"check": name, "passed": bool(condition), "detail": detail})
    marker = "PASS" if condition else "FAIL"
    print(f"[{marker}] {name}" + (f" -> {detail}" if detail is not None else ""), file=sys.stderr)


async def admin(*statements: str) -> None:
    engine = create_async_engine(
        get_settings().sqlalchemy_url("postgres"),
        poolclass=pool.NullPool,
        isolation_level="AUTOCOMMIT",
    )
    try:
        async with engine.connect() as connection:
            for statement in statements:
                await connection.execute(text(statement))
    finally:
        await engine.dispose()


async def query(database: str, sql: str, **params: Any) -> list[tuple]:
    engine = create_async_engine(get_settings().sqlalchemy_url(database), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            return [tuple(r) for r in await connection.execute(text(sql), params)]
    finally:
        await engine.dispose()


def cli(env: dict[str, str], *args: str) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(
        [sys.executable, "-m", "app.ingestion.cli", *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {"raw": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}
    return completed.returncode, payload


def main() -> int:
    settings = get_settings()
    database = f"{settings.database_name}_e2e"
    env = {**os.environ, "DATABASE_NAME": database, "LOG_LEVEL": "WARNING"}

    # 1. fresh database + migrations -----------------------------------------------------------
    asyncio.run(
        admin(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)', f'CREATE DATABASE "{database}"')
    )
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["database_url"] = settings.sqlalchemy_url(database)
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    version = asyncio.run(query(database, "SELECT version_num FROM alembic_version"))
    check("fresh database migrated to head", version == [("0003",)], version)

    provider = get_embedding_provider(settings)
    EXPECTED["space"] = (
        (provider.provider_name, provider.model, provider.dimension) if provider else None
    )
    check(
        "configured embedding provider loads (dimension read from the model)",
        provider is not None,
        EXPECTED["space"],
    )

    # 2. synthetic sources + CLI ingestion ----------------------------------------------------
    sources = Path(tempfile.mkdtemp(prefix="e2e_"))
    build(sources)

    def src(name: str) -> list[str]:
        return ["--file", str(sources / name), "--manifest", str(sources / f"{name}.manifest.json")]

    code, report = cli(env, "validate", *src("synth_2024_pdf.pdf"))
    check(
        "validate (dry run) synthetic PDF",
        code == 0 and report["valid"],
        {
            k: report["statistics"][k]
            for k in (
                "records_extracted",
                "pages_processed",
                "validation_errors",
                "validation_warnings",
            )
        },
    )
    code, report = cli(env, "ingest", *src("synth_2024_pdf.pdf"), "--embed")
    imported = report.get("import", {})
    check(
        "ingest synthetic 2024 PDF (+embeddings)",
        code == 0 and imported.get("status") == "imported",
        {
            "counts": imported.get("counts"),
            "embeddings": (report.get("embeddings") or {}).get("embedded"),
        },
    )
    pdf_dataset_id = imported.get("dataset_id")
    code, report = cli(env, "ingest", *src("synth_2025_json.json"), "--embed")
    check(
        "ingest synthetic 2025 JSON (+embeddings)",
        code == 0 and report["import"]["status"] == "imported",
    )
    code, report = cli(env, "ingest", *src("synth_paraphrase_json.json"), "--embed")
    embeddings = report.get("embeddings") or {}
    check(
        "ingest synthetic paraphrase dataset (+semantic embeddings)",
        code == 0 and report["import"]["status"] == "imported" and embeddings.get("failed") == 0,
        {
            k: embeddings.get(k)
            for k in (
                "provider",
                "model",
                "dimension",
                "documents",
                "embedded",
                "skipped_unchanged",
                "failed",
                "embed_ms",
                "write_ms",
                "duration_ms",
                "vector_index",
            )
        },
    )
    code, report = cli(env, "ingest", *src("synth_2024_pdf.pdf"))
    check(
        "re-ingest identical source is skipped",
        code == 0 and report["import"]["status"] == "skipped",
    )
    code, report = cli(env, "ingest", *src("synth_malformed.csv"))
    check(
        "malformed source -> VALIDATION_FAILED",
        code == 1 and report["import"]["dataset_status"] == "validation_failed",
        report["statistics"]["issue_codes"],
    )
    code, report = cli(env, "validate", *src("synth_restricted_2024.pdf"))
    check(
        "permission-restricted PDF refused",
        code == 3 and report["restricted"],
        [i["code"] for i in report["issues"]],
    )
    code, report = cli(env, "index", "--dataset-id", str(pdf_dataset_id))
    check(
        "re-index skips unchanged embeddings",
        code == 0 and report["embeddings"]["embedded"] == 0,
        report["embeddings"],
    )
    code, report = cli(
        env, "index", "--dataset-id", str(pdf_dataset_id), "--force", "--batch-size", "16"
    )
    check(
        "forced re-index regenerates every embedding in batches",
        code == 0
        and report["embeddings"]["embedded"] == report["embeddings"]["documents"]
        and report["embeddings"]["batches"] >= 4,
        {k: report["embeddings"][k] for k in ("embedded", "batches", "embed_ms", "write_ms")},
    )

    # 3. real HTTP server ---------------------------------------------------------------------
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        env=env,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base = f"http://127.0.0.1:{PORT}"
        with httpx.Client(base_url=base, timeout=30) as http:
            for _ in range(60):
                try:
                    if http.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    time.sleep(0.5)
            run_http_checks(http)
            run_semantic_http_checks(http)
    finally:
        server.terminate()
        server.wait(timeout=10)

    # 4. direct database integrity checks ----------------------------------------------------
    run_db_checks(database)

    failed = [c for c in CHECKS if not c["passed"]]
    print(
        json.dumps(
            {
                "database": database,
                "checks": CHECKS,
                "passed": len(CHECKS) - len(failed),
                "failed": len(failed),
            },
            indent=2,
            default=str,
        )
    )
    return 1 if failed else 0


def run_semantic_http_checks(http: httpx.Client) -> None:
    """Paraphrase queries share no word with their target: only semantics can find them."""
    ds = {"coding_system": "SYNTH-ICD", "version": "paraphrase-1"}
    found, shown = 0, []
    for code, _title, query in PARAPHRASE_PAIRS:
        body = http.get("/api/v1/icd/search", params={**ds, "q": query, "limit": 3}).json()
        top = body["results"][0] if body["results"] else {}
        found += top.get("code") == code
        shown.append(
            {
                "query": query,
                "expected": code,
                "top": top.get("code"),
                "record_id": top.get("record_id"),
                "semantic": top.get("scores", {}).get("semantic"),
                "semantic_raw": top.get("scores", {}).get("semantic_raw"),
                "lexical": top.get("scores", {}).get("lexical"),
                "hybrid": top.get("hybrid_score"),
                "status": body["semantic_status"],
                "model": (body.get("embedding_space") or {}).get("model"),
            }
        )
    check(
        "semantic hybrid search over HTTP (paraphrases, zero word overlap)",
        found >= 0.85 * len(PARAPHRASE_PAIRS) and all(r["status"] == "ok" for r in shown),
        {"top1": f"{found}/{len(PARAPHRASE_PAIRS)}", "results": shown[:6]},
    )
    response = http.post(
        "/api/v1/icd/suggest",
        json={"clinical_note": "Assessment: epilepsy. Denies hay fever.", **ds, "top_k": 3},
    )
    body = response.json()
    suggestion = body["suggestions"][0] if body["suggestions"] else {}
    scores = suggestion.get("retrieval_scores", {})
    check(
        "suggestion pipeline with real semantic retrieval (epilepsy -> P07)",
        response.status_code == 200
        and [s["code"] for s in body["suggestions"]] == ["P07"]
        and scores.get("semantic_status") == "ok"
        and suggestion.get("validation", {}).get("db_verified") is True,
        {
            "code": suggestion.get("code"),
            "title": suggestion.get("title"),
            "confidence": suggestion.get("confidence"),
            "scores": {
                k: scores.get(k)
                for k in (
                    "exact",
                    "lexical",
                    "fuzzy",
                    "semantic",
                    "semantic_raw",
                    "hierarchy",
                    "hybrid",
                    "rerank",
                )
            },
            "unmatched": [
                (u["clinical_concept"], u["concept_status"]) for u in body["unmatched_concepts"]
            ],
        },
    )


def run_http_checks(http: httpx.Client) -> None:
    ds = {"coding_system": "SYNTH-ICD", "version": "2024"}
    health = http.get("/health/db").json()
    check("GET /health/db", health == {"status": "ok", "database": "connected", "pgvector": True})
    datasets = http.get("/api/v1/icd/datasets", params={"coding_system": "SYNTH-ICD"}).json()
    check(
        "GET /api/v1/icd/datasets",
        {(d["version"], d["status"]) for d in datasets["datasets"]}
        == {
            ("2024", "ready"),
            ("2025", "ready"),
            ("paraphrase-1", "ready"),
            ("malformed", "validation_failed"),
        },
        [(d["version"], d["status"]) for d in datasets["datasets"]],
    )

    # 2024 was ingested from the PDF (tabular layout: no alphabetical index / synonyms);
    # 2025 from JSON (with source synonyms). Synonym-dependent checks therefore use 2025.
    ds25 = {"coding_system": "SYNTH-ICD", "version": "2025"}
    exact = http.get("/api/v1/icd/search", params={**ds, "q": "a01.0", "mode": "exact"}).json()
    check("search exact code", [r["code"] for r in exact["results"]] == ["A01.0"])
    no_synonyms = http.get(
        "/api/v1/icd/search", params={**ds, "q": "hypertension", "mode": "text"}
    ).json()
    check("PDF-derived 2024 has no invented synonyms", no_synonyms["results"] == [])
    lexical = http.get(
        "/api/v1/icd/search", params={**ds25, "q": "hypertension", "mode": "text"}
    ).json()
    check(
        "search lexical (source synonym, 2025)",
        lexical["results"][0]["code"] == "B00",
        lexical["results"][0]["scores"],
    )
    fuzzy = http.get("/api/v1/icd/search", params={**ds, "q": "chronic airway infecton"}).json()
    check(
        "search fuzzy (misspelling)",
        fuzzy["results"][0]["code"] == "A00.1",
        fuzzy["results"][0]["scores"],
    )
    hybrid = http.get(
        "/api/v1/icd/search", params={**ds25, "q": "acute heart failure", "limit": 3}
    ).json()
    check(
        "search hybrid incl. semantic (2025)",
        hybrid["semantic_available"] and hybrid["results"][0]["code"] == "B01.0",
        hybrid["results"][0]["scores"],
    )

    record = http.get("/api/v1/icd/codes/A00", params=ds).json()
    check(
        "GET code A00 (rules + provenance)",
        record["exclusions"][0]["target_code"] == "B15" and record["provenance"]["page_start"] >= 2,
        record["provenance"],
    )
    children = http.get("/api/v1/icd/codes/A01/children", params=ds).json()
    check(
        "GET children of A01 (2024)",
        [r["code"] for r in children["records"]] == ["A01.0", "A01.1", "A01.2", "A01.9"],
    )
    children_2025 = http.get(
        "/api/v1/icd/codes/A01/children", params={"coding_system": "SYNTH-ICD", "version": "2025"}
    ).json()
    check(
        "version isolation (A01.3 only in 2025)",
        "A01.3" in [r["code"] for r in children_2025["records"]],
    )
    ancestors = http.get("/api/v1/icd/codes/C00.20/ancestors", params=ds).json()
    check(
        "GET ancestors of C00.20",
        [r["code"] for r in ancestors["records"]] == ["III", "C00-C09", "C00", "C00.2"],
    )

    cases = [
        ("2024", "Assessment: Acute airway infection. Plan: fluids.", ["A00.0"]),
        ("2024", "Lobar consolidation on chest imaging.", ["A01.9"]),
        ("2024", "Diagnosis: lobar consolidation of the left lung.", ["A01.0"]),
        ("2024", "Airway infection in a 5-day-old newborn.", ["B15"]),
        ("2024", "Former smoker. Family history of diabetes.", ["D00.2", "D01"]),
        ("2024", "Denies cough. No evidence of airway infection.", []),
        ("2025", "Type 2 diabetes with kidney complication. CKD stage 3.", ["C00.20", "C12.3"]),
        ("2025", "Acute on chronic CHF.", ["B01.2"]),
        ("2025", "Lobar consolidation of multiple lobes.", ["A01.3"]),
    ]
    suggested: dict[str, set[str]] = {"2024": set(), "2025": set()}
    for version, note, expected in cases:
        response = http.post(
            "/api/v1/icd/suggest",
            json={
                "clinical_note": note,
                "coding_system": "SYNTH-ICD",
                "version": version,
                "top_k": 3,
            },
        )
        body = response.json()
        codes = [s["code"] for s in body["suggestions"]]
        suggested[version] |= set(codes) | {
            a["code"] for s in body["suggestions"] for a in s["alternatives"] if a["code"]
        }
        check(
            f"POST suggest [{version}]: {note[:42]!r}",
            response.status_code == 200 and codes == expected,
            [(s["code"], s["confidence"], s["missing_information"]) for s in body["suggestions"]],
        )
    CHECKS.append(
        {
            "check": "_suggested_codes",
            "passed": True,
            "detail": {k: sorted(v) for k, v in suggested.items()},
        }
    )

    errors = [
        (
            http.post(
                "/api/v1/icd/suggest",
                json={"clinical_note": "x", "coding_system": "ICD-99", "version": "1"},
            ),
            404,
            "UNSUPPORTED_CODING_SYSTEM",
        ),
        (
            http.post(
                "/api/v1/icd/suggest",
                json={"clinical_note": "x", "coding_system": "SYNTH-ICD", "version": "malformed"},
            ),
            409,
            "DATASET_NOT_READY",
        ),
        (
            http.post("/api/v1/icd/suggest", json={"clinical_note": " ", **ds}),
            422,
            "REQUEST_VALIDATION_ERROR",
        ),
        (http.get("/api/v1/icd/codes/Z99.9", params=ds), 404, "INVALID_ICD_CODE"),
        (http.post("/api/v1/admin/datasets/1/archive"), 404, "ADMIN_DISABLED"),
    ]
    for response, status, code in errors:
        check(
            f"error mapping {code}",
            response.status_code == status and response.json()["error"]["code"] == code,
        )
    check("request id header", bool(http.get("/health").headers.get("x-request-id")))


def run_db_checks(database: str) -> None:
    def q(sql: str, **params: Any) -> list[tuple]:
        return asyncio.run(query(database, sql, **params))

    rows = q(
        "SELECT id, version, status, source_checksum IS NOT NULL, source_page_count "
        "FROM icd_datasets ORDER BY id"
    )
    check(
        "dataset rows (hash, status, version)",
        len(rows) == 4 and all(r[3] for r in rows),
        rows,
    )
    counts = q(
        "SELECT d.version, count(n.id) FROM icd_datasets d LEFT JOIN icd_nodes n "
        "ON n.dataset_id = d.id GROUP BY d.version ORDER BY 1"
    )
    check(
        "records persisted per dataset",
        dict(counts) == {"2024": 70, "2025": 70, "paraphrase-1": 17, "malformed": 0},
        counts,
    )
    duplicates = q(
        "SELECT dataset_id, normalized_code FROM icd_nodes WHERE node_type IN "
        "('CATEGORY','SUBCATEGORY','CODE') GROUP BY 1, 2 HAVING count(*) > 1"
    )
    check("no duplicate codes per dataset", duplicates == [])
    orphans = q("SELECT code FROM icd_nodes WHERE parent_id IS NULL AND node_type <> 'CHAPTER'")
    check("no orphan records", orphans == [])
    broken = q(
        "SELECT c.code FROM icd_nodes c JOIN icd_nodes p ON p.id = c.parent_id "
        "WHERE p.dataset_id <> c.dataset_id"
    )
    check("parents always in the same dataset", broken == [])
    provenance = q("SELECT count(*) FROM icd_nodes WHERE source_locator IS NULL")
    check("every record has provenance", provenance == [(0,)])
    rules = q(
        "SELECT rule_type, count(*), count(target_node_id) FROM icd_rules r JOIN icd_datasets d "
        "ON d.id = r.dataset_id WHERE d.version = '2024' GROUP BY 1 ORDER BY 1"
    )
    check("coding rules persisted (type, count, resolved targets)", len(rules) >= 5, rules)
    embeddings = q(
        "SELECT embedding_provider, embedding_model, embedding_dimension, count(*), "
        "bool_and(vector_dims(embedding) = embedding_dimension), bool_and(embedding_normalized) "
        "FROM icd_search_documents WHERE embedding IS NOT NULL GROUP BY 1, 2, 3"
    )
    expected = EXPECTED["space"]
    check(
        "embeddings stored with provider + model + real dimension",
        len(embeddings) == 1
        and embeddings[0][:3] == expected
        and embeddings[0][3] == 70 + 70 + 17
        and embeddings[0][4] is True
        and embeddings[0][5] is True,
        embeddings,
    )
    hnsw = q(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE indexname LIKE 'ix_icd_search_documents_hnsw_%'"
    )
    check(
        "HNSW index matches the space (cosine opclass, provider/model/dimension predicate)",
        len(hnsw) == 1
        and "vector_cosine_ops" in hnsw[0][1]
        and f"vector({expected[2]})" in hnsw[0][1]
        and "embedding_provider" in hnsw[0][1],
        hnsw[0][0] if hnsw else None,
    )
    trgm = q("SELECT indexname FROM pg_indexes WHERE indexname LIKE '%trgm'")
    check("pg_trgm indexes exist", len(trgm) == 3, [r[0] for r in trgm])
    suggested = next(c["detail"] for c in CHECKS if c["check"] == "_suggested_codes")
    for version, codes in suggested.items():
        present = q(
            "SELECT n.code FROM icd_nodes n JOIN icd_datasets d ON d.id = n.dataset_id "
            "WHERE d.version = :version AND n.code = ANY(:codes)",
            version=version,
            codes=codes,
        )
        check(
            f"every returned code (incl. alternatives) exists in {version}",
            {r[0] for r in present} == set(codes),
            f"{len(codes)} codes",
        )
    runs = q("SELECT status, count(*) FROM icd_ingestion_runs GROUP BY 1 ORDER BY 1")
    check(
        "ingestion runs recorded",
        {r[0] for r in runs} >= {"completed", "skipped", "validation_failed"},
        runs,
    )


if __name__ == "__main__":
    sys.exit(main())
