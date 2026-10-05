"""Fresh-database clinical evaluation end to end (run in Docker).

    docker compose run --rm app python -m scripts.clinical_evaluation_e2e [--output-dir DIR]

Flow: fresh database -> alembic upgrade -> synthetic sources (2024, 2025, paraphrase, a second
coding system, malformed) -> CLI validate + ingest -> READY -> semantic index (configured
provider) -> non-READY datasets (pending, validation failed, archived) -> complete clinical
scenario evaluation -> real uvicorn: every scenario through POST /api/v1/icd/suggest, compared
with the service-level run, plus HTTP error/degradation checks -> reports.

Uses its own database (DATABASE_NAME + "_clinical"); the development database is untouched.
Exit status: 0 when every check and every safety gate passes, 1 otherwise.
Synthetic data only: nothing here measures real ICD-10-CA coding accuracy.
"""

import argparse
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
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.deps import load_provider
from app.core.config import get_settings
from app.core.database import create_engine_from_settings
from app.evaluation.clinical.report import markdown
from app.evaluation.clinical.run import run_clinical_evaluation
from app.evaluation.clinical.scenarios import load_scenarios
from app.schemas.dataset import DatasetCreate
from app.services.dataset_service import DatasetService
from app.synthetic.cli import build

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "tests" / "evaluation" / "clinical_scenarios.json"
PORT = 8110
CHECKS: list[dict[str, Any]] = []


def check(name: str, condition: bool, detail: Any = None) -> None:
    CHECKS.append({"check": name, "passed": bool(condition), "detail": detail})
    marker = "PASS" if condition else "FAIL"
    print(f"[{marker}] {name}" + (f" -> {detail}" if detail is not None else ""), file=sys.stderr)


async def _execute(database: str, *statements: str, autocommit: bool = False) -> list[tuple]:
    settings = get_settings()
    engine = create_async_engine(
        settings.sqlalchemy_url(database),
        poolclass=pool.NullPool,
        **({"isolation_level": "AUTOCOMMIT"} if autocommit else {}),
    )
    rows: list[tuple] = []
    try:
        async with engine.begin() as connection:
            for statement in statements:
                result = await connection.execute(text(statement))
                if result.returns_rows:
                    rows = [tuple(r) for r in result]
    finally:
        await engine.dispose()
    return rows


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
        return completed.returncode, json.loads(completed.stdout)
    except json.JSONDecodeError:
        return completed.returncode, {
            "raw": completed.stdout[-1500:],
            "err": completed.stderr[-1500:],
        }


def setup_database(database: str, env: dict[str, str]) -> None:
    settings = get_settings()
    asyncio.run(
        _execute(
            "postgres",
            f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)',
            f'CREATE DATABASE "{database}"',
            autocommit=True,
        )
    )
    config = Config(str(ROOT / "alembic.ini"))
    config.attributes["database_url"] = settings.sqlalchemy_url(database)
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    version = asyncio.run(_execute(database, "SELECT version_num FROM alembic_version"))
    check("fresh database migrated to head", version == [("0003",)], version)

    sources = Path(tempfile.mkdtemp(prefix="clinical_"))
    build(sources)

    def src(name: str) -> list[str]:
        return ["--file", str(sources / name), "--manifest", str(sources / f"{name}.manifest.json")]

    for name in (
        "synth_2024_json.json",
        "synth_2025_json.json",
        "synth_paraphrase_json.json",
        "synth_alt_2024_json.json",
    ):
        code, report = cli(env, "validate", *src(name))
        check(
            f"validate {name}",
            code == 0 and report.get("valid"),
            report.get("statistics", {}).get("records_extracted"),
        )
        code, report = cli(env, "ingest", *src(name), "--embed")
        embeddings = report.get("embeddings") or {}
        check(
            f"ingest {name} -> READY (+embeddings)",
            code == 0
            and report.get("import", {}).get("dataset_status") == "ready"
            and embeddings.get("failed") == 0
            and embeddings.get("embedded") == embeddings.get("documents"),
            {
                "counts": report.get("import", {}).get("counts", {}).get("nodes"),
                "embedded": embeddings.get("embedded"),
                "model": embeddings.get("model"),
            },
        )
    code, report = cli(env, "ingest", *src("synth_malformed.csv"))
    check(
        "malformed source -> VALIDATION_FAILED (non-READY)",
        code == 1 and report["import"]["dataset_status"] == "validation_failed",
    )
    # An ARCHIVED dataset: the second coding system, version 2023, ingested then archived.
    manifest = json.loads((sources / "synth_alt_2024_json.json.manifest.json").read_text())
    manifest["dataset"] = {**(manifest.get("dataset") or {}), "version": "2023"}
    archived_source = sources / "synth_alt_2023_json.json"
    document = json.loads((sources / "synth_alt_2024_json.json").read_text())
    document["dataset"]["version"] = "2023"
    archived_source.write_text(json.dumps(document))
    (sources / "synth_alt_2023_json.json.manifest.json").write_text(json.dumps(manifest))
    code, report = cli(env, "ingest", *src("synth_alt_2023_json.json"), "--embed")
    asyncio.run(
        _execute(
            database,
            "UPDATE icd_datasets SET status = 'archived' "
            "WHERE system = 'SYNTH-ALT' AND version = '2023'",
        )
    )
    # A PENDING dataset: registered identity, nothing ingested.
    asyncio.run(_register_pending(database))
    statuses = asyncio.run(
        _execute(database, "SELECT system, version, status FROM icd_datasets ORDER BY id")
    )
    check(
        "dataset lifecycle states (READY + pending + validation_failed + archived)",
        {s for *_, s in statuses} >= {"ready", "pending", "validation_failed", "archived"},
        statuses,
    )


async def _register_pending(database: str) -> None:
    settings = get_settings().model_copy(update={"database_name": database})
    engine = create_engine_from_settings(settings)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            await DatasetService(session).register(
                DatasetCreate(
                    system="SYNTH-ICD",
                    country="XX",
                    version="2026-draft",
                    publisher="icd-rag-service synthetic fixtures",
                    language="en",
                )
            )
    finally:
        await engine.dispose()


async def evaluate(database: str) -> dict[str, Any]:
    settings = get_settings().model_copy(update={"database_name": database})
    handle = load_provider(settings)
    engine = create_engine_from_settings(settings)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            return await run_clinical_evaluation(
                session,
                settings,
                handle.provider,
                load_scenarios(SCENARIOS),
                top_k=5,
                provider_status=handle.status,
                meta={"database": database, "scenario_file": str(SCENARIOS.relative_to(ROOT))},
            )
    finally:
        await engine.dispose()


def start_server(env: dict[str, str], port: int) -> subprocess.Popen[bytes]:
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5) as http:
        for _ in range(240):
            try:
                if http.get("/health").status_code == 200:
                    return server
            except httpx.TransportError:
                time.sleep(0.5)
    server.terminate()
    raise RuntimeError(f"server on port {port} did not start")


def _signature(suggestions: list[dict[str, Any]]) -> list[tuple]:
    return [
        (
            s["code"],
            s.get("confidence"),
            tuple(s.get("missing_information", [])),
            s.get("concept_status") or s.get("status"),
        )
        for s in suggestions
    ]


def http_scenarios(http: httpx.Client, report: dict[str, Any]) -> dict[str, Any]:
    """Every scenario through the real endpoint; results must equal the service-level run."""
    scenarios = {s.id: s for s in load_scenarios(SCENARIOS)}
    mismatches, statuses, latencies = [], {}, []
    for case in report["cases"]:
        scenario = scenarios[case["id"]]
        body = {
            "clinical_note": scenario.clinical_note,
            "coding_system": scenario.coding_system,
            "version": scenario.version,
            "top_k": 5,
        }
        if scenario.include_uncertain is not None:
            body["include_uncertain"] = scenario.include_uncertain
        started = time.perf_counter()
        response = http.post("/api/v1/icd/suggest", json=body)
        latencies.append((time.perf_counter() - started) * 1000)
        statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
        if response.status_code != 200 or "final" not in case:
            if "final" in case or response.status_code == 200:
                mismatches.append({"id": case["id"], "http_status": response.status_code})
            continue
        api = _signature(response.json()["suggestions"])
        service = _signature(case["final"]["suggestions"])
        if api != service:
            mismatches.append({"id": case["id"], "api": api, "service": service})
    latencies.sort()
    return {
        "scenarios": len(report["cases"]),
        "http_statuses": statuses,
        "mismatches": mismatches,
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p50": round(latencies[len(latencies) // 2], 2) if latencies else 0,
            "p95": round(latencies[int(0.95 * (len(latencies) - 1))], 2) if latencies else 0,
        },
    }


def http_checks(http: httpx.Client) -> None:
    ds = {"coding_system": "SYNTH-ICD", "version": "2024"}

    def post(**body: Any) -> httpx.Response:
        return http.post("/api/v1/icd/suggest", json=body)

    ok = post(clinical_note="Assessment: acute airway infection.", **ds)
    body = ok.json()
    check(
        "HTTP successful suggestion (DB-verified, dataset echoed)",
        ok.status_code == 200
        and [s["code"] for s in body["suggestions"]] == ["A00.0"]
        and body["suggestions"][0]["validation"]["db_verified"] is True
        and body["dataset"]["version"] == "2024",
        [(s["code"], s["confidence"]) for s in body["suggestions"]],
    )
    abstain = post(clinical_note="Patient denies chest pain and denies cough. Plan: review.", **ds)
    check(
        "HTTP abstention (negated findings only)",
        abstain.status_code == 200 and abstain.json()["suggestions"] == [],
        [u["concept_status"] for u in abstain.json()["unmatched_concepts"]],
    )
    for name, response, status, code in (
        (
            "invalid coding system",
            post(clinical_note="HTN.", coding_system="ICD-99", version="1"),
            404,
            "UNSUPPORTED_CODING_SYSTEM",
        ),
        (
            "wrong version",
            post(clinical_note="HTN.", coding_system="SYNTH-ICD", version="1999"),
            404,
            "UNSUPPORTED_DATASET_VERSION",
        ),
        (
            "dataset pending (not READY)",
            post(clinical_note="HTN.", coding_system="SYNTH-ICD", version="2026-draft"),
            409,
            "DATASET_NOT_READY",
        ),
        (
            "dataset validation failed (not READY)",
            post(clinical_note="HTN.", coding_system="SYNTH-ICD", version="malformed"),
            409,
            "DATASET_NOT_READY",
        ),
        (
            "dataset archived (not READY)",
            post(clinical_note="Ear canal irritation.", coding_system="SYNTH-ALT", version="2023"),
            409,
            "DATASET_NOT_READY",
        ),
        ("empty note", post(clinical_note="", **ds), 422, "REQUEST_VALIDATION_ERROR"),
        (
            "whitespace-only note",
            post(clinical_note=" \n\t ", **ds),
            422,
            "REQUEST_VALIDATION_ERROR",
        ),
        (
            "note above CLINICAL_NOTE_MAX_CHARS",
            post(clinical_note="cough " * 4000, **ds),
            422,
            "INVALID_CLINICAL_NOTE",
        ),
        (
            "note above the schema maximum",
            post(clinical_note="x" * 200_001, **ds),
            422,
            "REQUEST_VALIDATION_ERROR",
        ),
    ):
        payload = response.json()
        check(
            f"HTTP {name} -> {status} {code}",
            response.status_code == status
            and payload.get("error", {}).get("code") == code
            and "suggestions" not in payload,
            (response.status_code, payload.get("error", {}).get("code")),
        )
    noisy = (
        "Administrative: bed 4, insurance verified, transport booked. " * 120
        + "Assessment: Severe wheezing airway disorder. Denies chest pain. "
        + "Family history of diabetes. "
    )
    long_ok = post(clinical_note=noisy, **ds)
    codes = [s["code"] for s in long_ok.json().get("suggestions", [])]
    check(
        "HTTP long noisy note within limits (safe, no invented codes)",
        long_ok.status_code == 200 and set(codes) == {"A02.2", "D01"},
        {"chars": len(noisy), "codes": codes},
    )
    alt = post(
        clinical_note="Ear canal irritation.", coding_system="SYNTH-ALT", version="2024"
    ).json()
    check(
        "HTTP coding-system isolation (SYNTH-ALT A00 is its own record)",
        [s["code"] for s in alt["suggestions"]] == ["A00"]
        and alt["suggestions"][0]["icd_reference"]["coding_system"] == "SYNTH-ALT",
        alt["suggestions"][0]["title"] if alt["suggestions"] else None,
    )


def degraded_semantic_checks(env: dict[str, str]) -> None:
    """Provider that cannot load: optional mode degrades (200), required mode fails (503)."""
    broken = {
        **env,
        "EMBEDDING_MODEL": "synthetic-test/does-not-exist",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    body = {
        "clinical_note": "Assessment: acute airway infection.",
        "coding_system": "SYNTH-ICD",
        "version": "2024",
    }
    server = start_server({**broken, "SEMANTIC_RETRIEVAL_MODE": "optional"}, PORT + 1)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT + 1}", timeout=60) as http:
            response = http.post("/api/v1/icd/suggest", json=body)
            payload = response.json()
            suggestion = payload["suggestions"][0] if payload.get("suggestions") else {}
            check(
                "HTTP semantic provider unavailable, optional mode -> 200 degraded",
                response.status_code == 200
                and suggestion.get("code") == "A00.0"
                and suggestion.get("retrieval_scores", {}).get("semantic_status")
                == "provider_unavailable"
                and suggestion.get("retrieval_scores", {}).get("semantic") is None,
                suggestion.get("retrieval_scores", {}).get("semantic_status"),
            )
    finally:
        server.terminate()
        server.wait(timeout=15)
    server = start_server({**broken, "SEMANTIC_RETRIEVAL_MODE": "required"}, PORT + 2)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT + 2}", timeout=60) as http:
            response = http.post("/api/v1/icd/suggest", json=body)
            check(
                "HTTP semantic provider unavailable, required mode -> 503",
                response.status_code == 503
                and response.json()["error"]["code"] == "RETRIEVAL_ERROR",
                response.status_code,
            )
    finally:
        server.terminate()
        server.wait(timeout=15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/evaluation")
    parser.add_argument(
        "--setup-only", action="store_true", help="prepare the database and stop (debugging)"
    )
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    database = f"{settings.database_name}_clinical"
    env = {**os.environ, "DATABASE_NAME": database, "LOG_LEVEL": "WARNING"}
    setup_database(database, env)
    if args.setup_only:
        return 0 if all(c["passed"] for c in CHECKS) else 1

    report = asyncio.run(evaluate(database))
    gates = report["safety_gates"]
    for gate in gates:
        check(f"safety gate: {gate['gate']}", gate["passed"], gate["value"])

    server = start_server(env, PORT)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=60) as http:
            api = http_scenarios(http, report)
            check(
                f"API consistency: all {api['scenarios']} scenarios via POST /api/v1/icd/suggest "
                "match the service-level run",
                not api["mismatches"],
                {"statuses": api["http_statuses"], "mismatches": api["mismatches"][:5]},
            )
            http_checks(http)
    finally:
        server.terminate()
        server.wait(timeout=15)
    degraded_semantic_checks(env)

    report["api_verification"] = {**api, "checks": CHECKS}
    (output / "clinical_evaluation_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (output / "clinical_evaluation_report.md").write_text(markdown(report), encoding="utf-8")
    failed = [c for c in CHECKS if not c["passed"]]
    print(
        json.dumps(
            {
                "database": database,
                "checks": len(CHECKS),
                "passed": len(CHECKS) - len(failed),
                "failed": [c["check"] for c in failed],
                "safety_passed": report["safety_passed"],
                "scenarios_passed": report["final_selection"]["scenarios_passed"],
                "scenarios": report["final_selection"]["scenarios"],
                "report": str((output / "clinical_evaluation_report.json").relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 1 if failed or not report["safety_passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
