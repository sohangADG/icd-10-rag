# Runtime verification

Recorded on 2026-10-03; re-verified on 2026-10-04 after the pre-commit review, branch `feature/complete-icd-rag` (based on `main` @ `1717744`), using
Docker Compose with `pgvector/pgvector:pg17` and the app image (Python 3.12).

## 1. Commands

```bash
docker compose up -d db
docker compose build app
docker compose run --rm app pytest -v
docker compose run --rm app ruff check .
docker compose run --rm app ruff format --check .
docker compose run --rm app python -m scripts.runtime_e2e      # full E2E, own database
docker compose run --rm app python -m app.evaluation.cli run --synthetic --summary
```

## 2. Test suite

| Suite | Result |
|---|---|
| Phase 1 baseline, before changes | 38 passed |
| After implementation | 227 passed |
| After pre-commit review fixes | **276 passed, 0 failed, 0 skipped** (190 unit + 86 integration; 49 new regression tests), 1 third-party deprecation warning (Starlette TestClient/httpx) |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 137 files already formatted |

Two Phase 1 tests were deliberately updated:
- `test_suggestion_api_is_not_exposed` became `test_public_route_contract`. The suggest API is
  now in scope; the test still pins the exact route set and forbids upload/import routes.
- `test_embedding_requires_model_name` now sets `embedding_dimension`, so the new
  `embedding_has_dimension` check does not fire first. The intent is unchanged.

## 3. Docker runtime E2E (`scripts/runtime_e2e.py`): 48/48 checks passed

The PDF dry run reports 0 errors and 1 warning (`UNRESOLVED_CROSS_REFERENCE` for the deliberate
Z99 reference).

Flow: fresh database `icd_rag_e2e` → `alembic upgrade head` → synthetic sources → CLI
`validate` / `ingest --embed` → real `uvicorn` server → HTTP requests → direct SQL checks.

| Area | Checks |
|---|---|
| Migration | fresh DB at revision `0002` |
| Ingestion (CLI) | PDF dry run (55 records, 6 pages, 0 errors); PDF ingest + 55 embeddings; JSON 2025 ingest; identical re-ingest → `skipped`; malformed CSV → `validation_failed` (DUPLICATE_CODE, MALFORMED_CODE, MISSING_TITLE, MISSING_PARENT, HIERARCHY_CYCLE, HIERARCHY_LEVEL_INCONSISTENT, …); restricted PDF → `SOURCE_RESTRICTED`; re-index skips 55 unchanged embeddings |
| HTTP | `/health/db`; datasets list (2024 ready, 2025 ready, malformed validation_failed); search exact / lexical / fuzzy / hybrid with semantic; code, children, ancestors; version isolation (A01.3 only in 2025); 9 suggestion scenarios; error mapping (404/409/422, admin disabled); `X-Request-ID` |
| Suggestions | acute airway infection → A00.0 HIGH; no side → A01.9 LOW + missing `laterality`; left lung → A01.0 HIGH; newborn → B15 (A00 excluded); former smoker + family history → D00.2, D01; negated → none; T2 diabetes w/ kidney complication + CKD 3 → C00.20, C12.3; acute on chronic CHF → B01.2; multiple lobes (2025) → A01.3 |
| PostgreSQL | 3 dataset rows with SHA-256 + status + version; 55/55/0 records; no duplicate codes; no orphans; parents always same-dataset; every record has provenance; rules by type with resolved targets (EXCLUDE 4/4, USE_ADDITIONAL_CODE 4/4, CODE_FIRST 2/2, CODE_ALSO 1/1, SEE 2/1 — the unresolved one deliberately references Z99, which is not in the dataset —, SEE_ALSO 1/1, NOTE 2/0); 110 embeddings `hashing-v1-256`/256; HNSW index present; 3 pg_trgm indexes; every returned code (incl. alternatives) exists in its dataset; runs: completed 2, skipped 1, validation_failed 1 |

The development API container (`docker compose up -d app`, port 8000) was also called from the
host: `/health/db` returned `{"status":"ok","database":"connected","pgvector":true}`, and
`/api/v1/icd/datasets` returned the READY synthetic dataset with an `x-request-id` header.

## 4. Real ICD-10-CA 2022 PDF: licence guard

```
python -m app.ingestion.cli inspect --file data/sources/ICD10CA_2022_final.pdf      → exit 3
  page_count 2679, encrypted true, permission_value -3392,
  copy_extract false, extract_for_accessibility true, text_extraction_permitted false
python -m app.ingestion.cli validate --file data/sources/ICD10CA_2022_final.pdf ... → exit 3
  issues: SOURCE_RESTRICTED; records extracted: 0
```

Only the encryption dictionary and page tree were read. No page content was extracted.

## 5. Evaluation (synthetic, behaviour only)

See [evaluation.md](evaluation.md): final top-1 1.00, negative-case accuracy 1.00,
unsupported-code rate 0.00, rule-violation rate 0.00, retrieval recall@3 0.94.

## 6. Git / security

- `git check-ignore`: `.env`, all three `data/sources/*.pdf`, `data/synthetic/`, `.venv`,
  `.pytest_cache`, `.ruff_cache` are ignored. No PDF, XLSX, `.env` or synthetic output is tracked.
- `git diff --check`: clean.
- Secret scan (regex for API keys, AWS keys, private keys, tokens, password literals) over all
  changed and new files. The only match is the placeholder `"sk-secret"` passed to an HTTP mock
  transport in a unit test.
