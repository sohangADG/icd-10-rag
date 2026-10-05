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

## 7. Semantic embeddings (2026-10-04)

Real local model: `sentence_transformers` / `BAAI/bge-small-en-v1.5`, **384 dimensions read from
the model**, CPU, normalised, cosine distance. The model is cached in the `hf-cache` Docker
volume; the image has CPU-only torch 2.14.1, sentence-transformers 6.1.0, and is 2.22 GB.

| Check | Result |
|---|---|
| Default suite (hashing, deterministic, offline) | **325 passed, 5 skipped** (the opt-in real-model tests); 330 tests = 215 unit + 115 integration. Stable over 3 consecutive runs. |
| Real-model suite (`RUN_SEMANTIC_MODEL_TESTS=1 pytest -m semantic_model`) | **5 passed** |
| Runtime E2E (`scripts/runtime_e2e.py`, configured provider = sentence_transformers) | **53/53**, on two consecutive fresh-database runs |
| Migration | fresh DB at `0003`; dev DB upgraded 0002 → 0003 |

E2E semantic flow: paraphrase source → ingest → READY → 17 logical documents → real embeddings
(17 embedded, 0 failed, embed 447 ms, write 84 ms) → pgvector + HNSW (`vector_cosine_ops`,
provider/model/dimension predicate) → HTTP `/search`: 13/14 paraphrase queries return the target
first, with zero lexical overlap (e.g. "hypertension" → P00 *Elevated arterial pressure disorder*,
semantic 0.38, raw cosine 0.75, lexical 0) → HTTP `/suggest` "Assessment: epilepsy. Denies hay
fever." → **P07** *Recurrent seizure condition* (DB-verified; "hay fever" negated, not coded).
A forced re-index regenerated 55 vectors in 4 batches; the normal re-index skipped all 55 as
unchanged.

### Provider comparison (`python -m app.evaluation.cli compare`), retrieval stage, k = 3

Paraphrase suite (14 queries, no word shared with the target title):

| Mode | Provider | Recall@3 | MRR | Top-1 |
|---|---|---|---|---|
| lexical only | – | 0.071 | 0.071 | 0.071 |
| vector only | hashing | 0.143 | 0.186 | 0.071 |
| hybrid | hashing | 0.143 | 0.193 | 0.071 |
| vector only | sentence_transformers | **1.000** | **0.964** | **0.929** |
| hybrid | sentence_transformers | **1.000** | **0.964** | **0.929** |

Lexical-friendly synthetic suite (16 positive cases):

| Mode | Provider | Recall@3 | MRR | Top-1 |
|---|---|---|---|---|
| lexical only | – | 0.938 | 0.877 | 0.812 |
| vector only | hashing | 0.812 | 0.776 | 0.688 |
| hybrid | hashing | 0.938 | 0.877 | 0.812 |
| vector only | sentence_transformers | 0.938 | 0.919 | 0.875 |
| hybrid | sentence_transformers | 0.938 | 0.887 | 0.812 |

Full suggestion pipeline (reranking, rules, specificity, evidence gate, DB verification) with the
real model:
- **Synthetic suite:** final top-1 1.00, negative-case accuracy 1.00, unsupported-code rate 0.
- **Paraphrase suite:** final top-1 0.786. The other 3 cases (anaemia, eczema, insomnia) are
  **abstentions** with weak evidence; no wrong code was returned. Unsupported-code rate 0.

### Timings (tiny synthetic data; architecture check only, not production figures)

| Operation | sentence_transformers (CPU) | hashing |
|---|---|---|
| Embed 55 documents | 1.35 s (+0.13 s DB write) | 0.03 s (+0.14 s write) |
| Embed 17 documents | 0.36 s (+0.10 s write) | 0.003 s (+0.06 s write) |
| Vector-only query (embed query + pgvector) | mean 33–40 ms, p95 64–74 ms | mean 2.5 ms |
| Hybrid query (all components) | mean 17–23 ms, p95 25–36 ms* | mean 18–20 ms |
| Model load at startup (first run downloads ~130 MB) | ~36 s first time, then from cache | – |

\*Hybrid ran after vector-only on the same queries, so it was served by the query-embedding
cache. This shows the cache working; it is not a like-for-like latency comparison.

### Pre-commit review (2026-10-04)

- Fresh-database PostgreSQL checks:
  - 127 vectors, all `sentence_transformers` / `BAAI/bge-small-en-v1.5` / 384 / normalised;
  - 0 `vector_dims` mismatches;
  - 0 vectors without a content hash;
  - 0 un-embedded embeddable documents;
  - 0 duplicate documents;
  - 0 vector/record dataset mismatches;
  - 0 cross-dataset node links or rule targets;
  - one HNSW index (`vector_cosine_ops`, full space predicate), 3 pg_trgm indexes and the FTS GIN
    index;
  - Alembic at `0003` with both new CHECK constraints.
- Migration 0003, tested in `tests/integration/test_migration_0003.py`:
  - upgrading a populated 0002 database backfills provider and normalisation;
  - the dimension and space constraints reject invalid rows;
  - downgrade to 0002 keeps the data, and re-upgrade works.
- Docker: `docker compose build` succeeds. The image has CPU-only torch `2.14.1+cpu` and
  sentence-transformers 6.1.0. The model lives in the external `hf-cache` volume (129 MB), and
  git tracks no model files.
