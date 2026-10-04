# icd-rag-service

A standalone, version-aware, evidence-backed ICD coding RAG backend.

> **Status.** The system is implemented and verified end to end with **synthetic** data:
> multi-format ingestion, validation, versioned PostgreSQL storage, hybrid retrieval (full-text
> + trigram + pgvector), clinical concept extraction, rule and specificity validation, reranking,
> and a suggestion API that re-verifies every code against the database.
> **ICD-10-CA 2022 production data is pending a licensed dataset.** The CIHI PDF available
> locally is copy-protected and its Terms of Use forbid extraction, so the system refuses it.
> See [docs/licensing.md](docs/licensing.md).

## 1. What it does

```
licensed source file ─► adapters (PDF/text/CSV/TSV/XLSX/XML-ClaML/JSON) ─► normalized records
   ─► validation (per coding system) ─► hierarchy ─► transactional import ─► search index
clinical note ─► concept extraction (negation/uncertainty/history aware) ─► hybrid retrieval
   ─► rules (excludes/includes/instructions) ─► specificity guard ─► rerank
   ─► database re-verification ─► suggestions with evidence, alternatives, missing information
```

| | |
|---|---|
| Supported classifications | Validators for **ICD-10**, **ICD-10-CA**, **ICD-10-CM**, and the synthetic **SYNTH-ICD**. More systems can be added through `register_validator`. |
| Target version | ICD-10-CA **2022** (pending a licensed source). Any number of versions, languages and editions can coexist. |
| Source formats | Text-layer PDF, optional OCR PDF (explicit only), structured text, CSV, TSV, XLSX, XML (generic + WHO ClaML), JSON |
| Retrieval | Exact code, PostgreSQL full-text (weighted), pg_trgm fuzzy, source index terms, pgvector semantic, hierarchy context, configurable hybrid weights |
| API | `/api/v1/icd/{datasets, search, codes, suggest}` plus token-protected admin endpoints |

**Boundaries.** This service owns ICD dataset ingestion, structured storage, retrieval,
validation, concept extraction for coding, and explainable suggestions. It does **not** store
clinical notes. Suggestions are decision support, and a qualified coder makes the final choice.
"Confidence" measures evidence and retrieval support, not diagnostic certainty.

## 2. Documentation

| Topic | Document |
|---|---|
| Architecture & decisions | [docs/architecture.md](docs/architecture.md) |
| Ingestion pipeline, CLI, idempotency | [docs/ingestion.md](docs/ingestion.md) |
| Source adapters & mappings (PDF details) | [docs/source-adapters.md](docs/source-adapters.md) |
| Dataset identity, versions, lifecycle | [docs/dataset-versioning.md](docs/dataset-versioning.md) |
| Retrieval & embeddings | [docs/retrieval.md](docs/retrieval.md) |
| Reranking & confidence | [docs/reranking.md](docs/reranking.md) |
| Rules, specificity, hallucination guard | [docs/coding-validation.md](docs/coding-validation.md) |
| HTTP API | [docs/api.md](docs/api.md) |
| Evaluation framework | [docs/evaluation.md](docs/evaluation.md) |
| Security | [docs/security.md](docs/security.md) |
| Licensing (read before ingesting real data) | [docs/licensing.md](docs/licensing.md) |
| Verified runtime results | [docs/runtime-verification.md](docs/runtime-verification.md) |

### Layout

```
app/
  api/            routers (health, v1/icd, v1/admin), errors, middleware, deps
  clinical/       rule-based concept extraction (status, attributes, abbreviations)
  coding/         rules engine, specificity guard, reranker, suggestion service
  core/           config, database, logging (request ids), constants, exceptions, text utils
  evaluation/     metrics, runner, CLI
  indexing/       search documents, embedding providers
  ingestion/      adapters/, text/ (PDF pipeline), models, codes, hierarchy, validator,
                  pipeline, importer, service, cli
  models/         SQLAlchemy ORM
  repositories/   SQL access (datasets, records/hierarchy, search, ingestion writes)
  retrieval/      hybrid retriever
  schemas/        API models
  services/       dataset resolution, presenters
  synthetic/      original synthetic ICD-like dataset, renderers for every format, eval cases
migrations/       Alembic (0001 Phase 1 schema, 0002 ingestion/retrieval, 0003 embedding spaces)
scripts/          register_dataset.py, runtime_e2e.py
tests/unit/       no database required
tests/integration/ real PostgreSQL + pgvector; schema built by Alembic
data/sources/     licensed source files (git-ignored)
data/synthetic/   generated synthetic sources (git-ignored)
```

## 3. Technology stack

Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) · asyncpg · Alembic ·
PostgreSQL 15+ (17 recommended) · pgvector · pg_trgm · pdfplumber/pdfminer.six · openpyxl ·
defusedxml · httpx · pytest. **Docker Compose is the verified workflow** (§10). Windows-native
setup (§4–§9) is also supported where PostgreSQL + pgvector can be installed.

## 4. Prerequisites (Windows)

1. **Python 3.12+** — e.g. `winget install Python.Python.3.12` (provides the `py` launcher).
2. **PostgreSQL 15+** installed on Windows, listening on `localhost:5432`
   (e.g. the EDB installer: `winget install PostgreSQL.PostgreSQL.17`). Version 15+ is required
   for the `NULLS NOT DISTINCT` dataset-identity constraint.
3. **pgvector** installed into that PostgreSQL instance. There is no official Windows installer;
   the supported route is building it with the Visual Studio C++ build tools, from an
   **"x64 Native Tools Command Prompt for VS 2022" run as Administrator**:

   ```bat
   set "PGROOT=C:\Program Files\PostgreSQL\17"
   cd %TEMP%
   git clone --branch v0.8.7 https://github.com/pgvector/pgvector.git
   cd pgvector
   nmake /F Makefile.win
   nmake /F Makefile.win install
   ```

   Pick the latest release tag from https://github.com/pgvector/pgvector/tags. You do not run
   `CREATE EXTENSION` yourself: the first Alembic migration does it.

> **Smart App Control (Windows 11).** If Smart App Control is *On*, Windows blocks DLLs that are
> not signed by a trusted publisher. This has been observed to block the EDB PostgreSQL 17.11
> binaries (`initdb.exe` loading `libpq.dll` fails with exit code `0xC0E90002`,
> `STATUS_SYSTEM_INTEGRITY_POLICY_VIOLATION`), and it will also block a locally built, unsigned
> pgvector `vector.dll`. Check *Windows Security → App & browser control → Smart App Control*.
> Note that once Smart App Control is turned off, Windows cannot turn it back on without a
> reset/reinstall; alternatively use PostgreSQL on another host via `DATABASE_HOST`/`DATABASE_URL`.

## 5. Environment setup (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
copy .env.example .env
# edit .env: DATABASE_PASSWORD = the password you chose for the PostgreSQL `postgres` user
```

If PowerShell refuses to run `activate` (execution policy), either use `cmd.exe`
(`.venv\Scripts\activate.bat`) or call tools directly: `.venv\Scripts\python.exe -m pytest`.

Create the application database once (the `psql` client ships with PostgreSQL):

```powershell
& "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -U postgres -h localhost icd_rag
```

All settings are environment variables (see `.env.example`): `APP_NAME`, `APP_ENV`, `APP_HOST`,
`APP_PORT`, `LOG_LEVEL`, `LOG_FORMAT` (`json`|`console`), `DATABASE_HOST`, `DATABASE_PORT`,
`DATABASE_NAME`, `DATABASE_USER`, `DATABASE_PASSWORD`, or a single `DATABASE_URL`
(e.g. `postgresql://postgres:<password>@localhost:5432/icd_rag`) that overrides the parts.
`.env` is git-ignored; the password is held as a `SecretStr` and never logged.

## 6. Migrations

`0001` enables pgvector (`CREATE EXTENSION IF NOT EXISTS vector`) and creates the Phase 1 tables.
`0002` enables `pg_trgm` and extends the schema in place for ingestion, provenance and
retrieval. Existing rows are backfilled (`normalized_code`, the status mapping
`ingesting → processing`), and its downgrade refuses to run if it would have to invent page
numbers. If pgvector is not installed in PostgreSQL, the first step fails with
`extension "vector" is not available`; install pgvector (§4) first.

```powershell
alembic upgrade head        # apply
alembic current             # show revision
alembic downgrade base      # roll back everything (leaves the vector extension installed)
alembic upgrade head --sql  # print the SQL without a database
```

## 7. Register the ICD-10-CA 2022 dataset

Creates the dataset **identity record only** (`status = pending`, no checksum). Nothing is parsed.
Idempotent: a second run reports `"created": false` and returns the same row. Importing content is
done with the ingestion CLI (§11) once a licensed source is available. The importer reuses this
identity row.

```powershell
python -m scripts.register_dataset
```

## 8. Run the application

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

| Endpoint         | Purpose                                                    |
|------------------|------------------------------------------------------------|
| `GET /health`    | Liveness. `{"status":"ok","service":"icd-rag-service"}`    |
| `GET /health/db` | Readiness: DB connection + pgvector installed. 503 if not. |
| `/api/v1/icd/*`  | Datasets, search, code lookup/hierarchy, suggestions (§12, [docs/api.md](docs/api.md)) |
| `/docs`          | OpenAPI UI                                                 |

If the database is unreachable at startup, the app logs an error and still starts so `/health`
works; `/health/db` returns `503` until the database is available.

## 9. Tests and lint

```powershell
pytest -v                   # everything
pytest tests/unit -v        # no database needed
pytest -m integration -v    # needs PostgreSQL + pgvector
ruff check .
ruff format --check .
```

Integration tests create a separate database `<DATABASE_NAME>_test` (the DB role needs
`CREATEDB`), build it with `alembic upgrade head` (not `create_all`), run every test inside a
rolled-back transaction, and drop the database afterwards. A further test builds a scratch database
to verify `upgrade head → downgrade base → upgrade head`, and another asserts that Alembic
autogenerate finds no drift between the ORM models and the migrations. If PostgreSQL is
unreachable, integration tests **fail** (they are never silently skipped).

## 10. Docker (verified workflow)

```bash
docker compose up -d db                                  # pgvector/pgvector:pg17
docker compose build app
docker compose run --rm app alembic upgrade head
docker compose run --rm app pytest -v
docker compose run --rm app ruff check . && docker compose run --rm app ruff format --check .
docker compose run --rm app python -m scripts.runtime_e2e   # full E2E on its own database
docker compose run --rm -e RUN_SEMANTIC_MODEL_TESTS=1 app pytest -m semantic_model  # real model
docker compose up -d app                                 # http://127.0.0.1:8000/docs
```

Inside Compose the `app` service overrides `DATABASE_HOST=db`; outside Docker the default is
`localhost`.

## 11. Ingesting data

**Source files.** Put licensed source files in `data/sources/` (git-ignored). Restricted sources
may be ingested only when the operator has the rights to do so
([docs/licensing.md](docs/licensing.md)). Encrypted PDFs whose permissions forbid extraction are
refused, and OCR is never used automatically.

```bash
# Synthetic sources in every format, two versions, plus malformed and restricted samples
python -m app.synthetic.cli build --out data/synthetic

# Inspect: hash, pages, encryption/permissions. No content is read.
python -m app.ingestion.cli inspect --file data/synthetic/synth_2024_pdf.pdf

# Validate (dry run, no database): statistics + issues
python -m app.ingestion.cli validate --file data/synthetic/synth_2024_pdf.pdf \
    --manifest data/synthetic/synth_2024_pdf.pdf.manifest.json

# Import into PostgreSQL (transactional, idempotent) + search index + embeddings
python -m app.ingestion.cli ingest --file data/synthetic/synth_2024_json.json \
    --manifest data/synthetic/synth_2024_json.json.manifest.json --embed

python -m app.ingestion.cli datasets
python -m app.ingestion.cli index --dataset-id 1        # rebuild documents/embeddings
```

`ingest` refuses to run without a recorded licence basis (`dataset.licence.basis` in the manifest
or `--licence-basis "..."`). Manifests and field mappings for every format are described in
[docs/source-adapters.md](docs/source-adapters.md).

**When the licensed ICD-10-CA dataset arrives:** write a manifest (`coding_system: ICD-10-CA`,
`version: 2022`, `country: CA`, `language`, `publisher`, licence basis) and a column/element
mapping for the delivered format. Then run `validate`, review the statistics and issues, run the
golden-sample comparison (licensed-dataset tasks in the implementation report), and `ingest`.

## 12. Search and suggestions

```bash
curl "http://127.0.0.1:8000/api/v1/icd/search?coding_system=SYNTH-ICD&version=2024&q=acute%20heart%20failure"
curl "http://127.0.0.1:8000/api/v1/icd/codes/A00?coding_system=SYNTH-ICD&version=2024"
curl -X POST http://127.0.0.1:8000/api/v1/icd/suggest -H "Content-Type: application/json" \
  -d '{"clinical_note":"Diagnosis: lobar consolidation of the left lung. Denies cough.","coding_system":"SYNTH-ICD","version":"2024","top_k":3}'
```

See [docs/api.md](docs/api.md) for the full request/response contract and error codes.

## 13. Evaluation

```bash
python -m app.evaluation.cli run --synthetic --summary
python -m app.evaluation.cli run --cases cases.jsonl --out report.json
```

Metrics are reported separately for concept extraction, retrieval, reranking and final selection
([docs/evaluation.md](docs/evaluation.md)). Synthetic results validate **system behaviour only**.

## 14. Configuration

All settings are environment variables (see `.env.example`). Beyond the database settings (§5):
`EMBEDDING_PROVIDER` (`sentence_transformers` default | `openai` | `hashing` | `none`),
`EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`), `EMBEDDING_DIMENSION` (optional, validated
against the model), `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_NORMALIZE`,
`EMBEDDING_DISTANCE`, `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_SIMILARITY_FLOOR`,
`EMBEDDING_QUERY_CACHE_SIZE`, `SEMANTIC_RETRIEVAL_MODE` (`optional`|`required`),
`EMBEDDING_API_BASE_URL`, `EMBEDDING_API_KEY`, `RETRIEVAL_WEIGHT_*`,
`RETRIEVAL_CANDIDATES_PER_METHOD`, `RETRIEVAL_FUZZY_THRESHOLD`, `MAX_TOP_K`,
`CLINICAL_NOTE_MAX_CHARS`, `SUGGEST_UNCERTAIN_CONCEPTS`, `SUGGESTION_MIN_EVIDENCE`,
`SUGGESTION_MIN_LEXICAL_EVIDENCE`, `ADMIN_API_TOKEN`, `LOG_CLINICAL_TEXT` (debug only; refused
in production). Optional extra: `pip install '.[ocr]'`.

### Semantic embeddings

The default provider runs a local sentence-transformers model (`BAAI/bge-small-en-v1.5`, 384 dimensions, read from the model) on CPU. Outside Docker, install
CPU-only PyTorch first (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
The model is downloaded on first use into the Hugging Face cache (`HF_HOME`; in Docker the
`hf-cache` volume) and never committed. To switch model or provider, change the configuration
and rebuild embeddings without re-importing: `python -m app.ingestion.cli index --dataset-id N
[--provider ...] [--model ...] [--batch-size ...] [--force]`. Compare providers on synthetic data
with `python -m app.evaluation.cli compare --coding-system SYNTH-ICD --version paraphrase-1`. The
test suite uses the deterministic `hashing` provider; the real-model tests run with
`RUN_SEMANTIC_MODEL_TESTS=1`. Details: [docs/retrieval.md](docs/retrieval.md).

## 15. Known limitations

- **No real ICD-10-CA content.** Real extraction accuracy and coding accuracy are unmeasured
  until a licensed dataset is supplied. All results so far come from synthetic data.
- **The PDF layout profile is generic** (WHO tabular-list conventions). A real publication needs
  its own profile and golden-sample validation. Layout heuristics have limits: wrapped titles
  continue only after a comma, a connector word or an open parenthesis.
- **Concept extraction is rule-based** (lexicons and cue phrases). It does not resolve
  coreference or temporality beyond simple phrases, and it does not understand complex syntax.
  Laterality needs an anatomical word within three words. Negation/uncertainty/history scope
  ends at "but/however" or a new statement ("the patient has…"); other long-range scopes may
  be missed.
  The `ConceptExtractor` protocol allows an NLP/LLM replacement.
- **The default semantic model is a small general-English model** (`bge-small-en-v1.5`), not a
  medical one. On the synthetic paraphrase set it gets top-1 0.93 retrieval. Near-ties occur
  (e.g. "insomnia"), and those are abstained on by the evidence gate. The similarity floor (0.6)
  is calibrated for this model on synthetic data and must be re-measured for any other model. A
  medical-domain model can be substituted by configuration. Remote providers require licence
  permission.
- **One active embedding space per dataset**: switching model or provider re-embeds the dataset.
  Vectors of two models are never kept side by side for the same document.
- **The rule engine** matches exclusions by word overlap (≥ 75%). It does not apply
  dagger/asterisk pairing, sequencing logic or national coding standards (e.g. CIHI Canadian
  Coding Standards are not ingested).
- **ClaML `ModifierClass` expansion** is not supported.
- **No Redis cache or rate limiter** is included. Hooks are in place (`rate_limit_hook`).
- **The `icd_index_entries` hierarchy** (lead term → modifiers) is stored flat for source index
  terms.
