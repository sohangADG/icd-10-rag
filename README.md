# icd-rag-service

A standalone, version-aware ICD coding RAG service.

> **Status: Phase 1 — foundation only.** Database schema, migrations, health checks and dataset
> registration exist. No ICD content is parsed, stored, embedded, retrieved or suggested yet.

## 1. Purpose

The service will eventually:

```
Clinical note
  → (external system) extracts a documented clinical concept
  → POST concept to this service
  → retrieve candidate ICD codes, validate them against structured ICD data
  → return explainable candidates
  → (consuming application) user accepts/rejects, stores the accepted code
```

The first target dataset is **ICD-10-CA 2022** (CIHI), sourced from `ICD10CA_2022_final.pdf`
(classification) and `canadian-coding-standards-2022-en.pdf` (coding rules).

## 2. Architecture boundaries

**This service owns:** ICD dataset ingestion, structured ICD knowledge storage, retrieval,
ranking, validation and explainable code suggestions.

**This service does NOT own:** clinical notes, patients, concept extraction, the human
accept/reject decision, or persistence of accepted codes. Those belong to the consuming
application. No external application is integrated.

Design principles enforced by the schema:

- **Every dataset is a first-class, versioned entity** (`icd_datasets`). Identity is
  `(system, country, version, revision, edition, language)`, unique with `NULLS NOT DISTINCT`.
- **Every knowledge row carries `dataset_id`.** Cross-row references (parent node, rule target,
  index target, provenance links…) use composite foreign keys `(dataset_id, x_id) → (dataset_id, id)`,
  so the database itself rejects any link between e.g. ICD-10-CA 2022 and ICD-10-CM rows.
- **Relational data is authoritative.** `icd_search_documents.embedding` is a retrieval aid only;
  a vector hit never establishes a valid code — codes are always re-verified against `icd_nodes`.
- **Provenance is preserved** (`icd_source_refs`, `source_page*` columns).
- **History is never deleted:** retired codes keep their row with `status = disabled`.
- **Codes are free text** (no fixed length): Canadian 5th/6th-character extensions fit.
- The **Alphabetical Index** (`icd_index_entries`, hierarchical) is stored separately from the
  **Tabular List** (`icd_nodes`, `icd_terms`, `icd_rules`, `icd_relationships`).

### Layout

```
app/
  api/            HTTP routers (health.py; v1/ reserved for future ICD endpoints, currently empty)
  core/           config, database engine/session, logging, constants/enums
  models/         SQLAlchemy ORM models (one table per module)
  repositories/   data access (dataset, health)
  schemas/        Pydantic request/response models
  services/       business logic (dataset registration)
  main.py         FastAPI app + lifespan
migrations/       Alembic environment + versioned migrations
scripts/          operational scripts (register_dataset.py)
tests/unit/       no database required
tests/integration/ real PostgreSQL + pgvector; schema built by Alembic
docker/           Dockerfile for the optional app/dev container
data/sources/     place licensed source PDFs here (git-ignored)
```

## 3. Technology stack

Python 3.12 · FastAPI · Pydantic v2 / pydantic-settings · SQLAlchemy 2 (async) · asyncpg ·
Alembic · PostgreSQL 15+ (17 recommended) · pgvector · pytest / pytest-asyncio · JSON logging.
Local development is **Windows-native**; Docker is optional (see §11).

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

The initial migration enables pgvector (`CREATE EXTENSION IF NOT EXISTS vector`) and creates all
tables, indexes and constraints. If pgvector is not installed in PostgreSQL this step fails with
`extension "vector" is not available` — install pgvector (§4) first.

```powershell
alembic upgrade head        # apply
alembic current             # show revision
alembic downgrade base      # roll back everything (leaves the vector extension installed)
alembic upgrade head --sql  # print the SQL without a database
```

## 7. Register the ICD-10-CA 2022 dataset

Creates the dataset **identity record only** (`status = pending`, no checksum). Nothing is parsed.
Idempotent: a second run reports `"created": false` and returns the same row.

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

## 10. Optional: Docker

`docker-compose.yml` and `docker/Dockerfile` are kept for optional/future deployment; local
development does not need them. With Docker Desktop available:

```bash
docker compose up -d db                                  # pgvector/pgvector:pg17
docker compose run --rm app alembic upgrade head
docker compose run --rm app pytest -v
docker compose up -d app                                 # http://127.0.0.1:8000
```

Inside Compose the `app` service overrides `DATABASE_HOST=db`; outside Docker the default is
`localhost`.

## 11. Embeddings (design note)

`icd_search_documents.embedding` is a **dimensionless** `vector` column because no embedding model
has been chosen. `embedding_model` records which model produced a vector (a CHECK forbids a vector
without it). When a model is selected, a new migration will:

1. `ALTER TABLE icd_search_documents ALTER COLUMN embedding TYPE vector(<dim>)`
2. create an HNSW index (`USING hnsw (embedding vector_cosine_ops)`), which requires a fixed dimension.

No embeddings are generated in Phase 1.

## 12. Phase 1 limitations

**NOT IMPLEMENTED YET:**

- PDF parser
- ICD ingestion
- chunk generation
- embeddings
- lexical/vector retrieval
- reranking
- coding rules engine
- suggestion API (`POST /api/v1/icd/suggest` does not exist)
- consuming application integration

All ICD knowledge tables are empty by design; the only data this phase writes is the
ICD-10-CA 2022 dataset identity record.
