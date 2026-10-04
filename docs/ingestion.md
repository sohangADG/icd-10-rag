# Ingestion

```
Source → Source Adapter → Normalized Records → Structural Validation → Hierarchy Builder
       → Rule Validation → Database Import (transactional) → Index Generation → READY
       → (optional) Embeddings
```

## Commands

```bash
# Format facts only (hash, pages, encryption/permissions). Reads no content.
python -m app.ingestion.cli inspect  --file SRC [--manifest M]
# Dry run: parse + validate + statistics, no database
python -m app.ingestion.cli validate --file SRC --manifest M [--show-issues 50]
# Validate, import transactionally, build search index (+ embeddings)
python -m app.ingestion.cli ingest   --file SRC --manifest M [--embed] [--licence-basis "..."]
# Rebuild search documents / embeddings of a READY dataset
python -m app.ingestion.cli index    --dataset-id N [--no-embed]
python -m app.ingestion.cli datasets
```

Exit codes: `0` ok, `1` validation/import failure, `2` usage/licence error, `3` source restricted.
Output is JSON on stdout; logs go to stderr.

## Manifest

```json
{
  "adapter": "csv",
  "dataset": {"coding_system": "ICD-10-CA", "version": "2022", "country": "CA",
              "language": "en", "publisher": "...", "edition": null,
              "licence": {"basis": "..."}},
  "mapping": {"columns": {"code": "Code", "title": "Description"}},
  "hierarchy": {"infer_parents": false}
}
```

Dataset identity comes from the manifest, CLI flags (`--coding-system`, `--version`, ...), or an
identity embedded in the source (e.g. a JSON header). **It is never guessed from a filename.**
The SHA-256 and the page count are measured from the file.

## Pipeline stages

| Stage | Module | Output |
|---|---|---|
| Adapter selection | `adapters/__init__.py` | Explicit `adapter`, else the most confident detector. OCR is never auto-selected. |
| Extraction | `adapters/*` | `NormalizedICDRecord`s with `SourceProvenance` (page/row/sheet/element path/line) |
| Structural validation | `validator.py` | Code format per system, missing code/title/level, normalized duplicates, provenance, cross-references, empty dataset |
| Hierarchy | `hierarchy.py` | Source-provided parents first. Inference is opt-in and reported. Duplicates, orphans, cycles and level/code consistency are checked. |
| Hierarchy validation | `validator.py` | Empty sections, non-selectable leaves |
| Import | `importer.py` | Batched inserts (parents before children), set-based cross-reference resolution |
| Indexing | `app/indexing/indexer.py` | One search document per entity, weighted tsvector |
| Embeddings | `indexer.embed_documents` | Only new/changed documents. Labelled with model + dimension. HNSW index ensured. |

## Statistics (`validate` / `ingest`)

`records_extracted, chapters, blocks, categories, subcategories, codes, inclusion_terms,
exclusion_terms, instructions, index_terms, selectable_codes, duplicate_codes, orphan_records,
hierarchy_cycles, inferred_parent_links, validation_errors/warnings/info, issue_codes,
pages_processed, lines_processed, unparsed_lines, two_column_pages, pages_with_printed_number`.

No expected counts are hard-coded for real datasets.

## Validation severities

- **ERROR** blocks the dataset from becoming READY: the dataset is set to `VALIDATION_FAILED`,
  issues are persisted in `icd_ingestion_errors`, and **no content is written**.
- **WARNING** (e.g. unresolved cross-reference, empty section, code outside the parent range) and
  **INFO** (e.g. inferred parent, unparsed front-matter lines) are persisted with the run.

## Transactions, idempotency, retries

See the docstring of `app/ingestion/importer.py`. Summary:

| Situation | Result |
|---|---|
| Same identity + same SHA-256, already READY | `skipped` (run recorded as SKIPPED); nothing changes |
| Same SHA-256 under another identity | `DuplicateDatasetError` |
| READY identity, different file | `DuplicateDatasetError`. Publish it as a new version/revision/edition. |
| Previous attempt FAILED / VALIDATION_FAILED / stale PROCESSING | Leftover content deleted, re-imported in one transaction |
| Concurrent import of the same identity | Serialised by `pg_advisory_xact_lock`; a recent RUNNING run → `ImportInProgressError` |
| Exception during import | Full rollback; dataset `FAILED`, run `FAILED` with the error class |

## PDF / text specifics

See [source-adapters.md](source-adapters.md#pdf). A layout profile must be validated with golden
samples against the actual source before its output is trusted.
