# Dataset versioning

## Identity

A dataset is one release: `(system, country, version, revision, edition, language)`, unique with
`NULLS NOT DISTINCT`. For example, ICD-10-CA 2022 English and ICD-10-CA 2022 French are two
datasets. The API calls `system` `coding_system`. `modification` can name a national
modification separately.

Every knowledge row carries `dataset_id`. Cross-row references use composite foreign keys
`(dataset_id, x_id) → (dataset_id, id)`: parent, chapter, block, category, rule target, index
target and provenance. **PostgreSQL itself rejects any link between two datasets**
(tested in `test_database_rejects_cross_dataset_hierarchy_shortcuts`).

## Source identity

- `source_checksum` = SHA-256 of the source file. It is unique across datasets: one file backs at
  most one identity.
- `source_type`, `source_filename`, `source_identifier`, `source_uri`, `source_page_count`.
- `metadata.licence` records the operator's licence basis (`docs/licensing.md`).

## Lifecycle

```
PENDING → PROCESSING → INDEXING → READY → ARCHIVED
              ├→ VALIDATION_FAILED   (fatal validation errors, no content written)
              └→ FAILED              (unexpected error, transaction rolled back)
```

Only READY datasets are served. PROCESSING/INDEXING happen inside the import transaction, so
other sessions never see partially imported content. `icd_ingestion_runs.metadata.status_history`
records the transitions.

Phase 1's `ingesting` status was mapped onto `processing` by migration `0002`. The downgrade maps
it back.

## Publishing a new release

A new release is a **new dataset**: a different `version`, `revision` or `edition`. Re-importing
different bytes over a READY identity is refused, so history is never overwritten. Retired codes
can be kept with `status = disabled`.

## Selecting a dataset (API)

`dataset_id`, or `coding_system` + `version` (+ `country`, `language`). Errors are explicit:
`UNSUPPORTED_CODING_SYSTEM`, `UNSUPPORTED_DATASET_VERSION`, `DATASET_NOT_FOUND`,
`DATASET_NOT_READY` (incl. archived), `AMBIGUOUS_DATASET`.
