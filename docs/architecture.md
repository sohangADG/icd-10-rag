# Architecture

## Overview

```
                         ┌──────────────── ingestion (CLI, operator) ────────────────┐
 source file ──► Source Adapter ──► Normalized records ──► Validators ──► Hierarchy │
 (PDF/text/CSV/TSV/XLSX/XML/JSON)        (app/ingestion/models.py)                  │
                                   └──► Transactional importer ──► Index generation ─┘
                                                     │                    │
                                                     ▼                    ▼
                                PostgreSQL: icd_datasets, icd_nodes, icd_terms, icd_rules,
                                icd_index_entries, icd_source_refs, icd_search_documents
                                (tsvector + pg_trgm + pgvector), icd_ingestion_runs/errors
                                                     ▲
 clinical note ──► Concept extraction ──► Hybrid retrieval ──► Rule engine + specificity
   (POST /suggest)  (status-aware)        (exact/lexical/fuzzy/  guard ──► Reranker ──►
                                           semantic/hierarchy/   hierarchy resolution ──►
                                           index terms)          DB re-verification ──► response
```

## Layers and modules

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api` | Routers (`v1/icd.py`, `v1/admin.py`), error mapping, request-id/latency middleware, dependencies |
| Services | `app/services`, `app/coding`, `app/ingestion/service.py` | Dataset resolution, suggestion orchestration, ingestion orchestration |
| Domain logic | `app/clinical`, `app/coding`, `app/retrieval`, `app/indexing` | Concept extraction, rules, specificity, reranking, hybrid retrieval, search documents, embeddings |
| Ingestion | `app/ingestion` | Adapters, normalized models, code utilities, hierarchy, validators, pipeline, importer, CLI |
| Data access | `app/repositories` | SQL only; no knowledge of source formats |
| Persistence | `app/models`, `migrations/` | ORM models, Alembic migrations |
| Evaluation | `app/evaluation`, `app/synthetic` | Metrics, runner, CLI, synthetic dataset and cases |

Boundaries that are kept on purpose:
- **Adapters know formats; repositories know tables.** Only the importer maps normalized models
  (`app/ingestion/models.py`) to ORM rows.
- **Validation policy is per coding system** (`app/ingestion/validator.py`). Coding systems are
  never mixed.
- **Relational data is authoritative.** Search documents and vectors are retrieval aids. Every
  returned code is re-read from `icd_nodes` before it is returned.

## Key decisions

| Decision | Rationale |
|---|---|
| Extend the Phase 1 schema (migration `0002`) instead of new parallel tables | `icd_nodes` is the ICD record and `icd_rules` holds exclusions and instructions. `icd_terms` holds inclusions, synonyms and abbreviations, and `icd_index_entries` holds index terms. No duplicate concepts. |
| Exclusion text stays out of the lexical index and the embedding input | A query must never match a code through what that code excludes |
| Parent synonyms are inherited by subdivisions as *context* (weight B) | Lets "acute heart failure" reach "Acute heart pump weakness". Nothing is copied as the child's own term. |
| Dimensionless `vector` column + per-(model, dim) partial HNSW expression index | The embedding model is configurable. Vectors from different models are never compared. |
| Default embedding provider: local feature hashing | Works offline and in tests, with no vendor lock-in. Real semantic models plug in via `EMBEDDING_PROVIDER`. |
| Rule-based concept extraction behind a `ConceptExtractor` protocol | Deterministic, explainable, and testable. Replaceable by NLP/LLM extractors without touching retrieval or validation. |
| Single long import transaction + short status transactions | No partial READY data. Failures roll back completely. Retries are safe. |
| No upload/import HTTP endpoint | Ingestion is an operator action where licence attestation is recorded (`docs/licensing.md`) |
| Redis/caching not added | Not part of the existing architecture. Lookups are indexed and bounded. Clinical text must never be cached anyway. |

See the topic documents for detail: [ingestion](ingestion.md), [source adapters](source-adapters.md),
[dataset versioning](dataset-versioning.md), [retrieval](retrieval.md), [reranking](reranking.md),
[coding validation](coding-validation.md), [API](api.md), [evaluation](evaluation.md),
[security](security.md), [licensing](licensing.md), [runtime verification](runtime-verification.md).
