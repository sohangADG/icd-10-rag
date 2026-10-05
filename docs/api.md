# API

Base path `/api/v1/icd`. OpenAPI is at `/docs` and `/openapi.json`. Every response carries an
`X-Request-ID` header, which echoes a safe client value or is generated.

## Dataset selection
Query parameters (GET) or body fields (POST): `dataset_id`, or `coding_system` + `version`
[+ `country`, `language`].

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/health/db` | Liveness; readiness (PostgreSQL + pgvector) |
| GET | `/api/v1/icd/datasets` | List (`coding_system`, `version`, `country`, `language`, `status`, `limit≤500`, `offset`) |
| GET | `/api/v1/icd/datasets/{id}` | Detail: record counts by level, licence, latest ingestion run |
| GET | `/api/v1/icd/search` | `q` (1–200 chars), `mode=exact\|text\|hybrid`, `limit` (≤ `MAX_TOP_K`), `level`, `selectable_only` |
| GET | `/api/v1/icd/codes/{code}` | Record with hierarchy, inclusions, exclusions, instructions (with resolved target ids), synonyms, index terms, provenance |
| GET | `/api/v1/icd/codes/{code}/children` | Direct children (full records) |
| GET | `/api/v1/icd/codes/{code}/ancestors` | Root-first ancestors |
| POST | `/api/v1/icd/suggest` | Evidence-backed suggestions |
| POST | `/api/v1/admin/datasets/{id}/reindex` | Admin: rebuild documents + embeddings |
| POST | `/api/v1/admin/datasets/{id}/archive` | Admin: READY → ARCHIVED |

Admin endpoints return 404 unless `ADMIN_API_TOKEN` is set. Calls need an `X-Admin-Token` header,
which is compared in constant time. There is **no upload/import endpoint** (see
[licensing.md](licensing.md)).

## POST /suggest

```json
{"clinical_note": "Diagnosis: lobar consolidation of the left lung. Denies cough.",
 "coding_system": "SYNTH-ICD", "version": "2024", "top_k": 3,
 "country": null, "language": null, "dataset_id": null, "include_uncertain": null}
```
Unknown fields are rejected. `clinical_note` is limited to `CLINICAL_NOTE_MAX_CHARS`. The note is
processed in memory: it is **not stored and not logged**.

Response (abridged):
```json
{"dataset": {"id": 1, "coding_system": "SYNTH-ICD", "version": "2024", "status": "ready", ...},
 "clinical_concepts": [{"text": "lobar consolidation of the left lung", "status": "documented",
                        "concept_type": "diagnosis", "attributes": {"laterality": "left", ...},
                        "start": 11, "end": 47, "cues": [], "expansions": []}, ...],
 "suggestions": [{
   "clinical_concept": "lobar consolidation of the left lung",
   "concept_status": "documented", "concept_type": "diagnosis",
   "code": "A01.0", "title": "Lobar consolidation of left lung", "record_id": 17,
   "evidence": [{"type": "clinical_text", "text": "...", "start": 11, "end": 47},
                {"type": "matched_term", "text": "Lobar consolidation of left lung",
                 "match_type": "title", "score": 0.8857},
                {"type": "documented_specificity", "details": ["laterality=left"]}],
   "icd_reference": {"record_id": 17, "dataset_id": 1, "coding_system": "SYNTH-ICD",
                     "version": "2024", "code": "A01.0", "level": "SUBCATEGORY",
                     "is_selectable": true, "hierarchy": [...]},
   "alternatives": [{"code": "A01.9", "reason": "Alternative candidate", ...}],
   "missing_information": [],
   "confidence": "HIGH",
   "retrieval_scores": {"exact": 0.0, "lexical": 0.97, "fuzzy": 0.89, "semantic": 0.63,
                        "hierarchy": 0.75, "index_term": 0.0, "hybrid": 0.55, "rerank": 0.79,
                        "rerank_features": {...}},
   "validation": {"db_verified": true, "dataset_match": true, "selectable": true,
                  "resolution": "direct", "rule_status": "passed", "rule_checks": [...],
                  "specificity": {"status": "supported", "matched": ["laterality=left"], ...}},
   "source_provenance": {"source_filename": "...", "page_start": 2, "printed_page": "1",
                         "locator": {...}}}],
 "unmatched_concepts": [{"clinical_concept": "cough", "concept_status": "negated",
                         "reason": "Concept is negated in the documentation; ..."}],
 "confidence_note": "Confidence expresses ... It is not diagnostic certainty. ...",
 "duration_ms": 42}
```

## Errors

Every error has the same shape: `{"error": {"code", "message", "details", "request_id"}}`.
Submitted values are never echoed back.

| Code | HTTP |
|---|---|
| `REQUEST_VALIDATION_ERROR`, `INVALID_CLINICAL_NOTE` | 422 |
| `DATASET_NOT_FOUND`, `UNSUPPORTED_CODING_SYSTEM`, `UNSUPPORTED_DATASET_VERSION`, `INVALID_ICD_CODE`, `ADMIN_DISABLED` | 404 |
| `DATASET_NOT_READY`, `AMBIGUOUS_DATASET`, `DUPLICATE_DATASET`, `IMPORT_IN_PROGRESS` | 409 |
| `UNAUTHORIZED` | 401 |
| `LICENCE_RESTRICTION` | 403 |
| `RETRIEVAL_ERROR`, `EMBEDDING_PROVIDER_ERROR` | 503 |
| `INTERNAL_ERROR` | 500 (no internals) |
