# Retrieval

## Retrieval units

One search document per logical ICD entity (`icd_search_documents`, `document_type =
icd_record`). Arbitrary chunks of PDF text are never embedded. Each document keeps `node_id`
(the record id) and `dataset_id`, and holds three views:

| View | Used for | Contents |
|---|---|---|
| `content` | display / context | code + title, description, hierarchy, inclusions, synonyms, abbreviations, index terms, inherited parent terms, notes, instructions, subdivisions, **exclusions (labelled)**, dataset |
| `search_vector` | lexical search | A = code, title, synonyms, abbreviations, index terms · B = inclusions, description, inherited parent terms · C = hierarchy path, notes, subdivision titles |
| `semantic_text` | embeddings | **positive evidence only**: code + title, description, hierarchy path, inclusions, synonyms/abbreviations/index terms, inherited parent terms, self-describing notes |

Exclusions, and instructions that name *other* conditions (code first, use additional code,
code also, see, see also), are kept out of both `search_vector` and `semantic_text`. A query
therefore never retrieves a record through a condition the record excludes or merely refers to.
They stay available to the rule engine ([coding-validation.md](coding-validation.md)).

## Components (normalised to absolute 0..1 scales: `app/retrieval/scoring.py`)

| Component | Raw signal | Normalised score | Index |
|---|---|---|---|
| `exact` | code / normalised code equality | 1 or 0; **always ranks first** | `ix_icd_nodes_dataset_id_(normalized_)code` |
| `lexical` | weighted FTS: coverage of the query lexemes by A/B lexemes, plus `ts_rank_cd` density (normalisation 32) | 0.75·coverage + 0.25·density | GIN on `search_vector` |
| `fuzzy` | pg_trgm similarity / word similarity against titles | (sim − t)/(1 − t), clamped, t = `RETRIEVAL_FUZZY_THRESHOLD` | GIN trgm on `icd_nodes.title` |
| `index_term` | equality (1.0) or trigram similarity against source inclusion terms, synonyms, abbreviations and index terms | same as fuzzy | GIN trgm on `icd_terms` / `icd_index_entries` |
| `semantic` | vector similarity in the provider's embedding space | (sim − floor)/(1 − floor), clamped; raw value exposed as `semantic_raw` | per-space partial HNSW |
| `hierarchy` | share of query words in the ancestor titles | already 0..1 (context only, never evidence on its own) | ancestor chains, batch-loaded |

**Why explicit normalisation matters.** The scales were measured on the synthetic suites:
- *Semantic baseline.* With the default model, unrelated documents score a raw cosine of about
  0.57 (median), and correct targets about 0.73.
- *Trigram noise.* Short queries share a few trigrams with almost any title, giving 0.1–0.3
  similarity.

Without the floors, these baselines would let a component win by scale alone. Measured on 14
paraphrase queries: before the trigram floor, hybrid top-1 was 0.57 although vector-only was
0.93. With the floors, hybrid top-1 is 0.93 as well.

```
hybrid = Σ wᵢ·scoreᵢ / Σ wᵢ   over available components
```

The weights are configurable (`RETRIEVAL_WEIGHT_*`; defaults exact 0.30, lexical 0.25, fuzzy
0.20, semantic 0.15, hierarchy 0.05, index_term 0.05), and at least one weight must be positive.
A component that is *unavailable* (`null`, e.g. semantic retrieval off) has its weight dropped.
A component that is present but has no signal counts as 0.

## Candidate generation

1. For every query variant (the concept, abbreviation expansions, history prefixes), each method
   returns at most `RETRIEVAL_CANDIDATES_PER_METHOD` records.
2. The union is the pool. Every component is then computed for the whole pool with
   `node_id = ANY(:pool)` queries (bounded, no N+1).
3. Nodes and their ancestor chains are loaded in two queries (recursive CTE).
4. Filters: node types, selectable only, active only. Results are cut to `top_k` (≤ `MAX_TOP_K`).

Candidates are always rows of the resolved dataset. Retrieval never creates codes.

## Embedding providers

`EmbeddingProvider` (`app/indexing/embeddings.py`) exposes `provider_name`, `model`,
`dimension` and `normalized`, plus `embed()` for documents and `embed_queries()` for queries.

| `EMBEDDING_PROVIDER` | Use | Notes |
|---|---|---|
| `sentence_transformers` (**default**) | Real local semantic embeddings | Any sentence-transformers compatible model (`EMBEDDING_MODEL`, default `BAAI/bge-small-en-v1.5`, 384 dimensions, MIT licence). Runs on `EMBEDDING_DEVICE` (`cpu`, `cuda`, `mps`). The dimension is **read from the model**. |
| `openai` | Any OpenAI-compatible `/embeddings` API | Needs `EMBEDDING_DIMENSION` and `EMBEDDING_API_KEY` (never logged). Remote: refused unless the dataset licence allows remote processing. |
| `hashing` | Deterministic tests / offline development | Lexical feature hashing, **not semantic**. The default test suite uses it. |
| `none` | Semantic component disabled | — |

**Switching providers or models is configuration only.** A medical-domain model can be dropped
in with `EMBEDDING_MODEL=<model>`, followed by an index rebuild. Retrieval code does not change.
Recalibrate `EMBEDDING_SIMILARITY_FLOOR` for the new model (see below).

### Model download and caching

Models are downloaded on first use by the normal Hugging Face mechanism into `HF_HOME`. In
Docker this is `/opt/hf-cache`, backed by the named volume `hf-cache`, so it survives
rebuilds. Models are never baked into the image or committed (`.gitignore`/`.dockerignore`
exclude `.cache/`, `hf-cache/` and `models/`). The API loads the model once at startup. If it
cannot be loaded (e.g. no network on the first download), the failure is logged and the service
still starts with semantic status `provider_unavailable` (see failure policy).

### Dimensions, normalisation, distance

- **Dimension.** It is taken from the model. `EMBEDDING_DIMENSION`, when set, must match it, or
  the provider fails at start-up. Every vector is length-checked before storage: by the
  provider, by the indexer, and by the database CHECK
  `vector_dims(embedding) = embedding_dimension` (migration 0003). Vectors are **never
  truncated or padded**; a mismatch fails clearly.
- **Normalisation.** With `EMBEDDING_NORMALIZE=true` (default), *every* document and query
  vector is L2-normalised in one place (`EmbeddingProvider._checked`), so no record can be
  normalised differently from another. The flag is stored with each vector and is part of the
  embedding space.
- **Distance.** `EMBEDDING_DISTANCE` is `cosine` (default), `l2` or `inner_product`. The HNSW
  operator class and the query operator are generated from the same `VectorSpace`
  (`vector_cosine_ops` with `<=>`, `vector_l2_ops` with `<->`, `vector_ip_ops` with `<#>`), so a
  cosine index is never queried with L2 or the other way round.

### Embedding spaces and isolation

An **embedding space** is (provider, model, dimension, normalisation). Each vector is stored
with all four (`embedding_provider`, `embedding_model`, `embedding_dimension`,
`embedding_normalized`). Every vector query filters by dataset **and** by the full space
predicate, which is the same predicate as the space's partial HNSW index:

```sql
CREATE INDEX ix_icd_search_documents_hnsw_<hash> ON icd_search_documents
USING hnsw (((embedding)::vector(384)) vector_cosine_ops)
WHERE embedding_provider = 'sentence_transformers' AND embedding_model = 'BAAI/bge-small-en-v1.5'
  AND embedding_dimension = 384 AND embedding_normalized = true;
```

Vectors from different models, providers, dimensions or normalisation settings are therefore
never compared. Semantic retrieval is used only when the space covers **every** embeddable
record document of the dataset. A partially embedded dataset reports `incomplete_index` rather
than favouring its embedded part.

### Content hashing and rebuilding

`embedding_content_hash = SHA-256(space key + semantic_text)`. A document is re-embedded when
its semantic text **or** the provider, model, dimension or normalisation changes, and is skipped
otherwise. Rebuilding never re-imports the dataset:

```bash
python -m app.ingestion.cli index --coding-system SYNTH-ICD --version 2024          # changed only
python -m app.ingestion.cli index --dataset-id 1 --force                            # everything
python -m app.ingestion.cli index --dataset-id 1 --provider sentence_transformers \
    --model BAAI/bge-base-en-v1.5 --batch-size 64 --device cpu                       # switch model
```

Embedding runs in batches of `EMBEDDING_BATCH_SIZE`. Each batch is written and committed on its
own, with progress logged. If a run is interrupted or the provider fails, only the current batch
is rolled back; the next run skips the committed batches. The command reports `documents`,
`embedded`, `skipped_unchanged`, `skipped_empty`, `failed`, `batches`, `embed_ms`, `write_ms` and
`duration_ms`. Documents with empty semantic text are skipped and excluded from coverage.

### Similarity floor (calibration)

`EMBEDDING_SIMILARITY_FLOOR` (default: 0.6 for sentence-transformers, 0.0 for hashing) is the raw
similarity at or below which the semantic score is 0. The 0.6 default comes from measurements of
`BAAI/bge-small-en-v1.5` on the synthetic suites:
- unrelated documents: median 0.565, 75th percentile 0.611;
- correct targets: median 0.725, minimum 0.643.

Measure and set it again whenever the model changes.

### Query embedding cache

`EMBEDDING_QUERY_CACHE_SIZE` (default 256) is an in-process LRU of query vectors keyed by
SHA-256 of (embedding space, normalised query). The query text itself is never stored or logged.
There is no Redis dependency.

### Failure policy (`SEMANTIC_RETRIEVAL_MODE`)

| Situation | `semantic_status` | `optional` (default) | `required` |
|---|---|---|---|
| `EMBEDDING_PROVIDER=none` | `disabled` | lexical/fuzzy/exact only | lexical/fuzzy/exact only (explicit configuration) |
| model failed to load | `provider_unavailable` | degrade, warning logged | `RETRIEVAL_ERROR` 503 |
| dataset has no vectors in this space | `not_indexed` | degrade | 503 |
| only part of the dataset embedded | `incomplete_index` | degrade | 503 |
| query embedding or vector search failed | `error` | degrade | 503 |

Degrading means the semantic weight is dropped and the other signals rank as usual. The status
is returned with every search and suggestion. Nothing is ever invented.

For suggestions, a minimum-evidence gate sits on top of retrieval
(`SUGGESTION_MIN_EVIDENCE`, `SUGGESTION_MIN_LEXICAL_EVIDENCE`). Concepts whose best candidate
has only weak evidence, such as a semantic near-tie, are abstained on rather than guessed. See
[coding-validation.md §3b](coding-validation.md#3b-minimum-evidence-gate-abstention).

All metrics in these documents come from **synthetic** data and validate system behaviour
only. They say nothing about real ICD-10-CA coding accuracy.

## Search API

`GET /api/v1/icd/search?q=...&mode=exact|text|hybrid&limit=&level=&selectable_only=` returns,
per hit, `scores` (exact, lexical, fuzzy, semantic, hierarchy, index_term, semantic_raw) and
`hybrid_score`. Per response it returns `semantic_status` and `embedding_space` (provider, model,
dimension, normalized, distance, similarity_floor). Vectors are never returned (see
[api.md](api.md)).
