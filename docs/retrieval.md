# Retrieval

## Retrieval units

One search document per logical ICD entity (`icd_search_documents`, `document_type =
icd_record`). Arbitrary PDF chunks are never embedded. Each document keeps `node_id` (the record
id) and `dataset_id`, and contains:

`code + title · description · hierarchy path · inclusion terms · source synonyms / abbreviations
/ index terms · parent terms (inherited context) · instructions/notes · subdivision titles ·
exclusions (labelled, display only) · dataset/version`

`metadata` holds the code, level, selectability, the structured hierarchy path, the dataset and
the exclusions.

## Components (all scores in [0, 1], all returned)

| Component | Implementation | Index |
|---|---|---|
| `exact` | code or normalized code equality (`A00.0` = `A000`) | `ix_icd_nodes_dataset_id_(normalized_)code` |
| `lexical` | weighted tsvector: A = code/title/synonyms/index terms; B = inclusions/description/parent terms; C = hierarchy/notes/children. Score = 0.5·normalised `ts_rank_cd` + 0.5·coverage of the query lexemes by A/B lexemes. The query is OR-ed lexemes (no all-words requirement). Exclusions are **not** indexed. | GIN on `search_vector` |
| `fuzzy` | pg_trgm `similarity` / `word_similarity` against official titles (misspellings, word order) | GIN trgm on `icd_nodes.title` |
| `index_term` | exact normalized equality (1.0) or trigram similarity against source inclusion terms, synonyms, abbreviations and index terms | GIN trgm on `icd_terms.normalized_term`, `icd_index_entries.normalized_term` |
| `semantic` | cosine similarity of pgvector embeddings, same model + dimension only | partial HNSW on `(embedding::vector(dim))` per model |
| `hierarchy` | share of query words found in the candidate's ancestor titles | computed from the batch-loaded ancestor chains |

Text search configuration follows the dataset language (`english`, `french`, ...; `simple`
otherwise).

## Hybrid score

```
hybrid = Σ wᵢ·scoreᵢ / Σ wᵢ   over available components
```

The weights are configurable (`RETRIEVAL_WEIGHT_*`; defaults 0.30 / 0.25 / 0.20 / 0.15 / 0.05 /
0.05). When the configured embedding model has no vectors for the dataset, `semantic` is `null`
and its weight is dropped, not counted as zero. At least one weight must be positive
(validated at startup). **An exact code match always ranks first**, whatever the weights.

## Candidate generation

1. For every query variant (the concept, plus abbreviation expansions and history prefixes),
   each method returns at most `RETRIEVAL_CANDIDATES_PER_METHOD` records.
2. The union is the pool. Every component is then computed for the whole pool with
   `node_id = ANY(:pool)` queries (bounded, no N+1).
3. Nodes and their ancestor chains are loaded in two queries (recursive CTE).
4. Filters: node types, selectable only, active only. Results are cut to `top_k` (≤ `MAX_TOP_K`).

Candidates are always rows of the resolved dataset. Retrieval never creates codes.

## Embeddings

`EmbeddingProvider` interface (`app/indexing/embeddings.py`):

| `EMBEDDING_PROVIDER` | Notes |
|---|---|
| `hashing` (default) | Local, deterministic feature hashing (words, bigrams, character trigrams). Captures lexical/sub-word similarity only. |
| `openai` | Any OpenAI-compatible `/embeddings` endpoint. Uses `EMBEDDING_API_BASE_URL` and `EMBEDDING_API_KEY` (never logged). **Remote:** refused unless the dataset licence allows remote processing. |
| `sentence-transformers` | Local model (`pip install '.[local-embeddings]'`) |
| `none` | Semantic component disabled |

Stored per document: `embedding_model`, `embedding_dimension`, `embedded_at`, and
`embedding_content_hash` (SHA-256 of model + dimension + embedded text). An embedding space is
identified by **(model, dimension)**, never by the model name alone: the HNSW index predicate,
the semantic query and the skip check all include the dimension. Unchanged documents are skipped
on re-index. Changing the model *or only the dimension* (e.g. OpenAI `dimensions`) re-embeds
everything and creates a new HNSW index for that (model, dimension), so vectors of different
spaces are never mixed or compared. The query vector's length is validated.

HNSW is limited to ≤ 2000 dimensions. Above that, search falls back to an exact scan. `hnsw.iterative_scan = relaxed_order` keeps filtered ANN queries complete
(pgvector ≥ 0.8).

## Search API

`GET /api/v1/icd/search?q=...&mode=exact|text|hybrid&limit=&level=&selectable_only=` (see
[api.md](api.md)).
