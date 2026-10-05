# Reranking

`app/coding/reranker.py`. Every feature is returned in `retrieval_scores.rerank_features`.

```
rerank = 0.45·hybrid + 0.25·exact_terminology + 0.15·concept_overlap + 0.10·inclusion_match
       + 0.05·documented_specificity + 0.05·selectable
       − 0.20·unsupported_specificity − 0.10·less_specific_than_documented − 0.05·rule_warning
```

| Feature | Meaning |
|---|---|
| `hybrid` | Retrieval hybrid score ([retrieval.md](retrieval.md)) |
| `exact_terminology` | The concept (or an abbreviation expansion) equals the title or one of the record's own source terms, ignoring function words |
| `concept_overlap` | Best Jaccard overlap of concept words with the title or own terms |
| `inclusion_match` | The concept matches an inclusion term of the candidate |
| `documented_specificity` | The candidate's qualifiers (side, acuity, stage…) are documented |
| `selectable` | The candidate is a valid final code (leaf / selectable) |
| `unsupported_specificity` | The candidate needs a detail the note does not state |
| `less_specific_than_documented` | An "unspecified" variant although the detail *is* documented |
| `rule_warning` | A rule warning (e.g. another concept matches an exclusion) |

**Rejections are not penalties.** Exclusion conflicts, contradicted specificity and status-gate
failures remove the candidate (listed under `unmatched_concepts[].rejected_candidates` when
nothing else qualifies). Semantic similarity contributes only through `hybrid`, so a candidate
cannot win on vector similarity alone.

## Final selection

1. The highest accepted rerank wins. Ties prefer selectable codes, then code order.
2. Hierarchy-aware resolution (`SuggestionService._resolve`):
   - **supported**: use the candidate; if it is a non-selectable category, descend to the single
     child the documentation supports (often the "unspecified" one);
   - **unsupported**: go up to the classification parent and descend as above;
   - an "unspecified" result lists what the specific siblings would need (`missing_information`)
     and offers them as alternatives.
3. **Mandatory DB re-verification** (exists, same dataset, active). Otherwise the suggestion is
   dropped.

## Confidence

Confidence measures evidence and retrieval support, **not diagnostic certainty**.

- **HIGH**: rerank ≥ 0.55 and strong terminology evidence (exact, inclusion or source term, or
  lexical ≥ 0.8 together with fuzzy ≥ 0.8). There is no missing information, the resolution is
  direct, rules passed and the code is selectable.
- **MEDIUM**: rerank ≥ 0.35. Uncertain/suspected concepts are capped here.
- **LOW**: otherwise, and always when the final record is not selectable.
