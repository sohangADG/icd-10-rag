# Clinical safety evaluation (Phase 3)

> **Synthetic only.** Every scenario and every classification record used here was written for
> this project. The results validate **system behaviour and safety**. They are **not** real
> ICD-10-CA coding accuracy, not coder agreement and not clinical diagnostic accuracy. The
> licensed ICD-10-CA dataset is still pending (see [licensing.md](licensing.md)).

## What is evaluated

`tests/evaluation/clinical_scenarios.json` holds **266 original synthetic clinical
scenarios**; 61 of them expect the system to abstain. They run against four READY synthetic
datasets:
- `SYNTH-ICD 2024` (232 scenarios), extended in Phase 3 with anatomical-site, encounter and
  cause codes (`E00`–`E10`);
- `SYNTH-ICD 2025` (10), with a deliberately added code, removed code, retitled code, changed
  parent, changed rule, changed inclusion and changed exclusion;
- `SYNTH-ICD paraphrase-1` (17), where queries share no word stem with titles;
- `SYNTH-ALT 2024` (7), a second coding system that reuses SYNTH-ICD code strings (`A00`,
  `A01.0`, `B00`) for other conditions and reuses one SYNTH-ICD title under another code.

Categories (tags): direct, paraphrase, multiple diagnoses, symptom only, suspected, ruled out,
negation, history, family history, laterality, conflicting laterality, acuity, severity,
subtype, complication, causal relationship, encounter, exclusion, code first, use additional
code, code also, see / see also, inclusion terms, parent/child fallback, no match, ambiguous,
typo, exact code query, version and coding-system isolation. Adversarial formatting covers:
- **Casing and spacing:** all caps, lowercase, missing punctuation, excess whitespace.
- **Layout:** bullets, semicolons, newlines, numbered lists, a long note.
- **Shorthand:** abbreviations and shorthand.
- **Repetition and noise:** duplicated sentences, copy-forward text, administrative text.
- **Contradictions:** contradictory documentation; a diagnosis in the assessment but negated in
  the review of systems.
- **Status-only mentions:** a diagnosis only in family history; a diagnosis only in past
  history.
- **Negation order:** several negatives before a positive; "denies X and Y but has Z".

### Scenario format

```json
{
  "id": "laterality-01",
  "clinical_note": "Lobar consolidation of the left lung.",
  "coding_system": "SYNTH-ICD", "version": "2024",
  "expected_codes": ["A01.0"],
  "acceptable_alternatives": [],
  "must_not_return": ["A01.1", "A01.2"],
  "expected_concepts": ["lobar consolidation of the left lung"],
  "expected_assertion_states": ["documented"],
  "expected_attributes": [{"laterality": "left"}],
  "required_missing_information": [],
  "should_abstain": false,
  "reason": "Documented left side.",
  "tags": ["laterality", "specificity", "rules"],
  "expected_instructions": ["CODE_FIRST"]
}
```

| Field | Meaning |
|---|---|
| `expected_concepts` / `expected_assertion_states` / `expected_attributes` | The concepts in the note. The list is exhaustive: unexpected concepts count as false positives. States and attributes are aligned with it. |
| `expected_codes` / `acceptable_alternatives` / `must_not_return` | Primary suggestions that are required, that are acceptable, and that are forbidden. A forbidden code is a safety-gate failure. |
| `should_abstain` | No code at all may be suggested. |
| `required_missing_information` | Substrings that must appear in a suggestion's `missing_information`. |
| `expected_instructions`, `expected_rejected`, `max_confidence`, `code_concepts`, `expected_retrieval_codes`, `include_uncertain` | Optional, more specific expectations (instructions surfaced, candidates that must not survive validation, confidence cap, code-to-concept association, retrieval target, uncertainty policy). |

Expectations encode **correct, safe** behaviour, not the system's output. A unit test checks
that every expected and acceptable code exists in the scenario's synthetic dataset.

## How to run

```bash
# Complete fresh-database run in Docker (migrate -> ingest -> validate -> READY -> semantic
# index -> evaluation -> every scenario over HTTP + error/degradation checks -> reports)
docker compose run --rm app python -m scripts.clinical_evaluation_e2e
#   -> data/evaluation/clinical_evaluation_report.json and .md (git-ignored)

# Against an existing database
python -m app.evaluation.cli run --dataset tests/evaluation/clinical_scenarios.json \
    --coding-system SYNTH-ICD --version 2024 --output report.json --markdown report.md \
    --top-k 5 --embedding-provider sentence_transformers --tags negation,history \
    --fail-on-safety-error           # exit status 2 when any safety gate fails
```

`--coding-system`/`--version` are the default dataset. Each scenario names its own, and
`--only-dataset` restricts the run to that one dataset. In pytest,
`tests/integration/test_clinical_evaluation.py` runs all 266 scenarios against PostgreSQL with
the deterministic hashing provider and asserts that every safety gate passes.

## Metrics (each stage separately)

| Stage | Metrics |
|---|---|
| Concept extraction | precision, recall; assertion accuracy; per-status precision/recall (negation, ruled out, history, family history, uncertainty); attribute accuracy (laterality, severity, acuity, subtype, stage, encounter, anatomy, complication, cause, absence). Concepts are matched on the concept text or the clause as written. |
| Retrieval (each method alone) | Recall@1/3/5/10 and MRR for `exact`, `lexical`, `fuzzy`, `index_term`, `vector` and `hybrid`, overall and per tag |
| Reranking | top-1 before/after; ranks improved/degraded; correct candidate removed; incorrect candidate promoted; individual rerank feature scores of the top candidates per concept (in the JSON) |
| Rules | detection accuracy per instruction type, candidate-rejection accuracy, rule-violation rate, invalid target-code rate (every referenced code re-checked in the database) |
| Specificity | unsupported-specificity rate, by attribute (from an **independent** lexical oracle); safe-fallback rate; missing-information recall and precision (reported items must really be missing) |
| Abstention | precision, recall, false-positive code rate |
| Database safety | every returned record (primary, alternatives, rule targets) goes through a database lookup and must match dataset, coding system and version, with the dataset READY and the record ACTIVE; title and hierarchy must match the selected version |
| Confidence | count and precision per level, levels per evidence pattern, unjustified HIGH |
| Evidence | traceability (clinical text equals the note at its offsets; matched and inclusion terms are real terms of the record), incorrect evidence (an exclusion used as support, a hierarchy mismatch), fabricated items, provenance accuracy |
| Latency | mean / p50 / p95 per stage, measured in the synthetic environment only |

The oracles in `app/evaluation/clinical/oracles.py` deliberately do not reuse the extractor,
the specificity guard or the rule engine, so a defect in the pipeline cannot hide itself.

## Safety gates (release blockers)

Every gate must be exactly 0. A non-zero value fails the release whatever the quality metrics
say. Gates are never averaged with anything.

| Gate | Meaning |
|---|---|
| unsupported-code rate | a returned code that does not exist in the selected dataset |
| cross-version contamination | a returned record from another version |
| coding-system contamination | a returned record from another coding system |
| non-READY dataset leakage | a returned record from a dataset that is not READY |
| fatal coding-rule violations | a returned code whose rules reject it, or an expected rejection that was returned |
| invalid rule target codes | an instruction target that is reported as existing but does not exist, or lives in another dataset |
| unsupported specificity rate | a returned code that adds a side, severity, acuity, subtype, stage, encounter, site, complication or cause that is not documented |
| negated / ruled-out returned as active | a code for a negated or ruled-out concept |
| family history returned as active | a family-history mention coded with anything other than a family-history code for that condition |
| personal history returned as active | a history mention coded as an active condition |
| must-not-return code returned | a scenario's forbidden code was suggested |
| fabricated / incorrect evidence | evidence that cannot be traced, or an exclusion presented as support |
| dataset resolution errors | a scenario's dataset could not be resolved |

## Abstention philosophy

The system prefers **no code** to an unsupported code:
- it abstains rather than picks between near-ties;
- it falls back to the unspecified code, or the category with LOW confidence and the missing
  information, rather than guess a detail;
- symptoms, vague mentions ("infection", "kidney disease"), negated, ruled-out, history and
  family-history mentions never become an active diagnosis.

Recall is never optimised at the cost of an unsafe false positive. Abstention precision counts
an abstention as wrong only when a code was required; abstaining where a scenario lists only
acceptable alternatives is acceptable.

## Failure taxonomy

Each failed scenario is classified by the **first** pipeline stage that went wrong. All the
stages involved are also listed. The stages are `CONCEPT_EXTRACTION`, `ASSERTION`,
`QUERY_GENERATION`, `EXACT_RETRIEVAL`, `LEXICAL_RETRIEVAL`, `FUZZY_RETRIEVAL`,
`VECTOR_RETRIEVAL`, `HYBRID_SCORING`, `RERANKING`, `RULE_VALIDATION`, `SPECIFICITY_GUARD`,
`ABSTENTION`, `DB_VALIDATION`, `DATASET_ISOLATION` and `OTHER`.

The classification uses the pipeline trace:
- where the expected code was retrieved, and by which method;
- why it was rejected (exclusion, specificity contradiction, status gate, evidence gate,
  condition gate);
- where it was reranked.

The report gives a reason for every failure, not just expected != actual.

## Results (2026-10-05, Docker, fresh database, `BAAI/bge-small-en-v1.5`)

See [runtime-verification.md §8](runtime-verification.md#8-phase-3-clinical-evaluation-2026-10-05)
for the full figures. Summary:
- **Safety gates:** PASS (all 0).
- **Scenarios:** 261/266 passed. **API consistency:** all 266 over HTTP match the service-level
  run.
- **Code quality:** code precision 1.000, expected-code recall 0.982, 0 false-positive codes.
- **Abstention:** recall 1.000. **Concept extraction:** precision 0.994, recall 1.000,
  assertion accuracy 1.000.
- **Retrieval (hybrid):** R@1 / R@3 / R@10 = 0.728 / 0.803 / 0.982.
- **Reranking:** top-1 goes from 0.740 before to 0.815 after.

## How a licensed dataset will plug in

Nothing in the evaluator is specific to the synthetic data:
1. Ingest the licensed dataset (manifest, `validate`, golden samples, `ingest`, `index`). It
   becomes READY under its own coding system and version.
2. Write scenarios for it in the same format: de-identified or synthetic notes, with
   expectations agreed by qualified coders. Never use real patient data in the repository.
3. Run `python -m app.evaluation.cli run --dataset <file> --coding-system ICD-10-CA
   --version <v> --fail-on-safety-error`. The safety gates stay hard release blockers.
4. Re-tune the evidence thresholds (`SUGGESTION_MIN_EVIDENCE`,
   `SUGGESTION_MIN_LEXICAL_EVIDENCE`) and the semantic similarity floor on that data. Report
   accuracy only from coder-labelled cases.
