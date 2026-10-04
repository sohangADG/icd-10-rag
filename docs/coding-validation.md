# Coding validation

Three independent safeguards run before any code is returned.

## 1. Concept status gate (`SuggestionService`)

| Concept status | Policy |
|---|---|
| documented | Coded |
| suspected / uncertain | Coded only if `SUGGEST_UNCERTAIN_CONCEPTS` / `include_uncertain`. Confidence is capped at MEDIUM and a policy note is added. |
| history | Only history codes (title or own terms mention history/former/previous). The query is prefixed with "personal history of". |
| family_history | Only family-history codes |
| negated, ruled_out | Never coded; reported in `unmatched_concepts` |
| procedure (type) | Not coded with a diagnosis classification |

History or family-history codes are also refused for current-condition mentions.

## 2. Rule engine (`app/coding/rules.py`)

Rules of the candidate **and of its classification ancestors** apply.

| Rule | Behaviour |
|---|---|
| EXCLUDES | Rejects the candidate if the concept matches the exclusion text (≥ 75% of its content words). If the exclusion names one code, that code is scored as a candidate and offered as an alternative. A match against *another* concept in the note is a warning. |
| INCLUDES / inclusion terms | A match supports the candidate (rerank feature, confidence evidence) |
| CODE_FIRST / USE_ADDITIONAL_CODE / CODE_ALSO | Returned as instructions. The referenced code is checked to exist in the same dataset (`target_exists`). Never auto-added. |
| NOTE / SEE / SEE_ALSO / OTHER | Returned for the reviewer |

Ambiguous references (several codes, ranges) are shown as text and never turned into targets.

## 3. Specificity guard (`app/coding/specificity.py`)

A subdivision is compared with its classification parent. Whatever it **adds** must be
documented:

| Added qualifier | Requirement |
|---|---|
| laterality (left/right/both) | Same laterality documented |
| acuity (acute/chronic/acute on chronic) | Same acuity documented |
| severity (mild/moderate/severe) | Same severity documented |
| subtype ("type 2") / stage ("stage 3") | Same value documented |
| "with X" | A complication documented. If X is specific (e.g. kidney), X is documented. |
| "without X" | Absence explicitly documented ("without complication", "uncomplicated") |
| encounter (initial/subsequent/sequela) | Same encounter documented |
| anatomical site | Site documented |
| any other qualifier word | Present in the concept text |

Outcomes: **supported**, **unsupported** (fall back to an unspecified sibling or the parent and
list `missing_information`), or **contradicted** (rejected; e.g. documented right, code says
left). A concept equal to one of the candidate's own source terms (e.g. "smoker" listed under
"Current tobacco use") is supported by the classification itself. Nothing is inferred: laterality,
severity, subtype, cause and complications come only from the note's words. Laterality
counts only when an anatomical site follows within three words ("left knee", never "patient
left the clinic"). Contradicting values ("left and right", "acute and chronic") leave the
attribute unset and are reported as `"<attribute> (conflicting)"` missing information.

## 3a. Fallback and descent are checked too

A record the resolver reaches *without* retrieving it gets the same status gate and coding
rules as a retrieved candidate (including its ancestors' rules) before it can be returned or
offered as an alternative: the classification parent of an unsupported candidate, the
"unspecified" sibling, or the single child reached when descending from a category. If no
admissible record exists, the resolver moves to the next accepted candidate. An unsupported
candidate without a classification parent is never returned.

## 4. Final database verification (hallucination prevention)

```
candidate → IcdRepository.get_by_code(dataset_id, code, classification_only=True)
          → same record id? same dataset? status active? → return  |  otherwise → reject
```

Alternatives and rule targets also come from database rows only. The evaluation framework
measures the `unsupported_code_rate` by re-checking every returned code (0.0 on the synthetic
suite).
