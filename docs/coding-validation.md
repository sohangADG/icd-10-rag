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

A subdivision is compared with its **category**. Whatever it adds beyond the category, at any
level, must be documented. For example, `C00.10` adds "type 1" (through `C00.1`) and "with
kidney complication"; both must be in the note. Until Phase 3 only the immediate parent was
compared, so an intermediate subtype could slip through. Fallback (§3a) climbs to the nearest
ancestor whose own additions are documented, and descent compares children with the category
too. The documentation counts the concept, its abbreviation expansions, the clause as written
(e.g. "current" in "Current tobacco use") and the words its status stands for ("personal
history"). A **code written in the note is not documentation**: "A01.0" alone gives `A01.9`
(side undocumented), with `A01.0` offered as an alternative.

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

## 3b. Minimum evidence gate (abstention)

A candidate is suggested only if at least one retrieval signal reaches a minimum:

| Signal | Default threshold | Setting |
|---|---|---|
| lexical (≈ half of the concept's words covered) | 0.40 | `SUGGESTION_MIN_LEXICAL_EVIDENCE` |
| exact code, fuzzy, semantic, index term (calibrated above their noise floors) | 0.15 | `SUGGESTION_MIN_EVIDENCE` |

Hierarchy context never counts as evidence on its own. A concept that matches one of the
candidate's own inclusion terms passes regardless. Otherwise the concept is returned in
`unmatched_concepts` with the reason "Insufficient retrieval evidence". The system abstains:
it never falls back to whichever vector happens to be closest. Covered cases:
- a near-tie between two records' semantic scores ("insomnia" with bge-small: cosine 0.648 vs
  0.643);
- one shared generic word ("purple quokka **syndrome**");
- unknown terminology;
- a one-word ambiguous symptom ("Pain.").

### 3c. Condition support (`app/coding/condition.py`)

The evidence gate can be met through one shared word. "Generalized weakness" shares "weakness"
with *Heart pump weakness*, and "family history of hypertension" shares "family history" with
*Family history of glucose regulation disorder*. The Phase 3 evaluation found such codes being
returned. So the documentation must now name the candidate's **condition**. Each classification
text that states it is compared with the concept: the title, the source terms, and the
ancestors' titles and terms. Status, generic and subtype/stage words are left out.

| Verdict | Rule | Outcome |
|---|---|---|
| supported | some text has more than half of its condition words documented (misspellings tolerated), and every acuity, side or severity word in it is documented | normal path |
| partial | words are shared, but no text is supported: the documentation names a different or less specific condition | rejected |
| none | no word in common | needs a **meaning-based** semantic score ≥ `SUGGESTION_MIN_EVIDENCE`. Never allowed for history/family-history mentions or symptoms. Never allowed when other candidates share the note's words (all of them partial). |

Exact code, title, inclusion and source-term matches are exempt. The hashing test provider is
not meaning-based, so it never supports a candidate on its own.

### 3d. Contradicting documentation

When two suggestions in the same category document contradicting values, e.g. a left-lung
statement next to a right-lung imaging line, or copy-forward "chronic" next to today's "acute on
chronic":
- both are kept for the reviewer;
- both are set to LOW confidence;
- `validation.conflicting_documentation` names the attribute and the codes.

A repeated statement gives one suggestion, not two.

**The thresholds are tunable defaults, not medical truths.** They were chosen on the synthetic
suites to trade recall for precision: on the paraphrase suite, 3 of 14 concepts are abstained on
and none are wrongly coded. Re-tune them with the evaluation tooling against coder-labelled
cases for each real dataset and embedding model.

## 4. Final database verification (hallucination prevention)

```
candidate → IcdRepository.get_by_code(dataset_id, code, classification_only=True)
          → same record id? same dataset? status active? → return  |  otherwise → reject
```

Alternatives and rule targets also come from database rows only. The evaluation framework
measures the `unsupported_code_rate` by re-checking every returned code (0.0 on the synthetic
suite).
