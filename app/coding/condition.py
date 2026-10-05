"""Condition support: does the documentation name the condition a candidate code asserts?

Retrieval signals can be met through a single shared word. "Generalized weakness" shares
"weakness" with "Heart pump weakness"; "chest pain" shares "chest" with the index term "chest
infection"; "family history of hypertension" shares "family history" with "Family history of
glucose regulation disorder". The evidence gate cannot tell that the classification text then
names a DIFFERENT or MORE SPECIFIC condition than the one documented, so codes were invented
for symptoms, vague mentions and other conditions' family histories.

Every classification text that states the candidate's condition is examined: the candidate's
title and source terms (synonyms, inclusions, abbreviations, index terms) and the titles and
terms of its classification ancestors. Status words (family, history, personal...), generic
words (disorder, condition...) and subtype/stage tokens are not condition words.

* ``supported``: some text has MORE THAN HALF of its condition words documented (a misspelt
  word counts) and every acuity/side/severity word of that text is documented;
* ``partial``: texts share words with the documentation, but none is supported. The
  documentation names a different or less specific condition: the candidate is rejected;
* ``none``: no word in common anywhere. Only meaning-based (semantic) retrieval can support
  the candidate, and the evidence gate decides. History and family-history mentions are
  excluded from that path because their status words dominate the embedding.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.clinical.models import ClinicalConcept
from app.coding.specificity import stems

STATUS_WORDS = frozenset(
    {"family", "history", "personal", "past", "former", "previous", "prior", "hx"}
)
GENERIC_WORDS = frozenset(
    {
        *("a", "an", "and", "or", "of", "the", "in", "on", "to", "for", "by", "with", "without"),
        *("due", "not", "other", "nos", "specified", "unspecified", "type", "stage"),
        *("disorder", "disease", "condition", "syndrome", "complication", "involvement"),
    }
)
# Specificity words: a source text using one supports a concept only when it is documented
# ("chronic kidney disease" does not support "kidney disease").
ATTRIBUTE_WORDS = frozenset(
    {"acute", "chronic", "subacute", "left", "right", "bilateral", "both", "mild", "moderate"}
    | {"severe"}
)
MIN_FUZZY_LENGTH = 5
FUZZY_RATIO = 0.8


@dataclass(frozen=True)
class ConditionSupport:
    verdict: str  # supported | partial | none
    score: float
    text: str | None  # the classification text closest to the documentation


def _condition_words(text: str) -> set[str]:
    return {w for w in stems(text) - GENERIC_WORDS - STATUS_WORDS if not w.isdigit() and len(w) > 1}


def _documented(word: str, documented: set[str]) -> bool:
    if word in documented:
        return True
    if len(word) < MIN_FUZZY_LENGTH:
        return False
    return any(
        len(other) >= MIN_FUZZY_LENGTH and SequenceMatcher(None, word, other).ratio() >= FUZZY_RATIO
        for other in documented
    )


def condition_support(concept: ClinicalConcept, texts: Iterable[str]) -> ConditionSupport:
    documented = stems(concept.text)
    for variant in concept.expansions:
        documented |= stems(variant)
    best, best_text, overlap = 0.0, None, False
    for text in texts:
        words = _condition_words(text)
        if not words:
            continue
        matched = {w for w in words if _documented(w, documented)}
        overlap = overlap or bool(matched)
        if (words & ATTRIBUTE_WORDS) - documented:
            continue
        score = len(matched) / len(words)
        if score > best:
            best, best_text = score, text
    if best > 0.5:
        return ConditionSupport("supported", round(best, 4), best_text)
    return ConditionSupport("partial" if overlap else "none", round(best, 4), best_text)
