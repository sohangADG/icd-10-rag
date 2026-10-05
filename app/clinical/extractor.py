"""Rule-based clinical concept extraction (deterministic, no external services).

Pipeline: sections -> sentences -> clauses -> assertion status (NegEx-style scoped cues) ->
attributes -> concept type -> retrieval phrase (+ abbreviation expansions).

This extractor is intentionally conservative and transparent. It implements the
ConceptExtractor protocol, so an NLP/LLM-based extractor can replace it without touching
retrieval, reranking or validation. It never upgrades a mention to a confirmed diagnosis:
negated / ruled-out mentions are kept (as non-codable) so callers can see why nothing was coded.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from app.clinical import lexicons as lx
from app.clinical.models import (
    AssertionStatus,
    ClinicalAttributes,
    ClinicalConcept,
    ConceptType,
)
from app.core.text import tokenize

# Headers start a line or follow a sentence end ("... infection. Plan: ...").
_SECTION_RE = re.compile(
    r"(?:^|(?<=[.;]\s)|(?<=[.;]))[ \t]*("
    + "|".join(sorted(map(re.escape, lx.SECTION_HEADERS), key=len, reverse=True))
    + r")\s*:\s*",
    re.I | re.M,
)
# Clauses starting with these qualify the previous clause instead of naming a new concept.
_QUALIFIER_START = re.compile(
    r"^(?:with|without|complicated by|due to|secondary to|caused by|stage|type|"
    r"(?:initial|subsequent)\s+(?:encounter|visit)|sequelae?|uncomplicated|"
    r"no\s+complications?)\b",
    re.I,
)
_OR_SEPARATOR = re.compile(r"\bor\b", re.I)
_EX_PREFIX = re.compile(r"^ex[\s-]+", re.I)
_SITE_WORDS = frozenset(lx.ANATOMY)
_SIDE_OR_SITE = frozenset(lx.ANATOMY) | frozenset(lx.LATERALITY)
_TEMPORAL = re.compile(
    r"\b(?:today|yesterday|recently|currently|last (?:week|month|year)|this (?:week|month|year)|"
    r"(?:in|since) (?:19|20)\d{2}|(?:\d+|a|one|two|three|several) "
    r"(?:days?|weeks?|months?|years?) ago|"
    r"(?:for|over|x) (?:the )?(?:past |last )?(?:\d+|a|one|two|three|several|many) "
    r"(?:days?|weeks?|months?|years?))\b",
    re.I,
)
# Sentence boundaries: newline, ";" or "." not between digits (keeps "2.5").
_SENTENCE_RE = re.compile(r"\n+|;|(?<!\d)\.(?!\d)|(?<=\d)\.(?!\d)|(?<!\d)\.(?=\d)")
_CLAUSE_SPLIT_RE = re.compile(r",|\band\b|\bor\b|\bbut\b|\bhowever\b|\balso\b|\bplus\b", re.I)
_SCOPING_STATUSES = {
    AssertionStatus.NEGATED,
    AssertionStatus.UNCERTAIN,
    AssertionStatus.SUSPECTED,
    AssertionStatus.FAMILY_HISTORY,
}


class ConceptExtractor(Protocol):
    def extract(self, note: str) -> list[ClinicalConcept]: ...


@dataclass
class _Span:
    text: str
    start: int
    end: int


def _strip_prefix(text: str, prefixes: list[str]) -> tuple[str, str | None]:
    lowered = text.lower()
    for prefix in sorted(prefixes, key=len, reverse=True):
        if prefix == "?":
            if lowered.startswith("?"):
                return text[1:].strip(), prefix
            continue
        if re.match(rf"{re.escape(prefix)}\b", lowered):
            return text[len(prefix) :].strip(" :-"), prefix
    return text, None


def _sections(note: str) -> list[tuple[str | None, int, int]]:
    """(section, start, end) ranges. Text before the first header has section None."""
    matches = list(_SECTION_RE.finditer(note))
    if not matches:
        return [(None, 0, len(note))]
    ranges: list[tuple[str | None, int, int]] = []
    if matches[0].start() > 0:
        ranges.append((None, 0, matches[0].start()))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(note)
        ranges.append((lx.SECTION_HEADERS[match.group(1).lower()], match.end(), end))
    return ranges


def _split(text: str, offset: int, pattern: re.Pattern[str]) -> list[_Span]:
    spans: list[_Span] = []
    position = 0
    for match in pattern.finditer(text):
        spans.append(
            _Span(text[position : match.start()], offset + position, offset + match.start())
        )
        position = match.end()
    spans.append(_Span(text[position:], offset + position, offset + len(text)))
    result = []
    for span in spans:
        stripped = span.text.strip()
        if stripped:
            lead = len(span.text) - len(span.text.lstrip())
            result.append(_Span(stripped, span.start + lead, span.start + lead + len(stripped)))
    return result


def extract_attributes(text: str) -> ClinicalAttributes:
    attributes = ClinicalAttributes()
    words = tokenize(text)
    laterality: set[str] = set()
    severity: set[str] = set()
    for index, word in enumerate(words):
        if word in ("bilateral", "bilaterally"):
            laterality.add("bilateral")
        elif word in lx.LATERALITY and (
            any(w in lx.ANATOMY for w in words[index + 1 : index + 1 + lx.LATERALITY_WINDOW])
            or words[index + 1 : index + 2] == ["sided"]  # "right-sided"
        ):
            laterality.add(lx.LATERALITY[word])
        if word in lx.SEVERITY:
            severity.add(lx.SEVERITY[word])
        if word in lx.ANATOMY and lx.ANATOMY[word] not in attributes.anatomy:
            attributes.anatomy.append(lx.ANATOMY[word])
    if "bilateral" in laterality or {"left", "right"} <= laterality:
        # "left and right" is bilateral only when stated so; otherwise it is a conflict.
        if laterality == {"bilateral"}:
            attributes.laterality = "bilateral"
        else:
            attributes.conflicts.append("laterality")
    elif laterality:
        attributes.laterality = laterality.pop()
    if len(severity) == 1:
        attributes.severity = severity.pop()
    elif severity:
        attributes.conflicts.append("severity")
    if lx.ACUITY_PATTERNS[0][0].search(text):  # "acute on chronic" is one value
        attributes.acuity = lx.ACUITY_PATTERNS[0][1]
    else:
        acuity = {value for pattern, value in lx.ACUITY_PATTERNS[1:] if pattern.search(text)}
        if len(acuity) == 1:
            attributes.acuity = acuity.pop()
        elif acuity:
            attributes.conflicts.append("acuity")
    for pattern, value in lx.ENCOUNTER_PATTERNS:
        if pattern.search(text):
            attributes.encounter = value
            break
    if match := lx.SUBTYPE_PATTERN.search(text):
        raw = match.group(1).lower()
        attributes.subtype = "type " + {"i": "1", "ii": "2", "one": "1", "two": "2"}.get(raw, raw)
    if match := lx.STAGE_PATTERN.search(text):
        attributes.stage = match.group(1).lower()
    for match in lx.ABSENCE_PATTERN.finditer(text):
        attributes.explicit_absence.append((match.group(1) or "complication").lower())
    for match in lx.COMPLICATION_PATTERN.finditer(text):
        value = match.group(1).strip()
        if value:
            attributes.complications.append(value)
    attributes.causes = [m.group(1).strip() for m in lx.CAUSE_PATTERN.finditer(text)]
    return attributes


def _concept_type(text: str, section: str | None) -> ConceptType:
    if section == "procedures" or lx.PROCEDURE_PATTERN.search(text):
        return ConceptType.PROCEDURE
    if section == "assessment":
        return ConceptType.DIAGNOSIS  # the clinician's assessment, even if symptom-worded
    words = set(tokenize(text))
    if words & lx.SYMPTOMS or "shortness of breath" in text.lower():
        return ConceptType.SYMPTOM
    if section in {"assessment", "past_history", "family_history", "hpi", "chief_complaint"}:
        return ConceptType.DIAGNOSIS
    return ConceptType.DIAGNOSIS if len(words - lx.STOPWORDS) >= 1 else ConceptType.FINDING


def expand_abbreviations(text: str) -> list[str]:
    """Variants with standard clinical abbreviations spelled out (original kept separately)."""
    tokens = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", text)
    changed = False
    expanded = []
    for token in tokens:
        if token.isupper() and token in lx.ABBREVIATIONS:
            expanded.append(lx.ABBREVIATIONS[token])
            changed = True
        else:
            expanded.append(token)
    return ["".join(expanded)] if changed else []


class RuleBasedConceptExtractor:
    """Default ConceptExtractor implementation."""

    def __init__(self, *, max_clause_words: int = 20) -> None:
        self._max_words = max_clause_words

    def extract(self, note: str) -> list[ClinicalConcept]:
        concepts: list[ClinicalConcept] = []
        sentence_index = 0
        for section, start, end in _sections(note):
            if section in lx.SKIPPED_SECTIONS:
                continue
            for sentence in _split(note[start:end], start, _SENTENCE_RE):
                concepts.extend(self._sentence(sentence, section, sentence_index))
                sentence_index += 1
        return concepts

    def _sentence(
        self, sentence: _Span, section: str | None, sentence_index: int
    ) -> list[ClinicalConcept]:
        results: list[ClinicalConcept] = []
        # Scoped status: a leading negation/uncertainty/family cue applies to the following
        # clauses of the sentence until a terminator ("but", "however"...) separates them.
        scoped: AssertionStatus | None = None
        scoped_cue: str | None = None
        previous_end = sentence.start

        def merged(first: _Span, last: _Span) -> _Span:
            text = sentence.text[first.start - sentence.start : last.end - sentence.start]
            return _Span(text, first.start, last.end)

        # (clause span, concept text when it differs from the written clause)
        clauses: list[tuple[_Span, str | None]] = []
        pending: _Span | None = None  # qualifier-only fragment waiting for its condition
        for clause in _split(sentence.text, sentence.start, _CLAUSE_SPLIT_RE):
            if pending is not None:
                clause, pending = merged(pending, clause), None
            content = set(tokenize(clause.text)) - lx.STOPWORDS
            if content and content <= lx.QUALIFIER_WORDS:
                pending = clause
            elif clauses and (
                _QUALIFIER_START.match(clause.text)
                # "CKD, stage 3" qualifies; "HTN, type 2 diabetes" is a new condition.
                and not (
                    re.match(r"(?:type|stage)\b", clause.text, re.I)
                    and not re.fullmatch(r"(?:type|stage)\s*\w{1,4}", clause.text.strip(), re.I)
                )
            ):
                clauses[-1] = (merged(clauses[-1][0], clause), None)
            elif (
                clauses and content & _SITE_WORDS and content <= _SIDE_OR_SITE | lx.QUALIFIER_WORDS
            ):
                # A site-only fragment: "Arthritis of knee and hip" names arthritis of the hip
                # too (the condition is repeated with the new site); "Lobar pneumonia, both
                # lungs" qualifies the previous condition, which names no site of its own.
                previous, override = clauses[-1]
                head = _condition_head(override or previous.text)
                if head:
                    clauses.append((clause, f"{head} {clause.text}"))
                else:
                    clauses[-1] = (merged(previous, clause), None)
            else:
                clauses.append((clause, None))
        if pending is not None:
            # A trailing qualifier ("Asthma, severe") belongs to the condition before it.
            if clauses:
                clauses[-1] = (merged(clauses[-1][0], pending), None)
            else:
                clauses.append((pending, None))
        for clause, override in clauses:
            gap = sentence.text[previous_end - sentence.start : clause.start - sentence.start]
            previous_end = clause.end
            if lx.NEGATION_TERMINATORS.search(gap) or lx.NEW_STATEMENT.match(clause.text):
                scoped, scoped_cue = None, None
            concept = self._clause(clause, section, sentence_index, scoped, scoped_cue, override)
            if concept is None:
                continue
            # "X or Y": a differential. Neither alternative is an established diagnosis.
            if (
                _OR_SEPARATOR.search(gap)
                and results
                and results[-1].status == AssertionStatus.DOCUMENTED
                and concept.status == AssertionStatus.DOCUMENTED
            ):
                for item in (results[-1], concept):
                    item.status = AssertionStatus.UNCERTAIN
                    item.cues.append("or (differential)")
            # "History of X and Y": a history *prefix* scopes like negation does, so Y is never
            # silently promoted to a current diagnosis. ("former smoker" stays local.)
            scoping = concept.status in _SCOPING_STATUSES or (
                concept.status == AssertionStatus.HISTORY
                and bool(concept.cues)
                and concept.cues[0] in lx.HISTORY_PREFIXES
            )
            if scoping and concept.cues and "(scope)" not in concept.cues[0]:
                scoped, scoped_cue = concept.status, concept.cues[0]
            results.append(concept)
        return results

    def _clause(
        self,
        clause: _Span,
        section: str | None,
        sentence_index: int,
        scoped: AssertionStatus | None,
        scoped_cue: str | None,
        override: str | None = None,
    ) -> ClinicalConcept | None:
        text = (override or clause.text).strip(" .:-")
        if label := lx.LABEL_PREFIX.match(text):
            text = text[label.end() :]
        cues: list[str] = []
        status: AssertionStatus | None = None

        for pattern in lx.RULED_OUT_PATTERNS:
            if pattern.search(text):
                status = AssertionStatus.RULED_OUT
                cues.append("ruled out")
                text = pattern.sub("", text).strip(" .:-")
                break
        if status is None:
            family_text, cue = _strip_prefix(text, lx.FAMILY_PREFIXES)
            if cue is None and (match := lx.FAMILY_MEMBER.match(text)):
                family_text, cue = text[match.end() :], match.group(0).strip()
            if cue is not None:
                status, text = AssertionStatus.FAMILY_HISTORY, family_text
                cues.append(cue)
        if status is None:
            # Cues may follow a grammatical subject: "Patient denies X", "Pt has no X".
            subject = lx.SUBJECT_PREFIX.match(text)
            cue_text = text[subject.end() :] if subject else text
            for prefixes, value in (
                (lx.NEGATION_PREFIXES, AssertionStatus.NEGATED),
                (lx.UNCERTAIN_PREFIXES, AssertionStatus.UNCERTAIN),
                (lx.SUSPECTED_PREFIXES, AssertionStatus.SUSPECTED),
            ):
                stripped, cue = _strip_prefix(cue_text, prefixes)
                if cue is not None:
                    status, text = value, stripped
                    cues.append(cue)
                    break
            if status is None and (match := lx.NEGATION_INFIX.match(cue_text)):
                status, text = AssertionStatus.NEGATED, cue_text[match.end() :]
                cues.append(match.group(1).lower())
        if status is None and section == "family_history":
            # After explicit cues: "Family history: ... Patient denies X" stays negated.
            status = AssertionStatus.FAMILY_HISTORY
            cues.append("family history section")
        if status is None:
            stripped, cue = _strip_prefix(text, lx.HISTORY_PREFIXES)
            first_word = (tokenize(text) or [""])[0]
            if cue is not None:
                status, text = AssertionStatus.HISTORY, stripped
                cues.append(cue)
            elif first_word in lx.HISTORY_WORDS:
                # "former smoker": the cue word is part of the meaning, so it stays in the text.
                status = AssertionStatus.HISTORY
                cues.append(first_word)
            elif section == "past_history":
                status = AssertionStatus.HISTORY
                cues.append("past history section")
        if status is None and text.endswith("?"):
            status, text = AssertionStatus.UNCERTAIN, text.rstrip("?").strip()
            cues.append("?")
        if status is None and scoped is not None:
            status = scoped
            cues.append(f"{scoped_cue} (scope)")

        text, _ = _strip_prefix(text, lx.FILLER_PREFIXES)
        text = " ".join(_TEMPORAL.sub(" ", text).split()).strip(" .:-?,")
        words = [w for w in tokenize(text) if w not in lx.STOPWORDS and len(w) >= 2]
        if not words or len(tokenize(text)) > self._max_words:
            return None
        if all(w.isdigit() for w in words):
            return None
        expansions = expand_abbreviations(text)
        if (
            status == AssertionStatus.HISTORY
            and (former := _EX_PREFIX.sub("former ", text)) != text
        ):
            expansions.append(former)  # "ex-smoker" states "former smoker"
        return ClinicalConcept(
            text=text,
            concept_type=_concept_type(text, section),
            status=status or AssertionStatus.DOCUMENTED,
            attributes=_attributes(text, expansions),
            section=section,
            sentence_index=sentence_index,
            start=clause.start,
            end=clause.end,
            evidence=clause.text,
            cues=cues,
            expansions=expansions,
        )


def _attributes(text: str, expansions: list[str]) -> ClinicalAttributes:
    """Attributes of the text as written, completed by what an abbreviation states
    ("T2DM" -> type 2, "CKD" -> chronic). The written text wins; nothing is duplicated."""
    attributes = extract_attributes(text)
    for variant in expansions:
        extra = extract_attributes(variant)
        for name, value in extra:
            current = getattr(attributes, name)
            if isinstance(current, list):
                current.extend(v for v in value if v not in current)
            elif current is None and value is not None and name not in attributes.conflicts:
                setattr(attributes, name, value)
    return attributes


def _condition_head(text: str) -> str | None:
    """The condition part of a clause that names a site: the text before its first side or
    site word ("Arthritis of the left knee" -> "Arthritis of the"). None when the clause names
    no site, or starts with one."""
    for match in re.finditer(r"[A-Za-z]+", text):
        if match.group(0).lower() in _SITE_WORDS:
            break
    else:
        return None
    for match in re.finditer(r"[A-Za-z]+", text):
        if match.group(0).lower() in _SIDE_OR_SITE:
            head = text[: match.start()].strip(" ,;-")
            return head or None
    return None
