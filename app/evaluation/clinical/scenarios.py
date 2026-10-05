"""Clinical evaluation scenario format (JSON list or JSONL).

Every scenario states the expected behaviour of the whole pipeline, not only a final code:

    {
      "id": "lat-left-01",
      "clinical_note": "...",
      "coding_system": "SYNTH-ICD", "version": "2024",
      "expected_codes": ["A01.0"],            # codes that must be suggested (primary)
      "acceptable_alternatives": [],          # also acceptable as a primary suggestion
      "must_not_return": ["A01.1"],           # never acceptable as a primary suggestion
      "expected_concepts": ["..."],           # EXHAUSTIVE list of concepts in the note
      "expected_assertion_states": ["documented"],   # aligned with expected_concepts
      "expected_attributes": [{"laterality": "left"}],  # optional, aligned; null = absent
      "required_missing_information": [],     # substrings of missing_information
      "should_abstain": false,                # true: no code may be suggested at all
      "reason": "why this is the expected behaviour",
      "tags": ["laterality"]
    }

Optional: expected_retrieval_codes (codes retrieval must rank, when they differ from the final
codes: e.g. an exact code query whose final code depends on the documentation),
expected_instructions (rule types that must be surfaced, e.g. CODE_FIRST),
expected_rejected (codes that must not survive validation), max_confidence, code_concepts
(expected code -> concept it must be attached to), include_uncertain.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.clinical.models import AssertionStatus

CONFIDENCE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
RULE_TYPES = {
    "EXCLUDES",
    "INCLUDES",
    "CODE_FIRST",
    "USE_ADDITIONAL_CODE",
    "CODE_ALSO",
    "SEE",
    "SEE_ALSO",
    "NOTE",
}
ATTRIBUTES = {
    "laterality",
    "severity",
    "acuity",
    "subtype",
    "stage",
    "encounter",
    "anatomy",
    "complication",
    "cause",
    "absence",
}


class ClinicalScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    clinical_note: str = Field(min_length=1)
    coding_system: str = "SYNTH-ICD"
    version: str = "2024"
    expected_codes: list[str] = Field(default_factory=list)
    acceptable_alternatives: list[str] = Field(default_factory=list)
    must_not_return: list[str] = Field(default_factory=list)
    expected_concepts: list[str] = Field(default_factory=list)
    expected_assertion_states: list[AssertionStatus] = Field(default_factory=list)
    expected_attributes: list[dict[str, Any] | None] = Field(default_factory=list)
    required_missing_information: list[str] = Field(default_factory=list)
    should_abstain: bool = False
    reason: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    expected_retrieval_codes: list[str] = Field(default_factory=list)
    expected_instructions: list[str] = Field(default_factory=list)
    expected_rejected: list[str] = Field(default_factory=list)
    max_confidence: Literal["HIGH", "MEDIUM", "LOW"] | None = None
    code_concepts: dict[str, str] = Field(default_factory=dict)
    include_uncertain: bool | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "ClinicalScenario":
        if len(self.expected_assertion_states) != len(self.expected_concepts):
            raise ValueError(f"{self.id}: expected_assertion_states must align with concepts")
        if self.expected_attributes and len(self.expected_attributes) != len(
            self.expected_concepts
        ):
            raise ValueError(f"{self.id}: expected_attributes must align with concepts")
        for attributes in self.expected_attributes:
            unknown = set(attributes or {}) - ATTRIBUTES
            if unknown:
                raise ValueError(f"{self.id}: unknown attributes {sorted(unknown)}")
        if self.should_abstain and self.expected_codes:
            raise ValueError(f"{self.id}: an abstention scenario cannot expect codes")
        overlap = set(self.must_not_return) & (
            set(self.expected_codes) | set(self.acceptable_alternatives)
        )
        if overlap:
            raise ValueError(f"{self.id}: codes both expected and forbidden: {sorted(overlap)}")
        unknown_rules = set(self.expected_instructions) - RULE_TYPES
        if unknown_rules:
            raise ValueError(f"{self.id}: unknown rule types {sorted(unknown_rules)}")
        if set(self.code_concepts) - set(self.expected_codes):
            raise ValueError(f"{self.id}: code_concepts must refer to expected codes")
        return self

    @property
    def allowed_codes(self) -> set[str]:
        return set(self.expected_codes) | set(self.acceptable_alternatives)


def load_scenarios(path: Path) -> list[ClinicalScenario]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        rows = data["scenarios"] if isinstance(data, dict) else data
    scenarios = [ClinicalScenario.model_validate(row) for row in rows]
    ids = [s.id for s in scenarios]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate scenario ids: {duplicates}")
    return scenarios


def select(
    scenarios: list[ClinicalScenario],
    *,
    tags: set[str] | None = None,
    coding_system: str | None = None,
    version: str | None = None,
) -> list[ClinicalScenario]:
    """Filter by tag (any of) and, when given, by dataset."""
    chosen = scenarios
    if tags:
        chosen = [s for s in chosen if tags & set(s.tags)]
    if coding_system:
        chosen = [s for s in chosen if s.coding_system.upper() == coding_system.upper()]
    if version:
        chosen = [s for s in chosen if s.version == version]
    return chosen
