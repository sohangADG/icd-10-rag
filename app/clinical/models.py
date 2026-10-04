"""Structured clinical concept schemas (extraction output, retrieval input)."""

from enum import StrEnum

from pydantic import BaseModel, Field


class AssertionStatus(StrEnum):
    DOCUMENTED = "documented"
    SUSPECTED = "suspected"  # probable / likely / rule out
    UNCERTAIN = "uncertain"  # possible / query / ?
    HISTORY = "history"  # personal history / resolved / former
    FAMILY_HISTORY = "family_history"
    NEGATED = "negated"
    RULED_OUT = "ruled_out"


class ConceptType(StrEnum):
    DIAGNOSIS = "diagnosis"
    SYMPTOM = "symptom"
    FINDING = "finding"
    PROCEDURE = "procedure"


class ClinicalAttributes(BaseModel):
    """Only attributes literally present in the text; nothing is inferred."""

    laterality: str | None = None  # left | right | bilateral
    severity: str | None = None  # mild | moderate | severe
    acuity: str | None = None  # acute | chronic | acute on chronic | subacute
    anatomy: list[str] = Field(default_factory=list)
    encounter: str | None = None  # initial | subsequent | sequela | follow-up
    complications: list[str] = Field(default_factory=list)
    causes: list[str] = Field(default_factory=list)
    subtype: str | None = None  # e.g. "type 2"
    stage: str | None = None
    explicit_absence: list[str] = Field(default_factory=list)  # "without complication"
    # Attributes stated with contradicting values ("left and right", "acute and chronic"):
    # left unset so no code is selected on a guessed value.
    conflicts: list[str] = Field(default_factory=list)


class ClinicalConcept(BaseModel):
    text: str  # concept phrase used for retrieval (status cues removed)
    concept_type: ConceptType
    status: AssertionStatus
    attributes: ClinicalAttributes = Field(default_factory=ClinicalAttributes)
    section: str | None = None
    sentence_index: int
    start: int  # character offsets of the source clause in the note
    end: int
    evidence: str  # the clause as written
    cues: list[str] = Field(default_factory=list)  # status cues that were detected
    expansions: list[str] = Field(default_factory=list)  # abbreviation-expanded variants

    @property
    def codable(self) -> bool:
        return self.status not in {AssertionStatus.NEGATED, AssertionStatus.RULED_OUT}
