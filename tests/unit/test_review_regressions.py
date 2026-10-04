"""Regression tests for defects found in the pre-commit review (extraction, specificity,
configuration and retrieval guards)."""

import pytest

from app.clinical.extractor import RuleBasedConceptExtractor, extract_attributes
from app.clinical.models import AssertionStatus, ClinicalConcept, ConceptType
from app.coding.specificity import check_specificity
from app.core.config import Settings
from app.core.exceptions import RetrievalError
from app.repositories.search_repository import SearchRepository

EXTRACTOR = RuleBasedConceptExtractor()
A = AssertionStatus


def statuses(note: str) -> list[tuple[str, AssertionStatus]]:
    return [(c.text, c.status) for c in EXTRACTOR.extract(note)]


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        (
            "no fever, cough, or dyspnea",
            [("fever", A.NEGATED), ("cough", A.NEGATED), ("dyspnea", A.NEGATED)],
        ),
        ("history of diabetes", [("diabetes", A.HISTORY)]),
        ("family history of hypertension", [("hypertension", A.FAMILY_HISTORY)]),
        ("possible pneumonia", [("pneumonia", A.UNCERTAIN)]),
        ("rule out appendicitis", [("appendicitis", A.SUSPECTED)]),
        ("Assessment: pneumonia", [("pneumonia", A.DOCUMENTED)]),
        ("Family history: diabetes", [("diabetes", A.FAMILY_HISTORY)]),
        ("No fever but has cough", [("fever", A.NEGATED), ("cough", A.DOCUMENTED)]),
        (
            "Hypertension, type 2 diabetes and asthma",
            [
                ("Hypertension", A.DOCUMENTED),
                ("type 2 diabetes", A.DOCUMENTED),
                ("asthma", A.DOCUMENTED),
            ],
        ),
        ("CKD, stage 3", [("CKD, stage 3", A.DOCUMENTED)]),
    ],
)
def test_requested_extraction_cases(note: str, expected: list) -> None:
    assert statuses(note) == expected


def test_history_scope_never_promotes_to_a_current_diagnosis() -> None:
    # Defect E1: "hypertension" used to come back as DOCUMENTED.
    assert statuses("History of diabetes and hypertension") == [
        ("diabetes", A.HISTORY),
        ("hypertension", A.HISTORY),
    ]
    # A history *word* inside a concept stays local.
    assert statuses("Former smoker and hypertension") == [
        ("Former smoker", A.HISTORY),
        ("hypertension", A.DOCUMENTED),
    ]


def test_negation_scope_ends_at_a_new_statement() -> None:
    # Defect E2: "pneumonia" used to be NEGATED.
    assert statuses("No fever and the patient has pneumonia") == [
        ("fever", A.NEGATED),
        ("pneumonia", A.DOCUMENTED),
    ]


@pytest.mark.parametrize(
    ("note", "attribute"),
    [
        ("left and right knee pain", "laterality"),
        ("acute and chronic bronchitis", "acuity"),
        ("mild and severe asthma", "severity"),
    ],
)
def test_conflicting_qualifiers_are_one_concept_with_a_conflict(note: str, attribute: str) -> None:
    # Defect E3: qualifier-only fragments ("left", "acute", "mild") were separate concepts and
    # the first value won.
    (concept,) = EXTRACTOR.extract(note)
    assert concept.text == note
    assert attribute in concept.attributes.conflicts
    assert getattr(concept.attributes, attribute) is None


@pytest.mark.parametrize(
    ("text", "laterality"),
    [
        ("left knee pain", "left"),
        ("right knee pain", "right"),
        ("left lower lobe pneumonia", "left"),
        ("pneumonia affecting both lungs", "bilateral"),
        ("bilateral knee pain", "bilateral"),
        ("Patient has both hypertension", None),  # defect E4
        ("Patient left the clinic with otitis", None),  # defect E4
    ],
)
def test_laterality_requires_an_anatomical_context(text: str, laterality: str | None) -> None:
    assert extract_attributes(text).laterality == laterality


@pytest.mark.parametrize(
    ("text", "acuity"),
    [
        ("acute bronchitis", "acute"),
        ("chronic bronchitis", "chronic"),
        ("acute on chronic heart failure", "acute on chronic"),
        ("subacute cough", "subacute"),
    ],
)
def test_acuity_wording(text: str, acuity: str) -> None:
    assert extract_attributes(text).acuity == acuity


def _concept(text: str) -> ClinicalConcept:
    return ClinicalConcept(
        text=text,
        concept_type=ConceptType.DIAGNOSIS,
        status=A.DOCUMENTED,
        attributes=extract_attributes(text),
        sentence_index=0,
        start=0,
        end=len(text),
        evidence=text,
    )


@pytest.mark.parametrize(
    ("text", "title", "parent", "missing"),
    [
        (
            "left and right knee pain",
            "Pain in left knee",
            "Pain in knee",
            "laterality (conflicting)",
        ),
        ("knee pain", "Pain in left knee", "Pain in knee", "laterality"),
        ("asthma", "Severe asthma", "Asthma", "severity"),
        ("diabetes", "Type 2 diabetes", "Diabetes", "subtype"),
        (
            "type 2 diabetes",
            "Type 2 diabetes with kidney complication",
            "Type 2 diabetes",
            "complication (kidney complication)",
        ),
        ("bronchitis", "Chronic bronchitis", "Bronchitis", "acuity"),
        (
            "fracture of femur",
            "Fracture of femur, initial encounter",
            "Fracture of femur",
            "encounter",
        ),
        ("fracture", "Fracture of hip", "Fracture", "anatomical site (hip)"),
    ],
)
def test_missing_specificity_is_reported_not_invented(
    text: str, title: str, parent: str, missing: str
) -> None:
    check = check_specificity(_concept(text), title, parent)
    assert check.status == "unsupported"
    assert missing in check.missing


def test_retrieval_weights_cannot_all_be_zero() -> None:
    zero = {
        f"retrieval_weight_{n}": 0.0
        for n in ("exact", "lexical", "fuzzy", "semantic", "hierarchy", "index_term")
    }
    with pytest.raises(ValueError, match="RETRIEVAL_WEIGHT"):
        Settings(_env_file=None, **zero)


async def test_query_vector_dimension_is_validated() -> None:
    repository = SearchRepository(session=None)  # type: ignore[arg-type]
    with pytest.raises(RetrievalError, match="dimensions"):
        await repository.semantic(1, "m", 4, [0.1, 0.2], limit=5)
