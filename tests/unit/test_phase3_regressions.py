"""Regression tests for defects found by the Phase 3 clinical scenario evaluation (no database).

Each test names the evaluation scenario(s) that exposed the defect.
"""

import pytest

from app.clinical.extractor import RuleBasedConceptExtractor, extract_attributes
from app.clinical.models import AssertionStatus, ClinicalConcept, ConceptType
from app.coding.condition import condition_support
from app.coding.specificity import check_specificity

EXTRACTOR = RuleBasedConceptExtractor()
D, N, R, H, F, U = (
    AssertionStatus.DOCUMENTED,
    AssertionStatus.NEGATED,
    AssertionStatus.RULED_OUT,
    AssertionStatus.HISTORY,
    AssertionStatus.FAMILY_HISTORY,
    AssertionStatus.UNCERTAIN,
)


def concepts(note: str) -> list[tuple[str, AssertionStatus]]:
    return [(c.text, c.status) for c in EXTRACTOR.extract(note)]


def concept_of(note: str) -> ClinicalConcept:
    (only,) = EXTRACTOR.extract(note)
    return only


# --- assertion status (ruled-out-03/04/07/08, negation-03, family-10, history-06/10) ---------


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("AKI excluded.", [("AKI", R)]),
        ("Diabetes was excluded.", [("Diabetes", R)]),
        ("Workup negative for airway infection.", [("airway infection", N)]),
        ("CT negative for lobar consolidation.", [("lobar consolidation", N)]),
        ("Echo shows no evidence of heart failure.", [("heart failure", N)]),
        ("CXR: no consolidation.", [("consolidation", N)]),
        (
            "Patient denies hypertension, diabetes and kidney disease.",
            [("hypertension", N), ("diabetes", N), ("kidney disease", N)],
        ),
        ("Family history: diabetes. Patient denies diabetes.", [("diabetes", F), ("diabetes", N)]),
        ("Ex-smoker.", [("Ex-smoker", H)]),
        ("Quit smoking 10 years ago.", [("Quit smoking", H)]),
        ("Pneumonia or airway infection.", [("Pneumonia", U), ("airway infection", U)]),
        ("Denies fever or chills.", [("fever", N), ("chills", N)]),
    ],
)
def test_assertion_cues_found_by_the_clinical_evaluation(
    note: str, expected: list[tuple[str, AssertionStatus]]
) -> None:
    assert concepts(note) == expected


def test_label_prefixes_do_not_become_part_of_the_concept() -> None:
    assert concepts("Dx: CHF; HTN; AKI") == [("CHF", D), ("HTN", D), ("AKI", D)]
    assert concepts("Copied from prior note: chronic heart failure.") == [
        ("chronic heart failure", D)
    ]


def test_administrative_section_is_not_coded() -> None:
    assert concepts("Admin: insurance verified, parking validated. Assessment: mild asthma.") == [
        ("mild asthma", D)
    ]


# --- clause segmentation (encounter-01..06, laterality-06, severity-05/06, subtype-08,
#     complication-07, conflict-02, fallback-07) ---------------------------------------------


@pytest.mark.parametrize(
    ("note", "text", "attribute", "value"),
    [
        (
            "Fracture of forearm bone, initial encounter.",
            "Fracture of forearm bone, initial encounter",
            "encounter",
            "initial",
        ),
        (
            "Forearm fracture, subsequent encounter.",
            "Forearm fracture, subsequent encounter",
            "encounter",
            "subsequent",
        ),
        (
            "Forearm fracture, initial visit.",
            "Forearm fracture, initial visit",
            "encounter",
            "initial",
        ),
        ("Forearm fracture, sequela.", "Forearm fracture, sequela", "encounter", "sequela"),
        ("Lobar pneumonia, both lungs.", "Lobar pneumonia, both lungs", "laterality", "bilateral"),
        (
            "Wheezing airway disorder, severe.",
            "Wheezing airway disorder, severe",
            "severity",
            "severe",
        ),
        (
            "Diabetes type 1, uncomplicated.",
            "Diabetes type 1, uncomplicated",
            "explicit_absence",
            ["complication"],
        ),
        (
            "Type 2 diabetes, no complications.",
            "Type 2 diabetes, no complications",
            "explicit_absence",
            ["complications"],
        ),
    ],
)
def test_trailing_qualifiers_belong_to_the_condition(
    note: str, text: str, attribute: str, value: object
) -> None:
    only = concept_of(note)
    assert only.text == text and getattr(only.attributes, attribute) == value


def test_contradicting_qualifiers_stay_one_concept_with_a_conflict() -> None:
    for note, attribute in (
        ("Asthma, mild to moderate.", "severity"),
        ("Lobar consolidation left and right lung.", "laterality"),
    ):
        only = concept_of(note)
        assert attribute in only.attributes.conflicts


def test_coordinated_sites_expand_the_condition() -> None:
    extracted = EXTRACTOR.extract("Arthritis of knee and hip.")
    assert [(c.text, c.attributes.anatomy) for c in extracted] == [
        ("Arthritis of knee", ["knee"]),
        ("Arthritis of hip", ["hip"]),
    ]
    # The expanded concept's evidence is still the text actually written.
    assert extracted[1].evidence == "hip"


# --- attributes (laterality-10, subtype-06, format-18) ------------------------------------------


def test_sided_laterality_and_abbreviation_attributes() -> None:
    assert extract_attributes("Right-sided lobar consolidation").laterality == "right"
    assert concept_of("T2DM.").attributes.subtype == "type 2"
    assert concept_of("CKD stage 3.").attributes.acuity == "chronic"


def test_from_is_not_a_cause() -> None:
    assert extract_attributes("chronic heart failure from prior note").causes == []
    assert extract_attributes("infection due to animal bite").causes == ["animal bite"]


# --- specificity (direct-08, history-04, encounter-06, code-query-01) --------------------------


def test_words_written_in_the_clause_count_as_documentation() -> None:
    current = concept_of("Current tobacco use.")
    assert check_specificity(current, "Current tobacco use", "Tobacco exposure").supported
    history = concept_of("Personal history of tobacco use.")
    assert history.status == H
    assert check_specificity(
        history, "Personal history of tobacco use", "Tobacco exposure"
    ).supported


def test_encounter_wording_is_covered_by_the_encounter_attribute() -> None:
    visit = concept_of("Forearm fracture, initial visit.")
    check = check_specificity(
        visit, "Fracture of forearm bone, initial encounter", "Fracture of forearm bone"
    )
    assert check.supported, check.missing


def test_a_code_written_in_the_note_does_not_document_its_details() -> None:
    # code-query-01: specificity must come from clinical words, not from a pasted code.
    code = concept_of("A01.0")
    title, parent = "Lobar consolidation of left lung", "Lobar consolidation of lung"
    check = check_specificity(code, title, parent)
    assert not check.supported and check.missing == ["laterality"]


# --- condition support (symptom-*, ambiguous-*, family-04..07, conflict-04) ------------------


@pytest.mark.parametrize(
    ("note", "texts", "verdict"),
    [
        # one shared word with a more specific condition: not support
        ("Generalized weakness.", ["Heart pump weakness", "heart failure"], "partial"),
        ("Chest pain.", ["Airway infection", "chest infection"], "partial"),
        ("Infection.", ["Airway infection", "respiratory tract infection"], "partial"),
        (
            "Cough for three days.",
            ["Persistent cough syndrome", "chronic cough of unknown cause"],
            "partial",
        ),
        ("Kidney problem.", ["Acute kidney injury", "AKI"], "partial"),
        (
            "Code reviewed for left lung.",
            ["Lobar consolidation of left lung", "Lobar consolidation of lung"],
            "partial",
        ),
        # an undocumented acuity word in the source term does not support
        (
            "Kidney disease.",
            ["Chronic kidney impairment", "chronic kidney disease", "CKD"],
            "partial",
        ),
        # documented condition (majority of a source text, typos tolerated)
        ("Weak heart pump.", ["Heart pump weakness"], "supported"),
        (
            "Brochial pasage infection.",
            ["Airway infection", "bronchial passage infection"],
            "supported",
        ),
        ("Lobar consolidation.", ["Lobar consolidation of lung"], "supported"),
        ("CKD.", ["Chronic kidney impairment", "chronic kidney disease"], "supported"),
        # nothing in common: a meaning-only (semantic) match is still possible
        ("Epilepsy.", ["Recurrent seizure condition"], "none"),
    ],
)
def test_condition_support(note: str, texts: list[str], verdict: str) -> None:
    assert condition_support(concept_of(note), texts).verdict == verdict


def test_family_history_words_never_count_as_the_condition() -> None:
    family = concept_of("Family history of hypertension.")
    assert family.status == F
    texts = ["Family history of glucose regulation disorder", "family history of diabetes"]
    assert condition_support(family, texts).verdict == "none"
    diabetes = concept_of("Family history of diabetes.")
    assert condition_support(diabetes, texts).verdict == "supported"


def test_concept_type_is_unchanged_by_condition_support() -> None:
    assert concept_of("Generalized weakness.").concept_type == ConceptType.SYMPTOM
