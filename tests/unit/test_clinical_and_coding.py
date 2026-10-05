import pytest

from app.clinical.extractor import (
    RuleBasedConceptExtractor,
    expand_abbreviations,
    extract_attributes,
)
from app.clinical.models import AssertionStatus, ClinicalAttributes, ClinicalConcept, ConceptType
from app.coding.reranker import Reranker, RerankFeatures, concept_overlap, exact_terminology
from app.coding.rules import RuleEngine, inclusion_matches, matches_text
from app.coding.specificity import check_specificity
from app.coding.suggestion_service import queries_for
from app.core.constants import NodeType, RuleType, TermType
from app.models import IcdNode, IcdRule, IcdTerm
from app.retrieval.hybrid import ComponentScores, RetrievalCandidate, hybrid_score

EXTRACTOR = RuleBasedConceptExtractor()


def concepts(note: str) -> list[tuple[str, AssertionStatus]]:
    return [(c.text, c.status) for c in EXTRACTOR.extract(note)]


def concept(text: str, **attributes) -> ClinicalConcept:  # noqa: ANN003
    extracted = extract_attributes(text)
    for key, value in attributes.items():
        setattr(extracted, key, value)
    return ClinicalConcept(
        text=text,
        concept_type=ConceptType.DIAGNOSIS,
        status=AssertionStatus.DOCUMENTED,
        attributes=extracted,
        sentence_index=0,
        start=0,
        end=len(text),
        evidence=text,
        expansions=expand_abbreviations(text),
    )


# --- extraction ---------------------------------------------------------------------------------


def test_negation_scope_covers_the_list_until_a_terminator() -> None:
    assert concepts("Denies chest pain, fever or cough but reports chronic airway infection.") == [
        ("chest pain", AssertionStatus.NEGATED),
        ("fever", AssertionStatus.NEGATED),
        ("cough", AssertionStatus.NEGATED),
        ("chronic airway infection", AssertionStatus.DOCUMENTED),
    ]


@pytest.mark.parametrize(
    ("note", "status"),
    [
        ("No evidence of airway infection.", AssertionStatus.NEGATED),
        ("Pneumonia was ruled out.", AssertionStatus.RULED_OUT),
        ("Rule out lobar consolidation.", AssertionStatus.SUSPECTED),
        ("Probable airway infection.", AssertionStatus.SUSPECTED),
        ("Possible lobar consolidation.", AssertionStatus.UNCERTAIN),
        ("?AKI", AssertionStatus.UNCERTAIN),
        ("History of tobacco use.", AssertionStatus.HISTORY),
        ("Former smoker.", AssertionStatus.HISTORY),
        ("Family history of diabetes.", AssertionStatus.FAMILY_HISTORY),
        ("Mother had breast cancer.", AssertionStatus.FAMILY_HISTORY),
        ("Acute airway infection.", AssertionStatus.DOCUMENTED),
    ],
)
def test_assertion_status(note: str, status: AssertionStatus) -> None:
    extracted = EXTRACTOR.extract(note)
    assert extracted and extracted[0].status == status
    assert extracted[0].codable == (
        status not in {AssertionStatus.NEGATED, AssertionStatus.RULED_OUT}
    )


def test_sections_drive_status_and_skipping() -> None:
    note = (
        "PMH: HTN, without complication.\nFamily history: diabetes.\n"
        "Assessment: Acute airway infection. Plan: start antibiotics.\nMedications: insulin."
    )
    extracted = EXTRACTOR.extract(note)
    assert [(c.text, c.status, c.section) for c in extracted] == [
        ("HTN, without complication", AssertionStatus.HISTORY, "past_history"),
        ("diabetes", AssertionStatus.FAMILY_HISTORY, "family_history"),
        ("Acute airway infection", AssertionStatus.DOCUMENTED, "assessment"),
    ]
    assert extracted[0].attributes.explicit_absence == ["complication"]
    assert extracted[0].expansions == ["hypertension, without complication"]


def test_attributes_are_only_what_is_written() -> None:
    attrs = extract_attributes(
        "Severe acute on chronic left lung infection, type 2, stage 3, "
        "with kidney complication due to smoking, initial encounter"
    )
    assert attrs.severity == "severe"
    assert attrs.acuity == "acute on chronic"
    assert attrs.laterality == "left"
    assert attrs.subtype == "type 2" and attrs.stage == "3"
    assert attrs.encounter == "initial"
    assert attrs.complications == ["kidney complication"]
    assert attrs.causes and attrs.causes[0].startswith("smoking")
    assert "lung" in attrs.anatomy and "kidney" in attrs.anatomy
    assert extract_attributes("airway infection") == ClinicalAttributes(anatomy=["airway"])


def test_offsets_point_at_the_source_clause() -> None:
    note = "Assessment: Acute airway infection."
    extracted = EXTRACTOR.extract(note)[0]
    assert note[extracted.start : extracted.end] == "Acute airway infection"


def test_procedures_and_abbreviations() -> None:
    extracted = EXTRACTOR.extract("Underwent appendectomy. CKD stage 3.")
    assert extracted[0].concept_type == ConceptType.PROCEDURE
    assert extracted[1].expansions == ["chronic kidney disease stage 3"]
    assert expand_abbreviations("pe and htn") == []  # lower-case is not an abbreviation


def test_queries_for_history_and_family_history() -> None:
    family = EXTRACTOR.extract("Family history of diabetes.")[0]
    assert queries_for(family)[0] == "family history of diabetes"
    history = EXTRACTOR.extract("History of tobacco use.")[0]
    assert queries_for(history)[0] == "personal history of tobacco use"


# --- specificity ---------------------------------------------------------------------------------


def test_laterality_guard() -> None:
    parent = "Lobar consolidation of lung"
    no_side = concept("lobar consolidation")
    left = concept("lobar consolidation of the left lung")
    right = concept("lobar consolidation of the right lung")

    unsupported = check_specificity(no_side, "Lobar consolidation of left lung", parent)
    assert unsupported.status == "unsupported" and unsupported.missing == ["laterality"]
    assert check_specificity(left, "Lobar consolidation of left lung", parent).status == "supported"
    contradicted = check_specificity(right, "Lobar consolidation of left lung", parent)
    assert contradicted.status == "contradicted"
    unspecified = check_specificity(
        no_side, "Lobar consolidation of lung, unspecified side", parent
    )
    assert unspecified.supported and unspecified.is_unspecified_variant
    assert check_specificity(
        left, "Lobar consolidation of lung, unspecified side", parent
    ).less_specific_than_documented == ["laterality"]


@pytest.mark.parametrize(
    ("text", "title", "parent", "expected"),
    [
        ("acute heart failure", "Acute heart pump weakness", "Heart pump weakness", "supported"),
        ("heart failure", "Acute heart pump weakness", "Heart pump weakness", "unsupported"),
        (
            "chronic heart failure",
            "Acute heart pump weakness",
            "Heart pump weakness",
            "contradicted",
        ),
        (
            "acute on chronic CHF",
            "Acute on chronic heart pump weakness",
            "Heart pump weakness",
            "supported",
        ),
        (
            "mild wheezing",
            "Severe wheezing airway disorder",
            "Wheezing airway disorder",
            "contradicted",
        ),
        (
            "type 2 diabetes",
            "Type 1 glucose regulation disorder",
            "Glucose regulation disorder",
            "contradicted",
        ),
        (
            "CKD stage 3",
            "Chronic kidney impairment, stage 3",
            "Chronic kidney impairment",
            "supported",
        ),
        ("CKD", "Chronic kidney impairment, stage 3", "Chronic kidney impairment", "unsupported"),
        (
            "HTN",
            "Elevated blood pressure disorder without complication",
            "Elevated blood pressure disorder",
            "unsupported",
        ),
        (
            "HTN without complication",
            "Elevated blood pressure disorder without complication",
            "Elevated blood pressure disorder",
            "supported",
        ),
        (
            "HTN without complication",
            "Elevated blood pressure disorder with kidney involvement",
            "Elevated blood pressure disorder",
            "contradicted",
        ),
        (
            "type 2 diabetes with kidney complication",
            "Type 2 glucose regulation disorder with kidney complication",
            "Type 2 glucose regulation disorder",
            "supported",
        ),
        (
            "type 2 diabetes",
            "Type 2 glucose regulation disorder with kidney complication",
            "Type 2 glucose regulation disorder",
            "unsupported",
        ),
        ("smoker", "Exposure to second-hand tobacco smoke", "Tobacco exposure", "unsupported"),
    ],
)
def test_specificity_matrix(text: str, title: str, parent: str, expected: str) -> None:
    assert check_specificity(concept(text), title, parent).status == expected


def test_source_terms_support_specificity() -> None:
    check = check_specificity(
        concept("smoker"),
        "Current tobacco use",
        "Tobacco exposure",
        ["smoker", "tobacco dependence"],
    )
    assert check.supported and check.supported_by_term == "smoker"


def test_categories_without_classification_parent_are_supported() -> None:
    assert check_specificity(concept("anything"), "Airway infection", None).supported


# --- rules ---------------------------------------------------------------------------------------


def _node(node_id: int, code: str, title: str, node_type: NodeType = NodeType.CATEGORY) -> IcdNode:
    return IcdNode(
        id=node_id,
        dataset_id=1,
        code=code,
        title=title,
        node_type=node_type,
        depth=0,
        is_selectable=True,
    )


def _rule(node_id: int, rule_type: RuleType, text: str, target: str | None = None) -> IcdRule:
    return IcdRule(
        dataset_id=1, node_id=node_id, rule_type=rule_type, rule_text=text, target_code=target
    )


def test_exclusion_rejects_and_redirects() -> None:
    a00 = _node(1, "A00", "Airway infection")
    a000 = _node(2, "A00.0", "Acute airway infection", NodeType.SUBCATEGORY)
    b15 = _node(3, "B15", "Airway infection in the newborn")
    details = {
        1: {
            "terms": [],
            "rules": [_rule(1, RuleType.EXCLUDE, "airway infection in the newborn (B15)", "B15")],
        },
        2: {"terms": [], "rules": []},
    }
    newborn = concept("airway infection in a 5-day-old newborn")
    # The exclusion is on the category but covers its subdivision.
    evaluation = RuleEngine().evaluate(newborn, [], a000, [a00], details, {"B15": b15})
    assert evaluation.rejected and evaluation.redirect_codes == ["B15"]
    finding = evaluation.findings[0]
    assert (finding.severity, finding.attached_to, finding.target_record_id) == ("reject", "A00", 3)

    adult = concept("acute airway infection")
    assert not RuleEngine().evaluate(adult, [], a000, [a00], details, {"B15": b15}).rejected
    # A different concept matching the exclusion is a warning, not a rejection.
    warned = RuleEngine().evaluate(adult, [newborn], a000, [a00], details, {"B15": b15})
    assert warned.status == "warning" and not warned.rejected


def test_instructions_and_inclusions_are_reported() -> None:
    d00 = _node(1, "D00.0", "Current tobacco use", NodeType.SUBCATEGORY)
    details = {
        1: {
            "terms": [
                IcdTerm(
                    dataset_id=1,
                    node_id=1,
                    term="smoker",
                    normalized_term="smoker",
                    term_type=TermType.INCLUSION,
                    language="en",
                )
            ],
            "rules": [
                _rule(1, RuleType.CODE_FIRST, "Code first X (C12.-)", "C12"),
                _rule(1, RuleType.USE_ADDITIONAL_CODE, "Use additional code (Z99)", "Z99"),
                _rule(1, RuleType.NOTE, "A note."),
            ],
        }
    }
    evaluation = RuleEngine().evaluate(concept("smoker"), [], d00, [], details, {})
    assert evaluation.inclusion_match == "smoker" and evaluation.status == "passed"
    types = [(f.rule_type, f.target_exists) for f in evaluation.findings]
    assert types == [
        ("INCLUDES", None),
        ("CODE_FIRST", False),
        ("USE_ADDITIONAL_CODE", False),
        ("NOTE", None),
    ]


def test_text_matching_helpers() -> None:
    assert matches_text(concept("acute kidney injury"), "acute kidney injury (C13)", 0.75)
    assert not matches_text(concept("chronic kidney disease"), "acute kidney injury (C13)", 0.75)
    assert inclusion_matches(concept("smokers"), "smoker")
    assert not inclusion_matches(concept("former smoker"), "smoker")


# --- reranking / hybrid ------------------------------------------------------------------------


def _candidate(title: str, hybrid: float = 0.5, selectable: bool = True) -> RetrievalCandidate:
    node = _node(1, "X00", title)
    node.is_selectable = selectable
    return RetrievalCandidate(
        node=node, ancestors=[], scores=ComponentScores(), hybrid_score=hybrid
    )


def test_exact_terminology_and_overlap() -> None:
    left = concept("lobar consolidation of the left lung")
    assert exact_terminology(left, _candidate("Lobar consolidation of left lung"), []) == 1.0
    assert exact_terminology(concept("HTN"), _candidate("Elevated blood pressure"), ["HTN"]) == 1.0
    assert exact_terminology(concept("heart failure"), _candidate("Heart pump weakness"), []) == 0.0
    assert concept_overlap(
        concept("chronic heart failure"), _candidate("Heart pump weakness"), ["heart failure"]
    ) == pytest.approx(2 / 3)


def test_reranker_penalises_unsupported_specificity() -> None:
    reranker = Reranker()
    base = dict(
        hybrid=0.6,
        exact_terminology=0.0,
        concept_overlap=0.5,
        inclusion_match=0.0,
        documented_specificity=0.0,
        selectable=1.0,
        less_specific_than_documented=0.0,
        rule_warning=0.0,
    )
    supported = reranker.score(RerankFeatures(**base, unsupported_specificity=0.0))
    unsupported = reranker.score(RerankFeatures(**base, unsupported_specificity=1.0))
    assert supported - unsupported == pytest.approx(0.2)
    assert 0.0 <= unsupported <= supported <= 1.0


def test_hybrid_score_ignores_unavailable_components() -> None:
    weights = {
        "exact": 0.3,
        "lexical": 0.25,
        "fuzzy": 0.2,
        "semantic": 0.15,
        "hierarchy": 0.05,
        "index_term": 0.05,
    }
    without = ComponentScores(lexical=1.0, fuzzy=1.0, semantic=None)
    with_zero = ComponentScores(lexical=1.0, fuzzy=1.0, semantic=0.0)
    assert hybrid_score(without, weights) == pytest.approx(0.45 / 0.85, abs=1e-4)
    assert hybrid_score(with_zero, weights) == pytest.approx(0.45, abs=1e-4)
    assert hybrid_score(ComponentScores(), {}) == 0.0
