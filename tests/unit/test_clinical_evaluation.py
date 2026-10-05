"""The clinical scenario suite and its evaluation machinery (no database)."""

from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.clinical import oracles
from app.evaluation.clinical.report import markdown, safety_gates
from app.evaluation.clinical.scenarios import ClinicalScenario, load_scenarios, select
from app.synthetic.alt_system import ALT_CHAPTERS
from app.synthetic.dataset import CHAPTERS_2024, chapters_2025, flatten
from app.synthetic.paraphrase import paraphrase_chapters

SCENARIOS = Path(__file__).parents[1] / "evaluation" / "clinical_scenarios.json"
CODES = {
    ("SYNTH-ICD", "2024"): {r["code"] for r in flatten(CHAPTERS_2024)},
    ("SYNTH-ICD", "2025"): {r["code"] for r in flatten(chapters_2025())},
    ("SYNTH-ICD", "paraphrase-1"): {r["code"] for r in flatten(paraphrase_chapters())},
    ("SYNTH-ALT", "2024"): {r["code"] for r in flatten(ALT_CHAPTERS)},
}
CATEGORY_TAGS = {
    "direct", "paraphrase", "multiple", "symptom_only", "suspected", "ruled_out", "negation",
    "history", "family_history", "laterality", "conflict", "acuity", "severity", "subtype",
    "complication", "causal", "encounter", "exclusion", "code_first", "use_additional",
    "code_also", "inclusion", "fallback", "no_match", "ambiguous", "typo", "adversarial",
    "version", "isolation",
}  # fmt: skip


@pytest.fixture(scope="module")
def scenarios() -> list[ClinicalScenario]:
    return load_scenarios(SCENARIOS)


def test_suite_size_and_coverage(scenarios: list[ClinicalScenario]) -> None:
    assert len(scenarios) >= 200
    assert sum(s.should_abstain for s in scenarios) >= 30
    tags = Counter(t for s in scenarios for t in s.tags)
    assert set(tags) >= CATEGORY_TAGS, CATEGORY_TAGS - set(tags)


def test_every_expected_code_exists_in_its_synthetic_dataset(
    scenarios: list[ClinicalScenario],
) -> None:
    for scenario in scenarios:
        codes = CODES[(scenario.coding_system, scenario.version)]
        wanted = scenario.allowed_codes | set(scenario.expected_retrieval_codes)
        assert wanted <= codes, (scenario.id, wanted - codes)
        # must_not_return may name codes absent from the version (they must stay absent).
        assert set(scenario.expected_rejected) <= codes, scenario.id


def test_scenarios_are_original_synthetic_text(scenarios: list[ClinicalScenario]) -> None:
    # Scenario content names no real classification or third-party source.
    for scenario in scenarios:
        content = f"{scenario.clinical_note} {scenario.reason}".lower()
        assert "icd-10" not in content and "cihi" not in content, scenario.id


def test_scenario_validation_rejects_inconsistent_expectations() -> None:
    base = {"id": "x", "clinical_note": "n", "reason": "r"}
    with pytest.raises(ValidationError, match="abstention scenario"):
        ClinicalScenario(**base, should_abstain=True, expected_codes=["A00"])
    with pytest.raises(ValidationError, match="both expected and forbidden"):
        ClinicalScenario(**base, expected_codes=["A00"], must_not_return=["A00"])
    with pytest.raises(ValidationError, match="align"):
        ClinicalScenario(**base, expected_concepts=["a"], expected_assertion_states=[])
    with pytest.raises(ValidationError, match="unknown rule types"):
        ClinicalScenario(**base, expected_instructions=["CODE_LAST"])


def test_select_by_tag_and_dataset(scenarios: list[ClinicalScenario]) -> None:
    negation = select(scenarios, tags={"negation"})
    assert negation and all("negation" in s.tags for s in negation)
    alt = select(scenarios, coding_system="SYNTH-ALT", version="2024")
    assert alt and all(s.coding_system == "SYNTH-ALT" for s in alt)


# --- oracles -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "parent", "evidence", "attributes"),
    [
        (
            "Lobar consolidation of left lung",
            "Lobar consolidation of lung",
            "lobar pneumonia",
            ["laterality"],
        ),
        (
            "Lobar consolidation of left lung",
            "Lobar consolidation of lung",
            "left lung lobar pneumonia",
            [],
        ),
        (
            "Type 1 glucose regulation disorder with kidney complication",
            "Glucose regulation disorder",
            "diabetes with kidney complication",
            ["subtype"],
        ),
        (
            "Type 2 glucose regulation disorder with kidney complication",
            "Glucose regulation disorder",
            "T2DM, CKD",
            ["complication"],
        ),
        (
            "Fracture of forearm bone, sequela",
            "Fracture of forearm bone",
            "broken forearm",
            ["encounter"],
        ),
        (
            "Skin wound infection due to animal bite",
            "Skin wound infection",
            "infected dog bite",
            ["cause"],
        ),
        ("Joint inflammation of knee", "Joint inflammation", "arthritis", ["anatomy"]),
        ("Airway infection, unspecified", "Airway infection", "airway infection", []),
        ("Current tobacco use", "Tobacco exposure", "smoker", []),  # source term "smoker"
    ],
)
def test_independent_specificity_oracle(
    title: str, parent: str, evidence: str, attributes: list[str]
) -> None:
    findings = oracles.unsupported_specificity(title, parent, evidence, ["smoker"])
    assert [f.attribute for f in findings] == attributes


def test_missing_information_oracle() -> None:
    assert oracles.missing_item_is_genuine("laterality", "lobar pneumonia")
    assert not oracles.missing_item_is_genuine("laterality", "left lobar pneumonia")
    assert oracles.missing_item_is_genuine("laterality (conflicting)", "left and right lung")
    assert not oracles.missing_item_is_genuine("stage", "CKD stage 3")
    assert oracles.missing_item_is_genuine("anatomical site (knee)", "arthritis")


def test_concept_matching_uses_both_readings() -> None:
    matched = oracles.match_concepts(
        ["current smoker", "asthma"], [["smoker", "Current smoker"], ["asthma", "Asthma"]]
    )
    assert matched == {0: 0, 1: 1}
    # Too much extra context is not the same concept.
    assert oracles.match_concepts(["cough"], [["persistent cough due to airway infection"]]) == {}


def test_percentile_and_rate() -> None:
    assert oracles.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert oracles.percentile([5.0], 95) == 5.0
    assert oracles.rate(0, 0) is None and oracles.rate(1, 4) == 0.25


# --- safety gates --------------------------------------------------------------------------


def _report(**overrides: object) -> dict:
    report = {
        "database_safety": {
            "unsupported_code_rate": 0.0,
            "cross_version_contamination": 0,
            "coding_system_contamination": 0,
            "non_ready_leakage": 0,
            "dataset_resolution_errors": [],
        },
        "rules": {"fatal_rule_violations": 0, "invalid_target_codes": 0},
        "specificity": {"unsupported_specificity_rate": 0.0},
        "status_safety": {},
        "final_selection": {"must_not_return_violations": 0},
        "evidence": {"fabricated_evidence": 0, "incorrect_evidence": 0},
    }
    for path, value in overrides.items():
        section, key = path.split("__")
        report[section][key] = value
    return report


def test_safety_gates_pass_only_when_every_value_is_zero() -> None:
    assert all(g["passed"] for g in safety_gates(_report()))
    for path, value in (
        ("database_safety__unsupported_code_rate", 0.004),
        ("database_safety__cross_version_contamination", 1),
        ("database_safety__coding_system_contamination", 1),
        ("database_safety__non_ready_leakage", 1),
        ("rules__fatal_rule_violations", 1),
        ("specificity__unsupported_specificity_rate", 0.01),
        ("status_safety__negated_as_active", 1),
        ("status_safety__family_history_as_active", 1),
    ):
        gates = safety_gates(_report(**{path: value}))
        assert sum(not g["passed"] for g in gates) == 1, path


def test_markdown_states_the_synthetic_disclaimer() -> None:
    report = _report()
    report.update(
        meta={"disclaimer": "SYNTHETIC ONLY", "embedding_provider": "hashing"},
        dataset={"scenarios": 0, "abstention_scenarios": 0},
        retrieval={"semantic_status": {}, "by_mode": {}},
        final_selection={
            "scenarios_passed": 0,
            "scenarios": 0,
            "expected_code_recall": None,
            "code_precision": None,
            "exact_code_set_match": None,
            "must_not_return_violations": 0,
        },
        concept_extraction={
            "precision": None,
            "recall": None,
            "assertion_accuracy": None,
            "attribute_accuracy_overall": None,
        },
        reranking={
            "top1_before": None,
            "top1_after": None,
            "improved": 0,
            "degraded": 0,
            "correct_candidate_removed": 0,
            "incorrect_candidate_promoted": 0,
        },
        abstention={"precision": None, "recall": None, "false_positive_code_rate": None},
        failures={"by_primary_stage": {}},
        latency_ms={},
    )
    report["specificity"].update(
        safe_fallback_rate=None, missing_information_recall=None, missing_information_precision=None
    )
    report["safety_gates"] = safety_gates(report)
    report["safety_passed"] = True
    text = markdown(report)
    assert "SYNTHETIC ONLY" in text and "Safety gates: PASS" in text
