"""End-to-end regression tests (PostgreSQL) for safety defects found by the Phase 3 clinical
scenario evaluation. Each test names the scenario(s) that exposed the defect."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.coding.suggestion_service import SuggestionService
from app.core.config import get_settings
from app.indexing.embeddings import get_embedding_provider
from app.schemas.icd import SuggestRequest, SuggestResponse
from tests.integration.ingest import import_synthetic


@pytest.fixture
async def ready(session: AsyncSession, tmp_path: Path) -> None:
    await import_synthetic(session, tmp_path)


async def suggest(session: AsyncSession, note: str) -> SuggestResponse:
    settings = get_settings()
    return await SuggestionService(session, settings, get_embedding_provider(settings)).suggest(
        SuggestRequest(clinical_note=note, coding_system="SYNTH-ICD", version="2024")
    )


def codes(response: SuggestResponse) -> list[str]:
    return [s.code for s in response.suggestions]


@pytest.mark.usefixtures("ready")
@pytest.mark.parametrize(
    "note",
    [
        "Generalized weakness.",  # symptom-01: was B01.9 heart pump weakness
        "Chest pain.",  # symptom-02: was A00.9 airway infection
        "Cough for three days.",  # symptom-03: was A10 persistent cough syndrome
        "Wheezing.",  # symptom-07: was A02.9
        "Kidney problem.",  # symptom-12: was C13 acute kidney injury
        "Infection.",  # ambiguous-01: was A00.9
        "Kidney disease.",  # ambiguous-03: was C12.9 (chronic never documented)
    ],
)
async def test_vague_or_symptom_concepts_never_become_a_specific_diagnosis(
    session: AsyncSession, note: str
) -> None:
    response = await suggest(session, note)
    assert codes(response) == []
    assert response.unmatched_concepts


@pytest.mark.usefixtures("ready")
@pytest.mark.parametrize(
    "note",
    [
        "Family history of hypertension.",  # family-04: was D01 (family history of DIABETES)
        "Brother has asthma.",  # family-05
        "Father diagnosed with heart failure.",  # family-06
    ],
)
async def test_family_history_of_one_condition_is_never_another_conditions_code(
    session: AsyncSession, note: str
) -> None:
    assert codes(await suggest(session, note)) == []


@pytest.mark.usefixtures("ready")
async def test_family_history_inclusion_is_reported(session: AsyncSession) -> None:
    (suggestion,) = (await suggest(session, "Family history of diabetes.")).suggestions
    assert suggestion.code == "D01"
    assert "INCLUDES" in {c["rule_type"] for c in suggestion.validation["rule_checks"]}


@pytest.mark.usefixtures("ready")
async def test_specificity_added_above_the_parent_must_be_documented(
    session: AsyncSession,
) -> None:
    # subtype-03: C00.10 adds "type 1" (via C00.1) and "with kidney complication" (vs C00.1).
    (suggestion,) = (await suggest(session, "Diabetes with kidney complication.")).suggestions
    assert suggestion.code == "C00.9"
    assert "subtype" in suggestion.missing_information
    assert {a.code for a in suggestion.alternatives} >= {"C00.1", "C00.2"}


@pytest.mark.usefixtures("ready")
@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("AKI excluded.", []),  # ruled-out-03
        ("Workup negative for airway infection.", []),  # ruled-out-04
        ("Patient denies hypertension, diabetes and kidney disease.", []),  # negation-03
        ("Family history: hypertension. No personal history of hypertension.", []),  # format-12
        ("Personal history of tobacco use.", ["D00.2"]),  # history-04
        ("Ex-smoker.", ["D00.2"]),  # history-06: was D00 (current-exposure category)
        ("Current tobacco use.", ["D00.0"]),  # direct-08: 'current' was lost
        ("Fracture of forearm bone, initial encounter.", ["E01.0"]),  # encounter-01
        ("Forearm fracture, initial visit.", ["E01.0"]),  # encounter-06
        ("Type 1 diabetes, uncomplicated.", ["C00.19"]),  # subtype-08: was C00.1 + B00.9
        ("A01.0", ["A01.9"]),  # code-query-01: a pasted code documents no laterality
        ("Right lung lobar consolidation; code reviewed for left lung.", ["A01.1"]),  # conflict-04
        ("Severe asthma. Severe asthma.", ["A02.2"]),  # format-10: was suggested twice
    ],
)
async def test_scenario_regressions(session: AsyncSession, note: str, expected: list[str]) -> None:
    assert codes(await suggest(session, note)) == expected


@pytest.mark.usefixtures("ready")
async def test_quit_smoking_is_never_current_tobacco_use(session: AsyncSession) -> None:
    returned = codes(await suggest(session, "Quit smoking 10 years ago."))
    assert not {"D00", "D00.0", "D00.1"} & set(returned)


@pytest.mark.usefixtures("ready")
async def test_contradicting_documentation_is_flagged_and_never_confident(
    session: AsyncSession,
) -> None:
    # conflict-03 / format-18: was A01.0 HIGH next to a right-lung statement.
    for note in (
        "Lobar consolidation of the left lung. CXR shows right lung consolidation.",
        "Copied from prior note: chronic heart failure. Today: acute on chronic heart failure.",
    ):
        response = await suggest(session, note)
        assert len(response.suggestions) == 2
        for suggestion in response.suggestions:
            assert suggestion.confidence == "LOW"
            assert suggestion.validation["conflicting_documentation"]
