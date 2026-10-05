"""Synthetic evaluation cases for the synthetic classification.

They verify SYSTEM BEHAVIOUR (extraction, retrieval, reranking, rule and specificity handling)
end to end. They say nothing about real ICD-10-CA coding accuracy.
"""

from typing import Any

_BASE = {"expected_dataset": "SYNTH-ICD", "expected_version": "2024"}

SYNTHETIC_CASES: list[dict[str, Any]] = [
    {
        "case_id": "acuity-acute",
        "clinical_note": "Assessment: Acute airway infection. Plan: supportive care.",
        "expected_code": "A00.0",
        "expected_concept": "acute airway infection",
        "expected_status": "documented",
        "coder_code": "A00.0",
    },
    {
        "case_id": "acuity-chronic-typo",
        "clinical_note": "Patient has chronic airway infecton for two years.",
        "expected_code": "A00.1",
        "expected_concept": "chronic airway infecton",
        "notes": "misspelling handled by trigram matching",
    },
    {
        "case_id": "laterality-left",
        "clinical_note": "Diagnosis: lobar consolidation of the left lung.",
        "expected_code": "A01.0",
        "expected_concept": "lobar consolidation of the left lung",
        "coder_code": "A01.0",
    },
    {
        "case_id": "laterality-missing",
        "clinical_note": "Lobar consolidation on chest imaging.",
        "expected_code": "A01.9",
        "notes": "side not documented -> unspecified side, never left/right",
    },
    {
        "case_id": "synonym-pneumonia-bilateral",
        "clinical_note": "Lobar pneumonia affecting both lungs.",
        "expected_code": "A01.2",
    },
    {
        "case_id": "severity",
        "clinical_note": "Severe wheezing airway disorder exacerbation.",
        "expected_code": "A02.2",
    },
    {
        "case_id": "abbreviation-acute-on-chronic",
        "clinical_note": "Acute on chronic CHF.",
        "expected_code": "B01.2",
    },
    {
        "case_id": "explicit-without-complication",
        "clinical_note": "Known HTN without complication.",
        "expected_code": "B00.9",
    },
    {
        "case_id": "subtype-complication",
        "clinical_note": "Type 2 diabetes with kidney complication.",
        "expected_code": "C00.20",
    },
    {
        "case_id": "stage",
        "clinical_note": "CKD stage 3.",
        "expected_code": "C12.3",
        "coder_code": "C12.3",
    },
    {
        "case_id": "abbreviation-aki",
        "clinical_note": "AKI on admission.",
        "expected_code": "C13",
    },
    {
        "case_id": "inclusion-term",
        "clinical_note": "Current smoker, 20 cigarettes a day.",
        "expected_code": "D00.0",
    },
    {
        "case_id": "personal-history",
        "clinical_note": "Former smoker.",
        "expected_code": "D00.2",
        "expected_status": "history",
    },
    {
        "case_id": "family-history",
        "clinical_note": "Family history of diabetes.",
        "expected_code": "D01",
        "expected_status": "family_history",
    },
    {
        "case_id": "exclusion-newborn",
        "clinical_note": "Airway infection in a 5-day-old newborn.",
        "expected_code": "B15",
        "notes": "A00 excludes airway infection in the newborn (B15)",
    },
    {
        "case_id": "negated",
        "clinical_note": "Denies cough. No evidence of airway infection.",
        "expected_code": None,
        "expected_status": "negated",
    },
    {
        "case_id": "ruled-out",
        "clinical_note": "Lobar consolidation was ruled out.",
        "expected_code": None,
        "expected_status": "ruled_out",
    },
    {
        "case_id": "uncertain-right",
        "clinical_note": "Possible lobar consolidation of the right lung.",
        "expected_code": "A01.1",
        "expected_status": "uncertain",
    },
]


def synthetic_cases() -> list[dict[str, Any]]:
    return [{**_BASE, **case} for case in SYNTHETIC_CASES]
