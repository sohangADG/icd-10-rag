"""Original synthetic paraphrase dataset for testing semantic retrieval.

Every record title below was written for this project in deliberately unusual, descriptive
wording, and every evaluation query names the same idea with *different* words: no query shares
a word stem with its target title (enforced by tests). Exact, lexical and fuzzy matching
therefore cannot find the target; only meaning-based (semantic) retrieval can.

The dataset has no synonyms, inclusion terms or index terms on purpose. It is test data only
and must never be used for real coding.
"""

from typing import Any

from app.synthetic.dataset import SYNTHETIC_SYSTEM

PARAPHRASE_VERSION = "paraphrase-1"

DATASET_PARAPHRASE: dict[str, Any] = {
    "coding_system": SYNTHETIC_SYSTEM,
    "country": "XX",
    "version": PARAPHRASE_VERSION,
    "language": "en",
    "title": "Synthetic paraphrase test classification",
    "publisher": "icd-rag-service synthetic fixtures",
    "licence": {"basis": "synthetic", "note": "Original test data; no third-party content."},
    "extra": {"synthetic": True, "allow_remote_processing": True},
}

# (code, title, paraphrased query with no shared word stem)
PARAPHRASE_PAIRS: list[tuple[str, str, str]] = [
    ("P00", "Elevated arterial pressure disorder", "hypertension"),
    ("P01", "Sudden loss of kidney filtering function", "acute renal failure"),
    ("P02", "Persistent low mood disorder", "chronic depression"),
    ("P03", "Inflamed appendix", "appendicitis"),
    ("P04", "Broken thigh bone", "femur fracture"),
    ("P05", "Too few red blood cells", "anaemia"),
    ("P06", "Seasonal pollen sensitivity", "hay fever"),
    ("P07", "Recurrent seizure condition", "epilepsy"),
    ("P10", "Excess body fat condition", "obesity"),
    ("P11", "One-sided throbbing headache with visual aura", "migraine"),
    ("P12", "Itchy red skin patches", "eczema"),
    ("P13", "Gallbladder stones", "cholelithiasis"),
    ("P14", "Trouble falling or staying asleep", "insomnia"),
    ("P15", "Stomach acid backflow into the gullet", "gastro-oesophageal reflux disease"),
]


def paraphrase_chapters() -> list[dict[str, Any]]:
    first = [p for p in PARAPHRASE_PAIRS if p[0] < "P10"]
    second = [p for p in PARAPHRASE_PAIRS if p[0] >= "P10"]
    return [
        {
            "code": "I",
            "level": "CHAPTER",
            "title": "Synthetic paraphrase chapter",
            "range": ["P00", "P19"],
            "children": [
                {
                    "code": "P00-P09",
                    "level": "BLOCK",
                    "title": "Paraphrase block one",
                    "range": ["P00", "P09"],
                    "children": [
                        {"code": code, "level": "CATEGORY", "title": title}
                        for code, title, _ in first
                    ],
                },
                {
                    "code": "P10-P19",
                    "level": "BLOCK",
                    "title": "Paraphrase block two",
                    "range": ["P10", "P19"],
                    "children": [
                        {"code": code, "level": "CATEGORY", "title": title}
                        for code, title, _ in second
                    ],
                },
            ],
        }
    ]


def paraphrase_cases() -> list[dict[str, Any]]:
    """Retrieval evaluation cases: query -> expected code."""
    return [
        {
            "case_id": f"paraphrase-{code}",
            "clinical_note": query,
            "expected_code": code,
            "expected_dataset": SYNTHETIC_SYSTEM,
            "expected_version": PARAPHRASE_VERSION,
            "notes": f"paraphrase of '{title}' with no shared word stem",
        }
        for code, title, query in PARAPHRASE_PAIRS
    ]
