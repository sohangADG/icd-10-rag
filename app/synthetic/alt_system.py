"""A second, independent synthetic coding system with deliberately overlapping code strings.

SYNTH-ALT reuses SYNTH-ICD code strings (A00, A01, B00...) for *different* conditions, and
reuses one SYNTH-ICD title verbatim under a different code. It exists only to prove that
the coding system is part of every lookup: the same code string or title in another system
must never leak into results. Original test data; never use it for real coding.
"""

from typing import Any

ALT_SYSTEM = "SYNTH-ALT"

DATASET_ALT: dict[str, Any] = {
    "coding_system": ALT_SYSTEM,
    "country": "XX",
    "version": "2024",
    "language": "en",
    "title": "Synthetic alternative classification (contamination test fixture)",
    "publisher": "icd-rag-service synthetic fixtures",
    "publication_year": 2024,
    "licence": {"basis": "synthetic", "note": "Original test data; no third-party content."},
    "extra": {"synthetic": True, "allow_remote_processing": True},
}

ALT_CHAPTERS: list[dict[str, Any]] = [
    {
        "code": "I",
        "level": "CHAPTER",
        "title": "Synthetic alternative ear and eye conditions",
        "range": ["A00", "A09"],
        "children": [
            {
                "code": "A00-A09",
                "level": "BLOCK",
                "title": "Ear and eye irritation",
                "range": ["A00", "A09"],
                "children": [
                    # Same code strings as SYNTH-ICD, different meaning.
                    {"code": "A00", "level": "CATEGORY", "title": "Ear canal irritation"},
                    {
                        "code": "A01",
                        "level": "CATEGORY",
                        "title": "Eyelid swelling",
                        "children": [
                            {
                                "code": "A01.0",
                                "level": "SUBCATEGORY",
                                "title": "Eyelid swelling of left eye",
                            },
                            {
                                "code": "A01.1",
                                "level": "SUBCATEGORY",
                                "title": "Eyelid swelling of right eye",
                            },
                        ],
                    },
                ],
            },
        ],
    },
    {
        "code": "II",
        "level": "CHAPTER",
        "title": "Synthetic alternative circulation conditions",
        "range": ["B00", "B09"],
        "children": [
            {
                "code": "B00-B09",
                "level": "BLOCK",
                "title": "Alternative circulation block",
                "range": ["B00", "B09"],
                "children": [
                    {"code": "B00", "level": "CATEGORY", "title": "Cold hands and feet"},
                    # The SYNTH-ICD B00 title, under another code of another system.
                    {
                        "code": "B05",
                        "level": "CATEGORY",
                        "title": "Elevated blood pressure disorder",
                    },
                ],
            },
        ],
    },
]
