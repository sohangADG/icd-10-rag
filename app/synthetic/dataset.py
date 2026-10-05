"""Original synthetic ICD-like classification used to test and demonstrate the whole system.

Everything here was written for this project: condition names are deliberately invented
("glucose regulation disorder", "heart pump weakness"...) so the data cannot be mistaken for, and
does not reproduce, any real classification. It exercises every structural feature the system
supports: chapters, blocks, categories, subcategories and 5-character codes; inclusions,
exclusions, notes, code-first / use-additional-code / code-also instructions, see / see-also
references, source-provided synonyms, abbreviations and index terms; laterality, severity,
acuity, subtype and stage specificity; multiple versions.

Never use this data to make real coding decisions.
"""

import copy
from typing import Any

SYNTHETIC_SYSTEM = "SYNTH-ICD"

DATASET_2024: dict[str, Any] = {
    "coding_system": SYNTHETIC_SYSTEM,
    "country": "XX",
    "version": "2024",
    "language": "en",
    "title": "Synthetic ICD-like Classification (test fixture)",
    "publisher": "icd-rag-service synthetic fixtures",
    "publication_year": 2024,
    "licence": {"basis": "synthetic", "note": "Original test data; no third-party content."},
    "extra": {"synthetic": True, "allow_remote_processing": True},
}

# A record: code, level, title + optional rule lists. Children nest under "children".
CHAPTERS_2024: list[dict[str, Any]] = [
    {
        "code": "I",
        "level": "CHAPTER",
        "title": "Synthetic disorders of the airways",
        "range": ["A00", "A19"],
        "notes": ["This chapter groups synthetic airway conditions used for system testing."],
        "children": [
            {
                "code": "A00-A09",
                "level": "BLOCK",
                "title": "Infective airway disorders",
                "range": ["A00", "A09"],
                "children": [
                    {
                        "code": "A00",
                        "level": "CATEGORY",
                        "title": "Airway infection",
                        "inclusions": ["bronchial passage infection"],
                        "exclusions": ["airway infection in the newborn (B15)"],
                        "synonyms": ["respiratory tract infection"],
                        "index_terms": ["chest infection"],
                        "children": [
                            {
                                "code": "A00.0",
                                "level": "SUBCATEGORY",
                                "title": "Acute airway infection",
                                "inclusions": ["sudden-onset airway infection"],
                            },
                            {
                                "code": "A00.1",
                                "level": "SUBCATEGORY",
                                "title": "Chronic airway infection",
                                "inclusions": ["long-standing airway infection"],
                            },
                            {
                                "code": "A00.9",
                                "level": "SUBCATEGORY",
                                "title": "Airway infection, unspecified",
                            },
                        ],
                    },
                    {
                        "code": "A01",
                        "level": "CATEGORY",
                        "title": "Lobar consolidation of lung",
                        "synonyms": ["lobar pneumonia"],
                        "index_terms": ["pneumonia"],
                        "code_first": ["Code first any underlying airway infection (A00.-)"],
                        "children": [
                            {
                                "code": "A01.0",
                                "level": "SUBCATEGORY",
                                "title": "Lobar consolidation of left lung",
                            },
                            {
                                "code": "A01.1",
                                "level": "SUBCATEGORY",
                                "title": "Lobar consolidation of right lung",
                            },
                            {
                                "code": "A01.2",
                                "level": "SUBCATEGORY",
                                "title": "Lobar consolidation of both lungs",
                                "inclusions": ["bilateral lobar consolidation"],
                            },
                            {
                                "code": "A01.9",
                                "level": "SUBCATEGORY",
                                "title": "Lobar consolidation of lung, unspecified side",
                            },
                        ],
                    },
                    {
                        "code": "A02",
                        "level": "CATEGORY",
                        "title": "Wheezing airway disorder",
                        "synonyms": ["asthma"],
                        "use_additional_code": [
                            "Use additional code to identify exposure to tobacco smoke (D00.-)"
                        ],
                        "children": [
                            {
                                "code": "A02.0",
                                "level": "SUBCATEGORY",
                                "title": "Mild wheezing airway disorder",
                            },
                            {
                                "code": "A02.1",
                                "level": "SUBCATEGORY",
                                "title": "Moderate wheezing airway disorder",
                            },
                            {
                                "code": "A02.2",
                                "level": "SUBCATEGORY",
                                "title": "Severe wheezing airway disorder",
                            },
                            {
                                "code": "A02.9",
                                "level": "SUBCATEGORY",
                                "title": "Wheezing airway disorder, unspecified severity",
                            },
                        ],
                    },
                ],
            },
            {
                "code": "A10-A19",
                "level": "BLOCK",
                "title": "Other airway disorders",
                "range": ["A10", "A19"],
                "children": [
                    {
                        "code": "A10",
                        "level": "CATEGORY",
                        "title": "Persistent cough syndrome",
                        "inclusions": ["chronic cough of unknown cause"],
                        "exclusions": ["cough due to airway infection (A00.-)"],
                        "see_also": ["See also airway infection (A00.-)"],
                    },
                ],
            },
        ],
    },
    {
        "code": "II",
        "level": "CHAPTER",
        "title": "Synthetic disorders of the circulation",
        "range": ["B00", "B19"],
        "children": [
            {
                "code": "B00-B09",
                "level": "BLOCK",
                "title": "Blood pressure and heart pump disorders",
                "range": ["B00", "B09"],
                "children": [
                    {
                        "code": "B00",
                        "level": "CATEGORY",
                        "title": "Elevated blood pressure disorder",
                        "synonyms": ["hypertension", "high blood pressure"],
                        "abbreviations": ["HTN"],
                        "children": [
                            {
                                "code": "B00.0",
                                "level": "SUBCATEGORY",
                                "title": "Elevated blood pressure disorder with kidney involvement",
                                "use_additional_code": [
                                    "Use additional code to identify the stage of chronic kidney "
                                    "impairment (C12.-)"
                                ],
                            },
                            {
                                "code": "B00.9",
                                "level": "SUBCATEGORY",
                                "title": "Elevated blood pressure disorder without complication",
                            },
                        ],
                    },
                    {
                        "code": "B01",
                        "level": "CATEGORY",
                        "title": "Heart pump weakness",
                        "synonyms": ["heart failure", "cardiac failure"],
                        "abbreviations": ["CHF"],
                        "code_first": [
                            "Code first elevated blood pressure disorder with kidney involvement, "
                            "if present (B00.0)"
                        ],
                        "children": [
                            {
                                "code": "B01.0",
                                "level": "SUBCATEGORY",
                                "title": "Acute heart pump weakness",
                            },
                            {
                                "code": "B01.1",
                                "level": "SUBCATEGORY",
                                "title": "Chronic heart pump weakness",
                            },
                            {
                                "code": "B01.2",
                                "level": "SUBCATEGORY",
                                "title": "Acute on chronic heart pump weakness",
                            },
                            {
                                "code": "B01.9",
                                "level": "SUBCATEGORY",
                                "title": "Heart pump weakness, unspecified",
                            },
                        ],
                    },
                ],
            },
            {
                "code": "B10-B19",
                "level": "BLOCK",
                "title": "Conditions of the newborn",
                "range": ["B10", "B19"],
                "children": [
                    {
                        "code": "B15",
                        "level": "CATEGORY",
                        "title": "Airway infection in the newborn",
                        "inclusions": ["neonatal airway infection"],
                        "notes": [
                            "Use only for conditions arising in the first twenty-eight days after "
                            "birth."
                        ],
                    },
                ],
            },
        ],
    },
    {
        "code": "III",
        "level": "CHAPTER",
        "title": "Synthetic metabolic and kidney disorders",
        "range": ["C00", "C19"],
        "children": [
            {
                "code": "C00-C09",
                "level": "BLOCK",
                "title": "Glucose regulation disorders",
                "range": ["C00", "C09"],
                "children": [
                    {
                        "code": "C00",
                        "level": "CATEGORY",
                        "title": "Glucose regulation disorder",
                        "synonyms": ["diabetes mellitus", "diabetes"],
                        "abbreviations": ["DM"],
                        "code_also": [
                            "Code also any long-term use of glucose-lowering medication (D02)"
                        ],
                        "children": [
                            {
                                "code": "C00.1",
                                "level": "SUBCATEGORY",
                                "title": "Type 1 glucose regulation disorder",
                                "children": [
                                    {
                                        "code": "C00.10",
                                        "level": "CODE",
                                        "title": "Type 1 glucose regulation disorder with "
                                        "kidney complication",
                                        "use_additional_code": [
                                            "Use additional code to identify the stage of chronic "
                                            "kidney impairment (C12.-)"
                                        ],
                                    },
                                    {
                                        "code": "C00.19",
                                        "level": "CODE",
                                        "title": "Type 1 glucose regulation disorder without "
                                        "complication",
                                    },
                                ],
                            },
                            {
                                "code": "C00.2",
                                "level": "SUBCATEGORY",
                                "title": "Type 2 glucose regulation disorder",
                                "children": [
                                    {
                                        "code": "C00.20",
                                        "level": "CODE",
                                        "title": "Type 2 glucose regulation disorder with "
                                        "kidney complication",
                                        "use_additional_code": [
                                            "Use additional code to identify the stage of chronic "
                                            "kidney impairment (C12.-)"
                                        ],
                                    },
                                    {
                                        "code": "C00.29",
                                        "level": "CODE",
                                        "title": "Type 2 glucose regulation disorder without "
                                        "complication",
                                    },
                                ],
                            },
                            {
                                "code": "C00.9",
                                "level": "SUBCATEGORY",
                                "title": "Glucose regulation disorder, unspecified type",
                            },
                        ],
                    },
                ],
            },
            {
                "code": "C10-C19",
                "level": "BLOCK",
                "title": "Kidney disorders",
                "range": ["C10", "C19"],
                "children": [
                    {
                        "code": "C12",
                        "level": "CATEGORY",
                        "title": "Chronic kidney impairment",
                        "synonyms": ["chronic kidney disease"],
                        "abbreviations": ["CKD"],
                        "exclusions": ["acute kidney injury (C13)"],
                        "children": [
                            {
                                "code": "C12.1",
                                "level": "SUBCATEGORY",
                                "title": "Chronic kidney impairment, stage 1",
                            },
                            {
                                "code": "C12.2",
                                "level": "SUBCATEGORY",
                                "title": "Chronic kidney impairment, stage 2",
                            },
                            {
                                "code": "C12.3",
                                "level": "SUBCATEGORY",
                                "title": "Chronic kidney impairment, stage 3",
                            },
                            {
                                "code": "C12.9",
                                "level": "SUBCATEGORY",
                                "title": "Chronic kidney impairment, unspecified stage",
                            },
                        ],
                    },
                    {
                        "code": "C13",
                        "level": "CATEGORY",
                        "title": "Acute kidney injury",
                        "abbreviations": ["AKI"],
                        "exclusions": ["chronic kidney impairment (C12.-)"],
                        "see": ["See chronic kidney impairment for lasting loss of function (C12)"],
                    },
                ],
            },
        ],
    },
    {
        "code": "IV",
        "level": "CHAPTER",
        "title": "Synthetic factors influencing health status",
        "range": ["D00", "D09"],
        "children": [
            {
                "code": "D00-D09",
                "level": "BLOCK",
                "title": "Exposures, personal and family history",
                "range": ["D00", "D09"],
                "children": [
                    {
                        "code": "D00",
                        "level": "CATEGORY",
                        "title": "Tobacco exposure",
                        "children": [
                            {
                                "code": "D00.0",
                                "level": "SUBCATEGORY",
                                "title": "Current tobacco use",
                                "inclusions": ["smoker", "tobacco dependence"],
                            },
                            {
                                "code": "D00.1",
                                "level": "SUBCATEGORY",
                                "title": "Exposure to second-hand tobacco smoke",
                            },
                            {
                                "code": "D00.2",
                                "level": "SUBCATEGORY",
                                "title": "Personal history of tobacco use",
                                "inclusions": ["former smoker"],
                            },
                        ],
                    },
                    {
                        "code": "D01",
                        "level": "CATEGORY",
                        "title": "Family history of glucose regulation disorder",
                        "inclusions": ["family history of diabetes"],
                    },
                    {
                        "code": "D02",
                        "level": "CATEGORY",
                        "title": "Long-term use of glucose-lowering medication",
                        "synonyms": ["long-term insulin use"],
                        # Deliberately references a code outside this classification: the
                        # target must stay unresolved text, never become a code.
                        "see": ["See medication review guidance (Z99)"],
                    },
                ],
            },
        ],
    },
    # Anatomical-site, encounter and cause specificity (clinical-evaluation scenarios).
    {
        "code": "V",
        "level": "CHAPTER",
        "title": "Synthetic joint, bone and skin conditions",
        "range": ["E00", "E19"],
        "children": [
            {
                "code": "E00-E09",
                "level": "BLOCK",
                "title": "Joint and bone conditions",
                "range": ["E00", "E09"],
                "children": [
                    {
                        "code": "E00",
                        "level": "CATEGORY",
                        "title": "Joint inflammation",
                        "synonyms": ["arthritis"],
                        "inclusions": ["inflamed joint"],
                        "children": [
                            {
                                "code": "E00.0",
                                "level": "SUBCATEGORY",
                                "title": "Joint inflammation of knee",
                            },
                            {
                                "code": "E00.1",
                                "level": "SUBCATEGORY",
                                "title": "Joint inflammation of hip",
                            },
                            {
                                "code": "E00.9",
                                "level": "SUBCATEGORY",
                                "title": "Joint inflammation, unspecified site",
                            },
                        ],
                    },
                    {
                        "code": "E01",
                        "level": "CATEGORY",
                        "title": "Fracture of forearm bone",
                        "synonyms": ["broken forearm"],
                        "notes": [
                            "An encounter detail (initial, subsequent or sequela) is required for "
                            "this category."
                        ],
                        "children": [
                            {
                                "code": "E01.0",
                                "level": "SUBCATEGORY",
                                "title": "Fracture of forearm bone, initial encounter",
                            },
                            {
                                "code": "E01.1",
                                "level": "SUBCATEGORY",
                                "title": "Fracture of forearm bone, subsequent encounter",
                            },
                            {
                                "code": "E01.2",
                                "level": "SUBCATEGORY",
                                "title": "Fracture of forearm bone, sequela",
                            },
                        ],
                    },
                ],
            },
            {
                "code": "E10-E19",
                "level": "BLOCK",
                "title": "Skin and wound conditions",
                "range": ["E10", "E19"],
                "children": [
                    {
                        "code": "E10",
                        "level": "CATEGORY",
                        "title": "Skin wound infection",
                        "children": [
                            {
                                "code": "E10.0",
                                "level": "SUBCATEGORY",
                                "title": "Skin wound infection due to animal bite",
                            },
                            {
                                "code": "E10.1",
                                "level": "SUBCATEGORY",
                                "title": "Skin wound infection due to foreign body",
                            },
                            {
                                "code": "E10.9",
                                "level": "SUBCATEGORY",
                                "title": "Skin wound infection, unspecified cause",
                            },
                        ],
                    },
                ],
            },
        ],
    },
]

RULE_LIST_FIELDS = (
    "notes",
    "code_first",
    "use_additional_code",
    "code_also",
    "see",
    "see_also",
)
TERM_LIST_FIELDS = ("synonyms", "abbreviations", "index_terms")


def _find(chapters: list[dict[str, Any]], code: str) -> tuple[list[dict[str, Any]], int]:
    stack = [chapters]
    while stack:
        siblings = stack.pop()
        for index, node in enumerate(siblings):
            if node["code"] == code:
                return siblings, index
            stack.append(node.get("children", []))
    raise KeyError(code)


def chapters_2025() -> list[dict[str, Any]]:
    """Version 2025, with deliberate differences from 2024 (version-isolation tests):

    * added code A01.3; removed code D00.2; retitled code B00.9;
    * changed parent: A10 moves from block A10-A19 to the new block A10-A14;
    * changed rule: B01 "code first B00.0" becomes "code also C12.-";
    * changed inclusion: A00.1 "long-standing" becomes "persistent airway infection";
    * changed exclusion: A10 excludes cough due to the wheezing disorder, not to infection.
    """
    chapters = copy.deepcopy(CHAPTERS_2024)
    siblings, index = _find(chapters, "A01.2")
    siblings.insert(
        index + 1,
        {"code": "A01.3", "level": "SUBCATEGORY", "title": "Lobar consolidation of multiple lobes"},
    )
    siblings, index = _find(chapters, "B00.9")
    siblings[index]["title"] = "Elevated blood pressure disorder without organ involvement"
    siblings, index = _find(chapters, "D00.2")
    del siblings[index]

    siblings, index = _find(chapters, "A10-A19")
    block = siblings[index]
    block.update(code="A10-A14", title="Cough disorders", range=["A10", "A14"])
    siblings, index = _find(chapters, "B01")
    del siblings[index]["code_first"]
    siblings[index]["code_also"] = ["Code also chronic kidney impairment, if present (C12.-)"]
    siblings, index = _find(chapters, "A00.1")
    siblings[index]["inclusions"] = ["persistent airway infection"]
    siblings, index = _find(chapters, "A10")
    siblings[index]["exclusions"] = ["cough due to wheezing airway disorder (A02.-)"]
    return chapters


def dataset_2025() -> dict[str, Any]:
    return {**DATASET_2024, "version": "2025", "publication_year": 2025}


def flatten(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Depth-first flat records with explicit parent_code (source order preserved)."""
    flat: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], parent: dict[str, Any] | None) -> None:
        record = {k: v for k, v in node.items() if k != "children"}
        record["parent_code"] = parent["code"] if parent else None
        record["parent_level"] = parent["level"] if parent else None
        flat.append(record)
        for child in node.get("children", []):
            walk(child, node)

    for chapter in chapters:
        walk(chapter, None)
    return flat


def malformed_records() -> list[dict[str, Any]]:
    """Deliberately broken flat records for validation tests (one problem per record)."""
    return [
        {"code": "I", "level": "CHAPTER", "title": "Broken chapter", "parent_code": None},
        {"code": "A00-A09", "level": "BLOCK", "title": "Broken block", "parent_code": "I"},
        {"code": "A00", "level": "CATEGORY", "title": "Valid category", "parent_code": "A00-A09"},
        # duplicate code
        {
            "code": "A00",
            "level": "CATEGORY",
            "title": "Duplicate category",
            "parent_code": "A00-A09",
        },
        # malformed code
        {"code": "1AB", "level": "CATEGORY", "title": "Malformed code", "parent_code": "A00-A09"},
        # missing title
        {"code": "A01", "level": "CATEGORY", "title": "", "parent_code": "A00-A09"},
        # orphan: parent does not exist
        {"code": "A02.1", "level": "SUBCATEGORY", "title": "Orphan", "parent_code": "A02"},
        # cycle: A03.1 <-> A03.2
        {"code": "A03.1", "level": "SUBCATEGORY", "title": "Cycle one", "parent_code": "A03.2"},
        {"code": "A03.2", "level": "SUBCATEGORY", "title": "Cycle two", "parent_code": "A03.1"},
        # unresolved cross-reference (warning only)
        {
            "code": "A04",
            "level": "CATEGORY",
            "title": "Dangling reference",
            "parent_code": "A00-A09",
            "exclusions": ["something else (A99)"],
        },
    ]
