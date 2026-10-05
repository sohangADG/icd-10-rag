from datetime import date

import pytest

from app.core.constants import NodeType, SourceType
from app.ingestion.adapters.fields import FieldMapping, RecordAssembler
from app.ingestion.hierarchy import HierarchyBuilder, HierarchyConfig
from app.ingestion.models import DatasetMetadata, NormalizedICDRecord, SourceProvenance
from app.ingestion.validator import (
    ICD10CAValidator,
    ICD10CMValidator,
    SyntheticICDValidator,
    supported_coding_systems,
    validator_for,
)
from app.synthetic.dataset import CHAPTERS_2024, flatten, malformed_records

PROVENANCE = SourceProvenance(source_filename="t.csv", kind=SourceType.CSV, row=1)


def _records(rows: list[dict]) -> list[NormalizedICDRecord]:
    assembler = RecordAssembler(FieldMapping())
    for index, row in enumerate(rows, start=1):
        assembler.add_row(
            {
                **row,
                "range_start": (row.get("range") or [None, None])[0],
                "range_end": (row.get("range") or [None, None])[1],
            },
            SourceProvenance(source_filename="t.csv", kind=SourceType.CSV, row=index),
        )
    return assembler.records()


def _issue_codes(issues) -> set[str]:  # noqa: ANN001
    return {issue.code for issue in issues}


def test_explicit_hierarchy_is_reconstructed() -> None:
    records = _records(flatten(CHAPTERS_2024))
    result = HierarchyBuilder().build(records)

    assert not [i for i in result.issues if i.is_fatal]
    node = result.nodes["CODE:C0020"]
    assert node.parent_key == "SUBCATEGORY:C002"
    assert result.ancestors("CODE:C0020") == [
        "SUBCATEGORY:C002",
        "CATEGORY:C00",
        "BLOCK:C00-C09",
        "CHAPTER:III",
    ]
    assert (node.chapter_key, node.block_key, node.category_key) == (
        "CHAPTER:III",
        "BLOCK:C00-C09",
        "CATEGORY:C00",
    )
    assert node.depth == 4
    assert set(result.descendants("CATEGORY:C00")) >= {"SUBCATEGORY:C001", "CODE:C0010"}
    # Leaves are selectable, categories with subdivisions are not.
    assert node.is_selectable and not result.nodes["CATEGORY:C00"].is_selectable
    # Parents always precede children in insertion order.
    order = [n.record.key for n in result.ordered_for_insert()]
    assert order.index("CATEGORY:C00") < order.index("SUBCATEGORY:C002") < order.index("CODE:C0020")


def test_malformed_fixture_reports_every_problem() -> None:
    records = _records(malformed_records())
    hierarchy = HierarchyBuilder().build(records)
    validator = SyntheticICDValidator()
    metadata = DatasetMetadata(
        coding_system="SYNTH-ICD",
        country="XX",
        version="bad",
        language="en",
        publisher="t",
        source_sha256="0" * 64,
    )
    issues = (
        hierarchy.issues
        + validator.validate_records(records, metadata)
        + validator.validate_hierarchy(hierarchy)
    )
    codes = _issue_codes(issues)

    assert {
        "DUPLICATE_CODE",
        "MALFORMED_CODE",
        "MISSING_TITLE",
        "MISSING_PARENT",
        "HIERARCHY_CYCLE",
        "UNRESOLVED_CROSS_REFERENCE",
    } <= codes
    assert hierarchy.duplicates == 1 and hierarchy.cycles == 1 and hierarchy.orphans >= 1
    # The unresolved cross-reference is only a warning; the others are fatal.
    unresolved = [i for i in issues if i.code == "UNRESOLVED_CROSS_REFERENCE"]
    assert all(not i.is_fatal for i in unresolved)


def test_missing_parent_is_never_invented() -> None:
    records = _records(
        [
            {"code": "A00.1", "level": "SUBCATEGORY", "title": "child", "parent_code": "A00"},
        ]
    )
    result = HierarchyBuilder().build(records)
    assert result.nodes["SUBCATEGORY:A001"].parent_key is None
    assert "MISSING_PARENT" in _issue_codes(result.issues)
    assert len(result.nodes) == 1  # no placeholder parent created


def test_parent_inference_is_off_by_default_and_reported_when_enabled() -> None:
    rows = [
        {"code": "I", "level": "CHAPTER", "title": "chapter", "range": ["A00", "A09"]},
        {"code": "A00-A09", "level": "BLOCK", "title": "block", "range": ["A00", "A09"]},
        {"code": "A00", "level": "CATEGORY", "title": "category"},
        {"code": "A00.1", "level": "SUBCATEGORY", "title": "subcategory"},
    ]
    strict = HierarchyBuilder().build(_records(rows))
    assert "ORPHAN_RECORD" in _issue_codes(strict.issues)

    inferred = HierarchyBuilder(HierarchyConfig(infer_parents=True)).build(_records(rows))
    assert not [i for i in inferred.issues if i.is_fatal]
    assert inferred.nodes["SUBCATEGORY:A001"].parent_key == "CATEGORY:A00"
    assert inferred.nodes["CATEGORY:A00"].parent_key == "BLOCK:A00-A09"
    assert inferred.nodes["BLOCK:A00-A09"].parent_key == "CHAPTER:I"
    assert inferred.inferred_links == 3
    assert [i.code for i in inferred.issues].count("PARENT_INFERRED") == 3


def test_inference_never_links_to_a_nonexistent_code() -> None:
    rows = [{"code": "A00.1", "level": "SUBCATEGORY", "title": "lonely"}]
    result = HierarchyBuilder(HierarchyConfig(infer_parents=True)).build(_records(rows))
    assert result.nodes["SUBCATEGORY:A001"].parent_key is None
    assert result.inferred_links == 0


def test_level_inconsistency_and_code_mismatch() -> None:
    rows = [
        {"code": "A00", "level": "CATEGORY", "title": "category"},
        {"code": "B00.1", "level": "SUBCATEGORY", "title": "foreign", "parent_code": "A00"},
        {"code": "A01", "level": "CATEGORY", "title": "upside down", "parent_code": "A00.9"},
        {"code": "A00.9", "level": "SUBCATEGORY", "title": "sub", "parent_code": "A00"},
    ]
    codes = _issue_codes(HierarchyBuilder().build(_records(rows)).issues)
    assert "HIERARCHY_CODE_MISMATCH" in codes
    assert "HIERARCHY_LEVEL_INCONSISTENT" in codes


def test_flat_code_list_without_groupings_is_valid() -> None:
    rows = [{"code": "A00", "title": "one"}, {"code": "A01", "title": "two"}]
    result = HierarchyBuilder().build(_records(rows))
    assert not result.issues


@pytest.mark.parametrize(
    ("validator", "valid", "invalid"),
    [
        (ICD10CAValidator(), ["A00", "E10.52", "K50.808"], ["A00.0000", "1AB", "A0"]),
        (ICD10CMValidator(), ["S72.001A", "C7A.0", "T36.0X1A"], ["S72.00122", "72A"]),
        (SyntheticICDValidator(), ["A00", "C00.10"], ["C00.100", "C7A"]),
    ],
)
def test_per_system_code_formats(validator, valid: list[str], invalid: list[str]) -> None:  # noqa: ANN001
    def record(code: str) -> NormalizedICDRecord:
        return NormalizedICDRecord(
            key=code, code=code, level=NodeType.CATEGORY, title="t", provenance=PROVENANCE
        )

    assert all(validator.validate_code_format(record(code)) is None for code in valid)
    assert all(validator.validate_code_format(record(code)) is not None for code in invalid)


def test_validator_registry_does_not_mix_systems() -> None:
    assert validator_for("icd-10-ca").coding_systems == ("ICD-10-CA",)
    assert validator_for("ICD-10-CM").coding_systems == ("ICD-10-CM",)
    assert validator_for("ICD-11") is None
    assert {"ICD-10", "ICD-10-CA", "ICD-10-CM", "SYNTH-ICD"} <= set(supported_coding_systems())


def test_metadata_validation() -> None:
    validator = ICD10CAValidator()
    good = DatasetMetadata(
        coding_system="ICD-10-CA",
        country="CA",
        version="2022",
        language="en",
        publisher="p",
        source_sha256="a" * 64,
    )
    assert validator.validate_metadata(good) == []
    no_hash = good.model_copy(update={"source_sha256": None})
    wrong_system = good.model_copy(update={"coding_system": "ICD-10-CM"})
    inverted = good.model_copy(
        update={"effective_from": date(2023, 1, 1), "effective_to": date(2022, 1, 1)}
    )
    for metadata in (no_hash, wrong_system, inverted):
        assert {i.code for i in validator.validate_metadata(metadata)} == {
            "INVALID_DATASET_METADATA"
        }


def test_provenance_validation() -> None:
    validator = SyntheticICDValidator()
    metadata = DatasetMetadata(
        coding_system="SYNTH-ICD",
        country="XX",
        version="1",
        language="en",
        publisher="p",
        source_sha256="a" * 64,
        source_page_count=10,
    )
    page_out_of_range = NormalizedICDRecord(
        key="CATEGORY:A00",
        code="A00",
        level=NodeType.CATEGORY,
        title="t",
        provenance=SourceProvenance(
            source_filename="x.pdf", kind=SourceType.PDF, page_start=11, page_end=11
        ),
    )
    no_provenance = NormalizedICDRecord(
        key="CATEGORY:A01", code="A01", level=NodeType.CATEGORY, title="t"
    )
    codes = _issue_codes(validator.validate_records([page_out_of_range, no_provenance], metadata))
    assert {"INVALID_SOURCE_LOCATOR", "RECORD_WITHOUT_PROVENANCE"} <= codes


def test_empty_dataset_and_empty_sections() -> None:
    validator = SyntheticICDValidator()
    metadata = DatasetMetadata(
        coding_system="SYNTH-ICD",
        country="XX",
        version="1",
        language="en",
        publisher="p",
        source_sha256="a" * 64,
    )
    only_chapter = _records([{"code": "I", "level": "CHAPTER", "title": "empty"}])
    assert "EMPTY_DATASET" in _issue_codes(validator.validate_records(only_chapter, metadata))
    hierarchy = HierarchyBuilder().build(only_chapter)
    assert "EMPTY_SECTION" in _issue_codes(validator.validate_hierarchy(hierarchy))


def test_separate_rules_without_owner_are_errors() -> None:
    assembler = RecordAssembler(FieldMapping())
    assembler.add_row({"code": "A00", "title": "t"}, PROVENANCE)
    assembler.add_rule_row({"owner_code": "Z99", "rule_type": "NOTE", "text": "x"}, PROVENANCE)
    assembler.add_rule_row({"rule_type": "NOTE", "text": "x"}, PROVENANCE)
    assembler.add_rule_row({"owner_code": "A00", "rule_type": "WHATEVER", "text": "x"}, PROVENANCE)
    assert [i.code for i in assembler.issues] == [
        "RULE_WITHOUT_OWNER",
        "RULE_WITHOUT_OWNER",
        "UNKNOWN_RULE_TYPE",
    ]
