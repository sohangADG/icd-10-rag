import json
from pathlib import Path

import pytest

from app.core.constants import InstructionType, NodeType, SourceType
from app.ingestion.adapters import (
    ADAPTERS,
    AdapterError,
    CsvAdapter,
    JsonAdapter,
    SourceManifest,
    TsvAdapter,
    XmlAdapter,
    select_adapter,
)
from app.ingestion.adapters.base import sha256_of_file
from app.ingestion.pipeline import run_pipeline
from tests.helpers import manifest_for, write_source

ALL_FORMATS = ["json", "csv", "tsv", "xlsx", "xml", "text", "pdf"]
# Formats whose layout carries an alphabetical index / synonyms.
TERM_FORMATS = {"json", "csv", "tsv", "xlsx"}


def _structure(result) -> dict:  # noqa: ANN001
    hierarchy = result.hierarchy
    return {
        record.key: (
            record.title,
            hierarchy.nodes[record.key].parent_key,
            sorted(i.text for i in record.inclusions),
            sorted((e.text, e.target_code) for e in record.exclusions),
            sorted((i.instruction_type.value, i.text, i.target_code) for i in record.instructions),
        )
        for record in result.records
    }


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict:
    directory = tmp_path_factory.mktemp("reference")
    path, manifest = write_source(directory, "json")
    return _structure(run_pipeline(path, manifest))


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_every_adapter_reconstructs_the_same_classification(
    fmt: str, tmp_path: Path, reference: dict
) -> None:
    path, manifest = write_source(tmp_path, fmt)
    result = run_pipeline(path, manifest)

    assert result.is_valid, [i.message for i in result.fatal_issues]
    assert _structure(result) == reference
    stats = result.statistics()
    assert (stats["chapters"], stats["blocks"], stats["categories"]) == (5, 9, 16)
    assert (stats["subcategories"], stats["codes"]) == (36, 4)
    assert stats["index_terms"] == (20 if fmt in TERM_FORMATS else 0)
    assert result.metadata.coding_system == "SYNTH-ICD"
    assert result.metadata.source_sha256 == sha256_of_file(path)
    for record in result.records:
        assert record.provenance is not None
        assert record.provenance.source_filename.startswith("synth_2024")


def test_row_provenance_for_tabular_sources(tmp_path: Path) -> None:
    path, manifest = write_source(tmp_path, "xlsx")
    result = run_pipeline(path, manifest)
    first = result.records[0]
    assert (first.provenance.sheet, first.provenance.row) == ("Codes", 2)
    a00 = next(r for r in result.records if r.code == "A00")
    assert a00.exclusions[0].provenance.sheet == "Rules"


def test_versions_are_distinct(tmp_path: Path) -> None:
    v2024 = run_pipeline(*write_source(tmp_path, "json", "2024"))
    v2025 = run_pipeline(*write_source(tmp_path, "json", "2025"))
    codes_2024 = {r.code for r in v2024.records}
    codes_2025 = {r.code for r in v2025.records}
    # Added code and renamed block (A10 moved from A10-A19 to A10-A14); removed code.
    assert codes_2025 - codes_2024 == {"A01.3", "A10-A14"}
    assert codes_2024 - codes_2025 == {"D00.2", "A10-A19"}
    assert v2025.metadata.version == "2025"


def test_identity_is_never_taken_from_the_filename(tmp_path: Path) -> None:
    path = tmp_path / "ICD10CA_2022_final.csv"
    path.write_text("code,title\nA00,Something\n", encoding="utf-8")
    result = run_pipeline(path, SourceManifest())
    assert not result.is_valid
    assert result.fatal_issues[0].code == "INVALID_SOURCE"
    assert "metadata" in result.fatal_issues[0].message


def test_unsupported_coding_system_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("code,title\nA00,Something\n", encoding="utf-8")
    result = run_pipeline(path, manifest_for({"coding_system": "ICD-99"}))
    assert [i.code for i in result.fatal_issues] == ["UNSUPPORTED_CODING_SYSTEM"]


def test_csv_legacy_encoding_and_custom_columns(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_bytes(
        "Code;Libellé;Parent;Inclusions\nA00;Infection des voies aériennes;;fièvre|toux\n"
        "A00.1;Forme chronique;A00;\n".encode("cp1252")
    )
    manifest = manifest_for(
        mapping={
            "delimiter": ";",
            "columns": {
                "code": "Code",
                "title": "Libellé",
                "parent_code": "Parent",
                "inclusions": "Inclusions",
            },
        }
    )
    adapter = CsvAdapter(path, manifest)
    records = list(adapter.iterate_records())
    assert adapter.detected_encoding == "cp1252"
    assert records[0].title == "Infection des voies aériennes"
    assert [i.text for i in records[0].inclusions] == ["fièvre", "toux"]
    assert records[1].parent_code == "A00"
    assert records[1].level == NodeType.SUBCATEGORY


def test_tsv_inline_rules_are_typed(tmp_path: Path) -> None:
    path, manifest = write_source(tmp_path, "tsv")
    records = {r.code: r for r in TsvAdapter(path, manifest).iterate_records()}
    a01 = records["A01"]
    assert [(i.instruction_type, i.target_code) for i in a01.instructions] == [
        (InstructionType.CODE_FIRST, "A00")
    ]
    assert {t.kind.value for t in records["B00"].index_terms} == {"SYNONYM", "ABBREVIATION"}


def test_json_structured_rule_objects_keep_explicit_targets(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "code": "A00",
                        "title": "Parent",
                        "exclusions": [
                            {"text": "thing described in words", "target_code": "B15"},
                            {"text": "two codes (A01) and (A02)"},
                        ],
                        "children": [{"code": "A00.1", "title": "Child"}],
                    }
                ]
            }
        )
    )
    records = list(JsonAdapter(path, manifest_for()).iterate_records())
    parent, child = records
    assert [(e.text, e.target_code) for e in parent.exclusions] == [
        ("thing described in words", "B15"),
        ("two codes (A01) and (A02)", None),
    ]
    assert child.parent_code == "A00"
    assert child.provenance.element_path == "records[0]/children[0]"


def test_json_bad_records_path_is_an_adapter_error(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    path.write_text('{"records": {}}')
    with pytest.raises(AdapterError):
        list(JsonAdapter(path, manifest_for()).iterate_records())


def test_generic_xml_mapping_with_nesting(tmp_path: Path) -> None:
    path = tmp_path / "g.xml"
    path.write_text(
        """<Root>
          <Item code="A00"><Name>Parent</Name>
            <Excl>child of the newborn (B15)</Excl>
            <Kids><Item code="A00.1"><Name>Child</Name><Incl>inclusion one</Incl></Item></Kids>
          </Item>
        </Root>"""
    )
    manifest = manifest_for(
        mapping={
            "record_path": "./Item",
            "children_path": "Kids/Item",
            "fields": {
                "code": "@code",
                "title": "Name",
                "exclusions": "Excl",
                "inclusions": "Incl",
            },
        }
    )
    records = list(XmlAdapter(path, manifest).iterate_records())
    assert [(r.code, r.title, r.parent_code) for r in records] == [
        ("A00", "Parent", None),
        ("A00.1", "Child", "A00"),
    ]
    assert records[0].exclusions[0].target_code == "B15"
    assert records[1].inclusions[0].text == "inclusion one"


def test_xml_external_entities_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "evil.xml"
    path.write_text(
        '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        "<Root><Item code='A00'><Name>&x;</Name></Item></Root>"
    )
    manifest = manifest_for(
        mapping={"record_path": "./Item", "fields": {"code": "@code", "title": "Name"}}
    )
    with pytest.raises(AdapterError):
        list(XmlAdapter(path, manifest).iterate_records())


def test_claml_references_and_coding_hints(tmp_path: Path) -> None:
    path, manifest = write_source(tmp_path, "xml")
    records = {r.code: r for r in XmlAdapter(path, manifest).iterate_records()}
    assert records["A00"].exclusions[0].target_code == "B15"
    assert records["B01"].instructions[0].instruction_type == InstructionType.CODE_FIRST
    assert records["A10"].instructions[0].instruction_type == InstructionType.SEE_ALSO
    assert records["A00-A09"].level == NodeType.BLOCK


def test_adapter_detection_and_explicit_selection(tmp_path: Path) -> None:
    for fmt, source_type in (
        ("pdf", SourceType.PDF),
        ("xlsx", SourceType.XLSX),
        ("xml", SourceType.XML),
        ("csv", SourceType.CSV),
    ):
        path, _ = write_source(tmp_path, fmt)
        assert select_adapter(path).source_type == source_type
    path, _ = write_source(tmp_path, "pdf")
    forced = select_adapter(path, SourceManifest(adapter=SourceType.PDF_OCR))
    assert forced.source_type == SourceType.PDF_OCR
    unknown = tmp_path / "x.bin"
    unknown.write_bytes(b"\x00\x01")
    with pytest.raises(AdapterError):
        select_adapter(unknown)
    with pytest.raises(AdapterError):
        select_adapter(tmp_path / "missing.csv")
    assert set(ADAPTERS) == set(SourceType)
