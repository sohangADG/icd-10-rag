import pytest

from app.core.constants import InstructionType, NodeType
from app.ingestion.codes import (
    clean_code,
    code_in_range,
    find_code_references,
    level_from_code_shape,
    looks_like_code,
    normalize_code,
    parse_range,
    record_key,
    truncation_parents,
)
from app.ingestion.instructions import (
    make_exclusion,
    make_instruction,
    match_marker,
    parse_instruction_type,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("a00.0", "A000"), (" A00.0† ", "A000"), ("E10.52*", "E1052"), ("A00", "A00")],
)
def test_classification_codes_normalize_without_punctuation(raw: str, expected: str) -> None:
    assert normalize_code(raw) == expected


def test_grouping_codes_keep_their_range_hyphen() -> None:
    assert normalize_code("a00–a09", NodeType.BLOCK) == "A00-A09"
    assert normalize_code("A00-A09") == "A00-A09"
    assert normalize_code("iv", NodeType.CHAPTER) == "IV"


def test_clean_code_removes_decorations_and_whitespace() -> None:
    assert clean_code(" a 00 . 1 † ") == "A00.1"


def test_record_key_is_level_qualified() -> None:
    assert record_key(NodeType.CATEGORY, "A00", "1") == "CATEGORY:A00"
    assert record_key(NodeType.BLOCK, "A00-A09", "1") == "BLOCK:A00-A09"
    assert record_key(None, None, "7") == "UNKNOWN:#7"


def test_single_explicit_reference_becomes_target() -> None:
    references = find_code_references("airway infection in the newborn (B15)")
    assert references.single_target == "B15"


def test_dash_suffix_targets_the_category() -> None:
    references = find_code_references("Code first any underlying airway infection (A00.-)")
    assert references.codes == ["A00"]
    assert references.single_target == "A00"


def test_ambiguous_references_never_produce_a_target() -> None:
    two = find_code_references("infection (A00) or consolidation (A01)")
    assert two.single_target is None
    assert two.all_referenced == ["A00", "A01"]
    ranged = find_code_references("conditions of the newborn (B10-B19)")
    assert ranged.single_target is None
    assert ranged.ranges == [("B10", "B19")]


def test_text_without_codes_has_no_references() -> None:
    assert find_code_references("Use only for conditions of the first 28 days").codes == []


def test_ranges_and_containment() -> None:
    assert parse_range("A00-A09") == ("A00", "A09")
    assert parse_range("A00") is None
    assert code_in_range("A05.1", "A00", "A09")
    assert not code_in_range("A10", "A00", "A09")
    assert not code_in_range("B00", "A00", "A99")


def test_truncation_parents_nearest_first() -> None:
    assert truncation_parents("C00.10") == ["C00.1", "C00"]
    assert truncation_parents("A00.0") == ["A00"]
    assert truncation_parents("A00") == []


@pytest.mark.parametrize(
    ("code", "level"),
    [("A00", NodeType.CATEGORY), ("A00.0", NodeType.SUBCATEGORY), ("A00.00", NodeType.CODE)],
)
def test_level_from_code_shape(code: str, level: NodeType) -> None:
    assert level_from_code_shape(code) == level


def test_looks_like_code() -> None:
    assert looks_like_code("a01.0")
    assert not looks_like_code("airway")
    assert not looks_like_code("1AB")


@pytest.mark.parametrize(
    ("line", "kind", "exclusion_type", "remainder"),
    [
        (
            "Excludes: newborn infection (B15)",
            InstructionType.EXCLUDES,
            None,
            "newborn infection (B15)",
        ),
        ("Excludes1: x", InstructionType.EXCLUDES, "EXCLUDES1", "x"),
        ("Excludes2 y", InstructionType.EXCLUDES, "EXCLUDES2", "y"),
        ("Includes: bronchial infection", InstructionType.INCLUDES, None, "bronchial infection"),
        ("Note: use only for newborns", InstructionType.NOTE, None, "use only for newborns"),
    ],
)
def test_label_markers(
    line: str, kind: InstructionType, exclusion_type: str | None, remainder: str
) -> None:
    marker = match_marker(line)
    assert marker is not None and marker.is_label
    assert (marker.instruction_type, marker.exclusion_type, marker.remainder) == (
        kind,
        exclusion_type,
        remainder,
    )


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("Code first any underlying infection (A00.-)", InstructionType.CODE_FIRST),
        ("Use additional code to identify exposure (D00.-)", InstructionType.USE_ADDITIONAL_CODE),
        ("Code also any long-term medication use (D02)", InstructionType.CODE_ALSO),
        ("See also airway infection (A00.-)", InstructionType.SEE_ALSO),
        ("See airway infection", InstructionType.SEE),
    ],
)
def test_phrase_markers_keep_the_full_text(line: str, kind: InstructionType) -> None:
    marker = match_marker(line)
    assert marker is not None and not marker.is_label
    assert marker.instruction_type == kind and marker.remainder == line


def test_plain_text_is_not_a_marker() -> None:
    assert match_marker("bronchial passage infection") is None
    assert match_marker("Seen in clinic") is None  # "See" must be a whole word


def test_parse_instruction_type_labels() -> None:
    assert parse_instruction_type("use additional code") == InstructionType.USE_ADDITIONAL_CODE
    assert parse_instruction_type("EXCLUDES1") == InstructionType.EXCLUDES
    assert parse_instruction_type("inclusion") == InstructionType.INCLUDES
    assert parse_instruction_type("bogus") is None


def test_rule_builders_keep_text_and_only_explicit_targets() -> None:
    exclusion = make_exclusion("acute kidney injury (C13)")
    assert (exclusion.text, exclusion.target_code) == ("acute kidney injury (C13)", "C13")
    ambiguous = make_instruction(InstructionType.NOTE, "see A00 and A01")
    assert ambiguous.target_code is None and ambiguous.referenced_codes == ["A00", "A01"]
    explicit = make_exclusion("something", explicit_target="B15")
    assert explicit.target_code == "B15"
