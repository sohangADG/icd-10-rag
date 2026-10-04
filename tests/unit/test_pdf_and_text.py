from pathlib import Path

import pytest

from app.core.constants import InstructionType, NodeType, SourceType
from app.ingestion.adapters import AdapterError, SourceRestrictedError
from app.ingestion.adapters.layout_adapters import OcrPdfAdapter, PdfAdapter, TextAdapter
from app.ingestion.models import SourceProvenance
from app.ingestion.pipeline import run_pipeline
from app.ingestion.text.layout import PageText, TextLine
from app.ingestion.text.page_cleaner import PageCleaner, normalize_line
from app.ingestion.text.pdf_reader import Word, extract_pdf_pages, inspect_pdf_security
from app.ingestion.text.structure_parser import StructureParser, TextLayoutProfile
from app.synthetic.renderers import write_two_column_pdf
from tests.helpers import manifest_for, write_source


@pytest.fixture(scope="module")
def pdf_result(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201
    directory = tmp_path_factory.mktemp("pdf")
    path, manifest = write_source(directory, "pdf")
    adapter = PdfAdapter(path, manifest)
    records = list(adapter.iterate_records())
    return adapter, {r.code: r for r in records}


def test_pdf_pages_keep_physical_and_printed_numbers(pdf_result) -> None:  # noqa: ANN001
    adapter, _ = pdf_result
    pages = adapter.pages
    assert pages[0].physical_page == 1 and pages[0].printed_page is None  # unnumbered cover
    assert [(p.physical_page, p.printed_page) for p in pages[1:3]] == [(2, "1"), (3, "2")]
    assert all("SYNTH-ICD Tabular List" in " ".join(p.header_lines) for p in pages[1:])
    assert all(p.footer_lines and p.footer_lines[0].startswith("Page") for p in pages[1:])
    # Raw text is preserved; cleaned text no longer has the running header/footer.
    assert "Page 1" in pages[1].raw_text and "Page 1" not in pages[1].cleaned_text
    assert adapter.stats["pages_processed"] == len(pages)


def test_pdf_multiline_titles_are_joined(pdf_result) -> None:  # noqa: ANN001
    _, records = pdf_result
    assert records["C00.10"].title == "Type 1 glucose regulation disorder with kidney complication"
    assert records["A02.9"].title == "Wheezing airway disorder, unspecified severity"
    assert records["B00.0"].title == "Elevated blood pressure disorder with kidney involvement"


def test_pdf_hierarchy_inclusions_and_rules(pdf_result) -> None:  # noqa: ANN001
    _, records = pdf_result
    assert records["A00"].parent_code == "A00-A09" and records["A00-A09"].parent_code == "I"
    assert records["C00.20"].parent_code == "C00.2"
    assert [i.text for i in records["D00.0"].inclusions] == ["smoker", "tobacco dependence"]
    assert [(e.text, e.target_code) for e in records["A00"].exclusions] == [
        ("airway infection in the newborn (B15)", "B15")
    ]
    rule = records["C00.20"].instructions[0]
    assert rule.instruction_type == InstructionType.USE_ADDITIONAL_CODE
    # Wrapped across two lines, re-joined, explicit "C12.-" -> target C12.
    assert rule.text.endswith("chronic kidney impairment (C12.-)") and rule.target_code == "C12"
    assert records["I"].level == NodeType.CHAPTER and records["I"].range_start == "A00"


def test_pdf_hyphenation_is_repaired(pdf_result) -> None:  # noqa: ANN001
    _, records = pdf_result
    note = records["B15"].instructions[0]
    assert note.instruction_type == InstructionType.NOTE
    assert "after birth" in note.text and "af-" not in note.text


def test_pdf_provenance_and_page_continuation(pdf_result) -> None:  # noqa: ANN001
    _, records = pdf_result
    for record in records.values():
        provenance = record.provenance
        assert provenance.kind == SourceType.PDF
        assert 2 <= provenance.page_start <= provenance.page_end
        assert provenance.printed_page == str(provenance.page_start - 1)
    spanning = [r for r in records.values() if r.provenance.page_end > r.provenance.page_start]
    assert spanning, "at least one record continues onto the next page"
    assert all(r.raw_text for r in records.values())


def test_restricted_pdf_is_refused_without_reading_content(tmp_path: Path) -> None:
    path, manifest = write_source(tmp_path, "pdf", restricted=True)
    security = inspect_pdf_security(path)
    assert security.encrypted and not security.permissions["copy_extract"]
    assert not security.text_extraction_permitted

    adapter = PdfAdapter(path, manifest)
    inspection = adapter.inspect_metadata()
    assert inspection.encrypted and not inspection.text_extraction_permitted
    with pytest.raises(SourceRestrictedError):
        list(adapter.iterate_records())

    result = run_pipeline(path, manifest)
    assert result.restricted and [i.code for i in result.fatal_issues] == ["SOURCE_RESTRICTED"]
    assert result.records == []


def test_ocr_is_never_automatic_and_is_also_permission_checked(tmp_path: Path) -> None:
    path, manifest = write_source(tmp_path, "pdf", restricted=True)
    assert OcrPdfAdapter.detect(path) == 0.0

    def engine(image) -> list[Word]:  # noqa: ANN001
        raise AssertionError("OCR must not run on a restricted PDF")

    with pytest.raises(SourceRestrictedError):
        list(OcrPdfAdapter(path, manifest, ocr_engine=engine).iterate_records())


def test_ocr_path_uses_the_same_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, manifest = write_source(tmp_path, "pdf")
    lines = ["A00     Airway infection", "A00.1   Chronic airway infection"]

    def fake_pages(self):  # noqa: ANN001, ANN202
        yield 1, "image", 612.0, 792.0

    def engine(image) -> list[Word]:  # noqa: ANN001
        words = []
        for row, line in enumerate(lines):
            x = 56.0
            for token in line.split():
                words.append(Word(token, x, x + 6 * len(token), 100 + row * 14, 110 + row * 14))
                x += 6 * len(token) + 30
        return words

    monkeypatch.setattr(OcrPdfAdapter, "_render_pages", fake_pages)
    records = list(OcrPdfAdapter(path, manifest, ocr_engine=engine).iterate_records())
    assert [(r.code, r.title, r.parent_code) for r in records] == [
        ("A00", "Airway infection", None),
        ("A00.1", "Chronic airway infection", "A00"),
    ]


def test_ocr_without_dependencies_explains_how_to_enable_it(tmp_path: Path) -> None:
    path, manifest = write_source(tmp_path, "pdf")
    adapter = OcrPdfAdapter(path, manifest)
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        with pytest.raises(AdapterError, match="optional OCR"):
            adapter._engine()


def test_two_column_pages_are_read_column_by_column(tmp_path: Path) -> None:
    path = tmp_path / "two.pdf"
    expected = write_two_column_pdf(path)
    page = extract_pdf_pages(path)[0]
    assert page.columns == 2
    texts = [line.text for line in page.lines]
    assert texts[0].startswith("Chapter I")
    code_lines = [t for t in texts if t.startswith("A")]
    assert code_lines == [" ".join(e.split()) for e in expected]


def test_text_adapter_matches_layout_semantics(tmp_path: Path) -> None:
    path, manifest = write_source(tmp_path, "text")
    adapter = TextAdapter(path, manifest)
    records = {r.code: r for r in adapter.iterate_records()}
    assert adapter.inspect_metadata().page_count == len(adapter.pages)
    assert records["C00.19"].title == "Type 1 glucose regulation disorder without complication"
    assert records["A00"].provenance.printed_page is not None


# --- cleaner / parser unit cases ----------------------------------------------------------------


def _page(number: int, texts: list[str]) -> PageText:
    return PageText(number, [TextLine(t, number) for t in texts], "\n".join(texts))


def test_cleaner_removes_only_repeating_margins_and_normalises_text() -> None:
    pages = [
        _page(1, ["Running header", "A00 Thing one", "Page 1"]),
        _page(2, ["Running header", "A01 Thing two", "Page 2"]),
        _page(3, ["Running header", "A02 Thing ﬁve  here", "Page 3"]),
    ]
    cleaned = PageCleaner().clean(pages)
    assert [line.text for p in cleaned for line in p.lines] == [
        "A00 Thing one",
        "A01 Thing two",
        "A02 Thing five here",
    ]
    assert [p.printed_page for p in cleaned] == ["1", "2", "3"]


def test_dehyphenation_rules() -> None:
    pages = [_page(1, ["classifi-", "cation of x", "range A00-", "A09 kept", "Non-", "Smoker"])]
    texts = [line.text for line in PageCleaner().clean(pages)[0].lines]
    assert texts[0] == "classification" and texts[1] == "of x"
    assert "range A00-" in texts and "Non-" in texts  # not joined: code range / capital


def test_normalize_line() -> None:
    assert normalize_line("  é ­x  y z ") == "é x y z"


def test_parser_does_not_mistake_indented_items_for_blocks() -> None:
    profile = TextLayoutProfile()
    lines = [
        TextLine("Infective disorders (A00-A09)", 1, 0),
        TextLine("A00     Airway infection", 1, 0),
        TextLine("Excludes: infections of the newborn (B10-B19)", 1, 40),
        TextLine("other things (C00-C09)", 1, 90),
        TextLine("Note: a paragraph that", 1, 40),
        TextLine("continues here", 1, 70),
    ]
    page = PageText(1, lines, "")
    result = StructureParser(profile, "t.txt", SourceType.TEXT).parse([page])
    assert [r.code for r in result.records] == ["A00-A09", "A00"]
    a00 = result.records[1]
    assert [e.text for e in a00.exclusions] == [
        "infections of the newborn (B10-B19)",
        "other things (C00-C09)",
    ]
    assert all(e.target_code is None for e in a00.exclusions)  # ranges are not targets
    assert a00.instructions[0].text == "a paragraph that continues here"


def test_provenance_locator_is_compact() -> None:
    provenance = SourceProvenance(source_filename="a.pdf", kind=SourceType.PDF, page_start=3)
    assert provenance.to_locator() == {"kind": "pdf", "page_start": 3}


def test_layout_mapping_can_restrict_pages(tmp_path: Path) -> None:
    path, _ = write_source(tmp_path, "pdf")
    manifest = manifest_for(mapping={"first_page": 2, "last_page": 2})
    adapter = PdfAdapter(path, manifest)
    records = list(adapter.iterate_records())
    assert [p.physical_page for p in adapter.pages] == [2]
    assert records and all(r.provenance.page_start == 2 for r in records)
