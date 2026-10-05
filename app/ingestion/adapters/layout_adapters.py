"""Adapters for layout-based sources: text-layer PDF, optional OCR PDF, and structured text.

All three share one pipeline: page extraction -> PageCleaner -> StructureParser. They differ
only in how pages and positioned lines are obtained.

Mapping:
    {"layout": {...TextLayoutProfile overrides...},
     "cleaner": {"margin_lines": 2, "repeat_threshold": 0.5},
     "first_page": 1, "last_page": null,       # restrict to e.g. the tabular-list pages
     "column_detection": true,
     "encoding": "auto"}                        # text sources only
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from app.core.constants import SourceType
from app.ingestion.adapters.base import (
    AdapterError,
    SourceAdapter,
    SourceManifest,
    SourceRestrictedError,
)
from app.ingestion.adapters.delimited import read_text
from app.ingestion.models import NormalizedICDRecord, Severity, SourceInspection
from app.ingestion.text.layout import SPACE_WIDTH, PageText, TextLine
from app.ingestion.text.page_cleaner import CleanerConfig, PageCleaner
from app.ingestion.text.pdf_reader import (
    PdfSecurity,
    Word,
    extract_pdf_pages,
    inspect_pdf_security,
    words_to_page,
)
from app.ingestion.text.structure_parser import StructureParser, TextLayoutProfile


class LayoutMapping(BaseModel):
    layout: TextLayoutProfile = Field(default_factory=TextLayoutProfile)
    cleaner: CleanerConfig = Field(default_factory=CleanerConfig)
    first_page: int = Field(default=1, ge=1)
    last_page: int | None = Field(default=None, ge=1)
    column_detection: bool = True
    encoding: str = "auto"
    # Pages with fewer extracted characters than this count as having no text layer.
    min_chars_per_page: int = 20


class _LayoutAdapter(SourceAdapter):
    def __init__(self, path: Path, manifest: SourceManifest | None = None) -> None:
        super().__init__(path, manifest)
        self.mapping = LayoutMapping.model_validate(self.manifest.mapping)
        self.stats: dict[str, Any] = {}
        self.pages: list[PageText] = []

    def _extract_pages(self) -> list[PageText]:
        raise NotImplementedError

    def iterate_records(self) -> Iterator[NormalizedICDRecord]:
        pages = self._extract_pages()
        empty = [
            p.physical_page
            for p in pages
            if len(p.raw_text.strip()) < self.mapping.min_chars_per_page
        ]
        if pages and len(empty) == len(pages):
            self._report(
                Severity.ERROR,
                "NO_TEXT_LAYER",
                "No page has a usable text layer. If the operator is permitted to process this "
                "scanned document, select the OCR adapter explicitly (adapter: pdf_ocr).",
            )
        elif empty:
            self._report(
                Severity.WARNING,
                "PAGES_WITHOUT_TEXT",
                f"{len(empty)} page(s) have no usable text layer: {empty[:20]}.",
            )
        pages = PageCleaner(self.mapping.cleaner).clean(pages)
        self.pages = pages
        result = StructureParser(self.mapping.layout, self.path.name, self.source_type).parse(pages)
        self._issues.extend(result.issues)
        self.stats = {
            "pages_processed": result.pages_processed,
            "lines_processed": result.lines_processed,
            "unparsed_lines": result.unparsed_lines,
            "two_column_pages": sum(1 for p in pages if p.columns == 2),
            "pages_with_printed_number": sum(1 for p in pages if p.printed_page),
        }
        yield from result.records


class PdfAdapter(_LayoutAdapter):
    """Text-layer PDF. Refuses permission-restricted PDFs; never applies OCR implicitly."""

    source_type: ClassVar[SourceType] = SourceType.PDF
    extensions: ClassVar[tuple[str, ...]] = (".pdf",)

    @classmethod
    def detect(cls, path: Path) -> float:
        with path.open("rb") as handle:
            is_pdf = handle.read(5) == b"%PDF-"
        return 0.9 if is_pdf else 0.0

    def security(self) -> PdfSecurity:
        try:
            return inspect_pdf_security(self.path)
        except ValueError as exc:
            raise AdapterError(str(exc)) from exc

    def inspect_metadata(self) -> SourceInspection:
        security = self.security()
        inspection = super().inspect_metadata()
        inspection.page_count = security.page_count
        inspection.encrypted = security.encrypted
        inspection.text_extraction_permitted = security.text_extraction_permitted
        inspection.details = {
            "requires_password": security.requires_password,
            "permission_value": security.permission_value,
            "permissions": security.permissions,
        }
        return inspection

    def ensure_permitted(self) -> PdfSecurity:
        security = self.security()
        if security.requires_password:
            raise SourceRestrictedError(
                f"{self.path.name} requires a password to open; it will not be processed."
            )
        if not security.text_extraction_permitted:
            raise SourceRestrictedError(
                f"{self.path.name} is encrypted and its permissions do not allow content "
                f"extraction (P={security.permission_value}). It will not be processed. Obtain "
                "an unrestricted, licensed source (e.g. the publisher's data files)."
            )
        return security

    def _extract_pages(self) -> list[PageText]:
        self.ensure_permitted()
        return extract_pdf_pages(
            self.path,
            first_page=self.mapping.first_page,
            last_page=self.mapping.last_page,
            column_detection=self.mapping.column_detection,
        )


OcrEngine = Callable[[Any], list[Word]]
"""Takes a rendered page image (PIL.Image) and returns positioned words in PDF points."""


class OcrPdfAdapter(PdfAdapter):
    """Optional OCR path for scanned PDFs. Only used when explicitly selected (adapter:
    pdf_ocr); requires the `ocr` extra (pytesseract + pypdfium2) and a Tesseract binary.
    Subject to the same permission check as text extraction."""

    source_type: ClassVar[SourceType] = SourceType.PDF_OCR

    @classmethod
    def detect(cls, path: Path) -> float:
        return 0.0  # never auto-selected

    def __init__(
        self,
        path: Path,
        manifest: SourceManifest | None = None,
        *,
        ocr_engine: OcrEngine | None = None,
        render_scale: float = 300 / 72,
    ) -> None:
        super().__init__(path, manifest)
        self._ocr_engine = ocr_engine
        self._render_scale = render_scale

    def _engine(self) -> OcrEngine:
        if self._ocr_engine is not None:
            return self._ocr_engine
        try:
            import pytesseract  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AdapterError(
                "OCR requested but the optional OCR dependencies are not installed "
                "(pip install '.[ocr]' and a Tesseract binary)."
            ) from exc
        scale = self._render_scale

        def engine(image: Any) -> list[Word]:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            words: list[Word] = []
            for i, text in enumerate(data["text"]):
                if text and text.strip() and float(data["conf"][i]) >= 0:
                    left, top = data["left"][i] / scale, data["top"][i] / scale
                    words.append(
                        Word(
                            text,
                            left,
                            left + data["width"][i] / scale,
                            top,
                            top + data["height"][i] / scale,
                        )
                    )
            return words

        return engine

    def _render_pages(self) -> Iterator[tuple[int, Any, float, float]]:
        try:
            import pypdfium2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AdapterError(
                "OCR requested but pypdfium2 is not installed (pip install '.[ocr]')."
            ) from exc
        document = pypdfium2.PdfDocument(str(self.path))
        try:
            last = min(self.mapping.last_page or len(document), len(document))
            for number in range(self.mapping.first_page, last + 1):
                page = document[number - 1]
                width, height = page.get_size()
                image = page.render(scale=self._render_scale).to_pil()
                yield number, image, width, height
        finally:
            document.close()

    def _extract_pages(self) -> list[PageText]:
        self.ensure_permitted()
        engine = self._engine()
        pages: list[PageText] = []
        for number, image, width, height in self._render_pages():
            words = engine(image)
            pages.append(
                words_to_page(
                    words,
                    page_number=number,
                    width=width,
                    height=height,
                    raw_text=" ".join(w.text for w in words),
                    column_detection=self.mapping.column_detection,
                )
            )
        return pages


class TextAdapter(_LayoutAdapter):
    """Structured plain text. Form feeds (\\f) separate pages; indentation is significant."""

    source_type: ClassVar[SourceType] = SourceType.TEXT
    extensions: ClassVar[tuple[str, ...]] = (".txt", ".text")

    def inspect_metadata(self) -> SourceInspection:
        inspection = super().inspect_metadata()
        inspection.page_count = len(self._split_pages())
        return inspection

    def _split_pages(self) -> list[str]:
        text, _ = read_text(self.path, self.mapping.encoding)
        return text.replace("\r\n", "\n").split("\f")

    def _extract_pages(self) -> list[PageText]:
        pages: list[PageText] = []
        for number, page_text in enumerate(self._split_pages(), start=1):
            lines: list[TextLine] = []
            for row, raw_line in enumerate(page_text.split("\n")):
                expanded = raw_line.expandtabs(4)
                if not expanded.strip():
                    continue
                indent = len(expanded) - len(expanded.lstrip(" "))
                lines.append(TextLine(expanded.strip(), number, indent * SPACE_WIDTH, float(row)))
            pages.append(PageText(number, lines, page_text))
        return pages
