"""Text-layer PDF extraction (pdfplumber / pdfminer.six) with security-permission enforcement.

Permission policy (never bypassed):
* A PDF that needs a password to open is refused.
* An encrypted PDF whose permissions do not allow content copying/extraction (permission bit 5)
  is refused before any page content is read. The "extract for accessibility" permission
  (bit 10) is deliberately NOT treated as permission to ingest content.
* The same check guards the optional OCR path: rendering pages to images to OCR them would
  defeat the same restriction.

Only the document's security dictionary and page tree are read during inspection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pdfminer.pdfdocument import PDFDocument, PDFEncryptionError, PDFPasswordIncorrect
from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.pdftypes import resolve1

from app.ingestion.text.layout import PageText, TextLine

# PDF permission bits (ISO 32000-1, table 22), 1-based.
_PERMISSION_BITS = {
    "print": 3,
    "modify": 4,
    "copy_extract": 5,
    "annotate": 6,
    "fill_forms": 9,
    "extract_for_accessibility": 10,
    "assemble": 11,
    "print_high_quality": 12,
}


@dataclass(frozen=True)
class PdfSecurity:
    page_count: int
    encrypted: bool
    requires_password: bool
    permissions: dict[str, bool]
    permission_value: int | None

    @property
    def text_extraction_permitted(self) -> bool:
        return not self.requires_password and self.permissions.get("copy_extract", True)


def _decode_permissions(value: int) -> dict[str, bool]:
    unsigned = value & 0xFFFFFFFF
    return {name: bool((unsigned >> (bit - 1)) & 1) for name, bit in _PERMISSION_BITS.items()}


def inspect_pdf_security(path: Path) -> PdfSecurity:
    """Read encryption/permission flags and the page count — no page content is touched."""
    with path.open("rb") as handle:
        parser = PDFParser(handle)
        try:
            document = PDFDocument(parser, password="")
        except PDFPasswordIncorrect:
            return PdfSecurity(0, True, True, {}, None)
        except (PDFSyntaxError, PDFEncryptionError) as exc:
            raise ValueError(f"Unreadable PDF: {type(exc).__name__}") from exc
        encryption = document.encryption
        permission_value: int | None = None
        if encryption is not None:
            raw = resolve1(encryption[1].get("P"))
            permission_value = int(raw) if raw is not None else None
        permissions = (
            _decode_permissions(permission_value)
            if permission_value is not None
            else dict.fromkeys(_PERMISSION_BITS, True)
        )
        pages = resolve1(document.catalog.get("Pages"))
        page_count = int(resolve1(pages.get("Count"))) if pages else 0
        return PdfSecurity(
            page_count=page_count,
            encrypted=encryption is not None,
            requires_password=False,
            permissions=permissions,
            permission_value=permission_value,
        )


# --- word -> line reconstruction ----------------------------------------------------------------


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float


def _cluster_lines(words: list[Word], y_tolerance: float) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda w: (round(w.top, 1), w.x0)):
        if lines and abs(lines[-1][0].top - word.top) <= y_tolerance:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w.x0) for line in lines]


def _find_gutter(
    lines: list[list[Word]], width: float, height: float
) -> tuple[float, float] | None:
    """Locate a vertical gutter separating two text columns, or None for single-column pages.

    Lines in the top/bottom 8% (running headers/footers) are ignored. A gutter is the widest
    x-range in the middle 30-70% of the page crossed by at most 10% of body lines, with at
    least 25% of body lines having text on each side.
    """
    body = [line for line in lines if 0.08 * height < line[0].top < 0.92 * height]
    if len(body) < 6 or width <= 0:
        return None
    allowed_crossings = max(1, int(0.1 * len(body)))
    # Difference array over integer x: coverage[x] = number of body lines with a word strictly
    # spanning x (words within one line never overlap, so each line counts at most once).
    size = int(width) + 2
    diff = [0] * (size + 1)
    for line in body:
        for word in line:
            start, stop = int(word.x0) + 1, min(int(word.x1), size)
            if start < stop:
                diff[start] += 1
                diff[stop] -= 1
    coverage, running = [], 0
    for delta in diff[:size]:
        running += delta
        coverage.append(running)
    low, high = int(0.3 * width), min(int(0.7 * width), size - 1)
    best: tuple[float, float] | None = None
    run_start: int | None = None
    for x in range(low, high + 2):
        inside = x <= high and coverage[x] <= allowed_crossings
        if inside and run_start is None:
            run_start = x
        elif not inside and run_start is not None:
            if best is None or (x - run_start) > (best[1] - best[0]):
                best = (float(run_start), float(x))
            run_start = None
    if best is None or best[1] - best[0] < 8.0:
        return None
    left = sum(1 for line in body if any(w.x1 <= best[0] for w in line))
    right = sum(1 for line in body if any(w.x0 >= best[1] for w in line))
    if left < 0.25 * len(body) or right < 0.25 * len(body):
        return None
    return best


def _to_line(words: list[Word], page: int, column: int) -> TextLine:
    return TextLine(
        text=" ".join(w.text for w in words),
        page=page,
        x0=words[0].x0,
        top=words[0].top,
        column=column,
    )


def words_to_page(
    words: list[Word],
    *,
    page_number: int,
    width: float,
    height: float,
    raw_text: str,
    column_detection: bool = True,
    y_tolerance: float = 2.0,
) -> PageText:
    """Group words into reading-order lines; two-column pages are read left then right, with
    full-width lines (crossing the gutter) acting as section separators."""
    clustered = _cluster_lines(words, y_tolerance)
    gutter = _find_gutter(clustered, width, height) if column_detection else None
    if gutter is None:
        lines = [_to_line(line, page_number, 0) for line in clustered]
        return PageText(page_number, lines, raw_text, width, height, columns=1)

    gutter_mid = (gutter[0] + gutter[1]) / 2
    ordered: list[TextLine] = []
    left: list[TextLine] = []
    right: list[TextLine] = []
    for line in clustered:
        spanning = any(w.x0 < gutter_mid < w.x1 for w in line)
        if spanning:
            ordered.extend(left + right)
            left, right = [], []
            ordered.append(_to_line(line, page_number, 0))
            continue
        left_words = [w for w in line if w.x1 <= gutter_mid]
        right_words = [w for w in line if w.x0 >= gutter_mid]
        if left_words:
            left.append(_to_line(left_words, page_number, 0))
        if right_words:
            right.append(_to_line(right_words, page_number, 1))
    ordered.extend(left + right)
    return PageText(page_number, ordered, raw_text, width, height, columns=2)


def extract_pdf_pages(
    path: Path,
    *,
    first_page: int = 1,
    last_page: int | None = None,
    column_detection: bool = True,
) -> list[PageText]:
    """Extract text-layer pages. Callers must have checked inspect_pdf_security() first."""
    import pdfplumber  # local import: heavy, and only needed for PDF sources

    pages: list[PageText] = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        end = min(last_page or total, total)
        for number in range(max(first_page, 1), end + 1):
            page = pdf.pages[number - 1]
            raw_words: list[dict[str, Any]] = page.extract_words(
                x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False, use_text_flow=False
            )
            words = [
                Word(w["text"], float(w["x0"]), float(w["x1"]), float(w["top"]), float(w["bottom"]))
                for w in raw_words
            ]
            pages.append(
                words_to_page(
                    words,
                    page_number=number,
                    width=float(page.width),
                    height=float(page.height),
                    raw_text=page.extract_text() or "",
                    column_detection=column_detection,
                )
            )
            page.flush_cache()
    return pages
