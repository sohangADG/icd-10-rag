"""Conservative page cleanup: repeating headers/footers, printed page numbers, unicode and
whitespace normalization, and end-of-line hyphenation.

Nothing is removed unless it repeats across pages (headers/footers) or is pure typography
(soft hyphens, ligatures). Raw page text is preserved separately by the caller.
"""

import re
import unicodedata
from collections import Counter

from pydantic import BaseModel

from app.ingestion.text.layout import PageText, TextLine

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
_SPACES = re.compile(r"[ \t  -​  　]+")
_PAGE_NUMBER = re.compile(
    r"(?:^|\s)(?:page\s+)?(?P<num>\d{1,4}|[ivxlcdm]{1,7}|[A-Z]-\d{1,4})\s*$", re.I
)
_PAGE_NUMBER_LEADING = re.compile(r"^(?P<num>\d{1,4}|[ivxlcdm]{1,7})(?:\s|$)", re.I)
_CONTINUED = re.compile(r"\(\s*cont(?:inued|'d|\.)?\s*\)\s*$", re.I)


class CleanerConfig(BaseModel):
    # How many lines at the top/bottom of each page are header/footer candidates.
    margin_lines: int = 2
    # A candidate repeating on at least this share of pages (and >= 2 pages) is removed.
    repeat_threshold: float = 0.5
    dehyphenate: bool = True


def normalize_line(text: str) -> str:
    text = unicodedata.normalize("NFC", text.translate(_LIGATURES)).replace("­", "")
    return _SPACES.sub(" ", text).strip()


def _signature(text: str) -> str:
    """Header/footer identity: digits and roman page numbers wildcarded, case-folded."""
    lowered = text.lower()
    lowered = re.sub(r"\d+", "#", lowered)
    lowered = re.sub(r"(?<![a-z])[ivxlcdm]{1,7}$", "#", lowered)
    return lowered


class PageCleaner:
    def __init__(self, config: CleanerConfig | None = None) -> None:
        self.config = config or CleanerConfig()

    def clean(self, pages: list[PageText]) -> list[PageText]:
        for page in pages:
            page.lines = [
                TextLine(normalize_line(line.text), line.page, line.x0, line.top, line.column)
                for line in page.lines
            ]
            page.lines = [line for line in page.lines if line.text]
        repeating = self._repeating_signatures(pages)
        for page in pages:
            self._strip_margins(page, repeating)
        if self.config.dehyphenate:
            self._dehyphenate(pages)
        return pages

    def _candidates(self, page: PageText) -> tuple[list[TextLine], list[TextLine]]:
        n = self.config.margin_lines
        top = page.lines[:n]
        bottom = page.lines[-n:] if len(page.lines) > n else []
        return top, bottom

    def _repeating_signatures(self, pages: list[PageText]) -> set[str]:
        counts: Counter[str] = Counter()
        for page in pages:
            top, bottom = self._candidates(page)
            counts.update({_signature(line.text) for line in top + bottom})
        threshold = max(2, int(self.config.repeat_threshold * len(pages) + 0.999))
        repeating = {sig for sig, count in counts.items() if count >= threshold}
        # A bare page number ("#") repeats by construction; so does "page #".
        return repeating

    def _strip_margins(self, page: PageText, repeating: set[str]) -> None:
        top, bottom = self._candidates(page)
        top_ids = {id(line) for line in top}
        bottom_ids = {id(line) for line in bottom}
        kept: list[TextLine] = []
        for line in page.lines:
            in_margin = id(line) in top_ids or id(line) in bottom_ids
            if in_margin and _signature(line.text) in repeating:
                (page.header_lines if id(line) in top_ids else page.footer_lines).append(line.text)
                page.printed_page = page.printed_page or self._page_label(line.text)
                continue
            kept.append(line)
        # "(continued)" markers repeat a heading on continuation pages; drop the marker only.
        for line in kept:
            line.text = _CONTINUED.sub("", line.text).rstrip()
        page.lines = [line for line in kept if line.text]

    @staticmethod
    def _page_label(text: str) -> str | None:
        match = _PAGE_NUMBER.search(text) or _PAGE_NUMBER_LEADING.match(text)
        return match.group("num") if match else None

    @staticmethod
    def _dehyphenate(pages: list[PageText]) -> None:
        """Join "classifi-" + "cation" across line (and page) breaks.

        Only when the hyphen follows a letter and the next line starts with a lower-case letter,
        so code ranges ("A00-") and compound words starting with capitals are left alone.
        """
        lines = [line for page in pages for line in page.lines]
        for current, following in zip(lines, lines[1:], strict=False):
            text = current.text
            if (
                len(text) >= 2
                and text.endswith("-")
                and text[-2].isalpha()
                and following.text[:1].islower()
            ):
                word, _, rest = following.text.partition(" ")
                current.text = text[:-1] + word
                following.text = rest
        for page in pages:
            page.lines = [line for line in page.lines if line.text]
