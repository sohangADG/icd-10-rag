"""Layout primitives shared by the PDF, OCR and plain-text adapters."""

from dataclasses import dataclass, field


@dataclass
class TextLine:
    text: str
    page: int  # physical page, 1-based
    x0: float = 0.0  # left edge in points (PDF) or pseudo-points (text: spaces * SPACE_WIDTH)
    top: float = 0.0
    column: int = 0  # 0 = single column / left, 1 = right column


@dataclass
class PageText:
    """One physical page after extraction (raw) and, later, cleaning."""

    physical_page: int
    lines: list[TextLine]
    raw_text: str
    width: float = 0.0
    height: float = 0.0
    columns: int = 1
    printed_page: str | None = None
    header_lines: list[str] = field(default_factory=list)
    footer_lines: list[str] = field(default_factory=list)

    @property
    def cleaned_text(self) -> str:
        return "\n".join(line.text for line in self.lines)


# Plain-text sources express indentation in spaces; one space counts as this many points so the
# same indentation tolerances apply to PDF and text sources.
SPACE_WIDTH = 5.0
