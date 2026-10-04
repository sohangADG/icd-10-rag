"""Reconstruct logical ICD entities from cleaned text lines (PDF, OCR or plain text).

The parser is a line-oriented state machine driven by a configurable TextLayoutProfile. It
recognises, in document order:

* chapter headings, block headings (code ranges), and classification code lines;
* instruction markers (Includes / Excludes / Note / Code first / Use additional code /
  Code also / See / See also) and the items or paragraphs that follow them;
* multi-line titles, wrapped list items and records continuing across page boundaries.

Parents come from the document structure (chapter -> block -> code, and code-prefix nesting
of subdivisions in reading order), never from guessing. Headings must be flush-left within
their column so indented list items such as "infection (A00-A09)" are not mistaken for blocks.
"""

import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.core.constants import CLASSIFICATION_NODE_TYPES, InstructionType, NodeType, SourceType
from app.ingestion.codes import (
    CODE_TOKEN,
    clean_code,
    level_from_code_shape,
    normalize_code,
    record_key,
)
from app.ingestion.instructions import make_exclusion, make_instruction, match_marker
from app.ingestion.models import (
    NormalizedICDRecord,
    NormalizedInclusion,
    Severity,
    SourceProvenance,
    ValidationIssue,
)
from app.ingestion.text.layout import PageText, TextLine

_CONNECTORS = {
    "and",
    "or",
    "of",
    "with",
    "without",
    "in",
    "due",
    "to",
    "the",
    "by",
    "from",
    "for",
    "on",
    "at",
    "as",
    "a",
    "an",
    "other",
    "not",
}
_LIST_MODES = {InstructionType.INCLUDES, InstructionType.EXCLUDES}


class TextLayoutProfile(BaseModel):
    """Regexes and tolerances describing a printed tabular-list layout.

    Defaults follow the common WHO ICD-10 tabular-list conventions. A licensed source's profile
    must be confirmed with golden-sample validation before its output is trusted.
    """

    chapter_pattern: str = (
        r"^chapter\s+(?P<code>[IVXLC]+|\d{1,2})\b\s*[-–—:.]?\s*(?P<title>.*?)"
        rf"\s*(?:\((?P<start>{CODE_TOKEN})\s*-\s*(?P<end>{CODE_TOKEN})\))?$"
    )
    block_patterns: list[str] = [
        rf"^(?P<start>{CODE_TOKEN})\s*-\s*(?P<end>{CODE_TOKEN})\s+(?P<title>\S.*)$",
        rf"^(?P<title>[^()]+?)\s*\((?P<start>{CODE_TOKEN})\s*-\s*(?P<end>{CODE_TOKEN})\)$",
    ]
    code_pattern: str = rf"^(?P<code>{CODE_TOKEN})(?P<mark>[†‡*]?)\s+(?P<title>\S.*)$"
    # Lines indented under a code title before any marker are inclusion terms (WHO style).
    implicit_inclusions: bool = True
    # A heading must start within this many points of its column's left margin.
    heading_indent_tolerance: float = 3.0
    # A wrapped list item continues when indented this much deeper than the item start.
    hanging_indent: float = 6.0
    case_insensitive_headings: bool = True


@dataclass
class _OpenItem:
    mode: InstructionType
    exclusion_type: str | None
    text: str
    # Indent of the item's first line; None when the item began on its marker's line (the
    # marker's own indent says nothing about where wrapped item text would continue).
    indent: float | None
    page: int
    line_no: int
    implicit: bool = False
    last_page: int = 0


@dataclass
class ParseResult:
    records: list[NormalizedICDRecord]
    issues: list[ValidationIssue]
    unparsed_lines: int = 0
    lines_processed: int = 0
    pages_processed: int = 0
    page_labels: dict[int, str | None] = field(default_factory=dict)


class StructureParser:
    def __init__(
        self,
        profile: TextLayoutProfile,
        source_filename: str,
        source_kind: SourceType,
    ) -> None:
        flags = re.I if profile.case_insensitive_headings else 0
        self.profile = profile
        self.source_filename = source_filename
        self.source_kind = source_kind
        self._chapter_re = re.compile(profile.chapter_pattern, flags)
        self._block_res = [re.compile(p) for p in profile.block_patterns]
        self._code_re = re.compile(profile.code_pattern)

    # --- public -------------------------------------------------------------------------------

    def parse(self, pages: list[PageText]) -> ParseResult:
        self._records: list[NormalizedICDRecord] = []
        self._issues: list[ValidationIssue] = []
        self._chapter: NormalizedICDRecord | None = None
        self._block: NormalizedICDRecord | None = None
        self._stack: list[NormalizedICDRecord] = []
        self._owner: NormalizedICDRecord | None = None
        self._mode: str | InstructionType = "none"
        self._item: _OpenItem | None = None
        self._pending_exclusion_type: str | None = None
        self._labels = {page.physical_page: page.printed_page for page in pages}
        self._raw: dict[str, list[str]] = {}
        unparsed = 0
        line_no = 0
        for page in pages:
            margins = self._margins(page)
            for line in page.lines:
                line_no += 1
                indent = line.x0 - margins.get(line.column, 0.0)
                if not self._consume(line, indent, line_no):
                    unparsed += 1
        self._close_item()
        for record in self._records:
            record.raw_text = "\n".join(self._raw.get(record.key, [])) or None
        if unparsed:
            self._issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    code="UNPARSED_LINES",
                    message=f"{unparsed} line(s) outside any recognised record were skipped "
                    "(front matter, introductions, index pages...).",
                )
            )
        return ParseResult(
            records=self._records,
            issues=self._issues,
            unparsed_lines=unparsed,
            lines_processed=line_no,
            pages_processed=len(pages),
            page_labels=self._labels,
        )

    # --- line handling ------------------------------------------------------------------------

    @staticmethod
    def _margins(page: PageText) -> dict[int, float]:
        margins: dict[int, float] = {}
        for line in page.lines:
            margins[line.column] = min(margins.get(line.column, line.x0), line.x0)
        return margins

    def _consume(self, line: TextLine, indent: float, line_no: int) -> bool:
        text = line.text
        flush_left = indent <= self.profile.heading_indent_tolerance
        if flush_left:
            if match := self._chapter_re.match(text):
                self._start_chapter(match, line, line_no)
                return True
            for block_re in self._block_res:
                if match := block_re.match(text):
                    self._start_block(match, line, line_no)
                    return True
            if match := self._code_re.match(text):
                self._start_code(match, line, line_no)
                return True
        marker = match_marker(text)
        if marker is not None and self._owner is not None:
            self._close_item()
            self._mode = marker.instruction_type
            self._remember(line)
            self._pending_exclusion_type = marker.exclusion_type
            if marker.remainder:
                self._open_item(
                    marker.instruction_type,
                    marker.exclusion_type,
                    marker.remainder,
                    None,
                    line,
                    line_no,
                )
            return True
        if self._owner is None:
            return False
        self._remember(line)
        self._extend(text, indent, line, line_no)
        return True

    def _extend(self, text: str, indent: float, line: TextLine, line_no: int) -> None:
        owner = self._owner
        assert owner is not None
        if self._mode == "title_pending":
            owner.title = text
            self._mode = "title"
            return
        if self._mode == "title":
            if self._continues(owner.title or "", text):
                owner.title = f"{owner.title} {text}".strip()
                return
            if self.profile.implicit_inclusions and owner.level in CLASSIFICATION_NODE_TYPES:
                self._mode = InstructionType.INCLUDES
                self._open_item(
                    InstructionType.INCLUDES, None, text, indent, line, line_no, implicit=True
                )
                return
            self._mode = "description"
        if self._mode == "description":
            owner.description = f"{owner.description} {text}" if owner.description else text
            return
        if isinstance(self._mode, InstructionType):
            item = self._item
            if item is None:
                self._open_item(
                    self._mode, self._pending_exclusion_type, text, indent, line, line_no
                )
                return
            if self._mode in _LIST_MODES and not self._item_continues(item, text, indent):
                implicit = item.implicit
                self._close_item()
                self._open_item(
                    self._mode, item.exclusion_type, text, indent, line, line_no, implicit=implicit
                )
                return
            item.text = f"{item.text} {text}"
            item.last_page = line.page

    # --- record creation ----------------------------------------------------------------------

    def _provenance(self, line: TextLine, line_no: int) -> SourceProvenance:
        return SourceProvenance(
            source_filename=self.source_filename,
            kind=self.source_kind,
            page_start=line.page,
            page_end=line.page,
            printed_page=self._labels.get(line.page),
            line=line_no,
        )

    def _new_record(
        self,
        level: NodeType,
        code: str,
        title: str | None,
        line: TextLine,
        line_no: int,
        parent: NormalizedICDRecord | None,
    ) -> NormalizedICDRecord:
        self._close_item()
        record = NormalizedICDRecord(
            key=record_key(level, code, str(line_no)),
            code=code,
            normalized_code=normalize_code(code, level),
            title=title or None,
            level=level,
            parent_code=parent.code if parent else None,
            parent_level=parent.level if parent else None,
            sort_order=len(self._records) + 1,
            provenance=self._provenance(line, line_no),
        )
        self._records.append(record)
        self._owner = record
        self._mode = "title" if title else "title_pending"
        self._raw.setdefault(record.key, []).append(line.text)
        return record

    def _start_chapter(self, match: re.Match[str], line: TextLine, line_no: int) -> None:
        record = self._new_record(
            NodeType.CHAPTER, match.group("code").upper(), match.group("title"), line, line_no, None
        )
        if match.group("start"):
            record.range_start, record.range_end = match.group("start"), match.group("end")
        self._chapter, self._block, self._stack = record, None, []

    def _start_block(self, match: re.Match[str], line: TextLine, line_no: int) -> None:
        start, end = match.group("start"), match.group("end")
        record = self._new_record(
            NodeType.BLOCK, f"{start}-{end}", match.group("title"), line, line_no, self._chapter
        )
        record.range_start, record.range_end = start, end
        self._block, self._stack = record, []

    def _start_code(self, match: re.Match[str], line: TextLine, line_no: int) -> None:
        code = clean_code(match.group("code"))
        normalized = normalize_code(code)
        while self._stack and not (
            normalized.startswith(normalize_code(self._stack[-1].code or ""))
            and normalized != normalize_code(self._stack[-1].code or "")
        ):
            self._stack.pop()
        parent = self._stack[-1] if self._stack else (self._block or self._chapter)
        record = self._new_record(
            level_from_code_shape(code), code, match.group("title"), line, line_no, parent
        )
        if match.group("mark"):
            record.metadata["code_mark"] = match.group("mark")
        self._stack.append(record)

    # --- items --------------------------------------------------------------------------------

    def _open_item(
        self,
        mode: InstructionType,
        exclusion_type: str | None,
        text: str,
        indent: float | None,
        line: TextLine,
        line_no: int,
        *,
        implicit: bool = False,
    ) -> None:
        self._item = _OpenItem(
            mode, exclusion_type, text, indent, line.page, line_no, implicit, line.page
        )

    def _item_continues(self, item: _OpenItem, text: str, indent: float) -> bool:
        if self._continues(item.text, text):
            return True
        if item.indent is None:
            return False
        return text[:1].islower() and indent >= item.indent + self.profile.hanging_indent

    @staticmethod
    def _continues(previous: str, text: str) -> bool:
        """Whether `text` continues `previous` (wrapped title or list item)."""
        stripped = previous.rstrip()
        if not stripped:
            return False
        if stripped.count("(") > stripped.count(")") or stripped.count("[") > stripped.count("]"):
            return True
        if stripped.endswith((",", "-", "–", "/", ":")):
            return True
        last_word = stripped.rsplit(" ", 1)[-1].lower()
        return last_word in _CONNECTORS and text[:1].islower()

    def _close_item(self) -> None:
        item, owner = self._item, self._owner
        self._item = None
        if item is None or owner is None:
            return
        provenance = SourceProvenance(
            source_filename=self.source_filename,
            kind=self.source_kind,
            page_start=item.page,
            page_end=max(item.page, item.last_page),
            printed_page=self._labels.get(item.page),
            line=item.line_no,
        )
        text = item.text.strip()
        if item.mode is InstructionType.INCLUDES:
            owner.inclusions.append(
                NormalizedInclusion(
                    text=text,
                    provenance=provenance,
                    metadata={"marker": "implicit" if item.implicit else "Includes"},
                )
            )
        elif item.mode is InstructionType.EXCLUDES:
            owner.exclusions.append(
                make_exclusion(text, exclusion_type=item.exclusion_type, provenance=provenance)
            )
        else:
            owner.instructions.append(make_instruction(item.mode, text, provenance=provenance))

    def _remember(self, line: TextLine) -> None:
        owner = self._owner
        if owner is None:
            return
        self._raw.setdefault(owner.key, []).append(line.text)
        if owner.provenance and (owner.provenance.page_end or 0) < line.page:
            # The record continues on a later page: extend its page range.
            owner.provenance = owner.provenance.model_copy(update={"page_end": line.page})
