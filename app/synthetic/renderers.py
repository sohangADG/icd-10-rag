"""Render the synthetic classification into every supported source format.

Each writer returns the SourceManifest dict needed to ingest what it wrote, so tests can assert
that all adapters reconstruct the same classification (cross-adapter parity).
"""

import csv
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from app.ingestion.codes import find_code_references
from app.synthetic.dataset import RULE_LIST_FIELDS, TERM_LIST_FIELDS, flatten

_RULE_TYPES = {
    "exclusions": "EXCLUDES",
    "notes": "NOTE",
    "code_first": "CODE_FIRST",
    "use_additional_code": "USE_ADDITIONAL_CODE",
    "code_also": "CODE_ALSO",
    "see": "SEE",
    "see_also": "SEE_ALSO",
}
_TERM_TYPES = {"synonyms": "SYNONYM", "abbreviations": "ABBREVIATION", "index_terms": "INDEX_TERM"}


def _canonical(node: dict[str, Any]) -> dict[str, Any]:
    record = {k: v for k, v in node.items() if k not in {"range", "children"}}
    if "range" in node:
        record["range_start"], record["range_end"] = node["range"]
    return record


def _manifest(adapter: str, dataset: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    return {"adapter": adapter, "dataset": dataset, "mapping": mapping}


# --- JSON ------------------------------------------------------------------------------------


def write_json(
    path: Path, dataset: dict[str, Any], chapters: list[dict[str, Any]]
) -> dict[str, Any]:
    def convert(node: dict[str, Any]) -> dict[str, Any]:
        record = _canonical(node)
        if node.get("children"):
            record["children"] = [convert(child) for child in node["children"]]
        return record

    document = {"dataset": dataset, "chapters": [convert(c) for c in chapters]}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return _manifest(
        "json",
        {},
        {"dataset_path": ["dataset"], "records_path": ["chapters"], "children_key": "children"},
    )


# --- delimited / spreadsheet ------------------------------------------------------------------

_CODE_COLUMNS = [
    "code",
    "level",
    "title",
    "parent_code",
    "parent_level",
    "range_start",
    "range_end",
    "inclusions",
]


def _flat_rows(chapters: list[dict[str, Any]], *, rules_inline: bool) -> list[dict[str, Any]]:
    rows = []
    for record in flatten(chapters):
        record = _canonical(record)
        row = {column: record.get(column) or "" for column in _CODE_COLUMNS}
        row["inclusions"] = "|".join(record.get("inclusions", []))
        for field in TERM_LIST_FIELDS:
            row[field] = "|".join(record.get(field, []))
        if rules_inline:
            row["exclusions"] = "|".join(record.get("exclusions", []))
            for field in RULE_LIST_FIELDS:
                row[field] = "|".join(record.get(field, []))
        rows.append(row)
    return rows


def _rule_rows(chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in flatten(chapters):
        for field, rule_type in _RULE_TYPES.items():
            for text in record.get(field, []):
                rows.append(
                    {
                        "owner_code": record["code"],
                        "owner_level": record["level"],
                        "rule_type": rule_type,
                        "text": text,
                    }
                )
    return rows


def _write_delimited(path: Path, rows: list[dict[str, Any]], delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(
    path: Path, dataset: dict[str, Any], chapters: list[dict[str, Any]]
) -> dict[str, Any]:
    """Codes in one CSV (multi-valued cells), rules in a separate long-format CSV."""
    _write_delimited(path, _flat_rows(chapters, rules_inline=False), ",")
    rules_path = path.with_name(f"{path.stem}_rules.csv")
    _write_delimited(rules_path, _rule_rows(chapters), ",")
    return _manifest("csv", dataset, {"rules_file": rules_path.name})


def write_tsv(
    path: Path, dataset: dict[str, Any], chapters: list[dict[str, Any]]
) -> dict[str, Any]:
    """Everything in one TSV, rule lists as "|"-separated cells, custom column names."""
    rows = _flat_rows(chapters, rules_inline=True)
    renamed = [
        {("Code" if k == "code" else "Title" if k == "title" else k): v for k, v in r.items()}
        for r in rows
    ]
    _write_delimited(path, renamed, "\t")
    return _manifest("tsv", dataset, {"columns": {"code": "Code", "title": "Title"}})


def write_xlsx(
    path: Path, dataset: dict[str, Any], chapters: list[dict[str, Any]]
) -> dict[str, Any]:
    from openpyxl import Workbook

    workbook = Workbook()
    codes = workbook.active
    codes.title = "Codes"
    rows = _flat_rows(chapters, rules_inline=False)
    codes.append(list(rows[0]))
    for row in rows:
        codes.append([row[c] or None for c in rows[0]])
    rules = workbook.create_sheet("Rules")
    rule_rows = _rule_rows(chapters)
    rules.append(list(rule_rows[0]))
    for row in rule_rows:
        rules.append(list(row.values()))
    workbook.save(path)
    return _manifest("xlsx", dataset, {"records_sheet": "Codes", "rules_sheet": "Rules"})


# --- ClaML XML ----------------------------------------------------------------------------------


def _label(text: str) -> str:
    """Wrap explicit code references in <Reference> as ClaML does."""
    escaped = escape(text)
    for code in find_code_references(text).codes:
        printed = f"{code}.-" if f"{code}.-" in escaped else code
        escaped = escaped.replace(printed, f"<Reference>{printed}</Reference>", 1)
    return f'<Label xml:lang="en">{escaped}</Label>'


def write_claml(
    path: Path, dataset: dict[str, Any], chapters: list[dict[str, Any]]
) -> dict[str, Any]:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ClaML version="2.0.0">',
        f'  <Title name="SYNTH" version="{escape(dataset["version"])}">'
        f"{escape(dataset['title'])}</Title>",
    ]
    for record in flatten(chapters):
        kind = {"CHAPTER": "chapter", "BLOCK": "block"}.get(record["level"], "category")
        parts.append(f'  <Class code="{record["code"]}" kind="{kind}">')
        if record.get("parent_code"):
            parts.append(f'    <SuperClass code="{record["parent_code"]}"/>')
        parts.append(f'    <Rubric kind="preferred">{_label(record["title"])}</Rubric>')
        for text in record.get("inclusions", []):
            parts.append(f'    <Rubric kind="inclusion">{_label(text)}</Rubric>')
        for text in record.get("exclusions", []):
            parts.append(f'    <Rubric kind="exclusion">{_label(text)}</Rubric>')
        for text in record.get("notes", []):
            parts.append(f'    <Rubric kind="note">{_label(text)}</Rubric>')
        for field in ("code_first", "use_additional_code", "code_also", "see", "see_also"):
            for text in record.get(field, []):
                parts.append(f'    <Rubric kind="coding-hint">{_label(text)}</Rubric>')
        parts.append("  </Class>")
    parts.append("</ClaML>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return _manifest("xml", dataset, {"preset": "claml"})


# --- tabular-list text layout (text + PDF) -----------------------------------------------------

_BREAK_AFTER = {"with", "without", "of", "and", "or", "in", "to"}
TITLE_WIDTH = 45
NOTE_WIDTH = 60
PAGE_HEADER = "SYNTH-ICD Tabular List (synthetic test fixture)"


def _wrap(text: str, width: int, *, hyphenate: bool = False) -> list[str]:
    """Wrap after connector words / commas (how the parser recognises wrapped titles)."""
    if len(text) <= width:
        return [text]
    words = text.split(" ")
    if hyphenate:
        lines, current = [], ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= width:
                current = candidate
                continue
            room = width - len(current) - 2
            if len(word) >= 5 and room >= 2 and word.isalpha():
                lines.append(f"{current} {word[:room]}-")
                current = word[room:]
            else:
                lines.append(current)
                current = word
        return [*lines, current]
    for index in range(len(words) - 1, 0, -1):
        head = " ".join(words[:index])
        if len(head) <= width and (
            words[index - 1].lower() in _BREAK_AFTER or words[index - 1].endswith(",")
        ):
            return [head, *_wrap(" ".join(words[index:]), width)]
    return [text]


def layout_lines(chapters: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """(indent in spaces, text) lines of a WHO-style tabular list."""
    lines: list[tuple[int, str]] = []
    for record in flatten(chapters):
        level, code, title = record["level"], record["code"], record["title"]
        if level == "CHAPTER":
            start, end = record["range"]
            lines.append((0, f"Chapter {code} - {title} ({start}-{end})"))
        elif level == "BLOCK":
            start, end = record["range"]
            lines.append((0, f"{title} ({start}-{end})"))
        else:
            title_lines = _wrap(title, TITLE_WIDTH)
            lines.append((0, f"{code:<8}{title_lines[0]}"))
            lines.extend((8, extra) for extra in title_lines[1:])
            lines.extend((10, inclusion) for inclusion in record.get("inclusions", []))
        for index, exclusion in enumerate(record.get("exclusions", [])):
            lines.append((8, f"Excludes: {exclusion}") if index == 0 else (18, exclusion))
        for note in record.get("notes", []):
            wrapped = _wrap(f"Note: {note}", NOTE_WIDTH, hyphenate=True)
            lines.append((8, wrapped[0]))
            lines.extend((14, extra) for extra in wrapped[1:])
        for field in ("code_first", "use_additional_code", "code_also", "see", "see_also"):
            for text in record.get(field, []):
                wrapped = _wrap_plain(text, NOTE_WIDTH)
                lines.append((8, wrapped[0]))
                lines.extend((10, extra) for extra in wrapped[1:])
    return lines


def _wrap_plain(text: str, width: int) -> list[str]:
    lines, current = [], ""
    for word in text.split(" "):
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    return [*lines, current]


def paginate(lines: list[tuple[int, str]], per_page: int) -> list[list[tuple[int, str]]]:
    return [lines[i : i + per_page] for i in range(0, len(lines), per_page)]


COVER_LINES = [
    "Synthetic ICD-like Classification",
    "Test fixture generated by icd-rag-service",
    "Contains no real classification content",
]


def write_text(
    path: Path,
    dataset: dict[str, Any],
    chapters: list[dict[str, Any]],
    *,
    lines_per_page: int = 20,
) -> dict[str, Any]:
    """Cover page (unnumbered) + tabular pages with a running header and "Page N" footer."""
    pages = ["\n".join(COVER_LINES)]
    for number, page in enumerate(paginate(layout_lines(chapters), lines_per_page), start=1):
        body = [" " * indent + text for indent, text in page]
        pages.append("\n".join([PAGE_HEADER, "", *body, "", f"Page {number}"]))
    path.write_text("\f".join(pages) + "\n", encoding="utf-8")
    return _manifest("text", dataset, {})


def write_pdf(
    path: Path,
    dataset: dict[str, Any],
    chapters: list[dict[str, Any]],
    *,
    lines_per_page: int = 20,
    restricted: bool = False,
) -> dict[str, Any]:
    """Same layout as write_text, as a text-layer PDF. `restricted=True` encrypts it with
    copy/extract permission denied (to prove the adapter refuses such files)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.pdfencrypt import StandardEncryption
    from reportlab.pdfgen import canvas

    encrypt = (
        StandardEncryption(
            "", ownerPassword="fixture-owner", canPrint=0, canModify=0, canCopy=0, canAnnotate=0
        )
        if restricted
        else None
    )
    pdf = canvas.Canvas(str(path), pagesize=letter, encrypt=encrypt)
    pdf.setTitle(dataset.get("title", "Synthetic fixture"))
    _, height = letter
    left, line_height = 56.0, 14.0
    pdf.setFont("Helvetica", 10)
    for index, text in enumerate(COVER_LINES):
        pdf.drawString(left, height - 200 - index * 20, text)
    pdf.showPage()
    for number, page in enumerate(paginate(layout_lines(chapters), lines_per_page), start=1):
        pdf.setFont("Helvetica", 9)
        pdf.drawString(left, height - 36, PAGE_HEADER)
        pdf.setFont("Helvetica", 10)
        y = height - 72
        for indent, text in page:
            pdf.drawString(left + indent * 5.0, y, text)
            y -= line_height
        pdf.setFont("Helvetica", 9)
        pdf.drawString(left, 36, f"Page {number}")
        pdf.showPage()
    pdf.save()
    return _manifest("pdf", dataset, {})


def write_two_column_pdf(path: Path) -> list[str]:
    """One page: full-width heading, then two columns of code lines. Returns the expected
    reading order of the code lines."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    left_codes = [f"A0{i}     Left column entry {i}" for i in range(8)]
    right_codes = [f"A1{i}     Right column entry {i}" for i in range(8)]
    pdf = canvas.Canvas(str(path), pagesize=letter)
    _, height = letter
    pdf.setFont("Helvetica", 10)
    pdf.drawString(
        56, height - 72, "Chapter I - Two column synthetic chapter spanning the full page width"
    )
    y = height - 110
    for left_text, right_text in zip(left_codes, right_codes, strict=True):
        pdf.drawString(56, y, left_text)
        pdf.drawString(330, y, right_text)
        y -= 16
    pdf.showPage()
    pdf.save()
    return left_codes + right_codes


FORMAT_WRITERS = {
    "json": (write_json, ".json"),
    "csv": (write_csv, ".csv"),
    "tsv": (write_tsv, ".tsv"),
    "xlsx": (write_xlsx, ".xlsx"),
    "xml": (write_claml, ".xml"),
    "text": (write_text, ".txt"),
    "pdf": (write_pdf, ".pdf"),
}
