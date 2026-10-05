# Source adapters

Every adapter implements `SourceAdapter` (`app/ingestion/adapters/base.py`):

| Method | Purpose |
|---|---|
| `detect(path)` | Confidence 0–1 from extension/magic bytes (cheap) |
| `inspect_metadata()` | Format facts without importing: size, SHA-256, pages, encryption, sheets, columns… |
| `iterate_records()` | Normalized records in source order |
| `get_dataset_metadata()` | Manifest identity + measured facts. Raises if identity is incomplete. |
| `get_source_hash()` | SHA-256 (streamed) |
| `get_provenance()` | Document-level provenance |
| `validation_messages()` | Parser issues gathered while iterating |

Adapters never validate policy, infer hierarchy, or touch the database.

## Canonical fields

Record fields: `code, title, description, level, parent_code, parent_level, range_start,
range_end, selectable, status, sort_order, inclusions, exclusions, includes_notes, notes,
code_also, use_additional_code, code_first, see, see_also, other_instructions, index_terms,
synonyms, abbreviations`.

Long-format rule rows: `owner_code, owner_level, rule_type, text, target_code, exclusion_type`.
`rule_type` accepts any instruction type (`EXCLUDES`, `CODE_FIRST`, `Excludes1`…) or a term kind
(`INCLUSION`, `SYNONYM`, `ABBREVIATION`, `INDEX_TERM`).

Rule targets are set **only** when the source states exactly one code (an explicit `target_code`
column, a single `<Reference>`, or a single code in the text such as `(B15)` or `(A00.-)`).
Ranges and multiple codes keep `target_code = NULL` and list `referenced_codes`.

## CSV / TSV (`delimited.py`)

```json
{"adapter": "csv", "mapping": {
  "encoding": "auto", "delimiter": ",", "list_separator": "|",
  "columns": {"code": "Code", "title": "Description", "parent_code": "Parent"},
  "level_values": {"C": "CATEGORY"},
  "rules_file": "rules.csv", "index_terms_file": null,
  "rule_columns": {"owner_code": "Code", "rule_type": "Type", "text": "Text"}}}
```
`encoding: auto` tries UTF-8 (with BOM), then cp1252, then latin-1. Provenance: file + 1-based data row.

## XLSX (`xlsx.py`)
`records_sheet`, `header_row`, `rules_sheet`, `index_terms_sheet`, plus the same column mapping.
Provenance: sheet + row.

## JSON (`json_adapter.py`)
`dataset_path` (embedded identity), `records_path`, `children_key` (the parent comes from
nesting), and column mapping. Rule list items may be strings or `{"text", "target_code",
"exclusion_type"}` objects. Provenance: element path such as `records[0]/children[2]`.

## XML (`xml_adapter.py`)
- `preset: "claml"` (WHO ClaML): `<Class code kind>`, `<SuperClass>` for the explicit parent,
  and rubrics `preferred`/`inclusion`/`exclusion`/`note`/`coding-hint`/`definition`.
  A leading "Code first/Use additional code/Code also/See also" phrase in a note or hint keeps
  its specific type. `ModifierClass` expansion is not supported, and a warning is raised.
- `preset: "generic"`: `record_path`, `children_path`, `fields` (`@attr`, `child/path`,
  `child/@attr`), `namespaces`, `dataset_element`.
- Parsed with **defusedxml**, so external entities and entity-expansion attacks are refused.

## PDF (`layout_adapters.py`, `text/*`) {#pdf}

1. **Permission check first** (`inspect_pdf_security`). A PDF that needs a password, or whose
   copy/extract permission is off, raises `SourceRestrictedError` before any page content is read.
2. **Extraction** (pdfplumber/pdfminer): words → lines (y-clustering). Two-column pages are read
   left column then right column, with full-width lines as separators. The gutter is found with
   an O(words + width) coverage sweep. Physical page numbers are kept, along with the raw page
   text.
3. **Cleaning** (`page_cleaner.py`): repeating headers/footers (top/bottom lines whose
   digit-wildcarded signature repeats on ≥50% of pages) are removed and kept per page. The
   printed page label (e.g. `Page 12`) is taken from them. Unicode (NFC, ligatures, soft
   hyphens, NBSP) and whitespace are normalized. End-of-line hyphenation is repaired only for
   letter + `-` followed by a lower-case word (code ranges are left alone). `(continued)` markers
   are dropped.
4. **Structure** (`structure_parser.py`, configurable `TextLayoutProfile`):
   - Chapter, block and code lines must be flush-left in their column, so indented items like
     `infection (A00-A09)` are never read as blocks.
   - Parents come from the document structure: chapter → block → code. Subdivisions use code
     prefixes in reading order.
   - Instruction markers are recognised: Includes, Excludes(1/2), Note, Code first, Use
     additional code, Code also, See, See also. List items (includes/excludes) are split.
     Paragraphs (notes, instructions) are joined.
   - Wrapped titles continue after a comma, a trailing connector word, or an open parenthesis.
     This also works across a page break.
   - Indented lines under a code title before any marker are implicit inclusion terms (WHO
     style; configurable).
   - Records keep `page_start`/`page_end`, the printed page, the line number and the raw text.

Useful mapping options: `first_page`/`last_page` (e.g. only the tabular list),
`column_detection`, `cleaner.margin_lines`, `cleaner.repeat_threshold`, `layout.*` regexes,
`min_chars_per_page`.

**Pages without a text layer** are reported (`NO_TEXT_LAYER` / `PAGES_WITHOUT_TEXT`). The
adapter never OCRs on its own.

## OCR (`adapter: pdf_ocr`)
Optional, explicit only. Needs `pip install '.[ocr]'` (pytesseract, pypdfium2) plus a Tesseract
binary. Pages are rendered and OCR'd into positioned words, then go through the same
cleaner/parser. **The same permission check applies.**

## Structured text (`adapter: text`)
`\f` separates pages, and indentation (spaces) is significant. It uses the same cleaner and
parser as PDF.

## Adding an adapter
Subclass `SourceAdapter`, emit `NormalizedICDRecord`s (build them from canonical rows with
`RecordAssembler`), register the class in `ADAPTERS`, and add a parity test against the
synthetic dataset (`tests/unit/test_adapters.py`).
