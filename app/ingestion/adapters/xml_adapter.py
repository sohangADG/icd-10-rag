"""XML adapter. Two modes:

* ``preset: "claml"`` — WHO ClaML (Classification Markup Language), the standard XML exchange
  format for ICD classifications: <Class code kind> with <SuperClass> (explicit parent) and
  <Rubric kind="preferred|inclusion|exclusion|note|coding-hint|...">. <Reference> elements inside
  a rubric are explicit code references.
* ``preset: "generic"`` — configurable ElementTree paths:
    {"record_path": ".//Record", "children_path": "Children/Record",
     "fields": {"code": "@code", "title": "Title", "parent_code": "@parent",
                "inclusions": "Inclusions/Item", "exclusions": "Exclusions/Item"},
     "namespaces": {"c": "urn:example"}, "dataset_element": "Dataset"}
  A field path is "@attr", "child/path", or "child/path/@attr". List-valued canonical fields use
  every match (findall); scalar fields use the first.

Parsing uses defusedxml (no external entities / entity expansion attacks).
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Literal
from xml.etree.ElementTree import Element

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from pydantic import Field

from app.core.constants import InstructionType, NodeType, SourceType
from app.ingestion.adapters.base import AdapterError, SourceAdapter, SourceManifest
from app.ingestion.adapters.fields import FieldMapping, RecordAssembler, as_text
from app.ingestion.codes import (
    clean_code,
    find_code_references,
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
)

_LIST_FIELDS = {
    "inclusions",
    "exclusions",
    "includes_notes",
    "notes",
    "code_also",
    "use_additional_code",
    "code_first",
    "see",
    "see_also",
    "other_instructions",
    "index_terms",
    "synonyms",
    "abbreviations",
}

_CLAML_LEVELS = {"chapter": NodeType.CHAPTER, "block": NodeType.BLOCK}
_CLAML_INSTRUCTIONS = {
    "note": InstructionType.NOTE,
    "text": InstructionType.NOTE,
    "introduction": InstructionType.NOTE,
    "coding-hint": InstructionType.OTHER,
    "footnote": InstructionType.NOTE,
}


class XmlMapping(FieldMapping):
    preset: Literal["generic", "claml"] = "generic"
    record_path: str = ".//Record"
    children_path: str | None = None
    fields: dict[str, str] = Field(default_factory=dict)
    namespaces: dict[str, str] = Field(default_factory=dict)
    dataset_element: str | None = None


def _element_text(element: Element | None) -> str | None:
    if element is None:
        return None
    return as_text("".join(element.itertext()))


class XmlAdapter(SourceAdapter):
    source_type: ClassVar[SourceType] = SourceType.XML
    extensions: ClassVar[tuple[str, ...]] = (".xml", ".claml")

    def __init__(self, path: Path, manifest: SourceManifest | None = None) -> None:
        super().__init__(path, manifest)
        self.mapping = XmlMapping.model_validate(self.manifest.mapping)
        self._root: Element | None = None

    @classmethod
    def detect(cls, path: Path) -> float:
        if path.suffix.lower() not in cls.extensions:
            return 0.0
        with path.open("rb") as handle:
            head = handle.read(512).lstrip(b"\xef\xbb\xbf").lstrip()
        return 0.9 if head.startswith(b"<") else 0.1

    def _load(self) -> Element:
        if self._root is None:
            try:
                self._root = ElementTree.parse(self.path).getroot()
            except (ElementTree.ParseError, DefusedXmlException) as exc:
                raise AdapterError(f"Invalid or unsafe XML in {self.path.name}: {exc}") from exc
        return self._root

    # --- metadata -----------------------------------------------------------------------------

    def _embedded_dataset_metadata(self) -> dict[str, Any]:
        root = self._load()
        if self.mapping.preset == "claml":
            title = root.find("Title")
            if title is None:
                return {}
            values: dict[str, Any] = {"title": _element_text(title)}
            if title.get("version"):
                values["version"] = title.get("version")
            return {k: v for k, v in values.items() if v}
        if self.mapping.dataset_element:
            element = root.find(self.mapping.dataset_element, self.mapping.namespaces)
            if element is not None:
                values = dict(element.attrib)
                values.update({child.tag: _element_text(child) for child in element})
                return {k: v for k, v in values.items() if v}
        return {}

    def _inspection_details(self) -> dict[str, Any]:
        root = self._load()
        if self.mapping.preset == "claml":
            return {
                "root": root.tag,
                "classes": len(root.findall("Class")),
                "modifier_classes": len(root.findall("ModifierClass")),
            }
        return {
            "root": root.tag,
            "records": len(root.findall(self.mapping.record_path, self.mapping.namespaces)),
        }

    # --- records ------------------------------------------------------------------------------

    def iterate_records(self) -> Iterator[NormalizedICDRecord]:
        if self.mapping.preset == "claml":
            yield from self._iterate_claml()
        else:
            yield from self._iterate_generic()

    def _resolve(self, element: Element, path: str, *, many: bool) -> list[str]:
        ns = self.mapping.namespaces
        if path.startswith("@"):
            value = as_text(element.get(path[1:]))
            return [value] if value else []
        element_path, _, attribute = path.partition("/@")
        matches = element.findall(element_path, ns) if many else [element.find(element_path, ns)]
        values: list[str] = []
        for match in matches:
            if match is None:
                continue
            value = as_text(match.get(attribute)) if attribute else _element_text(match)
            if value:
                values.append(value)
        return values

    def _row(self, element: Element) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for field, path in self.mapping.fields.items():
            many = field in _LIST_FIELDS
            values = self._resolve(element, path, many=many)
            if values:
                row[field] = values if many else values[0]
        return row

    def _iterate_generic(self) -> Iterator[NormalizedICDRecord]:
        root = self._load()
        assembler = RecordAssembler(self.mapping)
        top = root.findall(self.mapping.record_path, self.mapping.namespaces)
        stack: list[tuple[Element, str, str | None]] = [
            (element, f"{self.mapping.record_path}[{i + 1}]", None)
            for i, element in reversed(list(enumerate(top)))
        ]
        while stack:
            element, element_path, parent_code = stack.pop()
            provenance = SourceProvenance(
                source_filename=self.path.name, kind=self.source_type, element_path=element_path
            )
            record = assembler.add_row(self._row(element), provenance, parent_code=parent_code)
            if self.mapping.children_path and record is not None:
                children = element.findall(self.mapping.children_path, self.mapping.namespaces)
                stack.extend(
                    (child, f"{element_path}/{self.mapping.children_path}[{i + 1}]", record.code)
                    for i, child in reversed(list(enumerate(children)))
                )
        self._issues.extend(assembler.issues)
        yield from assembler.records()

    def _iterate_claml(self) -> Iterator[NormalizedICDRecord]:
        root = self._load()
        assembler = RecordAssembler(self.mapping)
        if root.find("ModifierClass") is not None:
            self._report(
                Severity.WARNING,
                "MODIFIERS_NOT_EXPANDED",
                "ClaML ModifierClass elements are present; modifier-generated codes are not "
                "expanded by this adapter and must be supplied explicitly.",
            )
        for index, element in enumerate(root.findall("Class"), start=1):
            code = as_text(element.get("code"))
            kind = (element.get("kind") or "").lower()
            provenance = SourceProvenance(
                source_filename=self.path.name,
                kind=self.source_type,
                element_path=f"Class[{index}]",
            )
            if not code:
                self._report(
                    Severity.ERROR,
                    "MISSING_CODE",
                    "ClaML Class without a code attribute.",
                    locator=provenance.to_locator(),
                )
                continue
            if kind in _CLAML_LEVELS:
                level = _CLAML_LEVELS[kind]
                code = code.upper()
            else:
                code = clean_code(code)
                level = level_from_code_shape(code)
            superclass = element.find("SuperClass")
            record = NormalizedICDRecord(
                key=record_key(level, code, str(index)),
                code=code,
                normalized_code=normalize_code(code, level),
                level=level,
                parent_code=as_text(superclass.get("code")) if superclass is not None else None,
                provenance=provenance,
            )
            if "-" in code and level == NodeType.BLOCK:
                record.range_start, _, record.range_end = code.partition("-")
            self._apply_rubrics(element, record, provenance)
            assembler.add_record(record)
        self._issues.extend(assembler.issues)
        yield from assembler.records()

    @staticmethod
    def _apply_rubrics(
        element: Element, record: NormalizedICDRecord, provenance: SourceProvenance
    ) -> None:
        for rubric in element.findall("Rubric"):
            kind = (rubric.get("kind") or "").lower()
            for label in rubric.findall("Label"):
                text = _element_text(label)
                if not text:
                    continue
                references: list[str] = []
                for ref in label.iter("Reference"):
                    for code in find_code_references(
                        ref.get("code") or _element_text(ref) or ""
                    ).codes:
                        if code not in references:
                            references.append(code)
                explicit_target = references[0] if len(references) == 1 else None
                if kind == "preferred" and record.title is None:
                    record.title = text
                elif kind in {"preferredlong", "definition"}:
                    record.description = (
                        f"{record.description} {text}" if record.description else text
                    )
                elif kind == "inclusion":
                    record.inclusions.append(NormalizedInclusion(text=text, provenance=provenance))
                elif kind == "exclusion":
                    exclusion = make_exclusion(
                        text, explicit_target=explicit_target, provenance=provenance
                    )
                    if references:
                        exclusion.referenced_codes = references
                        exclusion.target_code = explicit_target
                    record.exclusions.append(exclusion)
                elif kind in _CLAML_INSTRUCTIONS:
                    # An explicit leading phrase ("Code first ...") is more specific than the
                    # rubric kind; label markers ("Note:") keep the rubric's type.
                    marker = match_marker(text)
                    instruction_type = (
                        marker.instruction_type
                        if marker is not None and not marker.is_label
                        else _CLAML_INSTRUCTIONS[kind]
                    )
                    record.instructions.append(
                        make_instruction(
                            instruction_type,
                            text,
                            explicit_target=explicit_target,
                            provenance=provenance,
                        )
                    )
