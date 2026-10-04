"""ORM -> API schema conversion shared by every ICD endpoint."""

from typing import Any

from app.core.constants import RULE_TO_INSTRUCTION, RuleType, TermType
from app.models import IcdDataset, IcdNode
from app.schemas.icd import (
    DatasetSummary,
    HierarchyItem,
    ICDRecordOut,
    RuleOut,
    SourceProvenanceOut,
)


def dataset_summary(dataset: IcdDataset) -> DatasetSummary:
    return DatasetSummary(
        id=dataset.id,
        coding_system=dataset.system,
        modification=dataset.modification,
        country=dataset.country,
        version=dataset.version,
        revision=dataset.revision,
        edition=dataset.edition,
        language=dataset.language,
        title=dataset.title,
        publisher=dataset.publisher,
        status=dataset.status.value,
        source_type=dataset.source_type.value if dataset.source_type else None,
        source_filename=dataset.source_filename,
        source_sha256=dataset.source_checksum,
        source_page_count=dataset.source_page_count,
        imported_at=dataset.imported_at,
    )


def hierarchy_item(node: IcdNode) -> HierarchyItem:
    return HierarchyItem(
        record_id=node.id, code=node.code, title=node.title, level=node.node_type.value
    )


def provenance(node: IcdNode, details: dict[str, Any] | None = None) -> SourceProvenanceOut:
    refs = (details or {}).get("source_refs", [])
    return SourceProvenanceOut(
        source_filename=refs[0].source_filename if refs else None,
        page_start=node.source_page_start,
        page_end=node.source_page_end,
        printed_page=refs[0].printed_page if refs else None,
        locator=node.source_locator,
    )


def _rule_out(rule: Any) -> RuleOut:
    instruction = RULE_TO_INSTRUCTION.get(rule.rule_type)
    return RuleOut(
        type=instruction.value if instruction else rule.rule_type.value,
        text=rule.rule_text,
        target_code=rule.target_code,
        target_record_id=rule.target_node_id,
        exclusion_type=rule.exclusion_type,
    )


def record_out(
    node: IcdNode, ancestors: list[IcdNode], details: dict[str, Any] | None
) -> ICDRecordOut:
    details = details or {}
    terms = details.get("terms", [])
    rules = details.get("rules", [])
    return ICDRecordOut(
        record_id=node.id,
        dataset_id=node.dataset_id,
        code=node.code,
        normalized_code=node.normalized_code,
        title=node.title,
        description=node.description,
        level=node.node_type.value,
        is_selectable=node.is_selectable,
        status=node.status.value,
        parent_record_id=node.parent_id,
        hierarchy=[hierarchy_item(a) for a in ancestors],
        inclusions=[t.term for t in terms if t.term_type == TermType.INCLUSION],
        exclusions=[_rule_out(r) for r in rules if r.rule_type == RuleType.EXCLUDE],
        instructions=[_rule_out(r) for r in rules if r.rule_type != RuleType.EXCLUDE],
        synonyms=[
            t.term for t in terms if t.term_type in (TermType.SYNONYM, TermType.ABBREVIATION)
        ],
        index_terms=[e.lead_term for e in details.get("index_terms", [])],
        provenance=provenance(node, details),
    )
