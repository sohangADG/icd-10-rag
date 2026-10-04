"""Database-free part of ingestion:

Source -> Source Adapter -> Normalized Records -> Structural Validation -> Hierarchy Builder
       -> Rule Validation -> PipelineResult (records + hierarchy + issues + statistics)

Used by `inspect`/`validate` (dry run) and as the first stage of `ingest`.
"""

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.constants import NodeType
from app.ingestion.adapters import (
    AdapterError,
    SourceManifest,
    SourceRestrictedError,
    select_adapter,
)
from app.ingestion.hierarchy import HierarchyBuilder, HierarchyResult
from app.ingestion.models import (
    DatasetMetadata,
    NormalizedICDRecord,
    Severity,
    SourceInspection,
    ValidationIssue,
)
from app.ingestion.validator import supported_coding_systems, validator_for

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    source_filename: str
    inspection: SourceInspection | None = None
    metadata: DatasetMetadata | None = None
    records: list[NormalizedICDRecord] = field(default_factory=list)
    hierarchy: HierarchyResult | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    adapter_stats: dict[str, Any] = field(default_factory=dict)
    restricted: bool = False
    duration_ms: int = 0

    @property
    def fatal_issues(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def is_valid(self) -> bool:
        return not self.fatal_issues and self.hierarchy is not None

    def statistics(self) -> dict[str, Any]:
        levels = Counter(r.level for r in self.records if r.level)
        severities = Counter(issue.severity for issue in self.issues)
        issue_codes = Counter(issue.code for issue in self.issues)
        hierarchy = self.hierarchy
        return {
            "dataset": self.metadata.coding_system if self.metadata else None,
            "version": self.metadata.version if self.metadata else None,
            "source_type": self.inspection.source_type.value if self.inspection else None,
            "source_sha256": self.inspection.sha256 if self.inspection else None,
            "pages_processed": self.adapter_stats.get("pages_processed"),
            "records_extracted": len(self.records),
            "chapters": levels[NodeType.CHAPTER],
            "blocks": levels[NodeType.BLOCK],
            "categories": levels[NodeType.CATEGORY],
            "subcategories": levels[NodeType.SUBCATEGORY],
            "codes": levels[NodeType.CODE],
            "inclusion_terms": sum(len(r.inclusions) for r in self.records),
            "exclusion_terms": sum(len(r.exclusions) for r in self.records),
            "instructions": sum(len(r.instructions) for r in self.records),
            "index_terms": sum(len(r.index_terms) for r in self.records),
            "selectable_codes": sum(1 for n in hierarchy.nodes.values() if n.is_selectable)
            if hierarchy
            else 0,
            "duplicate_codes": (hierarchy.duplicates if hierarchy else 0)
            + issue_codes["DUPLICATE_NORMALIZED_CODE"],
            "orphan_records": hierarchy.orphans if hierarchy else 0,
            "hierarchy_cycles": hierarchy.cycles if hierarchy else 0,
            "inferred_parent_links": hierarchy.inferred_links if hierarchy else 0,
            "validation_errors": severities[Severity.ERROR],
            "validation_warnings": severities[Severity.WARNING],
            "validation_info": severities[Severity.INFO],
            "issue_codes": dict(sorted(issue_codes.items())),
            **{k: v for k, v in self.adapter_stats.items() if k != "pages_processed"},
        }


def _fatal(code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=Severity.ERROR, code=code, message=message)


def run_pipeline(path: Path, manifest: SourceManifest | None = None) -> PipelineResult:
    started = time.perf_counter()
    manifest = manifest or SourceManifest()
    result = PipelineResult(source_filename=Path(path).name)
    try:
        adapter = select_adapter(Path(path), manifest)
        result.inspection = adapter.inspect_metadata()
        result.metadata = adapter.get_dataset_metadata()
    except AdapterError as exc:
        result.issues.append(_fatal("INVALID_SOURCE", str(exc)))
        return _finish(result, started)

    validator = validator_for(result.metadata.coding_system)
    if validator is None:
        result.issues.append(
            _fatal(
                "UNSUPPORTED_CODING_SYSTEM",
                f"No validator registered for {result.metadata.coding_system}; supported: "
                f"{', '.join(supported_coding_systems())}.",
            )
        )
        return _finish(result, started)
    result.issues.extend(validator.validate_metadata(result.metadata))

    try:
        result.records = list(adapter.iterate_records())
    except SourceRestrictedError as exc:
        result.restricted = True
        result.issues.append(_fatal("SOURCE_RESTRICTED", str(exc)))
        return _finish(result, started)
    except AdapterError as exc:
        result.issues.append(_fatal("PARSER_FAILURE", str(exc)))
        return _finish(result, started)
    result.issues.extend(adapter.validation_messages())
    result.adapter_stats = dict(getattr(adapter, "stats", {}) or {})

    result.issues.extend(validator.validate_records(result.records, result.metadata))
    hierarchy = HierarchyBuilder(manifest.hierarchy).build(result.records)
    result.issues.extend(hierarchy.issues)
    result.issues.extend(validator.validate_hierarchy(hierarchy))
    result.hierarchy = hierarchy
    return _finish(result, started)


def _finish(result: PipelineResult, started: float) -> PipelineResult:
    result.duration_ms = int((time.perf_counter() - started) * 1000)
    stats = result.statistics()
    logger.info(
        "source validated",
        extra={
            "source_filename": result.source_filename,
            "records": stats["records_extracted"],
            "errors": stats["validation_errors"],
            "warnings": stats["validation_warnings"],
            "restricted": result.restricted,
            "duration_ms": result.duration_ms,
        },
    )
    return result
