"""Structural and rule validation of normalized records, with per-coding-system validators.

Validators never mix coding systems: each system registers its own code-format rules.
Severity semantics: any ERROR blocks the dataset from becoming READY; WARNING and INFO are
reported and persisted with the ingestion run but do not block.
"""

import re
from collections import Counter
from typing import ClassVar

from app.core.constants import CLASSIFICATION_NODE_TYPES, NodeType
from app.ingestion.codes import clean_code, normalize_code, parse_range
from app.ingestion.hierarchy import HierarchyResult
from app.ingestion.models import (
    DatasetMetadata,
    NormalizedICDRecord,
    Severity,
    ValidationIssue,
)


def _issue(
    severity: Severity,
    code: str,
    message: str,
    record: NormalizedICDRecord | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        record_key=record.key if record else None,
        record_code=record.code if record else None,
        locator=record.provenance.to_locator() if record and record.provenance else None,
    )


class BaseICDValidator:
    """Rules common to every ICD-10-family classification."""

    coding_systems: ClassVar[tuple[str, ...]] = ()
    # Classification code format (after clean_code()).
    code_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")
    chapter_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^([IVXLC]+|[0-9]{1,2})$")
    block_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?-[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$"
    )

    # --- dataset metadata ---------------------------------------------------------------------

    def validate_metadata(self, metadata: DatasetMetadata) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if self.coding_systems and metadata.coding_system not in self.coding_systems:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "INVALID_DATASET_METADATA",
                    f"Validator {type(self).__name__} does not handle {metadata.coding_system}.",
                )
            )
        if not metadata.source_sha256:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "INVALID_DATASET_METADATA",
                    "Source SHA-256 fingerprint is missing.",
                )
            )
        if not re.fullmatch(r"[a-z]{2,3}(-[A-Za-z]{2,4})?", metadata.language):
            issues.append(
                _issue(
                    Severity.ERROR,
                    "INVALID_DATASET_METADATA",
                    f"Language {metadata.language!r} is not an ISO 639 code.",
                )
            )
        if (
            metadata.effective_from
            and metadata.effective_to
            and metadata.effective_to < metadata.effective_from
        ):
            issues.append(
                _issue(Severity.ERROR, "INVALID_DATASET_METADATA", "Effective range is inverted.")
            )
        return issues

    # --- records --------------------------------------------------------------------------------

    def validate_code_format(self, record: NormalizedICDRecord) -> ValidationIssue | None:
        if record.level is None or not record.code:
            return None
        code = clean_code(record.code)
        if record.level in CLASSIFICATION_NODE_TYPES:
            valid = bool(self.code_pattern.match(code))
        elif record.level == NodeType.CHAPTER:
            valid = bool(self.chapter_pattern.match(code))
        else:
            valid = bool(self.block_pattern.match(code)) or bool(self.code_pattern.match(code))
        if valid:
            return None
        return _issue(
            Severity.ERROR,
            "MALFORMED_CODE",
            f"Code {record.code!r} is not a valid {record.level} code for this coding system.",
            record,
        )

    def validate_records(
        self,
        records: list[NormalizedICDRecord],
        metadata: DatasetMetadata,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        classification = [r for r in records if r.level in CLASSIFICATION_NODE_TYPES]
        if not classification:
            issues.append(
                _issue(
                    Severity.ERROR,
                    "EMPTY_DATASET",
                    "The source produced no classification (category/code) records.",
                )
            )

        # Identical keys are reported by the hierarchy builder (DUPLICATE_CODE); here we catch
        # distinct spellings that collide after normalization ("A00.0" vs "A000").
        normalized_counts = Counter(
            normalize_code(code) for code in {(r.code or "") for r in classification} if code
        )
        reported_duplicates: set[str] = set()
        for record in records:
            if record.level is None:
                issues.append(
                    _issue(
                        Severity.ERROR, "INVALID_LEVEL", "Record has no hierarchy level.", record
                    )
                )
            if not record.code and record.level in CLASSIFICATION_NODE_TYPES:
                issues.append(
                    _issue(
                        Severity.ERROR, "MISSING_CODE", "Classification record has no code.", record
                    )
                )
            if not (record.title or "").strip():
                issues.append(
                    _issue(Severity.ERROR, "MISSING_TITLE", "Record has no title.", record)
                )
            format_issue = self.validate_code_format(record)
            if format_issue:
                issues.append(format_issue)
            if (
                record.code
                and record.level in CLASSIFICATION_NODE_TYPES
                and normalized_counts[normalize_code(record.code)] > 1
                and normalize_code(record.code) not in reported_duplicates
            ):
                reported_duplicates.add(normalize_code(record.code))
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "DUPLICATE_NORMALIZED_CODE",
                        f"Code {record.code!r} collides with another spelling of the same "
                        "code after normalization.",
                        record,
                    )
                )
            issues.extend(self._validate_provenance(record, metadata))
        issues.extend(self._validate_references(records))
        return issues

    @staticmethod
    def _validate_provenance(
        record: NormalizedICDRecord, metadata: DatasetMetadata
    ) -> list[ValidationIssue]:
        provenance = record.provenance
        if provenance is None:
            return [
                _issue(
                    Severity.ERROR,
                    "RECORD_WITHOUT_PROVENANCE",
                    "Record has no source provenance.",
                    record,
                )
            ]
        issues: list[ValidationIssue] = []
        pages = [p for p in (provenance.page_start, provenance.page_end) if p is not None]
        page_count = metadata.source_page_count
        if any(p < 1 for p in pages) or (page_count and any(p > page_count for p in pages)):
            issues.append(
                _issue(
                    Severity.ERROR,
                    "INVALID_SOURCE_LOCATOR",
                    f"Source page(s) {pages} outside 1..{page_count}.",
                    record,
                )
            )
        if (
            provenance.page_start is not None
            and provenance.page_end is not None
            and provenance.page_end < provenance.page_start
        ):
            issues.append(
                _issue(
                    Severity.ERROR,
                    "INVALID_SOURCE_LOCATOR",
                    "Source page range is inverted.",
                    record,
                )
            )
        if provenance.row is not None and provenance.row < 1:
            issues.append(
                _issue(Severity.ERROR, "INVALID_SOURCE_LOCATOR", "Source row must be >= 1.", record)
            )
        return issues

    def _validate_references(self, records: list[NormalizedICDRecord]) -> list[ValidationIssue]:
        """Cross-references must point at codes that exist in the same dataset."""
        known = {normalize_code(r.code) for r in records if r.code}
        known |= {normalize_code(r.code, r.level) for r in records if r.code}
        issues: list[ValidationIssue] = []
        for record in records:
            targets = [(e.target_code, "exclusion") for e in record.exclusions]
            targets += [(i.target_code, i.instruction_type.value) for i in record.instructions]
            targets += [(t.target_code, "index term") for t in record.index_terms]
            for target, kind in targets:
                if not target:
                    continue
                if parse_range(target) is None and not self.code_pattern.match(clean_code(target)):
                    issues.append(
                        _issue(
                            Severity.WARNING,
                            "INVALID_CROSS_REFERENCE",
                            f"{kind} target {target!r} is not a well-formed code.",
                            record,
                        )
                    )
                elif normalize_code(target) not in known and clean_code(target) not in known:
                    issues.append(
                        _issue(
                            Severity.WARNING,
                            "UNRESOLVED_CROSS_REFERENCE",
                            f"{kind} target {target!r} does not exist in this dataset.",
                            record,
                        )
                    )
        return issues

    # --- hierarchy ------------------------------------------------------------------------------

    def validate_hierarchy(self, hierarchy: HierarchyResult) -> list[ValidationIssue]:
        """Post-build checks. (Duplicates/orphans/cycles are reported by the builder itself.)"""
        issues: list[ValidationIssue] = []
        for node in hierarchy.nodes.values():
            record = node.record
            if record.level in (NodeType.CHAPTER, NodeType.BLOCK) and not node.children:
                issues.append(
                    _issue(
                        Severity.WARNING,
                        "EMPTY_SECTION",
                        f"{record.level} {record.key!r} has no child records.",
                        record,
                    )
                )
            if (
                record.level in CLASSIFICATION_NODE_TYPES
                and not node.is_selectable
                and not node.children
            ):
                issues.append(
                    _issue(
                        Severity.INFO,
                        "NON_SELECTABLE_LEAF",
                        f"Leaf {record.key!r} is marked non-selectable by the source.",
                        record,
                    )
                )
        return issues


class ICD10Validator(BaseICDValidator):
    """WHO ICD-10: A00-Z99, optional 4th (and in some chapters 5th) character."""

    coding_systems = ("ICD-10",)
    code_pattern = re.compile(r"^[A-Z][0-9]{2}(\.[0-9]{1,2})?$")


class ICD10CAValidator(BaseICDValidator):
    """ICD-10-CA (CIHI): WHO structure plus Canadian enhancements up to 6 characters.

    The pattern is format-level only (letter, two digits, up to three further characters after
    the dot). It must be reviewed against the licensed CIHI release before production use.
    """

    coding_systems = ("ICD-10-CA",)
    code_pattern = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,3})?$")


class ICD10CMValidator(BaseICDValidator):
    """ICD-10-CM (US): 3-7 characters, alphanumeric third character, X placeholders."""

    coding_systems = ("ICD-10-CM",)
    code_pattern = re.compile(r"^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
    chapter_pattern = re.compile(r"^([0-9]{1,2}|[IVXLC]+)$")


class SyntheticICDValidator(BaseICDValidator):
    """Synthetic ICD-like test classifications (SYNTH-ICD, and SYNTH-ALT for coding-system
    isolation tests). Never real clinical content."""

    coding_systems = ("SYNTH-ICD", "SYNTH-ALT")
    code_pattern = re.compile(r"^[A-Z][0-9]{2}(\.[0-9]{1,2})?$")


_REGISTRY: dict[str, type[BaseICDValidator]] = {
    system: validator
    for validator in (ICD10Validator, ICD10CAValidator, ICD10CMValidator, SyntheticICDValidator)
    for system in validator.coding_systems
}


def supported_coding_systems() -> list[str]:
    return sorted(_REGISTRY)


def validator_for(coding_system: str) -> BaseICDValidator | None:
    validator_cls = _REGISTRY.get(coding_system.upper())
    return validator_cls() if validator_cls else None


def register_validator(validator_cls: type[BaseICDValidator]) -> None:
    """Extension point for additional coding systems (e.g. other national modifications)."""
    for system in validator_cls.coding_systems:
        _REGISTRY[system.upper()] = validator_cls
