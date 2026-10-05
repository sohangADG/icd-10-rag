"""Hierarchy reconstruction over normalized records.

Parent resolution order:
1. The source-provided parent (explicit parent_code, XML/JSON nesting, PDF document structure).
2. Only if enabled (HierarchyConfig.infer_parents): inference by code truncation ("A00.1" ->
   "A00") and by range containment (category -> block -> chapter). Every inferred link must
   resolve to an existing record and is reported as an INFO issue; it is never silent.

The builder never creates missing parent records.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.core.constants import CLASSIFICATION_NODE_TYPES, NODE_TYPE_RANK, NodeType
from app.ingestion.codes import (
    clean_code,
    code_in_range,
    normalize_code,
    parse_range,
    truncation_parents,
)
from app.ingestion.models import NormalizedICDRecord, Severity, ValidationIssue


class HierarchyConfig(BaseModel):
    infer_parents: bool = False
    # Leaf classification records become selectable unless the source says otherwise.
    leaves_selectable: bool = True


@dataclass
class HierarchyNode:
    record: NormalizedICDRecord
    parent_key: str | None = None
    parent_inferred: bool = False
    depth: int = 0
    chapter_key: str | None = None
    block_key: str | None = None
    category_key: str | None = None
    children: list[str] = field(default_factory=list)
    is_selectable: bool = False


@dataclass
class HierarchyResult:
    nodes: dict[str, HierarchyNode]  # insertion order == source order
    issues: list[ValidationIssue]
    duplicates: int = 0
    orphans: int = 0
    cycles: int = 0
    inferred_links: int = 0

    def ordered_for_insert(self) -> list[HierarchyNode]:
        """Parents before children (by depth, then source order)."""
        order = {key: index for index, key in enumerate(self.nodes)}
        return sorted(self.nodes.values(), key=lambda n: (n.depth, order[n.record.key]))

    def ancestors(self, key: str) -> list[str]:
        result: list[str] = []
        seen = {key}
        current = self.nodes[key].parent_key
        while current is not None and current not in seen:
            result.append(current)
            seen.add(current)
            current = self.nodes[current].parent_key
        return result

    def descendants(self, key: str) -> Iterator[str]:
        stack = list(reversed(self.nodes[key].children))
        while stack:
            child = stack.pop()
            yield child
            stack.extend(reversed(self.nodes[child].children))


def _issue(
    severity: Severity, code: str, message: str, record: NormalizedICDRecord
) -> ValidationIssue:
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        record_key=record.key,
        record_code=record.code,
        locator=record.provenance.to_locator() if record.provenance else None,
    )


class HierarchyBuilder:
    def __init__(self, config: HierarchyConfig | None = None) -> None:
        self._config = config or HierarchyConfig()

    def build(self, records: list[NormalizedICDRecord]) -> HierarchyResult:
        nodes: dict[str, HierarchyNode] = {}
        issues: list[ValidationIssue] = []
        duplicates = 0
        for record in records:
            if record.key in nodes:
                duplicates += 1
                issues.append(
                    _issue(
                        Severity.ERROR,
                        "DUPLICATE_CODE",
                        f"Duplicate record {record.key!r}; the first occurrence is kept.",
                        record,
                    )
                )
                continue
            nodes[record.key] = HierarchyNode(record=record)

        by_code = self._index_by_code(nodes)
        has_grouping = any(
            n.record.level in (NodeType.CHAPTER, NodeType.BLOCK) for n in nodes.values()
        )
        result = HierarchyResult(nodes=nodes, issues=issues, duplicates=duplicates)

        for node in nodes.values():
            self._resolve_parent(node, by_code, has_grouping, result)
        self._break_cycles(result)
        self._finalise(result)
        return result

    @staticmethod
    def _index_by_code(nodes: dict[str, HierarchyNode]) -> dict[str, list[HierarchyNode]]:
        index: dict[str, list[HierarchyNode]] = {}
        for node in nodes.values():
            if node.record.code:
                key = normalize_code(node.record.code, node.record.level)
                index.setdefault(key, []).append(node)
                # Also index under the generic form so a parent_code given without level hits.
                generic = normalize_code(node.record.code)
                if generic != key:
                    index.setdefault(generic, []).append(node)
        return index

    def _lookup(
        self,
        by_code: dict[str, list[HierarchyNode]],
        code: str,
        level: NodeType | None,
        child: HierarchyNode,
    ) -> HierarchyNode | None:
        candidates = by_code.get(normalize_code(code, level), []) or by_code.get(
            normalize_code(code), []
        )
        candidates = [c for c in candidates if c is not child]
        if level is not None:
            candidates = [c for c in candidates if c.record.level == level] or candidates
        if len(candidates) > 1:
            # Prefer the candidate ranked directly above the child.
            child_rank = NODE_TYPE_RANK.get(child.record.level or NodeType.CODE, 4)
            above = [
                c
                for c in candidates
                if NODE_TYPE_RANK.get(c.record.level or NodeType.CODE, 4) < child_rank
            ]
            candidates = above or candidates
        return candidates[0] if candidates else None

    def _resolve_parent(
        self,
        node: HierarchyNode,
        by_code: dict[str, list[HierarchyNode]],
        has_grouping: bool,
        result: HierarchyResult,
    ) -> None:
        record = node.record
        if record.parent_code:
            parent = self._lookup(by_code, record.parent_code, record.parent_level, node)
            if parent is None:
                result.orphans += 1
                result.issues.append(
                    _issue(
                        Severity.ERROR,
                        "MISSING_PARENT",
                        f"Parent {record.parent_code!r} of {record.key!r} is not in the source.",
                        record,
                    )
                )
                return
            node.parent_key = parent.record.key
            self._check_level_consistency(node, parent, result)
            return

        if record.level == NodeType.CHAPTER:
            return
        if self._config.infer_parents:
            parent = self._infer_parent(node, by_code, result)
            if parent is not None:
                node.parent_key = parent.record.key
                node.parent_inferred = True
                result.inferred_links += 1
                result.issues.append(
                    _issue(
                        Severity.INFO,
                        "PARENT_INFERRED",
                        f"Parent of {record.key!r} inferred as {parent.record.key!r}.",
                        record,
                    )
                )
                self._check_level_consistency(node, parent, result)
                return
        if has_grouping:
            result.orphans += 1
            result.issues.append(
                _issue(
                    Severity.ERROR,
                    "ORPHAN_RECORD",
                    f"{record.key!r} has no parent although the source has chapters/blocks.",
                    record,
                )
            )

    def _infer_parent(
        self,
        node: HierarchyNode,
        by_code: dict[str, list[HierarchyNode]],
        result: HierarchyResult,
    ) -> HierarchyNode | None:
        record = node.record
        if record.code and record.level in CLASSIFICATION_NODE_TYPES:
            for candidate_code in truncation_parents(record.code):
                parent = self._lookup(by_code, candidate_code, None, node)
                if parent is not None and parent.record.level in CLASSIFICATION_NODE_TYPES:
                    return parent
        # Range containment: category -> block, block -> chapter.
        target_levels = {
            NodeType.CATEGORY: NodeType.BLOCK,
            NodeType.SUBCATEGORY: NodeType.BLOCK,
            NodeType.CODE: NodeType.BLOCK,
            NodeType.BLOCK: NodeType.CHAPTER,
        }
        target_level = target_levels.get(record.level) if record.level else None
        if target_level is None:
            return None
        probe = record.range_start or (record.code if record.code else None)
        if record.level == NodeType.BLOCK and record.code and not record.range_start:
            bounds = parse_range(record.code)
            probe = bounds[0] if bounds else probe
        if not probe:
            return None
        containing = [
            n
            for n in result.nodes.values()
            if n.record.level == target_level
            and self._range_of(n.record) is not None
            and code_in_range(probe, *self._range_of(n.record))  # type: ignore[misc]
        ]
        # Narrowest containing range wins; ties are ambiguous and not inferred.
        if len(containing) == 1:
            return containing[0]
        return None

    @staticmethod
    def _range_of(record: NormalizedICDRecord) -> tuple[str, str] | None:
        if record.range_start and record.range_end:
            return record.range_start, record.range_end
        return parse_range(record.code) if record.code else None

    @staticmethod
    def _check_level_consistency(
        node: HierarchyNode, parent: HierarchyNode, result: HierarchyResult
    ) -> None:
        child_level, parent_level = node.record.level, parent.record.level
        if child_level is None or parent_level is None:
            return
        child_rank, parent_rank = NODE_TYPE_RANK[child_level], NODE_TYPE_RANK[parent_level]
        if child_rank < parent_rank or (
            child_rank == parent_rank and child_level != NodeType.BLOCK
        ):
            result.issues.append(
                _issue(
                    Severity.ERROR,
                    "HIERARCHY_LEVEL_INCONSISTENT",
                    f"{child_level} {node.record.key!r} cannot be a child of "
                    f"{parent_level} {parent.record.key!r}.",
                    node.record,
                )
            )
            return
        if (
            child_level in CLASSIFICATION_NODE_TYPES
            and parent_level in CLASSIFICATION_NODE_TYPES
            and node.record.code
            and parent.record.code
            and not normalize_code(node.record.code).startswith(normalize_code(parent.record.code))
        ):
            result.issues.append(
                _issue(
                    Severity.WARNING,
                    "HIERARCHY_CODE_MISMATCH",
                    f"Code {clean_code(node.record.code)} does not extend its parent code "
                    f"{clean_code(parent.record.code)}.",
                    node.record,
                )
            )
        parent_range = HierarchyBuilder._range_of(parent.record)
        if (
            parent_range
            and child_level in CLASSIFICATION_NODE_TYPES
            and node.record.code
            and not code_in_range(node.record.code, *parent_range)
        ):
            result.issues.append(
                _issue(
                    Severity.WARNING,
                    "CODE_OUTSIDE_PARENT_RANGE",
                    f"Code {clean_code(node.record.code)} lies outside the range "
                    f"{parent_range[0]}-{parent_range[1]} of its parent.",
                    node.record,
                )
            )

    @staticmethod
    def _break_cycles(result: HierarchyResult) -> None:
        state: dict[str, int] = {}  # 1 = visiting, 2 = done
        for start in result.nodes:
            if state.get(start):
                continue
            path: list[str] = []
            current: str | None = start
            while current is not None and state.get(current) is None:
                state[current] = 1
                path.append(current)
                current = result.nodes[current].parent_key
            if current is not None and state.get(current) == 1:
                cycle = path[path.index(current) :]
                result.cycles += 1
                for key in cycle:
                    record = result.nodes[key].record
                    result.issues.append(
                        _issue(
                            Severity.ERROR,
                            "HIERARCHY_CYCLE",
                            f"{key!r} is part of a parent cycle: {' -> '.join(cycle)}.",
                            record,
                        )
                    )
                # Detach the cycle so traversal terminates; the dataset is blocked anyway.
                for key in cycle:
                    result.nodes[key].parent_key = None
            for key in path:
                state[key] = 2

    def _finalise(self, result: HierarchyResult) -> None:
        for node in result.nodes.values():
            if node.parent_key is not None:
                result.nodes[node.parent_key].children.append(node.record.key)
        for key, node in result.nodes.items():
            ancestors = result.ancestors(key)
            node.depth = len(ancestors)
            for ancestor_key in ancestors:
                level = result.nodes[ancestor_key].record.level
                if level == NodeType.CHAPTER and node.chapter_key is None:
                    node.chapter_key = ancestor_key
                elif level == NodeType.BLOCK and node.block_key is None:
                    node.block_key = ancestor_key
                elif level == NodeType.CATEGORY and node.category_key is None:
                    node.category_key = ancestor_key
            record = node.record
            if record.is_selectable is not None:
                node.is_selectable = record.is_selectable
            else:
                node.is_selectable = (
                    self._config.leaves_selectable
                    and record.level in CLASSIFICATION_NODE_TYPES
                    and not node.children
                )
