"""Read access to ICD records and their hierarchy. Every query is scoped to one dataset."""

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import CLASSIFICATION_NODE_TYPES, NodeType
from app.ingestion.codes import clean_code, normalize_code
from app.models import IcdIndexEntry, IcdNode, IcdRule, IcdSourceRef, IcdTerm

MAX_DESCENDANTS = 2000


class IcdRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_code(
        self, dataset_id: int, code: str, *, classification_only: bool = False
    ) -> IcdNode | None:
        """Exact lookup by printed code or normalized code ("A00.0" == "A000")."""
        cleaned = clean_code(code)
        statement = select(IcdNode).where(
            IcdNode.dataset_id == dataset_id,
            or_(
                IcdNode.code == cleaned,
                IcdNode.normalized_code == normalize_code(cleaned),
                IcdNode.normalized_code == normalize_code(cleaned, NodeType.BLOCK),
            ),
        )
        if classification_only:
            statement = statement.where(IcdNode.node_type.in_(CLASSIFICATION_NODE_TYPES))
        # Prefer classification records over grouping records on code collisions.
        rows = (await self._session.execute(statement)).scalars().all()
        rows = sorted(rows, key=lambda n: n.node_type not in CLASSIFICATION_NODE_TYPES)
        return rows[0] if rows else None

    async def get_by_codes(self, dataset_id: int, codes: Sequence[str]) -> dict[str, IcdNode]:
        """Batch exact lookup of classification codes. Keys are the codes as requested."""
        wanted = {normalize_code(c): c for c in codes if c}
        if not wanted:
            return {}
        rows = await self._session.execute(
            select(IcdNode).where(
                IcdNode.dataset_id == dataset_id,
                IcdNode.normalized_code.in_(list(wanted)),
                IcdNode.node_type.in_(CLASSIFICATION_NODE_TYPES),
            )
        )
        return {wanted[node.normalized_code]: node for node in rows.scalars()}  # type: ignore[index]

    async def get_many(self, dataset_id: int, node_ids: Sequence[int]) -> dict[int, IcdNode]:
        if not node_ids:
            return {}
        rows = await self._session.execute(
            select(IcdNode).where(IcdNode.dataset_id == dataset_id, IcdNode.id.in_(node_ids))
        )
        return {node.id: node for node in rows.scalars()}

    async def children(self, dataset_id: int, node_id: int) -> list[IcdNode]:
        rows = await self._session.execute(
            select(IcdNode)
            .where(IcdNode.dataset_id == dataset_id, IcdNode.parent_id == node_id)
            .order_by(IcdNode.sort_order, IcdNode.code)
        )
        return list(rows.scalars())

    async def children_of_many(
        self, dataset_id: int, node_ids: Sequence[int]
    ) -> dict[int, list[IcdNode]]:
        result: dict[int, list[IcdNode]] = {node_id: [] for node_id in node_ids}
        if not node_ids:
            return result
        rows = await self._session.execute(
            select(IcdNode)
            .where(IcdNode.dataset_id == dataset_id, IcdNode.parent_id.in_(node_ids))
            .order_by(IcdNode.sort_order, IcdNode.code)
        )
        for node in rows.scalars():
            result[node.parent_id].append(node)  # type: ignore[index]
        return result

    async def ancestors(self, dataset_id: int, node_id: int) -> list[IcdNode]:
        """Root-first ancestors via one recursive query (depth-bounded against bad data)."""
        rows = await self._session.execute(
            text(
                """
                WITH RECURSIVE up(id, parent_id, lvl) AS (
                    SELECT id, parent_id, 0 FROM icd_nodes
                    WHERE id = :node_id AND dataset_id = :dataset_id
                    UNION ALL
                    SELECT n.id, n.parent_id, up.lvl + 1 FROM icd_nodes n
                    JOIN up ON n.id = up.parent_id
                    WHERE n.dataset_id = :dataset_id AND up.lvl < 50
                )
                SELECT id, lvl FROM up WHERE lvl > 0 ORDER BY lvl DESC
                """
            ),
            {"node_id": node_id, "dataset_id": dataset_id},
        )
        ordered_ids = [row.id for row in rows]
        nodes = await self.get_many(dataset_id, ordered_ids)
        return [nodes[i] for i in ordered_ids if i in nodes]

    async def ancestors_of_many(
        self, dataset_id: int, node_ids: Sequence[int]
    ) -> dict[int, list[IcdNode]]:
        """Root-first ancestor chains for many nodes in two queries."""
        if not node_ids:
            return {}
        rows = await self._session.execute(
            text(
                """
                WITH RECURSIVE up(origin, id, parent_id, lvl) AS (
                    SELECT id, id, parent_id, 0 FROM icd_nodes
                    WHERE dataset_id = :dataset_id AND id = ANY(:node_ids)
                    UNION ALL
                    SELECT up.origin, n.id, n.parent_id, up.lvl + 1 FROM icd_nodes n
                    JOIN up ON n.id = up.parent_id
                    WHERE n.dataset_id = :dataset_id AND up.lvl < 50
                )
                SELECT origin, id, lvl FROM up WHERE lvl > 0 ORDER BY origin, lvl DESC
                """
            ),
            {"node_ids": list(node_ids), "dataset_id": dataset_id},
        )
        chains: dict[int, list[int]] = {node_id: [] for node_id in node_ids}
        for row in rows:
            chains[row.origin].append(row.id)
        nodes = await self.get_many(dataset_id, sorted({i for c in chains.values() for i in c}))
        return {origin: [nodes[i] for i in chain if i in nodes] for origin, chain in chains.items()}

    async def descendants(
        self, dataset_id: int, node_id: int, limit: int = MAX_DESCENDANTS
    ) -> list[IcdNode]:
        rows = await self._session.execute(
            text(
                """
                WITH RECURSIVE down(id, lvl) AS (
                    SELECT id, 0 FROM icd_nodes WHERE id = :node_id AND dataset_id = :dataset_id
                    UNION ALL
                    SELECT n.id, down.lvl + 1 FROM icd_nodes n JOIN down ON n.parent_id = down.id
                    WHERE n.dataset_id = :dataset_id AND down.lvl < 50
                )
                SELECT id FROM down WHERE lvl > 0 LIMIT :limit
                """
            ),
            {"node_id": node_id, "dataset_id": dataset_id, "limit": limit},
        )
        ids = [row.id for row in rows]
        nodes = await self.get_many(dataset_id, ids)
        return sorted(nodes.values(), key=lambda n: (n.depth, n.sort_order or 0, n.code or ""))

    async def details_of_many(
        self, dataset_id: int, node_ids: Sequence[int]
    ) -> dict[int, dict[str, Any]]:
        """Terms, rules, index terms and provenance for many nodes (four queries total)."""
        details: dict[int, dict[str, Any]] = {
            node_id: {"terms": [], "rules": [], "index_terms": [], "source_refs": []}
            for node_id in node_ids
        }
        if not node_ids:
            return details
        ids = list(node_ids)
        for term in (
            await self._session.execute(
                select(IcdTerm)
                .where(IcdTerm.dataset_id == dataset_id, IcdTerm.node_id.in_(ids))
                .order_by(IcdTerm.id)
            )
        ).scalars():
            details[term.node_id]["terms"].append(term)
        for rule in (
            await self._session.execute(
                select(IcdRule)
                .where(IcdRule.dataset_id == dataset_id, IcdRule.node_id.in_(ids))
                .order_by(IcdRule.id)
            )
        ).scalars():
            details[rule.node_id]["rules"].append(rule)  # type: ignore[index]
        for entry in (
            await self._session.execute(
                select(IcdIndexEntry)
                .where(
                    IcdIndexEntry.dataset_id == dataset_id, IcdIndexEntry.target_node_id.in_(ids)
                )
                .order_by(IcdIndexEntry.id)
            )
        ).scalars():
            details[entry.target_node_id]["index_terms"].append(entry)  # type: ignore[index]
        for ref in (
            await self._session.execute(
                select(IcdSourceRef)
                .where(IcdSourceRef.dataset_id == dataset_id, IcdSourceRef.node_id.in_(ids))
                .order_by(IcdSourceRef.id)
            )
        ).scalars():
            details[ref.node_id]["source_refs"].append(ref)  # type: ignore[index]
        return details

    async def count_by_type(self, dataset_id: int) -> dict[str, int]:
        rows = await self._session.execute(
            select(IcdNode.node_type, func.count())
            .where(IcdNode.dataset_id == dataset_id)
            .group_by(IcdNode.node_type)
        )
        return {node_type.value: count for node_type, count in rows}
