"""Minimal row builders for constraint tests. Values are synthetic test fixtures, not ICD content;
every row is rolled back at the end of its test."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import NodeType
from app.models import IcdDataset, IcdNode


async def make_dataset(session: AsyncSession, **overrides: Any) -> IcdDataset:
    values: dict[str, Any] = {
        "system": "TEST-ICD",
        "country": "XX",
        "version": "0000",
        "publisher": "Test Publisher",
        "language": "en",
    }
    values.update(overrides)
    dataset = IcdDataset(**values)
    session.add(dataset)
    await session.flush()
    return dataset


async def make_node(
    session: AsyncSession,
    dataset: IcdDataset,
    *,
    node_type: NodeType = NodeType.CATEGORY,
    code: str | None = "T00",
    depth: int = 0,
    parent: IcdNode | None = None,
    **overrides: Any,
) -> IcdNode:
    node = IcdNode(
        dataset_id=dataset.id,
        parent_id=parent.id if parent else None,
        node_type=node_type,
        code=code,
        title=f"Test node {code}",
        depth=depth,
        **overrides,
    )
    session.add(node)
    await session.flush()
    return node
