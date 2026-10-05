"""Test helpers: write synthetic sources in any format and build manifests for them."""

from pathlib import Path
from typing import Any

from app.ingestion.adapters import SourceManifest
from app.synthetic.dataset import CHAPTERS_2024, DATASET_2024, chapters_2025, dataset_2025
from app.synthetic.renderers import FORMAT_WRITERS


def synthetic(version: str = "2024") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if version == "2025":
        return dataset_2025(), chapters_2025()
    return DATASET_2024, CHAPTERS_2024


def write_source(
    directory: Path, fmt: str, version: str = "2024", **writer_kwargs: Any
) -> tuple[Path, SourceManifest]:
    writer, extension = FORMAT_WRITERS[fmt]
    dataset, chapters = synthetic(version)
    path = directory / f"synth_{version}_{fmt}{extension}"
    manifest = writer(path, dataset, chapters, **writer_kwargs)
    return path, SourceManifest.model_validate({**manifest, "base_dir": str(directory)})


def manifest_for(dataset_overrides: dict[str, Any] | None = None, **values: Any) -> SourceManifest:
    return SourceManifest.model_validate(
        {"dataset": {**DATASET_2024, **(dataset_overrides or {})}, **values}
    )
