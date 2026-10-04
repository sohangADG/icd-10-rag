"""Ingestion command line.

    python -m app.ingestion.cli inspect  --file SRC [--manifest M]
    python -m app.ingestion.cli validate --file SRC  --manifest M [--show-issues 50]
    python -m app.ingestion.cli ingest   --file SRC  --manifest M [--embed] [--licence-basis TEXT]
    python -m app.ingestion.cli index    --dataset-id N [--no-embed]
    python -m app.ingestion.cli datasets

Dataset identity comes from the manifest (or --coding-system/--version/... overrides); it is
never inferred from file names. `ingest` refuses to run unless a licence basis is recorded
(manifest dataset.licence.basis or --licence-basis): restricted sources may only be ingested by
operators who hold the rights to do so (docs/licensing.md).

Output is JSON on stdout; exit codes: 0 ok, 1 validation/import failure, 2 usage/licence error,
3 source restricted.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.core.database import create_engine_from_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.indexing.embeddings import get_embedding_provider
from app.ingestion.adapters import AdapterError, SourceManifest, select_adapter
from app.ingestion.pipeline import PipelineResult
from app.ingestion.service import IngestionService
from app.repositories.dataset_repository import DatasetRepository

_IDENTITY_FLAGS = ("coding_system", "version", "country", "language", "publisher", "edition")


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _manifest(args: argparse.Namespace) -> SourceManifest:
    manifest = SourceManifest.from_file(Path(args.manifest)) if args.manifest else SourceManifest()
    overrides = {
        flag: getattr(args, flag)
        for flag in _IDENTITY_FLAGS
        if getattr(args, flag, None) is not None
    }
    if overrides:
        manifest.dataset = {**manifest.dataset, **overrides}
    if getattr(args, "adapter", None):
        manifest.adapter = args.adapter
    return manifest


def _validation_payload(result: PipelineResult, show: int) -> dict[str, Any]:
    return {
        "source": result.source_filename,
        "valid": result.is_valid,
        "restricted": result.restricted,
        "statistics": result.statistics(),
        "metadata": result.metadata.model_dump(mode="json") if result.metadata else None,
        "issues": [issue.model_dump(mode="json") for issue in result.issues[:show]],
        "issues_truncated": max(0, len(result.issues) - show),
        "duration_ms": result.duration_ms,
    }


def cmd_inspect(args: argparse.Namespace) -> int:
    """Format-level facts only (hash, size, pages, encryption/permissions). No content read."""
    try:
        adapter = select_adapter(Path(args.file), _manifest(args))
        inspection = adapter.inspect_metadata()
    except AdapterError as exc:
        _print({"error": str(exc)})
        return 2
    payload = {"adapter": adapter.source_type.value, **inspection.model_dump(mode="json")}
    _print(payload)
    return 0 if inspection.text_extraction_permitted else 3


def cmd_validate(args: argparse.Namespace) -> int:
    result = IngestionService.validate(Path(args.file), _manifest(args))
    _print(_validation_payload(result, args.show_issues))
    if result.restricted:
        return 3
    return 0 if result.is_valid else 1


async def _with_session(callback):  # noqa: ANN001, ANN202 - small internal helper
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            return await callback(session)
    finally:
        await engine.dispose()


def cmd_ingest(args: argparse.Namespace) -> int:
    manifest = _manifest(args)
    licence = dict(manifest.dataset.get("licence") or {})
    if not licence:
        # Identity (and licence) may be embedded in the source itself (e.g. a JSON header).
        try:
            licence = dict(select_adapter(Path(args.file), manifest).get_dataset_metadata().licence)
        except AdapterError:
            licence = {}
    if args.licence_basis:
        licence["basis"] = args.licence_basis
        licence["confirmed_via"] = "cli"
    if not str(licence.get("basis") or "").strip():
        _print(
            {
                "error": "Refusing to ingest without a recorded licence basis. Add "
                "dataset.licence.basis to the manifest or pass --licence-basis, stating the "
                "rights under which this source may be processed (see docs/licensing.md)."
            }
        )
        return 2
    manifest.dataset = {**manifest.dataset, "licence": licence}
    provider = get_embedding_provider(get_settings()) if args.embed else None

    async def run(session):  # noqa: ANN001, ANN202
        return await IngestionService(session, provider).ingest(
            Path(args.file), manifest, embed=args.embed
        )

    try:
        result, outcome, index_stats = asyncio.run(_with_session(run))
    except AppError as exc:
        _print({"error": exc.code, "message": exc.message, "details": exc.details})
        return 1
    payload = _validation_payload(result, args.show_issues)
    payload["import"] = asdict(outcome)
    payload["embeddings"] = asdict(index_stats) if index_stats else None
    _print(payload)
    if result.restricted:
        return 3
    return 0 if outcome.status in {"imported", "skipped"} else 1


def cmd_index(args: argparse.Namespace) -> int:
    provider = None if args.no_embed else get_embedding_provider(get_settings())

    async def run(session):  # noqa: ANN001, ANN202
        return await IngestionService(session, provider).reindex(
            args.dataset_id, embed=not args.no_embed
        )

    try:
        payload = asyncio.run(_with_session(run))
    except AppError as exc:
        _print({"error": exc.code, "message": exc.message})
        return 1
    _print({"dataset_id": args.dataset_id, **payload})
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    async def run(session):  # noqa: ANN001, ANN202
        rows = await DatasetRepository(session).list_datasets(limit=500)
        return [
            {
                "id": d.id,
                "coding_system": d.system,
                "version": d.version,
                "country": d.country,
                "language": d.language,
                "status": d.status.value,
                "source_filename": d.source_filename,
                "source_sha256": d.source_checksum,
                "imported_at": d.imported_at,
            }
            for d in rows
        ]

    _print({"datasets": asyncio.run(_with_session(run))})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--file", required=True, help="source file")
        p.add_argument("--manifest", help="source manifest JSON (identity + mapping)")
        p.add_argument("--adapter", help="force an adapter (pdf, pdf_ocr, text, csv, ...)")
        for flag in _IDENTITY_FLAGS:
            p.add_argument(f"--{flag.replace('_', '-')}", dest=flag)
        p.add_argument("--show-issues", type=int, default=50)

    source_args(sub.add_parser("inspect", help="format/permission facts, no content read"))
    source_args(sub.add_parser("validate", help="dry run: parse + validate, no database"))
    ingest = sub.add_parser("ingest", help="validate, import transactionally, index")
    source_args(ingest)
    ingest.add_argument("--embed", action="store_true", help="also generate embeddings")
    ingest.add_argument("--licence-basis", help="operator statement of the licence/rights")
    index = sub.add_parser("index", help="rebuild search documents / embeddings")
    index.add_argument("--dataset-id", type=int, required=True)
    index.add_argument("--no-embed", action="store_true")
    sub.add_parser("datasets", help="list datasets")
    return parser


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    # Logs go to stderr so stdout stays machine-readable JSON.
    configure_logging(settings.log_level, settings.log_format, stream=sys.stderr)
    args = build_parser().parse_args(argv)
    handlers = {
        "inspect": cmd_inspect,
        "validate": cmd_validate,
        "ingest": cmd_ingest,
        "index": cmd_index,
        "datasets": cmd_datasets,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
