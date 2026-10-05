"""Evaluation command line.

    python -m app.evaluation.cli run --cases cases.jsonl [--k 3] [--out report.json]
    python -m app.evaluation.cli run --synthetic            # built-in synthetic cases

    # Clinical scenario suite: stage-separated metrics + release safety gates
    python -m app.evaluation.cli run --dataset tests/evaluation/clinical_scenarios.json
        --coding-system SYNTH-ICD --version 2024 [--output report.json] [--markdown report.md]
        [--top-k 5] [--embedding-provider sentence_transformers] [--tags negation,history]
        [--only-dataset] [--fail-on-safety-error]

The reports separate concept extraction, retrieval, reranking and final selection. Synthetic
scenarios measure system behaviour and safety only, never real coding accuracy.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api.deps import load_provider
from app.core.config import get_settings
from app.core.database import create_engine_from_settings
from app.core.logging import configure_logging
from app.evaluation.clinical.report import markdown
from app.evaluation.clinical.run import run_clinical_evaluation
from app.evaluation.clinical.scenarios import load_scenarios, select
from app.evaluation.models import EvalCase, load_cases
from app.evaluation.retrieval_comparison import compare_providers
from app.evaluation.runner import EvaluationRunner
from app.indexing.embeddings import get_embedding_provider
from app.services.dataset_resolver import DatasetResolver
from app.synthetic.eval_cases import synthetic_cases
from app.synthetic.paraphrase import paraphrase_cases


async def evaluate(cases: list[EvalCase], k: int) -> dict[str, Any]:
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            runner = EvaluationRunner(session, settings, get_embedding_provider(settings), k=k)
            return await runner.run(cases)
    finally:
        await engine.dispose()


async def compare(
    coding_system: str, version: str, cases: list[EvalCase], providers: list[str], k: int
) -> dict[str, Any]:
    settings = get_settings()
    built = [
        get_embedding_provider(settings.model_copy(update={"embedding_provider": name}))
        for name in providers
    ]
    engine = create_engine_from_settings(settings)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            dataset = await DatasetResolver(session).resolve(
                coding_system=coding_system, version=version
            )
            return await compare_providers(
                session, settings, dataset, cases, [p for p in built if p is not None], k=k
            )
    finally:
        await engine.dispose()


async def clinical(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    if args.embedding_provider:
        settings = settings.model_copy(
            update={"embedding_provider": args.embedding_provider.replace("-", "_")}
        )
    scenarios = load_scenarios(Path(args.dataset))
    chosen = select(
        scenarios,
        tags={t.strip() for t in args.tags.split(",") if t.strip()} if args.tags else None,
        coding_system=args.coding_system if args.only_dataset else None,
        version=args.version if args.only_dataset else None,
    )
    handle = load_provider(settings)
    engine = create_engine_from_settings(settings)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            return await run_clinical_evaluation(
                session,
                settings,
                handle.provider,
                chosen,
                top_k=args.top_k,
                provider_status=handle.status,
                meta={
                    "scenario_file": str(args.dataset),
                    "default_dataset": f"{args.coding_system} {args.version}",
                    "tags": args.tags,
                },
            )
    finally:
        await engine.dispose()


def _cases(name: str) -> list[EvalCase]:
    if name == "paraphrase":
        return [EvalCase.model_validate(c) for c in paraphrase_cases()]
    if name == "synthetic":
        return [EvalCase.model_validate(c) for c in synthetic_cases() if c.get("expected_code")]
    return load_cases(Path(name))


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging("WARNING", settings.log_format, stream=sys.stderr)
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", help="JSONL or CSV evaluation cases")
    source.add_argument("--synthetic", action="store_true", help="built-in synthetic cases")
    source.add_argument(
        "--paraphrase", action="store_true", help="built-in paraphrase (semantic) cases"
    )
    source.add_argument("--dataset", help="clinical scenario suite (JSON/JSONL)")
    run.add_argument("--k", type=int, default=3)
    run.add_argument("--out", "--output", dest="out", help="write the full JSON report here")
    run.add_argument("--markdown", help="clinical suite: also write a Markdown summary here")
    run.add_argument("--coding-system", default="SYNTH-ICD", help="clinical suite: default system")
    run.add_argument("--version", default="2024", help="clinical suite: default version")
    run.add_argument(
        "--only-dataset",
        action="store_true",
        help="clinical suite: run only the scenarios of --coding-system/--version",
    )
    run.add_argument("--top-k", type=int, default=5, help="clinical suite: suggestions top_k")
    run.add_argument("--embedding-provider", help="override EMBEDDING_PROVIDER for this run")
    run.add_argument("--tags", help="clinical suite: comma-separated tags (any of)")
    run.add_argument(
        "--fail-on-safety-error",
        action="store_true",
        help="exit with status 2 when any safety gate fails",
    )
    run.add_argument("--summary", action="store_true", help="omit per-case details on stdout")
    cmp = sub.add_parser(
        "compare",
        help="compare lexical-only / vector-only / hybrid retrieval per embedding provider "
        "(re-embeds the dataset for each provider; the last one stays active)",
    )
    cmp.add_argument("--coding-system", required=True)
    cmp.add_argument("--version", required=True)
    cmp.add_argument("--cases", default="paraphrase", help="paraphrase | synthetic | FILE")
    cmp.add_argument("--providers", default="hashing,sentence_transformers")
    cmp.add_argument("--k", type=int, default=3)
    cmp.add_argument("--out")
    cmp.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "compare":
        report = asyncio.run(
            compare(
                args.coding_system,
                args.version,
                _cases(args.cases),
                [p.strip() for p in args.providers.split(",") if p.strip()],
                args.k,
            )
        )
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        shown = {k: v for k, v in report.items() if not (args.summary and k == "details")}
        print(json.dumps(shown, indent=2, default=str))
        return 0

    if args.dataset:
        report = asyncio.run(clinical(args))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        if args.markdown:
            Path(args.markdown).write_text(markdown(report), encoding="utf-8")
        shown = {k: v for k, v in report.items() if not (args.summary and k == "cases")}
        print(json.dumps(shown, indent=2, default=str))
        if args.fail_on_safety_error and not report["safety_passed"]:
            return 2
        return 0

    if args.synthetic:
        cases = [EvalCase.model_validate(c) for c in synthetic_cases()]
    elif args.paraphrase:
        cases = _cases("paraphrase")
    else:
        cases = load_cases(Path(args.cases))
    report = asyncio.run(evaluate(cases, args.k))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    shown = {k: v for k, v in report.items() if not (args.summary and k == "details")}
    print(json.dumps(shown, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
