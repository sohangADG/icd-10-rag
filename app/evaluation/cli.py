"""Evaluation command line.

    python -m app.evaluation.cli run --cases cases.jsonl [--k 3] [--out report.json]
    python -m app.evaluation.cli run --synthetic            # built-in synthetic cases

The report separates concept extraction, retrieval, reranking and final selection.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.core.database import create_engine_from_settings
from app.core.logging import configure_logging
from app.evaluation.models import EvalCase, load_cases
from app.evaluation.runner import EvaluationRunner
from app.indexing.embeddings import get_embedding_provider
from app.synthetic.eval_cases import synthetic_cases


async def evaluate(cases: list[EvalCase], k: int) -> dict[str, Any]:
    settings = get_settings()
    engine = create_engine_from_settings(settings)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            runner = EvaluationRunner(session, settings, get_embedding_provider(settings), k=k)
            return await runner.run(cases)
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure_logging("WARNING", settings.log_format, stream=sys.stderr)
    parser = argparse.ArgumentParser(prog="python -m app.evaluation.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", help="JSONL or CSV evaluation cases")
    source.add_argument("--synthetic", action="store_true", help="built-in synthetic cases")
    run.add_argument("--k", type=int, default=3)
    run.add_argument("--out", help="write the full JSON report here")
    run.add_argument("--summary", action="store_true", help="omit per-case details on stdout")
    args = parser.parse_args(argv)

    cases = (
        [EvalCase.model_validate(c) for c in synthetic_cases()]
        if args.synthetic
        else load_cases(Path(args.cases))
    )
    report = asyncio.run(evaluate(cases, args.k))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    shown = {k: v for k, v in report.items() if not (args.summary and k == "details")}
    print(json.dumps(shown, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
