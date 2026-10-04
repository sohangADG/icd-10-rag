"""Evaluation case format.

JSONL (one object per line) or CSV with columns:
    clinical_note, expected_code, expected_dataset, expected_version, notes
optional: expected_concept, expected_status, coder_code

expected_code empty/null means "no code should be suggested" (e.g. negated findings).
"""

import csv
import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    case_id: str | None = None
    clinical_note: str = Field(min_length=1)
    expected_code: str | None = None
    expected_dataset: str
    expected_version: str
    notes: str | None = None
    expected_concept: str | None = None
    expected_status: str | None = None
    coder_code: str | None = None  # human coder's assignment, for agreement measurement


def load_cases(path: Path) -> list[EvalCase]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [EvalCase.model_validate({k: (v or None) for k, v in row.items()}) for row in rows]
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            data = json.loads(line)
            data.setdefault("case_id", f"line-{number}")
            cases.append(EvalCase.model_validate(data))
    return cases
