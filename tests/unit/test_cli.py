import json
from pathlib import Path

import pytest

from app.evaluation.cli import main as evaluation_main  # noqa: F401 - importable entry point
from app.ingestion.cli import main
from app.synthetic.cli import build


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("cli")
    build(directory)
    return directory


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_validate_dry_run(sources: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = sources / "synth_2024_pdf.pdf"
    code, payload = _run(
        ["validate", "--file", str(source), "--manifest", f"{source}.manifest.json"], capsys
    )
    assert code == 0 and payload["valid"]
    stats = payload["statistics"]
    assert (stats["dataset"], stats["version"], stats["records_extracted"]) == (
        "SYNTH-ICD",
        "2024",
        70,
    )
    assert stats["validation_errors"] == 0 and stats["pages_processed"] >= 5


def test_validate_reports_fatal_issues(sources: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = sources / "synth_malformed.csv"
    code, payload = _run(
        ["validate", "--file", str(source), "--manifest", f"{source}.manifest.json"], capsys
    )
    assert code == 1 and not payload["valid"]
    assert payload["statistics"]["validation_errors"] > 0


def test_inspect_restricted_pdf_reads_no_content(
    sources: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, payload = _run(["inspect", "--file", str(sources / "synth_restricted_2024.pdf")], capsys)
    assert code == 3
    assert payload["encrypted"] and not payload["text_extraction_permitted"]
    assert payload["details"]["permissions"]["copy_extract"] is False


def test_ingest_requires_a_licence_basis(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "x.csv"
    source.write_text("code,title\nA00,Something\n")
    code, payload = _run(
        [
            "ingest",
            "--file",
            str(source),
            "--coding-system",
            "SYNTH-ICD",
            "--version",
            "1",
            "--country",
            "XX",
            "--language",
            "en",
            "--publisher",
            "p",
        ],
        capsys,
    )
    assert code == 2 and "licence basis" in payload["error"]


def test_identity_overrides_from_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "x.csv"
    source.write_text("code,title\nA00,Something\nA01,Other\n")
    code, payload = _run(
        [
            "validate",
            "--file",
            str(source),
            "--coding-system",
            "SYNTH-ICD",
            "--version",
            "9",
            "--country",
            "XX",
            "--language",
            "en",
            "--publisher",
            "p",
        ],
        capsys,
    )
    assert code == 0 and payload["metadata"]["version"] == "9"
