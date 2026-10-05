"""Generate the synthetic fixture sources (all formats, two versions) with manifests.

    python -m app.synthetic.cli build --out data/synthetic

Writes e.g. synth_2024.json + synth_2024.json.manifest.json, synth_2024.pdf (+ manifest), ...,
synth_malformed.csv, synth_restricted_2024.pdf. Output is regenerable and git-ignored.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from app.synthetic.alt_system import ALT_CHAPTERS, DATASET_ALT
from app.synthetic.dataset import (
    CHAPTERS_2024,
    DATASET_2024,
    chapters_2025,
    dataset_2025,
    malformed_records,
)
from app.synthetic.paraphrase import DATASET_PARAPHRASE, paraphrase_chapters
from app.synthetic.renderers import FORMAT_WRITERS, write_json, write_pdf


def build(out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for version, dataset, chapters in (
        ("2024", DATASET_2024, CHAPTERS_2024),
        ("2025", dataset_2025(), chapters_2025()),
    ):
        for fmt, (writer, extension) in FORMAT_WRITERS.items():
            path = out / f"synth_{version}_{fmt}{extension}"
            manifest = writer(path, dataset, chapters)
            manifest_path = path.with_name(path.name + ".manifest.json")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            written += [path.name, manifest_path.name]

    # Paraphrase dataset for semantic-retrieval tests (titles share no words with the queries).
    paraphrase = out / "synth_paraphrase_json.json"
    manifest = write_json(paraphrase, DATASET_PARAPHRASE, paraphrase_chapters())
    (out / (paraphrase.name + ".manifest.json")).write_text(json.dumps(manifest, indent=2))
    written += [paraphrase.name, paraphrase.name + ".manifest.json"]

    # Second coding system with overlapping code strings (coding-system isolation tests).
    alt = out / "synth_alt_2024_json.json"
    manifest = write_json(alt, DATASET_ALT, ALT_CHAPTERS)
    (out / (alt.name + ".manifest.json")).write_text(json.dumps(manifest, indent=2))
    written += [alt.name, alt.name + ".manifest.json"]

    restricted = out / "synth_restricted_2024.pdf"
    manifest = write_pdf(restricted, DATASET_2024, CHAPTERS_2024, restricted=True)
    (out / (restricted.name + ".manifest.json")).write_text(json.dumps(manifest, indent=2))
    written.append(restricted.name)

    malformed = out / "synth_malformed.csv"
    fields = ["code", "level", "title", "parent_code", "exclusions"]
    with malformed.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in malformed_records():
            writer.writerow(
                {
                    **{k: record.get(k) or "" for k in fields},
                    "exclusions": "|".join(record.get("exclusions", [])),
                }
            )
    malformed_manifest = {
        "adapter": "csv",
        "dataset": {**DATASET_2024, "version": "malformed"},
        "mapping": {},
    }
    (out / "synth_malformed.csv.manifest.json").write_text(json.dumps(malformed_manifest, indent=2))
    written.append(malformed.name)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.synthetic.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--out", default="data/synthetic")
    args = parser.parse_args(argv)
    files = build(Path(args.out))
    print(json.dumps({"out": args.out, "files": files}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
