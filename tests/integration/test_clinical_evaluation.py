"""The complete clinical scenario suite against PostgreSQL (hashing provider, deterministic).

Safety gates must pass. Quality metrics are not asserted here: the hashing provider is not
meaning-based, so paraphrase scenarios cannot be found. The real-model run happens in the
Docker clinical E2E (scripts/clinical_evaluation_e2e.py).
"""

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.evaluation.clinical.run import run_clinical_evaluation
from app.evaluation.clinical.scenarios import load_scenarios
from app.indexing.embeddings import get_embedding_provider
from app.indexing.indexer import SearchIndexer
from app.ingestion.adapters import SourceManifest
from app.ingestion.importer import DatasetImporter
from app.ingestion.pipeline import run_pipeline
from app.models import IcdDataset
from app.synthetic.alt_system import ALT_CHAPTERS, DATASET_ALT
from app.synthetic.paraphrase import DATASET_PARAPHRASE, paraphrase_chapters
from app.synthetic.renderers import write_json
from tests.integration.ingest import import_synthetic

SCENARIOS = Path(__file__).parents[1] / "evaluation" / "clinical_scenarios.json"


async def _import_json(session: AsyncSession, path: Path, dataset: dict, chapters: list) -> None:
    manifest = SourceManifest.model_validate(
        {**write_json(path, dataset, chapters), "base_dir": str(path.parent)}
    )
    outcome = await DatasetImporter(session).import_result(run_pipeline(path, manifest))
    assert outcome.status == "imported"
    loaded = await session.get(IcdDataset, outcome.dataset_id)
    assert loaded is not None
    settings = get_settings()
    await SearchIndexer(session, get_embedding_provider(settings)).embed_documents(loaded)


async def test_complete_clinical_suite_passes_every_safety_gate(
    session: AsyncSession, tmp_path: Path
) -> None:
    await import_synthetic(session, tmp_path, "json", "2024")
    await import_synthetic(session, tmp_path, "json", "2025")
    await _import_json(
        session, tmp_path / "paraphrase.json", DATASET_PARAPHRASE, paraphrase_chapters()
    )
    await _import_json(session, tmp_path / "alt.json", DATASET_ALT, ALT_CHAPTERS)

    settings = get_settings()
    scenarios = load_scenarios(SCENARIOS)
    report = await run_clinical_evaluation(
        session, settings, get_embedding_provider(settings), scenarios
    )

    failed = [g for g in report["safety_gates"] if not g["passed"]]
    assert report["safety_passed"], failed
    assert report["dataset"]["scenarios"] == len(scenarios)
    assert report["database_safety"]["returned_codes_checked"] > 0
    assert report["database_safety"]["dataset_resolution_errors"] == []
    # Stage-separated sections are always present, measured on the same run.
    for section in ("concept_extraction", "retrieval", "reranking", "rules", "abstention"):
        assert report[section]
    assert set(report["retrieval"]["by_mode"]) == {
        "exact",
        "lexical",
        "fuzzy",
        "index_term",
        "vector",
        "hybrid",
    }
