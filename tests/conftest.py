"""Suite-wide test configuration.

The default test suite is deterministic and offline: it uses the hashing embedding provider,
whatever the local `.env` says (environment variables take precedence over `.env`). Tests that
load a real sentence-transformers model are marked `semantic_model` and run only when
RUN_SEMANTIC_MODEL_TESTS=1 (the model is downloaded into HF_HOME on first use).
"""

import os

import pytest

os.environ["EMBEDDING_PROVIDER"] = "hashing"
os.environ["EMBEDDING_MODEL"] = "hashing-v1"
os.environ["EMBEDDING_DIMENSION"] = "256"
os.environ["EMBEDDING_QUERY_PREFIX"] = ""
os.environ.pop("EMBEDDING_SIMILARITY_FLOOR", None)
os.environ["SEMANTIC_RETRIEVAL_MODE"] = "optional"

RUN_SEMANTIC = os.environ.get("RUN_SEMANTIC_MODEL_TESTS") == "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if RUN_SEMANTIC:
        return
    skip = pytest.mark.skip(reason="real-model test; set RUN_SEMANTIC_MODEL_TESTS=1 to run")
    for item in items:
        if "semantic_model" in item.keywords:
            item.add_marker(skip)
