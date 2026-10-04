"""Text normalization shared by ingestion (stored normalized terms) and retrieval (queries).

Both sides must normalize identically, so this lives in one place.
"""

import re
import unicodedata

_NON_WORD = re.compile(r"[^0-9a-z]+")

# Language code -> PostgreSQL text search configuration.
_TS_CONFIGS = {"en": "english", "fr": "french", "de": "german", "es": "spanish", "pt": "portuguese"}


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Lower-case, accent-free, punctuation collapsed to single spaces."""
    return _NON_WORD.sub(" ", strip_accents(text).lower()).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split(" ") if token]


def ts_config_for(language: str) -> str:
    """Text search configuration for a dataset language ('simple' when unsupported)."""
    return _TS_CONFIGS.get(language.split("-")[0].lower(), "simple")
