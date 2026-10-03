from datetime import date

import pytest
from pydantic import ValidationError

from app.core.constants import ICD10CA_2022
from app.schemas.dataset import DatasetCreate


def test_icd10ca_2022_identity_is_valid() -> None:
    dataset = DatasetCreate(**ICD10CA_2022)

    assert (dataset.system, dataset.country, dataset.version) == ("ICD-10-CA", "CA", "2022")
    assert dataset.revision is None


def test_inverted_effective_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetCreate(
            **ICD10CA_2022, effective_from=date(2023, 1, 1), effective_to=date(2022, 1, 1)
        )
