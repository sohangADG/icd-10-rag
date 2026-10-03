"""Register the ICD-10-CA 2022 dataset identity (metadata only; nothing is ingested).

Idempotent: running it again returns the existing record instead of creating a duplicate.

    python -m scripts.register_dataset
"""

import asyncio
import logging
import sys

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import get_settings
from app.core.constants import ICD10CA_2022
from app.core.database import create_engine_from_settings
from app.core.logging import configure_logging
from app.schemas.dataset import DatasetCreate
from app.services.dataset_service import DatasetService

logger = logging.getLogger("scripts.register_dataset")


async def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    engine = create_engine_from_settings(settings)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            result = await DatasetService(session).register(DatasetCreate(**ICD10CA_2022))
    except (SQLAlchemyError, OSError) as exc:
        logger.error(
            "dataset registration failed; is the database up and migrated?",
            extra={"error_type": type(exc).__name__},
        )
        return 1
    finally:
        await engine.dispose()

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
