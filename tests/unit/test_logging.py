import json
import logging

from app.core.logging import JsonFormatter


def test_json_formatter_emits_structured_fields() -> None:
    record = logging.makeLogRecord(
        {"name": "test", "levelname": "INFO", "msg": "dataset %s", "args": ("registered",)}
    )
    record.dataset_id = 7

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "dataset registered"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["dataset_id"] == 7
    assert "timestamp" in payload
