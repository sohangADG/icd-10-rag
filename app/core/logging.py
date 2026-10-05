import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

# Attributes present on every LogRecord; anything else came from `extra=` and becomes a field.
_STANDARD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}

# Correlation id of the HTTP request being served (set by RequestContextMiddleware).
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """Adds `request_id` to every record emitted while a request is being handled."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = request_id_var.get()
        if request_id is not None and not hasattr(record, "request_id"):
            record.request_id = request_id
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in vars(record).items() if key not in _STANDARD_ATTRS}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {key: value for key, value in vars(record).items() if key not in _STANDARD_ATTRS}
        return f"{line} {extras}" if extras else line


def configure_logging(level: str, fmt: str, stream: TextIO | None = None) -> None:
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    handler.addFilter(RequestIdFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Uvicorn installs its own handlers; route them through ours for a single consistent format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # Access logs include full URLs (query strings may carry search text): our own latency
    # middleware logs method/route/status instead.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
