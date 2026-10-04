"""Map application exceptions to JSON error responses with a stable shape:

    {"error": {"code": "DATASET_NOT_READY", "message": "...", "details": {...},
               "request_id": "..."}}

Unexpected exceptions return a generic 500 without internals (logged server-side).
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import AppError
from app.core.logging import request_id_var

logger = logging.getLogger(__name__)


def _body(code: str, message: str, details: object | None = None) -> dict[str, object]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id_var.get(),
        }
    }


async def _app_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    logger.info(
        "request failed",
        extra={"error_code": exc.code, "status_code": exc.status_code, "path": request.url.path},
    )
    return JSONResponse(_body(exc.code, exc.message, exc.details), status_code=exc.status_code)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    # Never echo submitted values back (they may contain clinical text): locations only.
    errors = [
        {"loc": list(error.get("loc", ())), "msg": error.get("msg"), "type": error.get("type")}
        for error in exc.errors()
    ]
    return JSONResponse(
        _body("REQUEST_VALIDATION_ERROR", "Invalid request", {"errors": errors}), status_code=422
    )


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled error", extra={"error_type": type(exc).__name__, "path": request.url.path}
    )
    return JSONResponse(_body("INTERNAL_ERROR", "Internal server error"), status_code=500)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unexpected)
