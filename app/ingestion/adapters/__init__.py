"""Source adapter registry and selection."""

from pathlib import Path

from app.core.constants import SourceType
from app.ingestion.adapters.base import (
    AdapterError,
    SourceAdapter,
    SourceManifest,
    SourceRestrictedError,
)
from app.ingestion.adapters.delimited import CsvAdapter, TsvAdapter
from app.ingestion.adapters.json_adapter import JsonAdapter
from app.ingestion.adapters.layout_adapters import OcrPdfAdapter, PdfAdapter, TextAdapter
from app.ingestion.adapters.xlsx import XlsxAdapter
from app.ingestion.adapters.xml_adapter import XmlAdapter

ADAPTERS: dict[SourceType, type[SourceAdapter]] = {
    SourceType.PDF: PdfAdapter,
    SourceType.PDF_OCR: OcrPdfAdapter,
    SourceType.TEXT: TextAdapter,
    SourceType.CSV: CsvAdapter,
    SourceType.TSV: TsvAdapter,
    SourceType.JSON: JsonAdapter,
    SourceType.XML: XmlAdapter,
    SourceType.XLSX: XlsxAdapter,
}


def select_adapter(path: Path, manifest: SourceManifest | None = None) -> SourceAdapter:
    """The manifest's explicit adapter wins; otherwise the most confident detector.

    OCR is never auto-selected: it must be requested with `"adapter": "pdf_ocr"`.
    """
    manifest = manifest or SourceManifest()
    if not Path(path).is_file():
        raise AdapterError(f"Source file not found: {Path(path).name}")
    if manifest.adapter is not None:
        return ADAPTERS[manifest.adapter](path, manifest)
    scored = sorted(
        ((cls.detect(Path(path)), cls) for cls in ADAPTERS.values()),
        key=lambda pair: pair[0],
        reverse=True,
    )
    confidence, adapter_cls = scored[0]
    if confidence <= 0:
        raise AdapterError(
            f'No adapter recognises {Path(path).name}; set "adapter" in the manifest.'
        )
    return adapter_cls(path, manifest)


__all__ = [
    "ADAPTERS",
    "AdapterError",
    "CsvAdapter",
    "JsonAdapter",
    "OcrPdfAdapter",
    "PdfAdapter",
    "SourceAdapter",
    "SourceManifest",
    "SourceRestrictedError",
    "TextAdapter",
    "TsvAdapter",
    "XlsxAdapter",
    "XmlAdapter",
    "select_adapter",
]
