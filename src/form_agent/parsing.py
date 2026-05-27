"""File parsing: PDF (digital + scanned) and plain text. OCR fallback for scanned PDFs/images.

Azure AI Document Intelligence (prebuilt-layout) is used automatically when
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is set, producing higher-quality text with
preserved table structure and reading order. Falls back to pypdf + Tesseract when
the endpoint is not configured or DI raises an error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# If a PDF page yields fewer than this many extractable characters,
# we treat it as scanned and run OCR on that page.
_OCR_FALLBACK_CHAR_THRESHOLD = 40


@dataclass
class ParsedDocument:
    source_path: str
    pages: list[str]                    # text per page (1 page for plain text/images)
    ocr_used: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(
            f"[Page {i + 1}]\n{p}" for i, p in enumerate(self.pages) if p.strip()
        )


def parse_file(path: str | Path) -> ParsedDocument:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(p)
    if suffix in {".txt", ".md"}:
        return ParsedDocument(source_path=str(p), pages=[p.read_text(encoding="utf-8", errors="ignore")])
    if suffix in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return _parse_image(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(path: Path) -> ParsedDocument:
    from .config import CONFIG

    if CONFIG.document_intelligence_endpoint:
        try:
            return _parse_with_document_intelligence(path)
        except Exception as e:
            logger.warning(
                "Document Intelligence failed for %s (%s); falling back to pypdf+Tesseract", path.name, e
            )

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    needs_ocr_indices: list[int] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("pypdf extraction failed on page %d: %s", i + 1, e)
            txt = ""
        pages.append(txt)
        if len(txt.strip()) < _OCR_FALLBACK_CHAR_THRESHOLD:
            needs_ocr_indices.append(i)

    ocr_used = False
    if needs_ocr_indices:
        ocr_pages = _ocr_pdf_pages(path, needs_ocr_indices)
        for idx, ocr_text in zip(needs_ocr_indices, ocr_pages):
            if ocr_text and len(ocr_text.strip()) > len(pages[idx].strip()):
                pages[idx] = ocr_text
                ocr_used = True
        if ocr_used:
            logger.info("OCR fallback used for %s on pages %s", path.name, [i + 1 for i in needs_ocr_indices])

    return ParsedDocument(source_path=str(path), pages=pages, ocr_used=ocr_used)


def _parse_image(path: Path) -> ParsedDocument:
    from .config import CONFIG

    if CONFIG.document_intelligence_endpoint:
        try:
            return _parse_with_document_intelligence(path)
        except Exception as e:
            logger.warning(
                "Document Intelligence failed for %s (%s); falling back to Tesseract", path.name, e
            )

    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pytesseract/Pillow required for image input") from e
    text = pytesseract.image_to_string(Image.open(path))
    return ParsedDocument(source_path=str(path), pages=[text], ocr_used=True)


def _parse_with_document_intelligence(path: Path) -> ParsedDocument:
    """Use Azure AI Document Intelligence (prebuilt-layout) to extract page text.

    Reconstructs ``pages`` from DI's paragraph and table content, preserving
    reading order and rendering tables as plain-text grids. Sets
    ``metadata["di_model"]`` for observability.
    """
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
    from azure.core.credentials import AzureKeyCredential
    import os

    from .config import CONFIG

    endpoint = CONFIG.document_intelligence_endpoint  # already checked before calling
    credential = AzureKeyCredential(os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"])
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=credential)

    with open(path, "rb") as fh:
        file_bytes = fh.read()

    poller = client.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=file_bytes),
    )
    result = poller.result()

    pages_text: list[str] = []
    di_pages = result.pages or []

    # Build a set of paragraph spans that belong to tables so we skip them
    # when iterating paragraphs (tables are rendered separately).
    table_span_offsets: set[int] = set()
    for table in result.tables or []:
        for region in table.bounding_regions or []:
            pass  # bounding regions are by page; we use cell content directly
        for cell in table.cells or []:
            for span in cell.spans or []:
                table_span_offsets.add(span.offset)

    for page in di_pages:
        parts: list[str] = []

        # Add paragraphs that belong to this page via bounding_regions
        page_num = page.page_number
        for para in result.paragraphs or []:
            # Check if paragraph is on this page via bounding_regions
            para_pages = [r.page_number for r in (para.bounding_regions or [])]
            if para_pages and page_num not in para_pages:
                continue
            para_spans = para.spans or []
            if not para_spans:
                continue
            offset = para_spans[0].offset
            if offset in table_span_offsets:
                continue
            if para.content:
                parts.append(para.content)

        # Append tables that appear on this page as plain-text grids
        for table in result.tables or []:
            in_this_page = any(
                r.page_number == page.page_number for r in (table.bounding_regions or [])
            )
            if not in_this_page:
                continue
            row_count = table.row_count or 0
            col_count = table.column_count or 0
            grid: list[list[str]] = [[""] * col_count for _ in range(row_count)]
            for cell in table.cells or []:
                r, c = cell.row_index or 0, cell.column_index or 0
                if r < row_count and c < col_count:
                    grid[r][c] = (cell.content or "").replace("\n", " ")
            parts.append("\n".join(" | ".join(row) for row in grid))

        pages_text.append("\n".join(parts).strip())

    # If DI returned no page objects (e.g. single-page image), treat content as one page
    if not pages_text and result.content:
        pages_text = [result.content]

    logger.info("Document Intelligence parsed %s: %d page(s)", path.name, len(pages_text))
    return ParsedDocument(
        source_path=str(path),
        pages=pages_text,
        ocr_used=True,  # DI always handles scanned content transparently
        metadata={"di_model": "prebuilt-layout"},
    )


def _ocr_pdf_pages(path: Path, indices: list[int]) -> list[str]:
    """Render given page indices to images and OCR them. Returns text per requested index."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:  # pragma: no cover
        logger.warning("pdf2image/pytesseract not available; skipping OCR for %s", path)
        return ["" for _ in indices]

    out: list[str] = []
    for idx in indices:
        try:
            images = convert_from_path(str(path), first_page=idx + 1, last_page=idx + 1, dpi=200)
            if not images:
                out.append("")
                continue
            text = pytesseract.image_to_string(images[0])
            out.append(text)
        except Exception as e:
            logger.warning("OCR failed on %s page %d: %s", path.name, idx + 1, e)
            out.append("")
    return out
