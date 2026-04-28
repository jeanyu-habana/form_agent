"""File parsing: PDF (digital + scanned) and plain text. OCR fallback for scanned PDFs/images."""
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


def _parse_image(path: Path) -> ParsedDocument:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pytesseract/Pillow required for image input") from e
    text = pytesseract.image_to_string(Image.open(path))
    return ParsedDocument(source_path=str(path), pages=[text], ocr_used=True)
