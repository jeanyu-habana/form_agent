from pathlib import Path
from unittest.mock import MagicMock, patch

from form_agent.parsing import parse_file


def test_parse_txt(tmp_path):
    p = tmp_path / "form.txt"
    p.write_text("Name: Alice\nAge: 30\n")
    doc = parse_file(p)
    assert "Alice" in doc.text
    assert len(doc.pages) == 1
    assert doc.ocr_used is False


def test_parse_unsupported(tmp_path):
    p = tmp_path / "form.xyz"
    p.write_text("nope")
    import pytest
    with pytest.raises(ValueError):
        parse_file(p)


def test_parse_missing(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        parse_file(tmp_path / "nope.pdf")


def test_document_intelligence_pdf(tmp_path, monkeypatch):
    """When DI endpoint is configured, _parse_with_document_intelligence is called."""
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://fake.cognitiveservices.azure.com/")

    from form_agent import config as cfg_mod
    cfg_mod.CONFIG = cfg_mod.Config.load()

    expected = MagicMock()
    expected.source_path = str(tmp_path / "form.pdf")
    expected.pages = ["Applicant: Alice\nHeader | Value"]
    expected.ocr_used = True
    expected.metadata = {"di_model": "prebuilt-layout"}

    pdf_path = tmp_path / "form.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("form_agent.parsing._parse_with_document_intelligence", return_value=expected) as mock_di:
        doc = parse_file(pdf_path)

    mock_di.assert_called_once()
    assert doc.ocr_used is True
    assert doc.metadata.get("di_model") == "prebuilt-layout"
    assert "Applicant" in doc.pages[0]

    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)
    cfg_mod.CONFIG = cfg_mod.Config.load()


def test_document_intelligence_fallback_on_error(tmp_path, monkeypatch):
    """If DI raises, parsing falls back to pypdf without crashing."""
    monkeypatch.setenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "https://fake.cognitiveservices.azure.com/")
    from form_agent import config as cfg_mod
    cfg_mod.CONFIG = cfg_mod.Config.load()

    # Write a minimal but valid single-page PDF that pypdf can handle.
    pdf_path = tmp_path / "form.pdf"
    pdf_path.write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj "
        b"3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f\n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )

    with patch("form_agent.parsing._parse_with_document_intelligence", side_effect=RuntimeError("DI failed")):
        doc = parse_file(pdf_path)

    # Should not raise; pypdf fallback produces a ParsedDocument.
    assert doc.source_path == str(pdf_path)
    assert isinstance(doc.pages, list)
    # Fallback should NOT set di_model metadata
    assert doc.metadata.get("di_model") is None

    monkeypatch.delenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", raising=False)
    cfg_mod.CONFIG = cfg_mod.Config.load()
