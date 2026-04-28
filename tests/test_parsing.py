from pathlib import Path

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
