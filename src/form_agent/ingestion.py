"""Convenience wrapper for ingestion (currently delegates to parsing.parse_file)."""
from __future__ import annotations

from pathlib import Path

from .parsing import ParsedDocument, parse_file


def ingest(path: str | Path) -> ParsedDocument:
    return parse_file(path)
