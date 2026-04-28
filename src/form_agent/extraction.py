"""LLM-driven schema inference + field extraction with citations and confidence."""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .llm import LLMClient
from .parsing import ParsedDocument

logger = logging.getLogger(__name__)


class FieldSpec(BaseModel):
    name: str = Field(description="snake_case field name")
    type: str = Field(description="one of: string, number, date, boolean, list, paragraph")
    description: str = ""


class ExtractedField(BaseModel):
    value: Any = None
    page: int | None = None
    snippet: str = ""
    confidence: float = 0.0


class ExtractionResult(BaseModel):
    form_type: str = "unknown"
    schema_: list[FieldSpec] = Field(default_factory=list, alias="schema")
    fields: dict[str, ExtractedField] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


_SCHEMA_SYS = (
    "You are an expert at analyzing forms. Given the text of a form, identify a concise schema of "
    "the most useful fields a downstream system should extract. Prefer 6-20 fields. Use snake_case names. "
    "Cover both structured fields (e.g. name, date_of_birth, total_amount) and 1-2 free-text/unstructured "
    "fields (e.g. notes, description, justification) when present."
)

_SCHEMA_USER_TMPL = (
    "Form text follows between triple backticks. Return JSON with keys:\n"
    '  "form_type": short label like "job_application" or "medical_intake"\n'
    '  "schema": list of {{"name","type","description"}} where type is one of '
    "string|number|date|boolean|list|paragraph.\n"
    "Form text:\n```\n{text}\n```"
)

_EXTRACT_SYS = (
    "You extract structured data from form text. For each requested field, return the value as it appears "
    "in the form (or null if absent), the page number it came from, a short verbatim snippet "
    "(<=200 chars) supporting the value, and a confidence in [0,1]. Be conservative: if uncertain, "
    "lower confidence. Never invent values."
)

_EXTRACT_USER_TMPL = (
    "Schema (JSON):\n{schema_json}\n\n"
    "Form text (page markers like [Page N] are included):\n```\n{text}\n```\n\n"
    'Return JSON: {{"fields": {{"<field_name>": {{"value": ..., "page": <int|null>, '
    '"snippet": "...", "confidence": 0.0-1.0}}, ...}} }}'
)


class Extractor:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def infer_schema(self, doc: ParsedDocument) -> tuple[str, list[FieldSpec]]:
        text = _truncate(doc.text, 12000)
        raw = self.llm.chat_json(
            [
                {"role": "system", "content": _SCHEMA_SYS},
                {"role": "user", "content": _SCHEMA_USER_TMPL.format(text=text)},
            ]
        )
        form_type = str(raw.get("form_type", "unknown"))
        schema_raw = raw.get("schema", [])
        schema: list[FieldSpec] = []
        for item in schema_raw:
            try:
                schema.append(FieldSpec(**item))
            except ValidationError as e:
                logger.warning("Invalid field spec dropped: %s (%s)", item, e)
        return form_type, schema

    def extract_fields(
        self, doc: ParsedDocument, schema: list[FieldSpec]
    ) -> dict[str, ExtractedField]:
        if not schema:
            return {}
        text = _truncate(doc.text, 16000)
        schema_json = [s.model_dump() for s in schema]
        raw = self.llm.chat_json(
            [
                {"role": "system", "content": _EXTRACT_SYS},
                {
                    "role": "user",
                    "content": _EXTRACT_USER_TMPL.format(schema_json=schema_json, text=text),
                },
            ]
        )
        out: dict[str, ExtractedField] = {}
        fields_raw = raw.get("fields", {}) or {}
        for spec in schema:
            item = fields_raw.get(spec.name) or {}
            try:
                out[spec.name] = ExtractedField(**item)
            except ValidationError:
                out[spec.name] = ExtractedField(value=item.get("value"), confidence=0.0)
        return out

    def run(self, doc: ParsedDocument) -> ExtractionResult:
        form_type, schema = self.infer_schema(doc)
        fields = self.extract_fields(doc, schema)
        return ExtractionResult(form_type=form_type, schema=schema, fields=fields)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2:]
    return f"{head}\n...[truncated]...\n{tail}"
