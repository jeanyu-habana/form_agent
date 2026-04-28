"""Single-form question answering with citations + confidence."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .llm import LLMClient
from .store import StoredForm


class Citation(BaseModel):
    field: str | None = None
    page: int | None = None
    snippet: str = ""


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0


_QA_SYS = (
    "You answer questions strictly using the provided form. If the form does not contain the answer, "
    'reply with "Not found in form" and confidence 0. Always cite either an extracted field name or a '
    "page number with a short verbatim snippet."
)

_QA_USER_TMPL = (
    "Form type: {form_type}\n"
    "Extracted fields (JSON):\n{fields_json}\n\n"
    "Form text:\n```\n{text}\n```\n\n"
    "Question: {question}\n\n"
    "Return JSON: {{\n"
    '  "answer": "...",\n'
    '  "citations": [{{"field": "<name or null>", "page": <int or null>, "snippet": "..."}}],\n'
    '  "confidence": 0.0-1.0\n'
    "}}"
)


class FormQA:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def ask(self, form: StoredForm, question: str) -> Answer:
        # Strip large snippets from extracted fields to keep prompt compact.
        compact_fields = {
            k: {"value": v.get("value"), "page": v.get("page"), "confidence": v.get("confidence")}
            for k, v in form.fields.items()
        }
        raw = self.llm.chat_json(
            [
                {"role": "system", "content": _QA_SYS},
                {
                    "role": "user",
                    "content": _QA_USER_TMPL.format(
                        form_type=form.form_type,
                        fields_json=json.dumps(compact_fields, indent=2, default=str),
                        text=_truncate(form.text, 14000),
                        question=question,
                    ),
                },
            ]
        )
        try:
            return Answer(**raw)
        except ValidationError:
            return Answer(answer=str(raw.get("answer", "")), confidence=0.0)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."
