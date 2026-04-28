"""Single-form structured summarization."""
from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from .config import CONFIG
from .llm import LLMClient
from .store import StoredForm


class Summary(BaseModel):
    tldr: list[str] = Field(default_factory=list, description="3 bullets")
    key_parties: list[str] = Field(default_factory=list)
    key_dates: list[str] = Field(default_factory=list)
    key_amounts: list[str] = Field(default_factory=list)
    obligations_or_actions: list[str] = Field(default_factory=list)
    risks_or_anomalies: list[str] = Field(default_factory=list)
    overall: str = ""


_SUM_SYS = (
    "You produce concise, faithful summaries of forms. Do not invent facts. "
    "If a section has no relevant content, return an empty list."
)

_SUM_USER_TMPL = (
    "Form type: {form_type}\n"
    "Extracted fields (JSON):\n{fields_json}\n\n"
    "Form text:\n```\n{text}\n```\n\n"
    "Return JSON with keys: tldr (list of exactly 3 short bullets), key_parties, key_dates, "
    "key_amounts, obligations_or_actions, risks_or_anomalies, overall (1-2 sentence narrative)."
)


class Summarizer:
    def __init__(self, llm: LLMClient, model: str | None = None) -> None:
        self.llm = llm
        self.model = model or CONFIG.summary_model

    def summarize(self, form: StoredForm) -> Summary:
        compact_fields = {k: v.get("value") for k, v in form.fields.items()}
        raw = self.llm.chat_json(
            [
                {"role": "system", "content": _SUM_SYS},
                {
                    "role": "user",
                    "content": _SUM_USER_TMPL.format(
                        form_type=form.form_type,
                        fields_json=json.dumps(compact_fields, indent=2, default=str),
                        text=_truncate(form.text, 14000),
                    ),
                },
            ],
            model=self.model,
        )
        try:
            return Summary(**raw)
        except ValidationError:
            return Summary(overall=str(raw))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."
