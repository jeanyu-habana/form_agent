"""Cross-form QA: routes between RAG synthesis and structured aggregation over stored fields."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .llm import LLMClient
from .qa import Citation
from .store import FormStore, StoredForm, VectorStore


class MultiAnswer(BaseModel):
    answer: str
    strategy: Literal["rag", "aggregate"] = "rag"
    citations: list[Citation] = Field(default_factory=list)
    forms_considered: list[str] = Field(default_factory=list)
    confidence: float = 0.0


_ROUTER_SYS = (
    "You decide how to answer a question over a collection of forms. "
    'Return JSON: {"strategy": "rag" | "aggregate", "reason": "..."}.\n'
    '- Use "aggregate" when the question asks for counts, totals, averages, comparisons, '
    "or filtering across forms based on structured fields.\n"
    '- Use "rag" for open-ended/factual questions that require reading specific form content.'
)


_AGG_SYS = (
    "You answer aggregation questions over a JSON list of forms (each with extracted fields). "
    "Compute the answer purely from the provided data. Cite the form ids you used. "
    "Never invent values. If data is insufficient, say so."
)

_AGG_USER_TMPL = (
    "Forms (JSON list of {{id, form_type, fields}}):\n{forms_json}\n\n"
    "Question: {question}\n\n"
    "Return JSON: {{\n"
    '  "answer": "...",\n'
    '  "citations": [{{"field": "<form_id:field_name or null>", "page": null, "snippet": "..."}}],\n'
    '  "confidence": 0.0-1.0\n'
    "}}"
)


_RAG_SYS = (
    "You answer questions using ONLY the provided excerpts from multiple forms. "
    "Cite the form id and page for each fact. If insufficient, say so."
)

_RAG_USER_TMPL = (
    "Excerpts (each tagged with form_id and page):\n{context}\n\n"
    "Question: {question}\n\n"
    "Return JSON: {{\n"
    '  "answer": "...",\n'
    '  "citations": [{{"field": "<form_id or null>", "page": <int or null>, "snippet": "..."}}],\n'
    '  "confidence": 0.0-1.0\n'
    "}}"
)


class MultiFormQA:
    def __init__(self, llm: LLMClient, store: FormStore, vectors: VectorStore) -> None:
        self.llm = llm
        self.store = store
        self.vectors = vectors

    def route(self, question: str) -> str:
        try:
            raw = self.llm.chat_json(
                [
                    {"role": "system", "content": _ROUTER_SYS},
                    {"role": "user", "content": f"Question: {question}"},
                ]
            )
            strategy = str(raw.get("strategy", "rag")).lower()
            return strategy if strategy in {"rag", "aggregate"} else "rag"
        except Exception:
            return "rag"

    def ask(self, question: str, top_k: int = 6) -> MultiAnswer:
        forms = self.store.list_forms()
        if not forms:
            return MultiAnswer(answer="No forms ingested.", strategy="rag", confidence=0.0)
        strategy = self.route(question)
        if strategy == "aggregate":
            return self._aggregate(question, forms)
        return self._rag(question, forms, top_k=top_k)

    def _aggregate(self, question: str, forms: list[StoredForm]) -> MultiAnswer:
        compact = [
            {
                "id": f.id,
                "form_type": f.form_type,
                "fields": {k: v.get("value") for k, v in f.fields.items()},
            }
            for f in forms
        ]
        raw = self.llm.chat_json(
            [
                {"role": "system", "content": _AGG_SYS},
                {
                    "role": "user",
                    "content": _AGG_USER_TMPL.format(
                        forms_json=json.dumps(compact, indent=2, default=str),
                        question=question,
                    ),
                },
            ]
        )
        try:
            ans = MultiAnswer(
                answer=raw.get("answer", ""),
                strategy="aggregate",
                citations=[Citation(**c) for c in raw.get("citations", []) if isinstance(c, dict)],
                forms_considered=[f.id for f in forms],
                confidence=float(raw.get("confidence", 0.0)),
            )
            return ans
        except (ValidationError, ValueError):
            return MultiAnswer(answer=str(raw), strategy="aggregate",
                               forms_considered=[f.id for f in forms])

    def _rag(self, question: str, forms: list[StoredForm], top_k: int) -> MultiAnswer:
        hits = self.vectors.query(question, top_k=top_k)
        if not hits:
            return MultiAnswer(answer="No relevant excerpts found.", strategy="rag",
                               forms_considered=[f.id for f in forms])
        context_parts = []
        used_form_ids: set[str] = set()
        for h in hits:
            meta = h.get("metadata") or {}
            fid = meta.get("form_id", "?")
            page = meta.get("page", "?")
            used_form_ids.add(fid)
            doc = (h.get("document") or "")[:1200]
            context_parts.append(f"[form_id={fid} page={page}]\n{doc}")
        context = "\n\n---\n\n".join(context_parts)
        raw = self.llm.chat_json(
            [
                {"role": "system", "content": _RAG_SYS},
                {
                    "role": "user",
                    "content": _RAG_USER_TMPL.format(context=context, question=question),
                },
            ]
        )
        try:
            return MultiAnswer(
                answer=raw.get("answer", ""),
                strategy="rag",
                citations=[Citation(**c) for c in raw.get("citations", []) if isinstance(c, dict)],
                forms_considered=sorted(used_form_ids),
                confidence=float(raw.get("confidence", 0.0)),
            )
        except (ValidationError, ValueError):
            return MultiAnswer(answer=str(raw), strategy="rag",
                               forms_considered=sorted(used_form_ids))
