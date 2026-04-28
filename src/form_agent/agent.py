"""High-level facade tying together ingestion, extraction, storage, QA, and summary."""
from __future__ import annotations

import logging
from pathlib import Path

from .extraction import Extractor
from .ingestion import ingest as ingest_file
from .llm import LLMClient
from .multi_qa import MultiAnswer, MultiFormQA
from .qa import Answer, FormQA
from .store import FormStore, StoredForm, VectorStore
from .summarize import Summary, Summarizer

logger = logging.getLogger(__name__)


class FormAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.store = FormStore()
        self.vectors = VectorStore(llm=self.llm)
        self.extractor = Extractor(self.llm)
        self.qa = FormQA(self.llm)
        self.summarizer = Summarizer(self.llm)
        self.multi_qa = MultiFormQA(self.llm, self.store, self.vectors)

    def ingest(self, path: str | Path) -> StoredForm:
        doc = ingest_file(path)
        result = self.extractor.run(doc)
        stored = self.store.save(doc, result)
        try:
            self.vectors.index_form(stored)
        except Exception as e:  # pragma: no cover - vector backend issues shouldn't block ingestion
            logger.warning("Vector indexing failed for %s: %s", stored.id, e)
        return stored

    def list_forms(self) -> list[StoredForm]:
        return self.store.list_forms()

    def get_form(self, form_id: str) -> StoredForm | None:
        return self.store.get(form_id)

    def ask(self, form_id: str, question: str) -> Answer:
        form = self._require(form_id)
        return self.qa.ask(form, question)

    def summarize(self, form_id: str) -> Summary:
        form = self._require(form_id)
        return self.summarizer.summarize(form)

    def ask_all(self, question: str, top_k: int = 6) -> MultiAnswer:
        return self.multi_qa.ask(question, top_k=top_k)

    def _require(self, form_id: str) -> StoredForm:
        form = self.store.get(form_id)
        if form is None:
            raise KeyError(f"Unknown form id: {form_id}")
        return form
