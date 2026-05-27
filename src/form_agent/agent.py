"""High-level facade tying together ingestion, extraction, storage, QA, and summary."""
from __future__ import annotations

import logging
from pathlib import Path

from opentelemetry import trace

from .extraction import Extractor
from .ingestion import ingest as ingest_file
from .llm import LLMClient
from .multi_qa import MultiAnswer, MultiFormQA
from .qa import Answer, FormQA
from .store import BlobStore, FormStore, StoredForm, VectorStore
from .summarize import Summary, Summarizer
from .telemetry import setup_telemetry

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


class FormAgent:
    def __init__(self, llm: LLMClient | None = None) -> None:
        setup_telemetry()
        self.llm = llm or LLMClient()
        self.store = FormStore()
        self.vectors = VectorStore(llm=self.llm)
        self.extractor = Extractor(self.llm)
        self.qa = FormQA(self.llm)
        self.summarizer = Summarizer(self.llm)
        self.multi_qa = MultiFormQA(self.llm, self.store, self.vectors)
        self._blobs = BlobStore()

    def ingest(self, path: str | Path) -> StoredForm:
        with _tracer.start_as_current_span("form_agent.ingest") as span:
            span.set_attribute("ingest.path", str(path))
            doc = ingest_file(path)
            span.set_attribute("ingest.pages", len(doc.pages))
            span.set_attribute("ingest.ocr_used", doc.ocr_used)
            result = self.extractor.run(doc)
            stored = self.store.save(doc, result)
            span.set_attribute("ingest.form_id", stored.id)
            span.set_attribute("ingest.form_type", stored.form_type)
            try:
                self.vectors.index_form(stored)
            except Exception as e:
                logger.warning("Vector indexing failed for %s: %s", stored.id, e)
            self._blobs.upload_form(stored)
        return stored

    def list_forms(self) -> list[StoredForm]:
        with _tracer.start_as_current_span("form_agent.list_forms") as span:
            forms = self.store.list_forms()
            span.set_attribute("list.count", len(forms))
            return forms

    def get_form(self, form_id: str) -> StoredForm | None:
        with _tracer.start_as_current_span("form_agent.get_form") as span:
            span.set_attribute("form.id", form_id)
            form = self.store.get(form_id)
            span.set_attribute("form.found", form is not None)
            return form

    def ask(self, form_id: str, question: str) -> Answer:
        with _tracer.start_as_current_span("form_agent.ask") as span:
            span.set_attribute("ask.form_id", form_id)
            span.set_attribute("ask.question", question)
            try:
                form = self._require(form_id)
                span.set_attribute("ask.form_type", form.form_type)
                answer = self.qa.ask(form, question)
                span.set_attribute("ask.confidence", answer.confidence)
                span.set_attribute("ask.citation_count", len(answer.citations))
                logger.info(
                    "ask form_id=%s confidence=%.2f citations=%d",
                    form_id, answer.confidence, len(answer.citations),
                )
                return answer
            except Exception as e:
                span.set_attribute("ask.error", str(e))
                span.set_status(trace.StatusCode.ERROR, str(e))
                logger.error("ask failed form_id=%s error=%s", form_id, e)
                raise

    def summarize(self, form_id: str) -> Summary:
        with _tracer.start_as_current_span("form_agent.summarize") as span:
            span.set_attribute("summarize.form_id", form_id)
            form = self._require(form_id)
            span.set_attribute("summarize.form_type", form.form_type)
            summary = self.summarizer.summarize(form)
            span.set_attribute("summarize.risk_count", len(summary.risks_and_anomalies))
            span.set_attribute("summarize.obligation_count", len(summary.obligations_and_actions))
            logger.info(
                "summarize form_id=%s form_type=%s",
                form_id, form.form_type,
            )
            return summary

    def ask_all(self, question: str, top_k: int = 6) -> MultiAnswer:
        with _tracer.start_as_current_span("form_agent.ask_all") as span:
            span.set_attribute("ask_all.question", question)
            span.set_attribute("ask_all.top_k", top_k)
            answer = self.multi_qa.ask(question, top_k=top_k)
            span.set_attribute("ask_all.strategy", answer.strategy)
            span.set_attribute("ask_all.confidence", answer.confidence)
            span.set_attribute("ask_all.forms_considered", answer.forms_considered)
            logger.info(
                "ask_all strategy=%s confidence=%.2f forms=%d",
                answer.strategy, answer.confidence, answer.forms_considered,
            )
            return answer

    def _require(self, form_id: str) -> StoredForm:
        form = self.store.get(form_id)
        if form is None:
            raise KeyError(f"Unknown form id: {form_id}")
        return form
