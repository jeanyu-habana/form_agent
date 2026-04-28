"""Thin OpenAI client wrapper. Single choke point for LLM access (mockable)."""
from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import CONFIG


class LLMClient:
    """Wraps chat + embeddings. Construct once, inject into components."""

    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.chat_model = chat_model or CONFIG.chat_model
        self.embed_model = embed_model or CONFIG.embed_model
        # Lazy import so tests can run without openai installed/configured.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or CONFIG.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(self, messages: Sequence[dict], *, model: str | None = None, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=model or self.chat_model,
            messages=list(messages),
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat_json(
        self,
        messages: Sequence[dict],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Chat completion forcing JSON object output."""
        resp = self._client.chat.completions.create(
            model=model or self.chat_model,
            messages=list(messages),
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]
