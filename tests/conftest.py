"""Shared test fixtures: a fake LLMClient that returns canned JSON."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure src/ is importable when running tests directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class FakeLLM:
    """Stand-in for LLMClient. Pops responses from a queue or uses defaults."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses: list[Any] = list(responses or [])
        self.calls: list[dict] = []

    def chat_json(self, messages, *, model=None, temperature=0.0):  # noqa: D401
        self.calls.append({"kind": "chat_json", "messages": list(messages), "model": model})
        if self.responses:
            return self.responses.pop(0)
        return {}

    def chat(self, messages, *, model=None, temperature=0.0):
        self.calls.append({"kind": "chat", "messages": list(messages), "model": model})
        if self.responses:
            return self.responses.pop(0)
        return ""

    def embed(self, texts):
        texts = list(texts)
        # deterministic 4-d embeddings based on length, no external deps
        return [[float(len(t) % 7), 0.1, 0.2, 0.3] for t in texts]


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def tmp_store_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("FORM_AGENT_STORE_DIR", str(tmp_path / "store"))
    # Force config reload
    from form_agent import config as cfg_mod
    cfg_mod.CONFIG = cfg_mod.Config.load()
    return tmp_path / "store"
