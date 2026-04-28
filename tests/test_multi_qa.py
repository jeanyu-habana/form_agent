"""Tests the router + aggregate path of MultiFormQA without touching Chroma."""
from unittest.mock import MagicMock

from form_agent.multi_qa import MultiFormQA
from form_agent.store import StoredForm


def _forms():
    return [
        StoredForm(
            id="exp-1", source_path="a.pdf", form_type="expense_report",
            pages=["Total: 100"], schema=[],
            fields={"total_amount": {"value": 100, "page": 1, "snippet": "Total: 100", "confidence": 0.9}},
        ),
        StoredForm(
            id="exp-2", source_path="b.pdf", form_type="expense_report",
            pages=["Total: 50"], schema=[],
            fields={"total_amount": {"value": 50, "page": 1, "snippet": "Total: 50", "confidence": 0.9}},
        ),
    ]


def test_router_aggregate(fake_llm):
    fake_llm.responses = [
        {"strategy": "aggregate", "reason": "totals"},
        {"answer": "150", "citations": [{"field": "exp-1:total_amount"}], "confidence": 0.99},
    ]
    store = MagicMock()
    store.list_forms.return_value = _forms()
    vectors = MagicMock()
    multi = MultiFormQA(fake_llm, store, vectors)
    ans = multi.ask("What is the total of all expense reports?")
    assert ans.strategy == "aggregate"
    assert "150" in ans.answer
    assert "exp-1" in ans.forms_considered
    vectors.query.assert_not_called()


def test_router_rag(fake_llm):
    fake_llm.responses = [
        {"strategy": "rag", "reason": "open question"},
        {"answer": "Alice has 9 years.", "citations": [{"field": "exp-1", "page": 1}], "confidence": 0.8},
    ]
    store = MagicMock()
    store.list_forms.return_value = _forms()
    vectors = MagicMock()
    vectors.query.return_value = [
        {"document": "Alice 9 years experience", "metadata": {"form_id": "exp-1", "page": 1}, "distance": 0.1},
    ]
    multi = MultiFormQA(fake_llm, store, vectors)
    ans = multi.ask("Who has the most experience?")
    assert ans.strategy == "rag"
    assert "Alice" in ans.answer
    vectors.query.assert_called_once()


def test_no_forms(fake_llm):
    store = MagicMock(); store.list_forms.return_value = []
    multi = MultiFormQA(fake_llm, store, MagicMock())
    ans = multi.ask("anything?")
    assert "No forms" in ans.answer
