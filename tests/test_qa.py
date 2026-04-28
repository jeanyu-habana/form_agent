from form_agent.qa import FormQA
from form_agent.store import StoredForm


def _form() -> StoredForm:
    return StoredForm(
        id="f-1",
        source_path="x.pdf",
        form_type="job_application",
        pages=["Name: Alice\nExperience: 9 years"],
        schema=[{"name": "applicant_name", "type": "string", "description": ""}],
        fields={"applicant_name": {"value": "Alice", "page": 1, "snippet": "Name: Alice", "confidence": 0.9}},
    )


def test_ask_returns_answer_with_citation(fake_llm):
    fake_llm.responses = [{
        "answer": "Alice",
        "citations": [{"field": "applicant_name", "page": 1, "snippet": "Name: Alice"}],
        "confidence": 0.92,
    }]
    qa = FormQA(fake_llm)
    ans = qa.ask(_form(), "What is the applicant's name?")
    assert ans.answer == "Alice"
    assert ans.confidence > 0.9
    assert ans.citations[0].field == "applicant_name"


def test_ask_handles_invalid_payload(fake_llm):
    fake_llm.responses = [{"answer": "fallback"}]  # missing citations -> ok (defaults), no validation error
    qa = FormQA(fake_llm)
    ans = qa.ask(_form(), "anything?")
    assert ans.answer == "fallback"
    assert ans.confidence == 0.0
