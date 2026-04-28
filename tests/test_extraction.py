from form_agent.extraction import Extractor, FieldSpec
from form_agent.parsing import ParsedDocument


def test_infer_schema_and_extract(fake_llm):
    fake_llm.responses = [
        {
            "form_type": "job_application",
            "schema": [
                {"name": "applicant_name", "type": "string", "description": "full name"},
                {"name": "years_experience", "type": "number", "description": ""},
                {"name": "junk", "type": "weirdtype"},  # malformed: missing description ok, type still str
            ],
        },
        {
            "fields": {
                "applicant_name": {"value": "Alice", "page": 1, "snippet": "Name: Alice", "confidence": 0.95},
                "years_experience": {"value": 9, "page": 1, "snippet": "9 years", "confidence": 0.8},
                # 'junk' missing on purpose - extractor must default it
            }
        },
    ]
    extractor = Extractor(fake_llm)
    doc = ParsedDocument(source_path="x.txt", pages=["Name: Alice\nExperience: 9 years"])
    result = extractor.run(doc)
    assert result.form_type == "job_application"
    assert any(s.name == "applicant_name" for s in result.schema_)
    assert result.fields["applicant_name"].value == "Alice"
    assert result.fields["applicant_name"].confidence > 0.9
    assert "junk" in result.fields
    assert result.fields["junk"].confidence == 0.0


def test_infer_schema_drops_invalid(fake_llm):
    fake_llm.responses = [{"form_type": "x", "schema": [{"no_name": "bad"}]}]
    extractor = Extractor(fake_llm)
    doc = ParsedDocument(source_path="x.txt", pages=["hi"])
    form_type, schema = extractor.infer_schema(doc)
    assert form_type == "x"
    assert schema == []
