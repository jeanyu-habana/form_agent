from form_agent.extraction import ExtractedField, ExtractionResult, FieldSpec
from form_agent.parsing import ParsedDocument
from form_agent.store import FormStore


def test_save_get_list_delete(tmp_store_dir):
    store = FormStore()
    doc = ParsedDocument(source_path="form.txt", pages=["Name: Alice"])
    result = ExtractionResult(
        form_type="t",
        schema=[FieldSpec(name="name", type="string")],
        fields={"name": ExtractedField(value="Alice", page=1, snippet="Name: Alice", confidence=0.9)},
    )
    saved = store.save(doc, result)
    assert saved.id

    fetched = store.get(saved.id)
    assert fetched is not None
    assert fetched.fields["name"]["value"] == "Alice"

    forms = store.list_forms()
    assert len(forms) == 1

    assert store.delete(saved.id) is True
    assert store.get(saved.id) is None
