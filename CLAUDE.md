# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common commands

```bash
# Install (editable) + system OCR deps (only needed for scanned PDFs/images)
pip install -r requirements.txt && pip install -e .
brew install tesseract poppler   # macOS

# Tests — fully mocked LLM, no live API or Tesseract required
pytest -q
pytest tests/test_qa.py::test_ask_returns_answer   # single test

# Generate synthetic sample forms (includes one scanned-PDF to exercise OCR)
python data/generate_samples.py

# CLI (installed by `pip install -e .`)
form-agent ingest data/samples/*.pdf
form-agent list
form-agent ask <form_id> "question..."
form-agent summarize <form_id>
form-agent ask-all "question..." [-k N]
form-agent show <form_id>     # dump stored JSON

# Optional UI
streamlit run ui/streamlit_app.py
```

## Architecture

The pipeline is **parse → extract → store → query**, with a single facade (`FormAgent` in `src/form_agent/agent.py`) wiring components together. All LLM access flows through one `LLMClient` (`llm.py`) so tests can inject a `FakeLLM` (see `tests/conftest.py`).

**Two-pass extraction is load-bearing** (`extraction.py`):
1. `infer_schema` — LLM call #1 invents a `FieldSpec` list per form (template-free; no hardcoded schemas).
2. `extract_fields` — LLM call #2 fills each field as `{value, page, snippet, confidence}`.

Downstream consumers (QA, summary, aggregate-router) read the structured fields rather than re-parsing raw text, which is what makes citations cheap and aggregations exact.

**Citations are a Pydantic contract**, not a convention. `qa.Answer`, `summarize.Summary`, and `multi_qa.MultiAnswer` validate the LLM JSON; on `ValidationError` the response is degraded to `confidence=0.0` rather than discarded. When adding a new LLM-backed response type, follow this same pattern.

**Storage is split** (`store.py`):
- `FormStore` — one JSON file per form under `<store_dir>/forms/<id>.json`. `id` is `<filename_stem>-<sha1[:12]>` of source path + first 512 chars.
- `VectorStore` — Chroma persistent client under `<store_dir>/chroma`, **one chunk per page** tagged with `form_id`, `page`, `form_type`. Cross-form retrieval filters via Chroma `where` clauses.

**Cross-form QA has a router** (`multi_qa.MultiFormQA`):
- `aggregate` strategy — compact `{id, form_type, fields}` JSON of every form is passed to the LLM. Use for counts/totals/comparisons across forms; computed from extracted values, not retrieved text.
- `rag` strategy — top-k Chroma page-chunks. Use for open-ended factual questions.
- The router itself is an LLM call (`_ROUTER_SYS` in `multi_qa.py`). When uncertain it falls back to `rag`.

**OCR fallback is per-page, not per-document** (`parsing._parse_pdf`): a PDF page yielding < `_OCR_FALLBACK_CHAR_THRESHOLD` (40) chars from `pypdf` is re-rendered via `pdf2image` and OCR'd via `pytesseract`. Hybrid PDFs (some digital, some scanned pages) work without re-OCR'ing everything.

## LLM provider: OpenAI vs Azure

`LLMClient` supports both, selected by `FORM_AGENT_LLM_PROVIDER` (`openai` | `azure`). Azure is **auto-detected** if `AZURE_OPENAI_ENDPOINT` is set and the provider isn't explicitly chosen (see `config.Config.load`). On Azure, the `model` argument is the **deployment name**, not a model name — set `AZURE_OPENAI_{CHAT,SUMMARY,EMBED}_DEPLOYMENT` instead of `FORM_AGENT_*_MODEL`. Both code paths use the same `chat`, `chat_json` (forces `response_format=json_object`), and `embed` methods.

## Config (env vars)

| Var | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI mode |
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` | triggers Azure auto-detect |
| `AZURE_OPENAI_{CHAT,SUMMARY,EMBED}_DEPLOYMENT` | Azure deployment names |
| `FORM_AGENT_CHAT_MODEL` / `FORM_AGENT_SUMMARY_MODEL` / `FORM_AGENT_EMBED_MODEL` | OpenAI model names |
| `FORM_AGENT_STORE_DIR` | Root for JSON + Chroma (default `.form_store`) |

`CONFIG` is loaded once at import time. Tests that need a fresh store directory must reload it (`tmp_store_dir` fixture in `conftest.py` shows the pattern: monkeypatch env then `cfg_mod.CONFIG = cfg_mod.Config.load()`).

## Testing notes

- `FakeLLM` (`tests/conftest.py`) pops canned responses from a queue. When testing a new flow, push responses in the same order the LLM is called (schema-infer → extract-fields → ...).
- Embeddings are stubbed deterministically (`[len(t) % 7, 0.1, 0.2, 0.3]`) — vector retrieval is structurally exercised but not semantically meaningful in tests.
- Vector indexing failures during `FormAgent.ingest` are swallowed with a warning so ingestion still succeeds if Chroma misbehaves (see `agent.py`). Don't tighten that without thinking about the test path.
