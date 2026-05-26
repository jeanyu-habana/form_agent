# LangGraph Architecture Sketch

This document maps the current custom orchestration to a hypothetical LangGraph version.

## Current State (Custom Python)

```
FormAgent (facade)
  ├─ ingest() → parsing → extraction → storage → indexing
  ├─ ask() → fetch form → FormQA → answer
  ├─ summarize() → fetch form → Summarizer → summary
  └─ ask_all() → list forms → router → (aggregate | rag) → multi-answer
```

## Proposed LangGraph Graphs

### 1. **Ingest Graph** (Stateful DAG)

**Nodes:**
- `parse_file_node` — routes by extension, OCR fallback
  - Current module: `parsing.parse_file(path)` → `ParsedDocument`
  - Input: `{"path": str}`
  - Output: `{"doc": ParsedDocument, "source_path": str}`

- `infer_schema_node` — LLM call #1 discovers form type + field schema
  - Current module: `extraction.Extractor.infer_schema(doc)` → `(form_type, [FieldSpec])`
  - Input: `{"doc": ParsedDocument}`
  - Output: `{"form_type": str, "schema": [FieldSpec]}`

- `extract_fields_node` — LLM call #2 fills fields with citations + confidence
  - Current module: `extraction.Extractor.extract_fields(doc, schema)` → `dict[str, ExtractedField]`
  - Input: `{"doc": ParsedDocument, "schema": [FieldSpec]}`
  - Output: `{"fields": dict[str, ExtractedField]}`

- `save_to_store_node` — persists JSON representation
  - Current module: `store.FormStore.save(doc, result)` → `StoredForm`
  - Input: `{"doc": ParsedDocument, "result": ExtractionResult}`
  - Output: `{"stored_form": StoredForm}`

- `index_vectors_node` — indexes page chunks in Chroma
  - Current module: `store.VectorStore.index_form(form)` → `None` (side effect)
  - Input: `{"form": StoredForm}`
  - Output: `{"indexed": True}`

**Edges:**
```
parse_file_node 
  → infer_schema_node 
    → extract_fields_node 
      → save_to_store_node 
        → index_vectors_node (parallel OK, or sequential for ordering)
```

**State:**
```python
class IngestState(TypedDict):
    path: str
    doc: ParsedDocument
    form_type: str
    schema: list[FieldSpec]
    fields: dict[str, ExtractedField]
    stored_form: StoredForm
    indexed: bool
```

---

### 2. **Single-Form QA Graph**

**Nodes:**
- `fetch_form_node` — retrieves from `FormStore`
  - Current module: `store.FormStore.get(form_id)` → `StoredForm | None`
  - Input: `{"form_id": str}`
  - Output: `{"form": StoredForm}`

- `answer_question_node` — generates answer with citations
  - Current module: `qa.FormQA.ask(form, question)` → `Answer`
  - Input: `{"form": StoredForm, "question": str}`
  - Output: `{"answer": Answer}`

**Edges:**
```
fetch_form_node → answer_question_node
```

**State:**
```python
class QAState(TypedDict):
    form_id: str
    question: str
    form: StoredForm
    answer: Answer
```

---

### 3. **Single-Form Summary Graph**

**Nodes:**
- `fetch_form_node` — (reusable from QA graph)
  - Current module: `store.FormStore.get(form_id)`

- `summarize_form_node` — structured summary
  - Current module: `summarize.Summarizer.summarize(form)` → `Summary`
  - Input: `{"form": StoredForm}`
  - Output: `{"summary": Summary}`

**Edges:**
```
fetch_form_node → summarize_form_node
```

**State:**
```python
class SummaryState(TypedDict):
    form_id: str
    form: StoredForm
    summary: Summary
```

---

### 4. **Cross-Form QA Graph** (Most Complex — Conditional Branching)

**Nodes:**
- `list_forms_node` — enumerate all ingested forms
  - Current module: `store.FormStore.list_forms()` → `list[StoredForm]`
  - Input: `{}`
  - Output: `{"forms": list[StoredForm]}`

- `router_node` — LLM decides strategy
  - Current module: `multi_qa.MultiFormQA.route(question)` → `"rag" | "aggregate"`
  - Input: `{"question": str}`
  - Output: `{"strategy": Literal["rag", "aggregate"]}`

- `aggregate_node` — structured pass over fields
  - Current module: `multi_qa.MultiFormQA._aggregate(question, forms)` → `MultiAnswer`
  - Input: `{"question": str, "forms": list[StoredForm], "strategy": "aggregate"}`
  - Output: `{"answer": MultiAnswer}`

- `rag_node` — retrieve + synthesize from top-k chunks
  - Current module: `multi_qa.MultiFormQA._rag(question, forms, top_k)` → `MultiAnswer`
  - Internally calls `vectors.query(question, top_k)` for retrieval
  - Input: `{"question": str, "forms": list[StoredForm], "top_k": int, "strategy": "rag"}`
  - Output: `{"answer": MultiAnswer}`

**Edges (with Conditional Routing):**
```
list_forms_node 
  → router_node 
    → conditional_edge:
        if strategy == "aggregate": aggregate_node
        if strategy == "rag": rag_node
```

**Conditional Function:**
```python
def route_strategy(state: MultiQAState) -> str:
    return state["strategy"]  # "rag" or "aggregate"
```

**State:**
```python
class MultiQAState(TypedDict):
    question: str
    top_k: int
    forms: list[StoredForm]
    strategy: Literal["rag", "aggregate"]
    answer: MultiAnswer
```

---

## Full LangGraph Code Sketch

```python
from langgraph.graph import StateGraph, START, END
from typing import Literal

# Define the ingest subgraph
ingest_graph = StateGraph(IngestState)
ingest_graph.add_node("parse_file", parse_file_node)
ingest_graph.add_node("infer_schema", infer_schema_node)
ingest_graph.add_node("extract_fields", extract_fields_node)
ingest_graph.add_node("save_to_store", save_to_store_node)
ingest_graph.add_node("index_vectors", index_vectors_node)

ingest_graph.add_edge(START, "parse_file")
ingest_graph.add_edge("parse_file", "infer_schema")
ingest_graph.add_edge("infer_schema", "extract_fields")
ingest_graph.add_edge("extract_fields", "save_to_store")
ingest_graph.add_edge("save_to_store", "index_vectors")
ingest_graph.add_edge("index_vectors", END)

ingest_runnable = ingest_graph.compile()

# Define the QA graph
qa_graph = StateGraph(QAState)
qa_graph.add_node("fetch_form", fetch_form_node)
qa_graph.add_node("answer_question", answer_question_node)

qa_graph.add_edge(START, "fetch_form")
qa_graph.add_edge("fetch_form", "answer_question")
qa_graph.add_edge("answer_question", END)

qa_runnable = qa_graph.compile()

# Define the cross-form QA graph with conditional routing
multi_qa_graph = StateGraph(MultiQAState)
multi_qa_graph.add_node("list_forms", list_forms_node)
multi_qa_graph.add_node("router", router_node)
multi_qa_graph.add_node("aggregate", aggregate_node)
multi_qa_graph.add_node("rag", rag_node)

multi_qa_graph.add_edge(START, "list_forms")
multi_qa_graph.add_edge("list_forms", "router")
multi_qa_graph.add_conditional_edges(
    "router",
    route_strategy,
    {
        "aggregate": "aggregate",
        "rag": "rag",
    }
)
multi_qa_graph.add_edge("aggregate", END)
multi_qa_graph.add_edge("rag", END)

multi_qa_runnable = multi_qa_graph.compile()
```

---

## Benefits of LangGraph Conversion

| Aspect | Current | LangGraph |
|--------|---------|-----------|
| **Parallelism** | Manual thread/async mgmt | Native via `add_edge(..., concurrent=True)` |
| **Error recovery** | Ad-hoc try/catch | Built-in retries + fallback edges |
| **Observability** | Print statements | Native trace/step introspection |
| **Persistence** | Manual checkpoint logic | Built-in checkpointing (resume mid-graph) |
| **Testing** | Mock LLMClient | Mock nodes directly in graph context |
| **Human-in-the-loop** | Not implemented | Native `interrupt()` nodes |
| **Async/Streaming** | Manual | First-class support |
| **Tool calling** | Manual OpenAI format | LangGraph agent scaffolding |

---

## Migration Path (Incremental)

**Phase 1 — Ingest** (lowest risk)
- Replace `FormAgent.ingest()` with `ingest_runnable.invoke({"path": ...})`
- Keep single-form QA/summary as-is

**Phase 2 — Single-Form QA/Summary** (medium risk)
- Replace `FormAgent.ask()` with `qa_runnable.invoke(...)`
- Replace `FormAgent.summarize()` with `summary_runnable.invoke(...)`

**Phase 3 — Cross-Form QA** (highest risk; most benefit)
- Replace `FormAgent.ask_all()` with `multi_qa_runnable.invoke(...)`
- Enables conditional branching visualization

**Phase 4 — Advanced Features**
- Add human-in-the-loop nodes (e.g., "confirm extraction?" before saving)
- Add retry/fallback edges (e.g., if schema inference fails, try simpler prompt)
- Add parallel indexing or streaming output

---

## When NOT to Use LangGraph

- If the pipeline is fully linear with no conditionals → current approach is fine.
- If observability/persistence is not a priority → overhead may not justify it.
- If you need ultra-lightweight dependencies → LangGraph adds ~100MB.

## When to Use LangGraph

- ✅ You need **conditional branching** (already present: router in `ask_all`)
- ✅ You want **native observability** for production deployments
- ✅ You plan **human-in-the-loop** features (e.g., "confirm these extracted fields?")
- ✅ You want to scale to **100+ nodes** across multiple workflows
- ✅ You need **retry/fallback logic** (e.g., if LLM extraction fails, try again with simplified prompt)

---

## Current vs. LangGraph: Code Volume

**Current** (3-line FormAgent.ingest):
```python
def ingest(self, path: str | Path) -> StoredForm:
    doc = ingest_file(path)
    result = self.extractor.run(doc)
    stored = self.store.save(doc, result)
    try:
        self.vectors.index_form(stored)
    except Exception as e:
        logger.warning("Vector indexing failed for %s: %s", stored.id, e)
    return stored
```

**LangGraph** (15+ lines, but gains tracing + retry):
```python
async def ingest_workflow(path: str) -> StoredForm:
    result = await ingest_runnable.ainvoke({"path": path})
    return result["stored_form"]
    # Automatic tracing, step inspection, checkpointing, parallelism
```

---

## Recommendation

Keep current architecture as-is for now, but prepare for LangGraph if:
1. You add **human review** nodes (e.g., "approve extracted fields?")
2. You need production observability / audit trails
3. You scale to **multiple concurrent ingest/QA jobs**

If none of these apply, the current clean Python design is superior (lighter, faster, no magic).
