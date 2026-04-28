"""Minimal Streamlit UI for the Intelligent Form Agent."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from form_agent.agent import FormAgent

st.set_page_config(page_title="Intelligent Form Agent", layout="wide")


@st.cache_resource(show_spinner=False)
def get_agent() -> FormAgent:
    return FormAgent()


agent = get_agent()

st.title("Intelligent Form Agent")
st.caption("Extract, query, and summarize forms with LLM-powered understanding.")

# --- Sidebar: ingestion + form selection ---------------------------------
with st.sidebar:
    st.header("Forms")
    upload = st.file_uploader("Upload a form (PDF/TXT/PNG/JPG)",
                              type=["pdf", "txt", "png", "jpg", "jpeg"])
    if upload is not None and st.button("Ingest"):
        suffix = Path(upload.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(upload.read())
            tmp_path = tmp.name
        with st.spinner("Parsing + extracting..."):
            try:
                form = agent.ingest(tmp_path)
                st.success(f"Ingested as `{form.id}` ({form.form_type})")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    forms = agent.list_forms()
    options = {f"{f.id}  ({f.form_type})": f.id for f in forms}
    selected_label = st.selectbox("Select a form", list(options.keys()) or ["(none)"])
    selected_id = options.get(selected_label) if options else None

# --- Main area: tabs ----------------------------------------------------
tab_fields, tab_ask, tab_sum, tab_multi = st.tabs(
    ["Extracted Fields", "Ask (single)", "Summary", "Ask across all"]
)

with tab_fields:
    if selected_id:
        form = agent.get_form(selected_id)
        st.subheader(f"{form.form_type}  —  {Path(form.source_path).name}")
        st.caption(f"id: `{form.id}`  ·  pages: {len(form.pages)}  ·  ocr_used: {form.metadata.get('ocr_used')}")
        st.json({k: v for k, v in form.fields.items()})
        with st.expander("Raw text"):
            st.text(form.text[:8000])
    else:
        st.info("Upload and ingest a form to get started.")

with tab_ask:
    if selected_id:
        q = st.text_input("Question", key="ask_q",
                          placeholder="e.g. What is the candidate's most recent employer?")
        if q and st.button("Ask", key="ask_btn"):
            with st.spinner("Thinking..."):
                ans = agent.ask(selected_id, q)
            st.markdown(f"**Answer:** {ans.answer}")
            st.caption(f"confidence: {ans.confidence:.2f}")
            if ans.citations:
                st.markdown("**Citations**")
                for c in ans.citations:
                    st.markdown(f"- field=`{c.field}`  page=`{c.page}`  · _{c.snippet}_")
    else:
        st.info("Select a form first.")

with tab_sum:
    if selected_id and st.button("Summarize", key="sum_btn"):
        with st.spinner("Summarizing..."):
            s = agent.summarize(selected_id)
        st.markdown("### TL;DR")
        for b in s.tldr:
            st.markdown(f"- {b}")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Key parties**");  [st.markdown(f"- {x}") for x in s.key_parties]
            st.markdown("**Key dates**");    [st.markdown(f"- {x}") for x in s.key_dates]
            st.markdown("**Key amounts**");  [st.markdown(f"- {x}") for x in s.key_amounts]
        with cols[1]:
            st.markdown("**Obligations / actions**"); [st.markdown(f"- {x}") for x in s.obligations_or_actions]
            st.markdown("**Risks / anomalies**");     [st.markdown(f"- {x}") for x in s.risks_or_anomalies]
        if s.overall:
            st.markdown("### Overall")
            st.write(s.overall)

with tab_multi:
    st.caption(f"{len(forms)} form(s) ingested.")
    q = st.text_input("Question across all forms", key="multi_q",
                      placeholder="e.g. What is the total of all expense reports?")
    top_k = st.slider("Top-k chunks (RAG)", 2, 12, 6)
    if q and st.button("Ask all", key="multi_btn"):
        with st.spinner("Routing + answering..."):
            ans = agent.ask_all(q, top_k=top_k)
        st.markdown(f"**Strategy:** `{ans.strategy}`")
        st.markdown(f"**Answer:** {ans.answer}")
        st.caption(f"confidence: {ans.confidence:.2f}  ·  forms considered: {len(ans.forms_considered)}")
        if ans.citations:
            st.markdown("**Citations**")
            for c in ans.citations:
                st.markdown(f"- field=`{c.field}`  page=`{c.page}`  · _{c.snippet}_")
