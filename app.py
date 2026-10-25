import os

import streamlit as st
from dotenv import load_dotenv

from rag_app.pipeline import build_rag_chain, load_uploaded_documents

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px;
       background: linear-gradient(120deg,#172554,#312e81);
       color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2.2rem;}
.hero p {opacity: .88; margin-bottom: 0;}
[data-testid="stChatMessage"] {border-radius: 12px;}
</style>
<div class="hero">
  <h1>📚 RAG Document Assistant</h1>
  <p>Ask questions about your documents. Answers include retrieved source references.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption("Your API key stays in your local .env file.")
    api_key_present = bool(os.getenv("OPENROUTER_API_KEY"))
    st.write("API key:", "✅ Found" if api_key_present else "⚠️ Not configured")
    model = st.text_input(
        "OpenRouter model",
        value=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    )
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Upload PDF/TXT files\n2. Build the index\n3. Ask questions\n4. Review sources")
    st.caption("First run may download the local embedding model.")

uploads = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="You can upload multiple PDF or TXT files.",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "indexed_names" not in st.session_state:
    st.session_state.indexed_names = []

col1, col2 = st.columns([1, 2])
with col1:
    build_clicked = st.button(
        "🔎 Build / Refresh document index", type="primary", use_container_width=True
    )
with col2:
    st.caption("Upload documents first. Building a new index replaces the current session index.")

if build_clicked:
    if not uploads:
        st.warning("Please upload at least one PDF or TXT file.")
    elif not api_key_present:
        st.error("Missing OPENROUTER_API_KEY. Copy .env.example to .env and add your key.")
    else:
        try:
            with st.spinner("Reading documents and building the vector index..."):
                docs = load_uploaded_documents(uploads)
                if not docs:
                    st.error("No readable text was found in the uploaded files.")
                else:
                    st.session_state.rag_chain = build_rag_chain(
                        docs, api_key=os.getenv("OPENROUTER_API_KEY"), model=model
                    )
                    st.session_state.indexed_names = [f.name for f in uploads]
                    st.session_state.messages = []
                    st.success(
                        f"Indexed {len(docs)} document page/section(s) from {len(uploads)} file(s)."
                    )
        except Exception as exc:
            st.error(f"Could not build the index: {exc}")
            st.info(
                "Check your API key, internet connection, package installation, "
                "and whether the PDFs contain selectable text."
            )

if st.session_state.indexed_names:
    st.success("Current session documents: " + ", ".join(st.session_state.indexed_names))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources used"):
                for i, source in enumerate(msg["sources"], start=1):
                    page_text = f" — page {source['page']}" if source.get("page") else ""
                    st.markdown(f"**Source {i}:** `{source['source']}`{page_text}")
                    st.caption(source["snippet"])

question = st.chat_input("Ask something about your uploaded documents...")
if question:
    if st.session_state.rag_chain is None:
        st.warning("Upload documents and build the index before asking questions.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and preparing an answer..."):
                try:
                    result = st.session_state.rag_chain.invoke(question)
                    answer = result.get(
                        "answer", "I couldn't generate an answer from the available context."
                    )
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("📎 Sources used", expanded=True):
                            for i, source in enumerate(sources, start=1):
                                page_text = f" — page {source['page']}" if source.get("page") else ""
                                st.markdown(f"**Source {i}:** `{source['source']}`{page_text}")
                                st.caption(source["snippet"])
                    st.session_state.messages.append({
                        "role": "assistant", "content": answer, "sources": sources
                    })
                except Exception as exc:
                    st.error(f"Answer generation failed: {exc}")
                    st.caption("Verify your OpenRouter model name and API key, then try again.")
