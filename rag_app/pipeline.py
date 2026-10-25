from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_uploaded_documents(uploaded_files: Iterable) -> list[Document]:
    """Load PDF and TXT uploads into LangChain Documents."""
    documents: list[Document] = []
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded.getvalue())
            temp_path = temp.name

        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(temp_path).load()
            else:
                loaded = TextLoader(
                    temp_path, encoding="utf-8", autodetect_encoding=True
                ).load()

            for doc in loaded:
                doc.metadata["source"] = uploaded.name
                if "page" in doc.metadata:
                    # PyPDFLoader uses zero-based page numbers.
                    doc.metadata["page"] = int(doc.metadata["page"]) + 1
            documents.extend(loaded)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    return documents


def build_rag_chain(documents: list[Document], api_key: str, model: str | None = None):
    """Create an in-memory Chroma index and an invoke-compatible RAG function."""
    if not documents:
        raise ValueError("No documents were provided.")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("Documents contain no extractable text.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_document_assistant",
    )
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model=model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a careful document question-answering assistant.
Answer using only the supplied context. If the context does not contain the answer,
say that you could not find enough information in the uploaded documents.
Do not invent facts or citations. Keep answers clear and concise.

Context:
{context}"""),
        ("human", "{question}"),
    ])

    def answer(question: str) -> dict:
        retrieved = retriever.invoke(question)
        if not retrieved:
            return {
                "answer": "I couldn't find relevant information in the uploaded documents.",
                "sources": [],
            }

        context_parts = []
        for index, doc in enumerate(retrieved, start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page")
            page_text = f", page {page}" if page else ""
            context_parts.append(
                f"[Source {index}: {source}{page_text}]\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)
        response = (prompt | llm).invoke(
            {"context": context, "question": question}
        )

        sources = []
        seen = set()
        for doc in retrieved:
            source = str(doc.metadata.get("source", "Unknown source"))
            page = doc.metadata.get("page")
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            snippet = " ".join(doc.page_content.split())
            if len(snippet) > 350:
                snippet = snippet[:347] + "..."
            sources.append({
                "source": source,
                "page": page,
                "snippet": snippet,
            })

        return {"answer": response.content, "sources": sources}

    return _RAGRunnable(answer)


class _RAGRunnable:
    """Small invoke-compatible wrapper used by the Streamlit UI."""
    def __init__(self, function):
        self.function = function

    def invoke(self, question: str) -> dict:
        return self.function(question)
