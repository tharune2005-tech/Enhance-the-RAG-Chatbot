"""Streamlit RAG chatbot with runtime PDF/CSV ingest and session chat history."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FAQ_CSV, FAQ_DRIVE_URL, LLM_MODEL, google_api_key
from src.ingest import load_csv_path, load_file
from src.knowledge_base import KnowledgeBase, build_embeddings
from src.llm import build_llm
from src.rag_chain import ConversationalRAG

st.set_page_config(
    page_title="UnsaidTalks RAG Chatbot",
    page_icon="💬",
    layout="wide",
)


def _file_id(name: str, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"{name}:{len(data)}:{digest}"


@st.cache_resource(show_spinner="Loading embedding model (first run downloads MiniLM)...")
def _embeddings():
    return build_embeddings()


@st.cache_resource(show_spinner="Building the UnsaidTalks FAQ knowledge base...")
def _load_knowledge_base():
    kb = KnowledgeBase(embeddings=_embeddings())
    if kb.load_local() and kb.document_count > 0:
        return kb
    if not FAQ_CSV.exists():
        FAQ_CSV.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(FAQ_DRIVE_URL, FAQ_CSV)
    docs = load_csv_path(FAQ_CSV)
    kb.add_documents(docs, label="unsaidtalks_faq.csv")
    return kb


def _ensure_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_ids" not in st.session_state:
        st.session_state.ingested_ids = set()
    if "api_key" not in st.session_state:
        st.session_state.api_key = google_api_key() or ""


def _rag(kb: KnowledgeBase, api_key: str) -> ConversationalRAG:
    llm = build_llm(api_key=api_key)
    return ConversationalRAG(llm, kb.as_retriever())


def _reset_conversation() -> None:
    st.session_state.messages = []


def main() -> None:
    _ensure_session()
    kb = _load_knowledge_base()

    st.title("UnsaidTalks RAG Chatbot")
    st.caption(
        "Session RAG pipeline plus runtime PDF/CSV uploads and conversation memory. "
        "The knowledge base updates in place — no app restart."
    )

    with st.sidebar:
        st.header("Settings")
        st.session_state.api_key = st.text_input(
            "Google API key",
            value=st.session_state.api_key,
            type="password",
            help="Create a key at Google AI Studio. Stored only in this browser session.",
        )
        st.caption(f"Model: `{LLM_MODEL}` · Embeddings: MiniLM-L6-v2 · Store: FAISS")

        st.divider()
        st.header("Knowledge base")
        st.metric("Indexed chunks", kb.document_count)
        if kb.ingested:
            st.write("Indexed sources")
            for item in kb.ingested:
                st.caption(f"{item['label']} — {item['chunks']} chunks")

        uploads = st.file_uploader(
            "Upload PDF or CSV",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            help="Files are chunked, embedded, and merged into FAISS immediately.",
        )
        if uploads:
            for uploaded in uploads:
                data = uploaded.getvalue()
                fid = _file_id(uploaded.name, data)
                if fid in st.session_state.ingested_ids:
                    continue
                try:
                    docs = load_file(uploaded.name, data)
                    added = kb.add_documents(docs, label=uploaded.name)
                    st.session_state.ingested_ids.add(fid)
                    st.success(f"Indexed {uploaded.name} ({added} chunks). Ask about it now.")
                except Exception as exc:
                    st.error(f"{uploaded.name}: {exc}")

        st.divider()
        st.header("Conversation")
        st.write(f"{len(st.session_state.messages)} messages in this session")
        if st.button("Start new session", use_container_width=True):
            _reset_conversation()
            st.rerun()
        st.caption("New session clears chat history only. Uploaded documents stay indexed.")

    if not st.session_state.api_key:
        st.info("Add a Google API key in the sidebar to start chatting.")
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources") or []
            if sources:
                with st.expander("Retrieved sources"):
                    for src in sources:
                        st.markdown(f"**{src['label']}**")
                        st.caption(src["snippet"])

    prompt = st.chat_input("Ask about UnsaidTalks or an uploaded document")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            rag = _rag(kb, st.session_state.api_key)
            history = [m for m in st.session_state.messages[:-1] if m["role"] in {"user", "assistant"}]
            result = rag.ask(prompt, history)
            answer = result["answer"]
            sources = []
            for doc in result["source_documents"]:
                meta = doc.metadata
                label = meta.get("source", "unknown")
                if meta.get("page"):
                    label = f"{label} · page {meta['page']}"
                sources.append(
                    {
                        "label": label,
                        "snippet": doc.page_content[:400].replace("\n", " "),
                    }
                )
            st.markdown(answer)
            if sources:
                with st.expander("Retrieved sources"):
                    for src in sources:
                        st.markdown(f"**{src['label']}**")
                        st.caption(src["snippet"])
            if result["standalone_question"] != prompt:
                st.caption(f"Search query: {result['standalone_question']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as exc:
            err = f"Could not generate an answer: {exc}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})


if __name__ == "__main__":
    main()
