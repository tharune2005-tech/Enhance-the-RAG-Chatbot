#!/usr/bin/env python3
"""Generate notebooks/RAG_Unsaidtalks_Enhanced.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [],
}


def md(text: str) -> None:
    nb["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip("\n").split("\n")],
        }
    )
    nb["cells"][-1]["source"][-1] = nb["cells"][-1]["source"][-1].rstrip("\n")


def code(text: str) -> None:
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines]
    if source:
        source[-1] = source[-1].rstrip("\n")
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source,
        }
    )


md(
    """# UnsaidTalks RAG Chatbot — Optional Activity

This notebook extends the [live-session RAG chatbot](https://colab.research.google.com/drive/1qSSBFI9J83u_Ij1vSwvCeKfpX0j8WbYB?usp=sharing).

The session built a **static** pipeline: load the UnsaidTalks FAQ CSV → MiniLM embeddings → FAISS → `RetrievalQA`.

| Task | What you will add |
| --- | --- |
| **1. Runtime knowledge-base updates** | Upload **PDF** and **CSV** files, embed them, and merge into FAISS **without restarting** |
| **2. Chat history** | Keep the full conversation and use it for follow-up questions. Reset only on a **new session** |

A Streamlit app with the same behaviour lives at `app.py` in this repository."""
)

md("## 1. Install dependencies")

code(
    r"""%pip install -q langchain langchain-core langchain-community langchain-google-genai \
    langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers \
    pandas pypdf python-dotenv"""
)

md("## 2. API key and Gemini (same as the session)")

code(
    r"""import os
from pathlib import Path

try:
    from google.colab import userdata

    os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    IN_COLAB = True
except Exception:
    IN_COLAB = False
    try:
        from dotenv import load_dotenv

        for candidate in (Path.cwd() / ".env", Path.cwd().parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                break
    except ImportError:
        pass

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Set GOOGLE_API_KEY (Colab userdata, .env, or os.environ).")

from langchain_google_genai import ChatGoogleGenerativeAI

MODEL_NAME = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0.1)
print(f"LLM ready: {MODEL_NAME}")"""
)

md("## 3. Load the UnsaidTalks FAQ (session code)")

code(
    r"""from langchain_core.documents import Document
import pandas as pd

file_id = "1h1jHbpjZ_OwvP7H-5K39Ot8OXIwEGhJx"
direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"

local_faq = Path("data/unsaidtalks_faq.csv")
if not local_faq.exists():
    local_faq = Path("../data/unsaidtalks_faq.csv")

if local_faq.exists():
    df = pd.read_csv(local_faq)
else:
    df = pd.read_csv(direct_url)

print(df.columns.tolist())
print(df.head(2))

documents = []
for _, row in df.iterrows():
    documents.append(
        Document(
            page_content=f"Question: {row['prompt']}\nAnswer: {row['response']}",
            metadata={"source": str(row["prompt"]), "type": "faq"},
        )
    )
print(f"Loaded {len(documents)} FAQ documents")"""
)

md("## 4. Embeddings + FAISS vector store")

code(
    r"""from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

instructor_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb_file_path = "faiss_index"
vectordb = FAISS.from_documents(documents=documents, embedding=instructor_embeddings)
vectordb.save_local(vectordb_file_path)
print(f"Indexed {vectordb.index.ntotal} vectors")"""
)

md("## 5. Session-style RetrievalQA (single-turn, no memory)")

code(
    r"""from langchain_core.prompts import PromptTemplate

from langchain.chains import RetrievalQA

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

prompt_template = '''Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}'''

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    input_key="query",
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

result1 = chain.invoke({"query": "How would Unsaidtalks will help me in cracking placements?"})
print("Q:", result1["query"])
print("A:", result1["result"])"""
)

md(
    """---

# Task 1 — Runtime knowledge-base updates

Accept **PDF** and **CSV** uploads, chunk them, and call `vectordb.add_documents(...)` so FAISS grows **in the same process**. No kernel restart."""
)

code(
    r"""import io
from pypdf import PdfReader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)


def load_csv_bytes(data: bytes, filename: str) -> list[Document]:
    df_new = pd.read_csv(io.BytesIO(data))
    cols = {c.lower().strip(): c for c in df_new.columns}
    pair = None
    for left, right in (("prompt", "response"), ("question", "answer"), ("q", "a")):
        if left in cols and right in cols:
            pair = (cols[left], cols[right])
            break
    docs = []
    if pair:
        q_col, a_col = pair
        for _, row in df_new.iterrows():
            docs.append(
                Document(
                    page_content=f"Question: {row[q_col]}\nAnswer: {row[a_col]}",
                    metadata={"source": filename, "type": "csv"},
                )
            )
        return docs
    for idx, row in df_new.iterrows():
        body = "\n".join(f"{c}: {row[c]}" for c in df_new.columns if pd.notna(row[c]))
        docs.append(Document(page_content=body, metadata={"source": filename, "row": int(idx) + 1}))
    return docs


def load_pdf_bytes(data: bytes, filename: str) -> list[Document]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(page_content=text, metadata={"source": filename, "page": i + 1, "type": "pdf"}))
    if not pages:
        raise ValueError(f"No text in {filename}")
    return splitter.split_documents(pages)


def ingest_file(filename: str, data: bytes) -> int:
    '''Parse a PDF or CSV and merge it into the live FAISS index.'''
    global retriever, chain
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        new_docs = load_pdf_bytes(data, filename)
    elif suffix == ".csv":
        new_docs = load_csv_bytes(data, filename)
    else:
        raise ValueError("Only PDF and CSV are supported")
    vectordb.add_documents(new_docs)
    vectordb.save_local(vectordb_file_path)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    print(f"Added {len(new_docs)} chunks from {filename}. Index size: {vectordb.index.ntotal}")
    return len(new_docs)"""
)

md(
    """### Upload files (Colab widget) or ingest the sample files in this repo

In Colab, run the next cell and choose a `.pdf` / `.csv`. Locally, the sample files under `data/` are ingested automatically if no upload is provided."""
)

code(
    r"""uploaded = {}
if IN_COLAB:
    from google.colab import files

    print("Choose one or more PDF / CSV files")
    uploaded = files.upload()

if uploaded:
    for name, data in uploaded.items():
        ingest_file(name, data)
else:
    samples = [
        Path("data/sample_runtime.csv"),
        Path("data/sample_intern_handbook.pdf"),
        Path("../data/sample_runtime.csv"),
        Path("../data/sample_intern_handbook.pdf"),
    ]
    seen = set()
    for path in samples:
        if path.exists() and path.name not in seen:
            ingest_file(path.name, path.read_bytes())
            seen.add(path.name)
    if not seen:
        print("No sample files found. Upload a PDF or CSV in Colab, or run from the repo root.")"""
)

md("Ask questions that are **only** answered by the newly uploaded documents:")

code(
    r"""for q in [
    "What is the weekend DSA bootcamp fee?",
    "Who is the weekend DSA bootcamp mentor?",
    "What is the intern stipend at UnsaidTalks?",
    "What is the intern referral code?",
]:
    result = chain.invoke({"query": q})
    print("Q:", q)
    print("A:", result["result"])
    print("---")"""
)

md(
    """---

# Task 2 — Chat history

The session `RetrievalQA` chain is **stateless**. Follow-ups like “how much is it?” fail because “it” never reaches the retriever.

We keep a `chat_history` list for the whole session, rewrite the latest turn into a standalone search query, retrieve, then generate with **context + full history**. `new_session()` is the only reset."""
)

code(
    r"""from langchain_core.messages import HumanMessage, SystemMessage

chat_history = []  # survives across cells until new_session()


def _text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


def format_history(history, limit=12) -> str:
    if not history:
        return "(no previous turns)"
    lines = []
    for turn in history[-limit:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def standalone_question(question: str) -> str:
    if not chat_history:
        return question
    prompt = (
        f"Chat history:\n{format_history(chat_history)}\n\n"
        f"Latest user message: {question}\n\n"
        "Rewrite as a standalone search query. Return only the query."
    )
    rewritten = _text(
        llm.invoke(
            [
                SystemMessage(content="Rewrite follow-up questions as standalone search queries. Do not answer."),
                HumanMessage(content=prompt),
            ]
        )
    ).strip()
    return rewritten or question


def ask(question: str) -> str:
    '''Contextual RAG turn. Appends to chat_history automatically.'''
    global retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    query = standalone_question(question)
    docs = retriever.invoke(query)
    context = "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )
    system = (
        "You are a RAG assistant for UnsaidTalks and user-uploaded documents. "
        "Answer from CONTEXT only. Use CHAT HISTORY for follow-ups. "
        "If the answer is not in the context, say: I don't know.\n\n"
        f"CONTEXT:\n{context}"
    )
    user = f"CHAT HISTORY:\n{format_history(chat_history)}\n\nQUESTION: {question}\n(search query: {query})"
    answer = _text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])).strip()
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    print(f"search: {query}")
    return answer


def new_session() -> None:
    '''Clear conversation memory. Does not drop the FAISS index.'''
    chat_history.clear()
    print("New session started. Knowledge base is unchanged.")"""
)

md("### Multi-turn conversation (history is required for the follow-ups)")

code(
    r"""print(ask("What is the intern stipend at UnsaidTalks?"))
print("---")
print(ask("When is it paid?"))
print("---")
print(ask("What is the referral code for that internship?"))
print(f"\nTurns stored: {len(chat_history)}")"""
)

md("### New session — history is wiped, uploaded documents stay")

code(
    r"""new_session()
print(ask("When is it paid?"))
print("The follow-up should no longer resolve 'it' from the intern stipend turn.")
print(f"Turns stored: {len(chat_history)}")"""
)

md(
    """## Streamlit app

For a chat UI with a file picker and a **Start new session** button:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `data/sample_runtime.csv` and `data/sample_intern_handbook.pdf` from the sidebar, then continue the same follow-up conversation in the browser."""
)

out = Path(__file__).resolve().parent.parent / "notebooks" / "RAG_Unsaidtalks_Enhanced.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
# fix md() trailing — rewrite cells to join sources properly
for cell in nb["cells"]:
    src = cell["source"]
    if isinstance(src, list) and src:
        # ensure last line has no extra issues
        cell["source"] = src

out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({len(nb['cells'])} cells)")
