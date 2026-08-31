"""FAISS knowledge base that can grow at runtime without restarting the app."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import EMBEDDING_MODEL, FAISS_DIR, RETRIEVE_K


def build_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


class KnowledgeBase:
    def __init__(self, persist_dir: Path | str | None = None, embeddings=None):
        self.persist_dir = Path(persist_dir or FAISS_DIR)
        self.embeddings = embeddings or build_embeddings()
        self.vectordb: FAISS | None = None
        self.ingested: list[dict] = []

    @property
    def document_count(self) -> int:
        if self.vectordb is None:
            return 0
        return int(self.vectordb.index.ntotal)

    def load_local(self) -> bool:
        index_file = self.persist_dir / "index.faiss"
        if not index_file.exists():
            return False
        self.vectordb = FAISS.load_local(
            str(self.persist_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        sources_file = self.persist_dir / "sources.json"
        if sources_file.exists():
            self.ingested = json.loads(sources_file.read_text())
        return True

    def save(self) -> None:
        if self.vectordb is None:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vectordb.save_local(str(self.persist_dir))
        (self.persist_dir / "sources.json").write_text(json.dumps(self.ingested, indent=2))

    def add_documents(self, documents: list[Document], label: str | None = None) -> int:
        if not documents:
            raise ValueError("No documents to add.")
        if self.vectordb is None:
            self.vectordb = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectordb.add_documents(documents)
        self.ingested.append(
            {
                "label": label or documents[0].metadata.get("source", "upload"),
                "chunks": len(documents),
            }
        )
        self.save()
        return len(documents)

    def as_retriever(self, k: int | None = None):
        if self.vectordb is None:
            raise RuntimeError("Knowledge base is empty. Upload a PDF or CSV first.")
        return self.vectordb.as_retriever(search_kwargs={"k": k or RETRIEVE_K})
