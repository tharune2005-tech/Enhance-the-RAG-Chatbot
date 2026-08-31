"""Turn PDF and CSV bytes into LangChain Documents, then chunk PDFs."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

FAQ_COLUMN_PAIRS = (
    ("prompt", "response"),
    ("question", "answer"),
    ("query", "response"),
    ("q", "a"),
)


def _simple_split(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return [c for c in chunks if c]


def _splitter():
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            return None

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _match_faq_columns(columns: list[str]) -> tuple[str, str] | None:
    lowered = {c.lower().strip(): c for c in columns}
    for left, right in FAQ_COLUMN_PAIRS:
        if left in lowered and right in lowered:
            return lowered[left], lowered[right]
    return None


def csv_dataframe_to_documents(df: pd.DataFrame, source: str) -> list[Document]:
    """FAQ-style prompt/response rows become one document each (session format)."""
    documents: list[Document] = []
    pair = _match_faq_columns(list(df.columns))
    if pair:
        q_col, a_col = pair
        for _, row in df.iterrows():
            question = str(row.get(q_col, "")).strip()
            answer = str(row.get(a_col, "")).strip()
            if not question and not answer:
                continue
            documents.append(
                Document(
                    page_content=f"Question: {question}\nAnswer: {answer}",
                    metadata={"source": source, "type": "csv", "title": question[:120]},
                )
            )
        return documents

    for idx, row in df.iterrows():
        lines = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        if not lines:
            continue
        documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={"source": source, "type": "csv", "row": int(idx) + 1},
            )
        )
    return documents


def load_csv_bytes(data: bytes, filename: str) -> list[Document]:
    df = pd.read_csv(io.BytesIO(data))
    return csv_dataframe_to_documents(df, filename)


def load_csv_path(path: str | Path) -> list[Document]:
    path = Path(path)
    df = pd.read_csv(path)
    return csv_dataframe_to_documents(df, path.name)


def load_pdf_bytes(data: bytes, filename: str) -> list[Document]:
    reader = PdfReader(io.BytesIO(data))
    pages: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={"source": filename, "type": "pdf", "page": i + 1},
            )
        )
    if not pages:
        raise ValueError(f"No extractable text in PDF: {filename}")
    splitter = _splitter()
    if splitter is None:
        split_pages: list[Document] = []
        for page in pages:
            for chunk in _simple_split(page.page_content):
                split_pages.append(Document(page_content=chunk, metadata=dict(page.metadata)))
        return split_pages
    return splitter.split_documents(pages)


def load_pdf_path(path: str | Path) -> list[Document]:
    path = Path(path)
    return load_pdf_bytes(path.read_bytes(), path.name)


def load_file(filename: str, data: bytes) -> list[Document]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return load_pdf_bytes(data, filename)
    if suffix == ".csv":
        return load_csv_bytes(data, filename)
    raise ValueError(f"Unsupported file type '{suffix}'. Upload a PDF or CSV.")
