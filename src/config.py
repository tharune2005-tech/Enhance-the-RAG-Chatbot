"""Paths and model defaults shared by the Streamlit app and notebook helpers."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FAISS_DIR = Path(os.getenv("FAISS_DIR", ROOT / "faiss_index"))
FAQ_CSV = DATA_DIR / "unsaidtalks_faq.csv"
FAQ_DRIVE_ID = "1h1jHbpjZ_OwvP7H-5K39Ot8OXIwEGhJx"
FAQ_DRIVE_URL = f"https://drive.google.com/uc?export=download&id={FAQ_DRIVE_ID}"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "4"))
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
LLM_TEMPERATURE = 0.1


def google_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
