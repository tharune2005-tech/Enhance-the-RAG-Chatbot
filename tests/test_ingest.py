"""Offline checks for PDF/CSV ingest — no API key required."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest import load_csv_path, load_pdf_path


class IngestTests(unittest.TestCase):
    def test_faq_csv_uses_question_answer_format(self) -> None:
        docs = load_csv_path(ROOT / "data" / "unsaidtalks_faq.csv")
        self.assertGreaterEqual(len(docs), 100)
        self.assertTrue(docs[0].page_content.startswith("Question:"))
        self.assertIn("Answer:", docs[0].page_content)

    def test_runtime_csv_contains_bootcamp_fee(self) -> None:
        docs = load_csv_path(ROOT / "data" / "sample_runtime.csv")
        blob = "\n".join(d.page_content for d in docs)
        self.assertIn("2499", blob)
        self.assertIn("Priya Sharma", blob)

    def test_runtime_pdf_contains_stipend(self) -> None:
        docs = load_pdf_path(ROOT / "data" / "sample_intern_handbook.pdf")
        self.assertGreaterEqual(len(docs), 1)
        blob = "\n".join(d.page_content for d in docs)
        self.assertIn("15000", blob)
        self.assertIn("UT-INTERN-2026", blob)


if __name__ == "__main__":
    unittest.main()
