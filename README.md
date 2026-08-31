# UnsaidTalks RAG Chatbot

Extended Retrieval-Augmented Generation chatbot from the UnsaidTalks live session. The original notebook indexed a static FAQ CSV once. This repo adds the two production pieces from the optional activity: **runtime knowledge-base updates** and **conversation memory**.

Starter notebook: [RAG Chatbot (Colab)](https://colab.research.google.com/drive/1qSSBFI9J83u_Ij1vSwvCeKfpX0j8WbYB?usp=sharing)

## What this adds

| Task | Behaviour |
| --- | --- |
| **1. Runtime knowledge updates** | Upload **PDF** or **CSV** in the running app. Files are parsed, chunked, embedded with MiniLM, and merged into FAISS immediately. You can ask questions about the new file without restarting. |
| **2. Chat history** | The full session transcript is kept and passed into retrieval + generation. Follow-ups such as “how much is it?” use the previous turns. History clears only when you start a **new session**. |

The default knowledge base is the same UnsaidTalks FAQ used in the session (`prompt` / `response` columns, 500 rows).

## Architecture

```
PDF / CSV upload          UnsaidTalks FAQ CSV
        │                         │
        ▼                         ▼
   ingest.py  ─────────────►  FAISS (MiniLM embeddings)
                                  │
User question + chat history      │
        │                         ▼
        ├─ rewrite follow-up ──► retriever (top-k)
        │                         │
        └─ Gemini  ◄──── context + full history
```

Same stack as the session: **LangChain**, **HuggingFace `all-MiniLM-L6-v2`**, **FAISS**, **Gemini**. Retrieval still refuses to invent an answer when the context does not contain it.

## Quick start

Python **3.11 or 3.12** (the embedding stack is not reliable on 3.14 yet).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# paste GOOGLE_API_KEY from https://aistudio.google.com/apikey
streamlit run app.py
```

The first launch downloads the MiniLM weights and builds `faiss_index/` from `data/unsaidtalks_faq.csv`. Later launches reuse that index.

### Try the two tasks

1. Ask: *How would UnsaidTalks help me crack placements?*
2. Upload `data/sample_runtime.csv` and ask: *What is the weekend DSA bootcamp fee?*
3. Upload `data/sample_intern_handbook.pdf` and ask: *What is the intern stipend?*
4. Follow up: *When is it paid?* then *What is the referral code?* — these only work because chat history is kept.
5. Click **Start new session** and ask *When is it paid?* again. The bot should no longer know which “it” you mean from the previous chat.

## Project layout

```
app.py                          Streamlit chatbot
src/ingest.py                   PDF + CSV loaders
src/knowledge_base.py           FAISS load / save / add_documents
src/rag_chain.py                History-aware retrieve-then-generate
src/llm.py                      Gemini client
notebooks/RAG_Unsaidtalks_Enhanced.ipynb
data/unsaidtalks_faq.csv        Session FAQ
data/sample_runtime.csv         Extra CSV for Task 1
data/sample_intern_handbook.pdf Extra PDF for Task 1
```

## Notebook

[`notebooks/RAG_Unsaidtalks_Enhanced.ipynb`](notebooks/RAG_Unsaidtalks_Enhanced.ipynb) rebuilds the live-session pipeline, then implements both optional tasks with Colab file upload. Open it in Jupyter or Colab after installing `requirements.txt`.

## Configuration

| Variable | Purpose |
| --- | --- |
| `GOOGLE_API_KEY` | Gemini key (required) |
| `GOOGLE_MODEL` | Defaults to `gemini-2.5-flash`. The session used `gemini-3.5-flash` if your project has access. |
| `FAISS_DIR` | Optional override for the index directory |

## Notes

- FAQ-style CSVs (`prompt`/`response` or `question`/`answer`) become one document per row, matching the session. Other CSVs become one document per row with `column: value` text. PDFs are split into overlapping chunks.
- **New session** clears messages only. Uploaded files stay in FAISS until you delete `faiss_index/` and restart.
- The original session `RetrievalQA` prompt is preserved in spirit: answer from context, otherwise *I don't know.*
