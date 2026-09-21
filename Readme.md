# Table RAG — build from scratch

A minimal, working pipeline: **extract → standardize → embed/store → retrieve**,
for tables inside xlsx, docx, and pdf files.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

PDF fallback parsing needs Ghostscript for Camelot:

- Windows: install from ghostscript.com, add to PATH
- (pdfplumber alone needs nothing extra and handles most clean, bordered tables)

## Environment variables

```bash
export LLM_BASE_URL="https://api.openai.com/v1"     # or your gateway's base URL
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"

export EMBED_BASE_URL="https://api.openai.com/v1"
export EMBED_API_KEY="sk-..."                        # defaults to LLM_API_KEY if unset
export EMBED_MODEL="text-embedding-3-small"

export CHROMA_DIR="./chroma_store"                   # local persistent vector store
```

If no API key is set, both the summarizer and embedder fall back to offline
stubs so you can smoke-test the pipeline end-to-end before wiring up a real
model — useful for getting the plumbing right before you've settled on a
provider or gotten API access sorted out.

## Run

```bash
uvicorn app.api:app --reload --port 8000
```

Ingest a file:

```bash
curl -F "file=@quarterly_report.xlsx" http://localhost:8000/ingest
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was North America revenue in Q3?"}'
```

## Adapting to a different provider

- **Any OpenAI-compatible endpoint**: if your provider exposes
  `/chat/completions` and `/embeddings` routes in the OpenAI request/response
  shape, just point `LLM_BASE_URL` / `EMBED_BASE_URL` at it — no code changes
  needed. If the shape differs, only `standardize.summarize_table` and
  `embed_store.get_embedding` need edits; everything else is decoupled from
  the provider.
- **A different vector store**: if you outgrow local Chroma, swap
  `embed_store.py`'s Chroma calls for another vector DB's client (Qdrant,
  Pinecone, pgvector, etc.) — keep the `store_table` / `query` function
  signatures the same so nothing else in the pipeline changes.
- **Scaling ingestion**: for a large personal document set, add a simple
  queue (Celery/RQ) in front of `/ingest` rather than processing
  synchronously in the request.

## Design notes

- We embed the LLM-generated **summary** of each table, not the raw table
  cells — cells carry little standalone semantic signal, so summaries
  retrieve better.
- Markdown is stored as the retrievable payload, so the LLM answering the
  question sees the actual table, not just the summary.
- Merged/blank header cells are forward-filled before Markdown conversion
  (`standardize._forward_fill_headers`) so multi-row/merged headers don't
  collapse into empty column names.
- Retrieval always returns `source_file` + `location` so answers are
  traceable back to the exact sheet/table/page.
