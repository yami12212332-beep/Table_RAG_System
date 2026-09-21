"""
FastAPI entrypoint.

Run:
    uvicorn app.api:app --reload --port 8000

Endpoints:
    POST /ingest   - upload xlsx/docx/pdf, extracts+embeds all tables in it
    POST /ask      - ask a question, get an answer grounded in stored tables
"""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.extract import extract
from app.standardize import standardize
from app.embed_store import store_table
from app.retrieve import answer_question

app = FastAPI(title="Table RAG")


class Question(BaseModel):
    question: str
    top_k: int = 4


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    tables = extract(tmp_path)
    stored_ids = []
    for t in tables:
        t.source_file = file.filename  # keep the original name, not the temp path
        record = standardize(t)
        stored_ids.append(store_table(record))

    return {"file": file.filename, "tables_found": len(tables), "chunk_ids": stored_ids}


@app.post("/ask")
async def ask(q: Question):
    return answer_question(q.question, top_k=q.top_k)


@app.get("/health")
async def health():
    return {"status": "ok"}
