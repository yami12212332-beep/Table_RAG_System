"""
Embedding & vector storage layer.

Uses Chroma with a persistent local directory — no external DB service to
stand up. Swap `get_embedding` for a different embeddings provider (Azure
OpenAI, Cohere, a local model, etc.) if you'd rather not call OpenAI directly.
"""
import os
import uuid
import requests
import chromadb

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://api.openai.com/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY", ""))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection("table_chunks")


def get_embedding(text: str) -> list:
    if not EMBED_API_KEY:
        # Deterministic offline fallback (NOT semantically meaningful — dev only).
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in h][:32]

    resp = requests.post(
        f"{EMBED_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
        json={"model": EMBED_MODEL, "input": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def store_table(record: dict) -> str:
    """
    record comes from standardize.standardize():
      {"markdown": ..., "summary": ..., "source_file": ..., "location": ...}

    We embed the summary, and keep markdown + source metadata for retrieval.
    """
    chunk_id = str(uuid.uuid4())
    vector = get_embedding(record["summary"])
    _collection.add(
        ids=[chunk_id],
        embeddings=[vector],
        documents=[record["markdown"]],
        metadatas=[
            {
                "summary": record["summary"],
                "source_file": record["source_file"],
                "location": record["location"],
            }
        ],
    )
    return chunk_id


def query(question: str, top_k: int = 4) -> list:
    vector = get_embedding(question)
    results = _collection.query(query_embeddings=[vector], n_results=top_k)
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"markdown": doc, **meta})
    return hits
