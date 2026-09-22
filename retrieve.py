"""
Retrieval & querying layer.
Fetches the most relevant table chunks, then asks the LLM to answer using
only those chunks — with source attribution so answers stay traceable.
"""
from embed_store import query
from standardize import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
import requests


SYSTEM_PROMPT = (
    "You are a data analysis assistant. Use ONLY the provided Markdown table "
    "data to answer the user's question. If the tables don't contain the "
    "answer, say so explicitly rather than guessing. Cite the source_file "
    "and location for any figure you quote."
)


def answer_question(question: str, top_k: int = 4) -> dict:
    hits = query(question, top_k=top_k)
    if not hits:
        return {"answer": "No relevant tables found for this question.", "sources": []}

    context = "\n\n---\n\n".join(
        f"Source: {h['source_file']} ({h['location']})\n{h['markdown']}" for h in hits
    )

    if not LLM_API_KEY:
        return {
            "answer": "[No LLM configured — showing raw retrieved tables instead]\n\n" + context,
            "sources": [{"source_file": h["source_file"], "location": h["location"]} for h in hits],
        }

    resp = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nTables:\n{context}"},
            ],
            "temperature": 0,
        },
        timeout=30,
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    return {
        "answer": answer,
        "sources": [{"source_file": h["source_file"], "location": h["location"]} for h in hits],
    }
