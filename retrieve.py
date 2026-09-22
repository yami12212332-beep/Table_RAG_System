"""
Retrieval & querying layer.
Fetches the most relevant table chunks, then asks the LLM to answer using
only those chunks — with source attribution so answers stay traceable.
"""
from embed_store import query
from standardize import _client, LLM_MODEL


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

    if _client is None:
        return {
            "answer": "[No LLM configured — showing raw retrieved tables instead]\n\n" + context,
            "sources": [{"source_file": h["source_file"], "location": h["location"]} for h in hits],
        }
    print(context)
    completion = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nTables:\n{context}"},
        ],
        temperature=0,
    )
    answer = completion.choices[0].message.content.strip()
    return {
        "answer": answer,
        "sources": [{"source_file": h["source_file"], "location": h["location"]} for h in hits],
    }