SYSTEM_PROMPT = (
    "You are an assistant that answers **only** based on the provided excerpts. "
    "Always cite the sources with their IDs in the format [DOC x]. "
    "Answer in English clearly and concisely."
)

def build_messages(question: str, contexts: list[str]):
    context_block = "\n\n".join(f"[DOC {i}]\n{c}" for i, c in enumerate(contexts))
    user = f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer using only the provided context and include source citations."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user}
    ]
