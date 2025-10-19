from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "You are an assistant that answers only based on the provided excerpts. "
    "Always include [DOC x] citations that correspond to the provided snippets. "
    "Answer in English clearly and concisely."
)

def build_messages(question: str, contexts: List[Dict[str, Any]]):
    context_block = "\n\n".join(
        f"[DOC {i+1}]\n{c.get('content','')}" for i, c in enumerate(contexts)
    )
    user = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the provided context and include source citations like [DOC x]."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
