SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir des extraits fournis. "
    "Cite les sources avec leurs IDs (format [DOC x]). Réponds en français concis."
)
def build_messages(question:str, contexts:list[str]):
    context_block = "\n\n".join(f"[DOC {i}]\n{c}" for i,c in enumerate(contexts,1))
    user = f"Contexte:\n{context_block}\n\nQuestion: {question}\nRéponds avec les sources [DOC x]."
    return [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":user}
    ]
