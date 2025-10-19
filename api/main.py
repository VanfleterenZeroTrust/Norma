import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from azure_clients import ModelsClient
from retrieval import Retriever
from prompts import build_messages

load_dotenv()
app=FastAPI(title="RAG Azure Student")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

models=ModelsClient()
retriever=Retriever()

class AskBody(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
async def ask(body: AskBody):
    q=body.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")
    vec = models.embed([q])[0]
    hits = retriever.hybrid(q, vec, k=5)
    contexts=[h["content"] for h in hits]
    messages=build_messages(q, contexts)
    answer = await models.chat(messages)
    return {"answer": answer, "sources": hits}
