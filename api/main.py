from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from retrieval import retrieve
from prompts import build_messages
from azure_clients import chat_completion

app = FastAPI(title="RAG Student API", version="1.0")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        contexts: List[Dict[str, Any]] = retrieve(req.question)
        messages = build_messages(req.question, contexts)
        answer = await chat_completion(messages)
        return AskResponse(
            answer=(answer or "").strip(),
            sources=[c["id"] for c in contexts],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok"}
