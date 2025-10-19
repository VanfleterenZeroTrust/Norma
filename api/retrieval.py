import os, httpx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

LLM_ENDPOINT=os.environ["LLM_ENDPOINT"]
LLM_API_KEY=os.environ["LLM_API_KEY"]
LLM_MODEL=os.environ.get("LLM_MODEL_NAME","phi-4-mini-instruct")

# Local embeddings for user queries
_EMBEDDER_MODEL = os.environ.get("EMBEDDER_MODEL_ID", "intfloat/multilingual-e5-small")
_embedder = SentenceTransformer(_EMBEDDER_MODEL)

class ModelsClient:
    def __init__(self):
        self.headers_llm={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type":"application/json"}

    def embed(self, texts):
        # e5 family expects 'query:' prefix for queries
        return _embedder.encode([f"query: {t}" for t in texts], normalize_embeddings=True).tolist()

    async def chat(self, messages):
        payload={"model": LLM_MODEL, "messages": messages, "temperature": 0.2}
        async with httpx.AsyncClient(timeout=60) as client:
            r=await client.post(LLM_ENDPOINT, headers=self.headers_llm, json=payload)
            r.raise_for_status()
            data=r.json()
            return data["choices"][0]["message"]["content"]
