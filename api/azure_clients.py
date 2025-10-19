import os
from dotenv import load_dotenv
from typing import List
import httpx
from fastembed import TextEmbedding
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
AZURE_SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "docs")

LLM_MODEL = os.environ.get("LLM_MODEL_NAME", "phi-4-mini-instruct")
LLM_ENDPOINT = os.environ["LLM_ENDPOINT"]
LLM_API_KEY = os.environ["LLM_API_KEY"]

# Embedding (384 dims for MiniLM)
_EMBED_MODEL_ID = os.environ.get(
    "EMBEDDER_MODEL_ID", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_embedder = TextEmbedding(model_name=_EMBED_MODEL_ID)

def embed_query(text: str) -> List[float]:
    vec = next(_embedder.embed([text], batch_size=1))
    # ensure native Python floats (not numpy types)
    return [float(x) for x in vec]

def get_search_client() -> SearchClient:
    return SearchClient(
        AZURE_SEARCH_ENDPOINT,
        AZURE_SEARCH_INDEX,
        AzureKeyCredential(AZURE_SEARCH_ADMIN_KEY),
    )

async def chat_completion(messages):
    payload = {"model": LLM_MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 800}
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(LLM_ENDPOINT, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
