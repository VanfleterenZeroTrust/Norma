from typing import List, Dict, Any
import os
import httpx
from azure_clients import embed_query

TOP_K = 4

# Read env once
AZURE_SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/")
AZURE_SEARCH_ADMIN_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "docs")

def retrieve(query: str) -> List[Dict[str, Any]]:
    qvec = embed_query(query)

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2024-07-01"
    payload = {
        "search": "",
        "top": TOP_K,
        "select": "id,content",
        "vectorQueries": [
            {
                "kind": "vector",
                "vector": qvec,
                "k": TOP_K,
                "fields": "embedding",
            }
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_ADMIN_KEY,
    }

    with httpx.Client(timeout=30) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    out: List[Dict[str, Any]] = []
    for item in data.get("value", []):
        out.append({
            "id": item.get("id", ""),
            "content": item.get("content", ""),
        })
    return out
