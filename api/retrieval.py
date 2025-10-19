from typing import List, Dict, Any
from azure.search.documents import SearchClient
from azure_clients import embed_query, get_search_client

TOP_K = 4

def retrieve(query: str) -> List[Dict[str, Any]]:
    client: SearchClient = get_search_client()
    qvec = embed_query(query)

    # Utilisation d'un dict REST-compatible (pour toutes versions du SDK)
    vq = {
        "kind": "vector",
        "vector": qvec,
        "k": TOP_K,
        "fields": "embedding",
    }

    # Recherche purement vectorielle (tu pourras repasser à search_text=query ensuite)
    results = client.search(
        search_text="",
        vector_queries=[vq],
        select=["id", "content"],
        top=TOP_K,
    )

    out: List[Dict[str, Any]] = []
    for r in results:
        out.append({"id": r["id"], "content": r["content"]})
    return out
