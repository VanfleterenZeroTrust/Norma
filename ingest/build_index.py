import os
import glob
import base64
import numpy as np
from dotenv import load_dotenv
from fastembed import TextEmbedding
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchField,
    VectorSearch,
)
from azure.search.documents import SearchClient
from chunkers import pdf_to_chunks

# =========================
#   Config & environment
# =========================
load_dotenv()
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "docs")

# =========================
#   Embeddings (FastEmbed)
# =========================
# Modèle multilingue FR/EN — 384 dimensions
EMBED_MODEL_ID = os.environ.get(
    "EMBEDDER_MODEL_ID", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
VECTOR_DIM = 384
_embedder = TextEmbedding(model_name=EMBED_MODEL_ID)

def embed_texts(texts):
    """
    Retourne une liste de list[float] JSON-serializable (pas de numpy types).
    """
    embs = []
    for vec in _embedder.embed(texts, batch_size=64):
        arr = np.asarray(vec, dtype=float).ravel()           # ndarray float64
        embs.append([float(x) for x in arr.tolist()])        # -> list[float]
    return embs

# =========================
#   Helpers
# =========================
def safe_key(raw: str) -> str:
    """
    Encode un identifiant arbitraire en Base64 URL-safe (sans '='),
    conforme aux contraintes d'Azure Search pour les clés.
    """
    b = raw.encode("utf-8")
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")

# =========================
#   Index Azure AI Search
# =========================
idx_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))

# Compat noms de classes HNSW selon la version du SDK
try:
    from azure.search.documents.indexes.models import HnswVectorSearchAlgorithmConfiguration as _Hnsw
except Exception:
    from azure.search.documents.indexes.models import HnswAlgorithmConfiguration as _Hnsw  # type: ignore

HNSW_NAME = "hnsw"
vector_search = VectorSearch(algorithm_configurations=[_Hnsw(name=HNSW_NAME)])

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchField(name="content", type=SearchFieldDataType.String, searchable=True),
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=VECTOR_DIM,
        # ⚠️ Sur ta version du SDK, il faut utiliser 'vector_search_configuration'
        vector_search_configuration=HNSW_NAME,
    ),
]

index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)

print(f"Creating index '{INDEX_NAME}'...")
try:
    idx_client.delete_index(INDEX_NAME)
except Exception:
    pass
idx_client.create_index(index)

# =========================
#   Read PDFs & chunk
# =========================
docs = []
for path in glob.glob("./data/*.pdf"):
    print(f"Processing {path} ...")
    for ch in pdf_to_chunks(path):
        # ch["id"] peut contenir ./, ., etc. -> on le rend sûr ici
        raw_id = str(ch.get("id", "")) or f"{path}"
        ch["id"] = safe_key(raw_id)
        docs.append(ch)

print(f"Total chunks to upload: {len(docs)}")

# =========================
#   Embed & upload batches
# =========================
client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
BATCH = 32

for i in range(0, len(docs), BATCH):
    batch = docs[i : i + BATCH]
    embs = embed_texts([d.get("text", "") for d in batch])

    # Construire des documents JSON-safe (pas de numpy.*) + ids sûrs
    clean_docs = []
    for d, e in zip(batch, embs):
        content = d.pop("text", "")
        raw_id = str(d.get("id", ""))
        safe_id = safe_key(raw_id) if raw_id else safe_key(content[:50] + str(i))

        clean_docs.append({
            "id": safe_id,                      # clé conforme (letters/digits/_/-/=)
            "content": str(content),            # force string
            "embedding": [float(x) for x in e], # list[float] natifs
        })

    client.upload_documents(clean_docs)
    print(f"Uploaded {len(clean_docs)} docs ({i + len(clean_docs)}/{len(docs)})")

print("✅ Done! Index built successfully.")
