import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI()

# ======================
# LOAD DOCUMENTS
# ======================

documents = []
with open("docs.txt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        documents.append({
            "id": i,
            "content": line.strip(),
            "metadata": {"source": "contracts"}
        })

# ======================
# MODELS
# ======================

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# precompute embeddings
doc_embeddings = embed_model.encode(
    [d["content"] for d in documents],
    normalize_embeddings=True
)

# ======================
# REQUEST MODEL
# ======================

class Query(BaseModel):
    query: str
    k: int = 6
    rerank: bool = True
    rerankK: int = 4

# ======================
# SEARCH ENDPOINT
# ======================

@app.post("/")
def search(q: Query):

    start = time.time()

    # embed query
    q_emb = embed_model.encode([q.query], normalize_embeddings=True)[0]

    # cosine similarity
    sims = np.dot(doc_embeddings, q_emb)

    # top-k
    idx = np.argsort(sims)[::-1][:q.k]
    candidates = [documents[i] for i in idx]
    scores = [float(sims[i]) for i in idx]

    # normalize 0-1
    scores = [(s+1)/2 for s in scores]

    # reranking
    if q.rerank:
        pairs = [(q.query, d["content"]) for d in candidates]
        rerank_scores = rerank_model.predict(pairs)

        # normalize
        rerank_scores = (rerank_scores - rerank_scores.min()) / (
            rerank_scores.max() - rerank_scores.min() + 1e-9
        )

        combined = list(zip(candidates, rerank_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        combined = combined[:q.rerankK]

        results = [
            {
                "id": d["id"],
                "score": round(float(s),3),
                "content": d["content"],
                "metadata": d["metadata"]
            }
            for d,s in combined
        ]

    else:
        results = [
            {
                "id": d["id"],
                "score": round(scores[i],3),
                "content": d["content"],
                "metadata": d["metadata"]
            }
            for i,d in enumerate(candidates[:q.rerankK])
        ]

    latency = int((time.time()-start)*1000)

    return {
        "results": results,
        "reranked": q.rerank,
        "metrics":{
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
