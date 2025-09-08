from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, json, time, pathlib, os
from textnorm import normalize
from sentence_transformers import SentenceTransformer, CrossEncoder

APP_DIR = pathlib.Path("models")
faq = json.loads((APP_DIR/"faq_train.json").read_text(encoding="utf-8"))
q_emb = np.load(APP_DIR/"q_emb.npy")
model_name = (APP_DIR/"model_name.txt").read_text(encoding="utf-8").strip()
embedder = SentenceTransformer(model_name)

# Optional cross-encoder re-ranker (comment out if torch not installed)
USE_RERANKER = True
reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
reranker = CrossEncoder(reranker_model) if USE_RERANKER else None

CONFIDENCE_THRESH = 0.35  # cosine threshold; tweak per dataset

class AskReq(BaseModel):
    question: str
    top_k: int = 5
    return_alternatives: int = 3

app = FastAPI()

@app.get("/health")
def health(): return {"ok": True, "n_items": len(faq)}

def search(query: str, top_k: int):
    qv = embedder.encode([normalize(query)], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = q_emb @ qv
    idx = np.argsort(-sims)[:top_k]
    hits = [{"i": int(i), "q": faq[i]["q"], "a": faq[i]["a"], "score": float(sims[i])} for i in idx]
    return hits

@app.post("/ask")
def ask(payload: AskReq):
    t0 = time.perf_counter()
    hits = search(payload.question, payload.top_k)

    # Optional re-ranking
    if reranker:
        pairs = [(payload.question, h["q"]) for h in hits]
        rerank_scores = reranker.predict(pairs).tolist()
        for h, s in zip(hits, rerank_scores): h["rerank"] = float(s)
        hits.sort(key=lambda x: x.get("rerank", x["score"]), reverse=True)

    best = hits[0] if hits else None
    latency = (time.perf_counter()-t0)*1000

    # Low-confidence fallback
    if not best or best["score"] < CONFIDENCE_THRESH:
        return {
            "answer": "I’m not fully sure. Could you rephrase or ask a more specific question?",
            "matches": hits[:payload.return_alternatives],
            "latency_ms": latency,
            "uncertain": True
        }

    return {
        "answer": best["a"],
        "matches": hits[:payload.return_alternatives],
        "latency_ms": latency,
        "uncertain": False
    }
