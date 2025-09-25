# ml/server.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import io, base64, wave
import numpy as np, json, time, pathlib
from textnorm import normalize
from sentence_transformers import SentenceTransformer, CrossEncoder
from tts_provider import Pyttsx3Provider

# ---------- Paths / models ----------
BASE_DIR = pathlib.Path(__file__).resolve().parent
APP_DIR = BASE_DIR / "models"   # contains faq_train.json, q_emb.npy, model_name.txt

faq = json.loads((APP_DIR / "faq_train.json").read_text(encoding="utf-8"))
q_emb = np.load(APP_DIR / "q_emb.npy")  # shape: (N, d)
model_name = (APP_DIR / "model_name.txt").read_text(encoding="utf-8").strip()
embedder = SentenceTransformer(model_name)

# TTS (local, pluggable)
tts = Pyttsx3Provider()  # you can pass voice_id=... , rate=175 later

# Optional cross-encoder re-ranker
USE_RERANKER = True
reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
reranker = None
if USE_RERANKER:
    try:
        reranker = CrossEncoder(reranker_model)
        print(f"[info] Reranker loaded: {reranker_model}")
    except Exception as e:
        print(f"[warn] Reranker unavailable: {e}")
        reranker = None

CONFIDENCE_THRESH = 0.35  # cosine threshold; tune per dataset

# ---------- FastAPI ----------
app = FastAPI(title="AvatarChatbotDemo")

# CORS for local dev (relax; tighten later if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------
class AskReq(BaseModel):
    question: str
    top_k: int = 5
    return_alternatives: int = 3

class TTSReq(BaseModel):
    text: str

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "n_items": len(faq)}

# ---------- Search / Ask ----------
def search(query: str, top_k: int):
    q = normalize(query or "")
    qv = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = q_emb @ qv
    idx = np.argsort(-sims)[: max(1, top_k)]
    return [
        {"i": int(i), "q": faq[i]["q"], "a": faq[i]["a"], "score": float(sims[i])}
        for i in idx
    ]

@app.post("/ask")
def ask(payload: AskReq):
    t0 = time.perf_counter()
    hits = search(payload.question, payload.top_k)

    # Optional re-ranking
    if reranker and hits:
        try:
            pairs = [(payload.question, h["q"]) for h in hits]
            scores = reranker.predict(pairs).tolist()
            for h, s in zip(hits, scores):
                h["rerank"] = float(s)
            hits.sort(key=lambda x: x.get("rerank", x["score"]), reverse=True)
        except Exception as e:
            print(f"[warn] reranker failed: {e}")

    best = hits[0] if hits else None
    latency = (time.perf_counter() - t0) * 1000.0

    if not best or best["score"] < CONFIDENCE_THRESH:
        return {
            "answer": "I’m not fully sure. Could you rephrase or ask a more specific question?",
            "matches": hits[: payload.return_alternatives],
            "latency_ms": latency,
            "uncertain": True,
        }

    return {
        "answer": best["a"],
        "matches": hits[: payload.return_alternatives],
        "latency_ms": latency,
        "uncertain": False,
    }

# ---------- TTS ----------
@app.post("/tts")
def tts_json(req: TTSReq):
    """Return JSON: { sample_rate, channels, pcm16 (base64 of int16 mono) }"""
    out = tts.tts(req.text)
    if "error" in out:
        raise HTTPException(status_code=503, detail=out["error"])
    return out

@app.post("/tts_wav")
def tts_wav(req: TTSReq):
    """Return a proper WAV (audio/wav) with header (download in Swagger)."""
    out = tts.tts(req.text)
    if "error" in out:
        raise HTTPException(status_code=503, detail=out["error"])

    pcm = np.frombuffer(base64.b64decode(out["pcm16"]), dtype=np.int16)
    sr = int(out.get("sample_rate", 22050))
    ch = int(out.get("channels", 1))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)      # 16-bit PCM
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    audio = buf.getvalue()

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="tts.wav"'}
    )
