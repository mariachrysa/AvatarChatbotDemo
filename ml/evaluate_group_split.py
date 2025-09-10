# ml/evaluate_group_split.py
import json, pathlib, random, math, numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "faq_train.json"  # all items (with paraphrases mixed in)
EMBEDDER_ID = "sentence-transformers/all-mpnet-base-v2"

USE_RERANKER = True
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-12-v2"
try:
    from sentence_transformers import CrossEncoder
    RERANKER = CrossEncoder(RERANKER_ID) if USE_RERANKER else None
except Exception as e:
    print("[warn] Reranker unavailable -> embedding-only:", e)
    RERANKER = None

random.seed(0)

items = json.loads(DATA.read_text(encoding="utf-8"))

# --- group by canonical answer string ---
groups = defaultdict(list)
for it in items:
    groups[it["a"].strip()].append(it["q"].strip())

# --- make a group-aware split: ensure every group has at least 1 train item ---
answers = list(groups.keys())
random.shuffle(answers)

test_frac = 0.2
target_test_groups = max(1, int(len(answers) * test_frac))

test_answers = set(answers[:target_test_groups])
train_q, train_a = [], []
test_q,  test_a  = [], []

for a, qs in groups.items():
    qs_shuf = qs[:]
    random.shuffle(qs_shuf)
    if a in test_answers:
        # put 1 example in train (so answer exists), rest in test
        if len(qs_shuf) == 1:
            # degenerate group: keep in train to avoid unreachable test
            train_q.append(qs_shuf[0]); train_a.append(a)
        else:
            train_q.append(qs_shuf[0]); train_a.append(a)
            for q in qs_shuf[1:]:
                test_q.append(q); test_a.append(a)
    else:
        # all in train
        for q in qs_shuf:
            train_q.append(q); train_a.append(a)

print(f"[split] train Qs={len(train_q)}  test Qs={len(test_q)}  groups={len(groups)}")

# --- embeddings ---
embedder = SentenceTransformer(EMBEDDER_ID)
train_emb = embedder.encode(train_q, convert_to_numpy=True, normalize_embeddings=True)

def search(query: str, top_k: int = 10):
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = train_emb @ qv
    idx = np.argsort(-sims)[:top_k]
    hits = [{"q": train_q[i], "a": train_a[i], "score": float(sims[i])} for i in idx]
    return hits

def evaluate(top_k=10):
    ranks = []
    for q, gold_a in zip(test_q, test_a):
        hits = search(q, top_k=top_k)
        if RERANKER:
            pairs = [(q, h["q"]) for h in hits]
            scores = RERANKER.predict(pairs).tolist()
            for h, s in zip(hits, scores): h["rerank"] = float(s)
            hits.sort(key=lambda h: h.get("rerank", h["score"]), reverse=True)

        r = math.inf
        for i, h in enumerate(hits, start=1):
            if h["a"].strip() == gold_a.strip():
                r = i; break
        ranks.append(r)

    n = len(ranks)
    top1 = sum(1 for r in ranks if r == 1) / n if n else 0.0
    r3   = sum(1 for r in ranks if r <= 3) / n if n else 0.0
    r10  = sum(1 for r in ranks if r <= 10) / n if n else 0.0
    mrr  = sum((1/r) for r in ranks if r != math.inf) / n if n else 0.0
    return top1, r3, r10, mrr, n

if __name__ == "__main__":
    top1, r3, r10, mrr, n = evaluate()
    mode = "Embed+Rerank" if RERANKER else "Embed-only"
    print(f"[{mode}] Top-1={top1:.2f}  R@3={r3:.2f}  R@10={r10:.2f}  MRR={mrr:.2f}  (N={n})")
