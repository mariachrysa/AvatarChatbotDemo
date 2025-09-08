import json, numpy as np, pathlib
from textnorm import normalize
from sentence_transformers import SentenceTransformer

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "faq_train.json"
OUT = pathlib.Path("models"); OUT.mkdir(parents=True, exist_ok=True)

model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name)

faq = json.loads(DATA.read_text(encoding="utf-8"))
questions = [normalize(x["q"]) for x in faq]

emb = model.encode(questions, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
np.save(OUT / "q_emb.npy", emb)
(OUT / "faq_train.json").write_text(json.dumps(faq, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "model_name.txt").write_text(model_name, encoding="utf-8")
print("Index built:", emb.shape)
