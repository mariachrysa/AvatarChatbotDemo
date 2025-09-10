# ml/evaluate_server_http.py
import json, pathlib, requests, math

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "faq_test.json"

ASK_URL = "http://127.0.0.1:8000/ask"  # must be running: uvicorn server:app ...

def top1_accuracy():
    test = json.loads(TEST.read_text(encoding="utf-8"))
    correct = 0
    for x in test:
        resp = requests.post(ASK_URL, json={"question": x["q"], "top_k": 5}).json()
        pred = (resp.get("answer") or "").strip()
        gold = x["a"].strip()
        correct += int(pred == gold)
    return correct, len(test)

if __name__ == "__main__":
    ok, n = top1_accuracy()
    print(f"[SERVER /ask] Top-1 exact-match: {ok}/{n} = {ok/n:.2f}")
