import json, pathlib
from collections import Counter
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "faq_train.json"
items = json.loads(DATA.read_text(encoding="utf-8"))
counts = Counter(x["a"].strip() for x in items)
print("Groups by size ->", Counter(counts.values()))
print("\nAnswers with only one phrasing:")
singletons = [a for a,c in counts.items() if c==1]
for a in singletons: print("-", a)
