import re

ABBREV = {
    r"\bER\b": "emergency department",
    r"\bA&E\b": "emergency department",
    r"\bED\b": "emergency department",
    r"\bdept\b": "department",
    r"\bMRI\b": "magnetic resonance imaging",
    r"\bCT\b": "computed tomography",
    r"\bICU\b": "intensive care unit",
    r"\bID\b": "identification",
    r"\bMon[\u2013-]Fri\b": "Mon-Fri",
    r"\bMon to Fri\b": "Mon-Fri"
}
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[’']", "'", s)
    s = re.sub(r"\s+", " ", s)
    for pat, repl in ABBREV.items():
        s = re.sub(pat, repl, s)
    return s
