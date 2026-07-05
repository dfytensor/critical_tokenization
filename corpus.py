"""Corpus loading + train/val/test splitting for critical-tokenization experiments."""
import json
import os
import urllib.request

CACHE = os.path.join(os.path.dirname(__file__), "corpus_cache")
os.makedirs(CACHE, exist_ok=True)

CN_JSONL = r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"
EN_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",
    "https://www.gutenberg.org/files/11/11-0.txt",
    "https://www.gutenberg.org/files/2701/2701-0.txt",
]


def _strip_gutenberg(raw: str) -> str:
    for tag in ("*** START OF", "*** START OF THE PROJECT GUTENBERG"):
        i = raw.find(tag)
        if i != -1:
            raw = raw[i:]
            j = raw.find("***")
            if j != -1:
                raw = raw[j + 3:]
            break
    for tag in ("*** END OF", "*** END OF THE PROJECT GUTENBERG"):
        i = raw.find(tag)
        if i != -1:
            raw = raw[:i]
            break
    return raw.strip()


def load_chinese(max_chars: int = 2_000_000) -> str:
    cache = os.path.join(CACHE, f"cn_{max_chars}.txt")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return f.read()
    parts = []
    total = 0
    with open(CN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text", "")
            if not t:
                continue
            parts.append(t)
            total += len(t)
            if total >= max_chars:
                break
    text = "\n".join(parts)[:max_chars]
    with open(cache, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def load_english(max_chars: int = 1_500_000) -> str:
    cache = os.path.join(CACHE, f"en_{max_chars}.txt")
    if os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return f.read()
    chunks = []
    total = 0
    for url in EN_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[corpus] skip {url}: {e}")
            continue
        raw = _strip_gutenberg(raw)
        chunks.append(raw)
        total += len(raw)
        if total >= max_chars:
            break
    text = "\n".join(chunks)[:max_chars]
    with open(cache, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def split_text(text: str, ratios=(0.6, 0.2, 0.2)):
    n = len(text)
    a = int(n * ratios[0])
    b = int(n * (ratios[0] + ratios[1]))
    return text[:a], text[a:b], text[b:]


def normalize(text: str) -> str:
    out = []
    for ch in text:
        if ch.isspace():
            out.append(" ")
        elif ch.isprintable():
            out.append(ch)
    return "".join(out)
