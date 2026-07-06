"""Cross-language universality of the alpha~1 critical vocab.

Fetch a few Gutenberg texts per language (Latin/Cyrillic/CJK mix), measure
char-level alpha, byte alpha, and BPE V* (alpha~1). Tests whether 'critical
unit = alpha~1' generalizes across scripts/morphology.
Resumable: caches fetched text per language.
"""
import os
import json
import time
import urllib.request
import importlib.util
import numpy as np

_CACHE = r"F:\OpenASH2605\critical_tokenization\corpus_cache\crosslang"
os.makedirs(_CACHE, exist_ok=True)
_spec = importlib.util.spec_from_file_location("ct_tok", r"F:\OpenASH2605\critical_tokenization\tokenizers.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)

LANGS = {
    "de": [5200, 2229, 2591, 6327, 33667],
    "fr": [135, 4650, 19942, 30163, 13951],
    "es": [2000, 57270, 14420, 64060],
    "ru": [28774, 22574, 30716, 59506],
    "it": [1012, 35690, 47487],
    "pt": [33670, 55852, 39065],
}


def strip(raw):
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


def fetch_lang(lang, ids, target=4_000_000):
    cache = os.path.join(_CACHE, f"{lang}.txt")
    if os.path.exists(cache) and os.path.getsize(cache) > target * 0.5:
        return open(cache, encoding="utf-8").read()[:target]
    parts = []
    total = 0
    for bid in ids:
        if total >= target:
            break
        for url in (f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt",
                    f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt"):
            try:
                with urllib.request.urlopen(url, timeout=20) as r:
                    raw = r.read().decode("utf-8", errors="ignore")
                break
            except Exception:
                raw = ""
        if not raw:
            continue
        t = strip(raw)
        if len(t) < 2000:
            continue
        parts.append(t)
        total += len(t)
        time.sleep(0.2)
    text = "\n\n".join(parts)[:target]
    if len(text) > 100000:
        open(cache, "w", encoding="utf-8").write(text)
    return text


def za(counts):
    f = np.array(sorted(counts.values(), reverse=True), dtype=float)
    if len(f) < 10:
        return float("nan")
    tot = f.sum()
    cum = np.cumsum(f) / tot
    R = max(10, min(int(np.searchsorted(cum, 0.95)) + 1, len(f)))
    return float(-np.polyfit(np.log(np.arange(1, R + 1)), np.log(f[:R]), 1)[0])


def cz(ids):
    c = {}
    for i in ids:
        c[i] = c.get(i, 0) + 1
    return c


def measure(text):
    out = {}
    if len(text) < 100000:
        return None
    bpe_train = text[:1_500_000]
    val = text[-400_000:]
    ct = bt.CharTokenizer(text)
    out["distinct_chars"] = len(set(text))
    out["char_V"] = ct.actual_vocab_size
    out["char_alpha"] = za(cz(ct.encode(val)))
    out["byte_alpha"] = za(cz([b + 1 for b in val.encode("utf-8")]))
    m = bt.BpeTokenizer(bpe_train, 8000)
    M = len(m.merges)
    base = len(m.base_chars)
    best_v, best_a = None, 9
    sweep = []
    for nm in [0, 500, 1000, 2000, 3500, max(0, M - 1)]:
        if nm > M:
            continue
        e = m if nm >= M else m.restrict_to(range(nm))
        a = za(cz(e.encode(val)))
        V = base + nm + 1
        sweep.append((V, round(a, 3)))
        if abs(a - 1.0) < best_a:
            best_a, best_v = abs(a - 1.0), V
    out["bpe_sweep"] = sweep
    out["vstar_bpe"] = best_v
    out["vstar_alpha"] = round(1.0 - best_a + 1.0, 3) if best_v else None
    return out


def main():
    results = {}
    for lang, ids in LANGS.items():
        print(f"\n== {lang} ==", flush=True)
        text = fetch_lang(lang, ids)
        print(f"  fetched {len(text):,} chars", flush=True)
        r = measure(text)
        if r is None:
            print("  too little data, skip", flush=True)
            continue
        results[lang] = r
        print(f"  distinct_chars={r['distinct_chars']} char_V={r['char_V']} char_a={r['char_alpha']:.3f} byte_a={r['byte_alpha']:.3f}", flush=True)
        print(f"  BPE V*={r['vstar_bpe']}  sweep={r['bpe_sweep']}", flush=True)
        json.dump(results, open(os.path.join(_CACHE, "results.json"), "w"), indent=2)
    print("\n==== SUMMARY ====")
    for lang, r in results.items():
        print(f"  {lang}: char_a={r['char_alpha']:.3f} (V={r['char_V']})  byte_a={r['byte_alpha']:.3f}  BPE V*={r['vstar_bpe']}")


if __name__ == "__main__":
    main()
