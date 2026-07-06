"""Measure real V* (alpha~1 critical vocab) on a LARGE Chinese corpus (minimind, 50M chars).

Tests whether the 'Chinese critical unit = char, alpha~1' finding holds at scale,
and what the real V* number is. Pure CPU (numpy + stdlib tokenizers).
"""
import os
import sys
import json
import importlib.util
import numpy as np

_CT = r"F:\OpenASH2605\critical_tokenization\tokenizers.py"
_spec = importlib.util.spec_from_file_location("ct_tok", _CT)
base_tok = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base_tok)

CN_JSONL = r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"


def load_cn(n):
    parts = []
    total = 0
    with open(CN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except Exception:
                continue
            if t:
                parts.append(t)
                total += len(t)
                if total >= n:
                    break
    return "\n".join(parts)[:n]


def zipf_alpha(counts):
    items = sorted(counts.values(), reverse=True)
    freqs = np.array(items, dtype=np.float64)
    total = freqs.sum()
    cum = np.cumsum(freqs) / total
    R = int(np.searchsorted(cum, 0.95)) + 1
    R = max(10, min(R, len(freqs)))
    ranks = np.arange(1, R + 1, dtype=np.float64)
    lf, lr = np.log(freqs[:R]), np.log(ranks)
    slope = np.polyfit(lr, lf, 1)[0]
    ss_res = np.sum((lf - (slope * lr + np.polyfit(lr, lf, 1)[1])) ** 2)
    ss_tot = np.sum((lf - lf.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return float(-slope), float(r2)


def count_zipf(ids):
    c = {}
    for i in ids:
        c[i] = c.get(i, 0) + 1
    return c


def main():
    N = 50_000_000
    print(f"loading {N/1e6:.0f}M Chinese chars...", flush=True)
    text = load_cn(N)
    bpe_train = text[:5_000_000]
    val = text[-500_000:]
    print(f"  loaded {len(text):,} chars; distinct chars in full = {len(set(text))}", flush=True)

    print("\n== char-level ==", flush=True)
    ct = base_tok.CharTokenizer(text)
    ids = ct.encode(val)
    a, r2 = zipf_alpha(count_zipf(ids))
    print(f"  char vocab={ct.actual_vocab_size}  alpha={a:.3f} (r2={r2:.3f})  cpt={len(val)/len(ids):.3f}", flush=True)

    print("\n== byte-level (UTF-8) ==", flush=True)
    b_ids = [b + 1 for b in val.encode("utf-8")]
    a_b, r2_b = zipf_alpha(count_zipf(b_ids))
    print(f"  byte vocab=257  alpha={a_b:.3f} (r2={r2_b:.3f})  cpt={len(val)/len(b_ids):.3f}", flush=True)

    print("\n== BPE above char: sweep vocab, find V* (alpha~1) ==", flush=True)
    master = base_tok.BpeTokenizer(bpe_train, 18000)
    M = len(master.merges)
    base = len(master.base_chars)
    print(f"  master: base_chars={base} merges={M} (V_max={master.actual_vocab_size})", flush=True)
    levels = sorted(set([0, 500, 1000, 2000, 3000, 5000, 8000, max(0, M - 1)]))
    levels = [l for l in levels if l <= M]
    rows = []
    for nmerge in levels:
        enc = master if nmerge >= M else master.restrict_to(range(nmerge))
        v_ids = enc.encode(val)
        a, r2 = zipf_alpha(count_zipf(v_ids))
        cpt = len(val) / max(len(v_ids), 1)
        V = base + nmerge + 1
        rows.append((V, a, r2, cpt, nmerge))
        print(f"  merges={nmerge:5d}  V={V:6d}  alpha={a:.3f} (r2={r2:.3f})  cpt={cpt:.3f}", flush=True)

    char_a, _ = zipf_alpha(count_zipf(ct.encode(val)))
    best = min(rows, key=lambda r: abs(r[1] - 1.0))
    print(f"\n>> char-level: V={ct.actual_vocab_size}  alpha={char_a:.3f}", flush=True)
    print(f">> BPE V* (alpha closest to 1.0): V={best[0]}  alpha={best[1]:.3f}  (merges={best[4]})", flush=True)
    print(f">> byte-level alpha={a_b:.3f}", flush=True)

    out = dict(N=N, distinct_chars=len(set(text)), char_vocab=ct.actual_vocab_size,
               char_alpha=char_a, byte_alpha=a_b, rows=rows, vstar_bpe=best[0])
    with open(r"F:\OpenASH2605\critical_tokenization\results\real_vstar_cn.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved results/real_vstar_cn.json")


if __name__ == "__main__":
    main()
