"""Measurable quantities behind the critical-tokenization framework.

Operationalization of the framework's abstract terms:
  D_T   -> Higuchi fractal dimension of the frequency-RANK token sequence
           (ranks are ordinal/meaningful, unlike arbitrary token IDs),
           plus the power-law exponent gamma of mutual information decay
           I(d) ~ d^-gamma (the classic long-range-correlation criticality test).
  alpha_T -> Zipf exponent of token rank-frequency (log-log regression slope).
  L(z)  -> n-gram (order-k) cross-entropy on a held-out split, reported as
           bits-per-character (BPC) = bits/token / chars-per-token.
  L(T)  -> vocab description cost = |V| * log2(|V_base|) bits.
  delta_critic = |D_T - D_lang| + |alpha_T - 1|   (D_lang estimated empirically).
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Sequence
import numpy as np


def freq_rank_sequence(ids: Sequence[int], counts: Counter) -> np.ndarray:
    rank_of = {}
    for r, (tok, _) in enumerate(counts.most_common(), start=1):
        rank_of[tok] = r
    return np.fromiter((rank_of.get(t, len(rank_of) + 1) for t in ids), dtype=np.float64)


def zipf_alpha(counts: Counter, mass: float = 0.95):
    items = counts.most_common()
    freqs = np.array([c for _, c in items], dtype=np.float64)
    total = freqs.sum()
    cum = np.cumsum(freqs) / total
    R = int(np.searchsorted(cum, mass)) + 1
    R = max(10, min(R, len(freqs)))
    ranks = np.arange(1, R + 1, dtype=np.float64)
    f = freqs[:R]
    mask = f > 0
    lf, lr = np.log(f[mask]), np.log(ranks[mask])
    slope, intercept = np.polyfit(lr, lf, 1)
    pred = slope * lr + intercept
    ss_res = np.sum((lf - pred) ** 2)
    ss_tot = np.sum((lf - lf.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": float(-slope), "r2": float(r2), "R": R, "n_types": len(freqs)}


def higuchi_fd(data: Sequence[float], kmax: int = 32) -> float:
    x = np.asarray(data, dtype=np.float64)
    N = len(x)
    if N < kmax * 2:
        kmax = max(2, N // 2)
    Lk = np.zeros(kmax)
    for k in range(1, kmax + 1):
        Lm = 0.0
        for m in range(k):
            n_max = (N - m - 1) // k
            if n_max < 1:
                continue
            idx = m + np.arange(1, n_max + 1) * k
            ll = np.sum(np.abs(np.diff(x[idx])))
            ll = (ll * (N - 1)) / (n_max * k)
            Lm += ll
        Lk[k - 1] = Lm / k
    ks = np.arange(1, kmax + 1, dtype=np.float64)
    slope = np.polyfit(np.log(ks), np.log(Lk + 1e-12), 1)[0]
    return float(-slope)


def mutual_information(ids: Sequence[int], max_lag: int = 16, vocab_size: int = 0):
    arr = np.asarray(ids, dtype=np.int64)
    N = len(arr)
    if vocab_size <= 0:
        vocab_size = int(arr.max()) + 1
    uniq, ucnt = np.unique(arr, return_counts=True)
    p1 = np.ones(vocab_size + 1) * 1e-12
    p1[uniq] = ucnt / N
    mis = []
    for d in range(1, max_lag + 1):
        a = arr[: N - d]
        b = arr[d:]
        key = a * (vocab_size + 1) + b
        kk, cc = np.unique(key, return_counts=True)
        denom = N - d
        pa = p1[a]
        pb = p1[b]
        pab = cc / denom
        idx_a = kk // (vocab_size + 1)
        idx_b = kk % (vocab_size + 1)
        ratio = pab / (p1[idx_a] * p1[idx_b])
        mi = float(np.sum(pab * np.log2(ratio)))
        mis.append(max(mi, 1e-9))
    mis = np.array(mis)
    lags = np.arange(1, max_lag + 1, dtype=np.float64)
    mask = mis > 1e-8
    if mask.sum() >= 3:
        gamma = -np.polyfit(np.log(lags[mask]), np.log(mis[mask]), 1)[0]
    else:
        gamma = float("nan")
    return {"mi": mis, "lags": lags, "gamma": float(gamma)}


def block_entropy(ids: Sequence[int], max_n: int = 5):
    arr = list(ids)
    N = len(arr)
    out = []
    for n in range(1, max_n + 1):
        grams = Counter(tuple(arr[i : i + n]) for i in range(N - n + 1))
        total = sum(grams.values())
        H = -sum((c / total) * np.log2(c / total) for c in grams.values())
        out.append((n, float(H)))
    return out


def ngram_bpc(train_ids: Sequence[int], test_ids: Sequence[int], order: int = 2,
              vocab_size: int = 0, lam: float = 0.5,
              weights=(0.15, 0.30, 0.55)):
    train = list(train_ids)
    test = list(test_ids)
    if vocab_size <= 0:
        vocab_size = max(max(train), max(test)) + 1
    V = vocab_size
    uni = Counter(train)
    N = len(train)
    bi = Counter()
    bic = Counter()
    for i in range(1, len(train)):
        bi[(train[i - 1], train[i])] += 1
        bic[train[i - 1]] += 1
    tri = Counter()
    tric = Counter()
    for i in range(2, len(train)):
        tri[(train[i - 2], train[i - 1], train[i])] += 1
        tric[(train[i - 2], train[i - 1])] += 1
    w0, w1, w2 = weights
    log2 = np.log2
    bits = 0.0
    n = 0
    uni_get = uni.get
    bi_get = bi.get
    bic_get = bic.get
    tri_get = tri.get
    tric_get = tric.get
    for i in range(2, len(test)):
        w = test[i]
        h1 = test[i - 1]
        h0 = test[i - 2]
        p0 = (uni_get(w, 0) + lam) / (N + lam * V)
        p1 = (bi_get((h1, w), 0) + lam) / (bic_get(h1, 0) + lam * V)
        p2 = (tri_get((h0, h1, w), 0) + lam) / (tric_get((h0, h1), 0) + lam * V)
        p = w0 * p0 + w1 * p1 + w2 * p2
        bits += -log2(p)
        n += 1
    bits_per_token = bits / max(n, 1)
    return {"bits_per_token": float(bits_per_token), "n_eval": n}


def unigram_bpc(train_ids: Sequence[int], test_ids: Sequence[int], vocab_size: int = 0):
    train = list(train_ids)
    test = list(test_ids)
    if vocab_size <= 0:
        vocab_size = max(max(train), max(test)) + 1
    V = vocab_size
    uni = Counter(train)
    N = len(train)
    log2 = np.log2
    bits = 0.0
    for w in test:
        bits += -log2((uni.get(w, 0) + 0.5) / (N + 0.5 * V))
    return bits / max(len(test), 1)


def chars_per_token(text: str, ids: Sequence[int]) -> float:
    return len(text) / max(len(ids), 1)


def vocab_cost(vocab_size: int, base_vocab: int) -> float:
    return float(vocab_size * np.log2(max(base_vocab, 2)))
