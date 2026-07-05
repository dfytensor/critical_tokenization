"""Shared utils for the LLM critical-tokenization verification (Chinese corpus).

Tokenizers span fine->coarse with char (alpha~1.0) as the INTERIOR critical point:
  byte (UTF-8 bytes)  <  char (alpha=1.0)  <  bpe (alpha<1)
so a U-shaped BPC-vs-alpha curve would support the critical-tokenization thesis.
"""
import os
import sys
import json
import importlib.util
import numpy as np
import torch


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CT_TOK_PATH = r"F:\OpenASH2605\critical_tokenization\tokenizers.py"
base_tok = _load_module(_CT_TOK_PATH, "ct_tokenizers_internal")

CN_JSONL = r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"
WORK = r"F:\OpenASH2605\critical_tokenization\llm_verify"
CACHE_DIR = os.path.join(WORK, "caches")
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.insert(0, r"F:\rwkv\frsmash_v36")
from frsmash_v36 import FRSMASHv36


class ByteTokenizer:
    name = "byte"

    def __init__(self, text: str):
        self.actual_vocab_size = 257

    def encode(self, text: str):
        return [b + 1 for b in text.encode("utf-8")]


def load_cn_text(n_chars):
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
            if not t:
                continue
            parts.append(t)
            total += len(t)
            if total >= n_chars:
                break
    return "\n".join(parts)[:n_chars]


EN_FILE = r"F:\OpenASH2605\critical_tokenization\llm_verify\en_large.txt"


def load_en_text(n_chars):
    with open(EN_FILE, encoding="utf-8") as f:
        return f.read()[:n_chars]


def model_total_params(vocab, hidden, layers, heads=4, n_slots=4):
    m = FRSMASHv36(vocab, hidden, heads, layers, n_slots=n_slots)
    del m
    return None


def _count_params(vocab, hidden, layers, heads=4, n_slots=4):
    m = FRSMASHv36(vocab, hidden, heads, layers, n_slots=n_slots)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


def pick_hidden(vocab, target=8_000_000, layers=4, heads=4, h_grid=None):
    if h_grid is None:
        h_grid = [128, 160, 192, 224, 256, 288, 320, 352]
    best_h, best_d = None, 1e18
    best_n = 0
    for h in h_grid:
        if h % heads != 0:
            continue
        n = _count_params(vocab, h, layers, heads)
        d = abs(n - target)
        if d < best_d:
            best_d, best_h, best_n = d, h, n
    return best_h, best_n


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
    return float(-slope)
