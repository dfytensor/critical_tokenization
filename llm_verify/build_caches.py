"""Build per-tokenizer id caches from the SAME text (train/val split).

EN: BPE-family sweep with V*~1500 (alpha~1.0) INTERIOR -> clean U-test.
CN: byte/char/bpe sweep (char alpha~1.0).
Stores int32 train/val id tensors + meta (vocab, cpt, alpha, hidden, params).
"""
import os
import sys
import json
import time
from collections import Counter
import torch

sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
import common as C

LANGS = {
    "en": dict(
        loader=C.load_en_text,
        configs=[("char", dict(kind="char")),
                 ("bpe500", dict(kind="bpe", vocab=500)),
                 ("bpe1500", dict(kind="bpe", vocab=1500)),
                 ("bpe3000", dict(kind="bpe", vocab=3000)),
                 ("bpe6000", dict(kind="bpe", vocab=6000))],
        train_chars=18_000_000, val_chars=1_000_000, bpe_train=2_000_000),
    "cn": dict(
        loader=C.load_cn_text,
        configs=[("byte", dict(kind="byte")),
                 ("char", dict(kind="char")),
                 ("bpe8000", dict(kind="bpe", vocab=8000)),
                 ("bpe10000", dict(kind="bpe", vocab=10000)),
                 ("bpe12000", dict(kind="bpe", vocab=12000))],
        train_chars=18_000_000, val_chars=1_000_000, bpe_train=1_500_000),
}

TARGET_PARAMS = 8_000_000
LAYERS = 4
HEADS = 4


def make_tokenizer(kind, train_text, vocab, bpe_train):
    if kind == "byte":
        return C.ByteTokenizer(train_text)
    if kind == "char":
        return C.base_tok.CharTokenizer(train_text)
    if kind == "bpe":
        return C.base_tok.BpeTokenizer(train_text[:bpe_train], vocab)
    raise ValueError(kind)


def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    cfg = LANGS[lang]
    out_dir = os.path.join(C.WORK, f"caches_{lang}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"loading {lang} text...", flush=True)
    full = cfg["loader"](cfg["train_chars"] + cfg["val_chars"])
    train_text = full[:cfg["train_chars"]]
    val_text = full[cfg["train_chars"]:cfg["train_chars"] + cfg["val_chars"]]
    print(f"  train={len(train_text):,} val={len(val_text):,}", flush=True)

    summary = {}
    for name, tc in cfg["configs"]:
        t0 = time.time()
        print(f"\n=== {name} ({tc['kind']}) ===", flush=True)
        tok = make_tokenizer(tc["kind"], train_text, tc.get("vocab"), cfg["bpe_train"])
        V = tok.actual_vocab_size
        tr = tok.encode(train_text)
        print(f"  vocab={V} train_tokens={len(tr):,}", flush=True)
        va = tok.encode(val_text)
        cpt = len(val_text) / len(va)
        alpha = C.zipf_alpha(Counter(va))
        hidden, nparams = C.pick_hidden(V, TARGET_PARAMS, LAYERS, HEADS)
        print(f"  val_tokens={len(va):,} cpt={cpt:.3f} alpha={alpha:.3f} H={hidden} P={nparams/1e6:.2f}M", flush=True)
        torch.save({"train": torch.tensor(tr, dtype=torch.int32),
                    "val": torch.tensor(va, dtype=torch.int32)},
                   os.path.join(out_dir, f"{name}.pt"))
        summary[name] = dict(kind=tc["kind"], vocab=V, cpt=cpt, alpha=alpha,
                             hidden=hidden, params=nparams,
                             train_tokens=len(tr), val_tokens=len(va))
        print(f"  done {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nalpha by tokenizer:", {k: round(v["alpha"], 3) for k, v in summary.items()})
    print("->", out_dir)


if __name__ == "__main__":
    main()
