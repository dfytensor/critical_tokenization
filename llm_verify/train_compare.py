"""Train FRSMASH v3.6 on each tokenizer's cache with EQUAL params & EQUAL steps,
log val BPC vs step (and vs chars-seen) to test the critical-tokenization thesis.

Per-tokenizer: params equalized via hidden (from meta). BPC = val_CE_bits / cpt.
"""
import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
import common as C
from frsmash_v36 import FRSMASHv36

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
CACHE_DIR = os.path.join(C.WORK, f"caches_{LANG}")
OUT_DIR = os.path.join(C.WORK, f"runs_{LANG}")
os.makedirs(OUT_DIR, exist_ok=True)

SEQ = 512
BATCH = 32
STEPS = 1000
EVAL_EVERY = 50
LR = 5e-4
HEADS = 4
LAYERS = 4
LOG2E = math.log2(math.e)
DEV = "cuda"


def make_batches(train_ids, n):
    starts = torch.randint(0, len(train_ids) - SEQ - 1, (n,))
    seqs = torch.stack([train_ids[s:s + SEQ + 1] for s in starts])
    return seqs[:, :-1].long(), seqs[:, 1:].long()


@torch.no_grad()
def eval_bpc(model, val_ids, cpt, batch=64, max_windows=1024):
    model.eval()
    n = len(val_ids)
    starts = list(range(0, n - SEQ - 1, SEQ))
    if len(starts) > max_windows:
        starts = starts[:max_windows]
    total_loss = 0.0
    total_tok = 0
    for i in range(0, len(starts), batch):
        idxs = starts[i:i + batch]
        seqs = torch.stack([val_ids[s:s + SEQ + 1] for s in idxs])
        x = seqs[:, :-1].long().to(DEV)
        y = seqs[:, 1:].long().to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
        loss = F.cross_entropy(o.reshape(-1, o.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tok += y.numel()
    model.train()
    bits_per_token = (total_loss / max(total_tok, 1)) * LOG2E
    return bits_per_token / cpt


def train_one(name, meta):
    V = meta["vocab"]
    H = meta["hidden"]
    cpt = meta["cpt"]
    print(f"\n##### {name}: V={V} H={H} cpt={cpt:.3f} alpha={meta['alpha']:.3f} P={meta['params']/1e6:.2f}M", flush=True)
    data = torch.load(os.path.join(CACHE_DIR, f"{name}.pt"), weights_only=False)
    train_ids = data["train"].to(DEV)
    val_ids = data["val"]
    model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=4).to(DEV)
    nparams = sum(p.numel() for p in model.parameters())
    assert abs(nparams - meta["params"]) / meta["params"] < 0.01, (nparams, meta["params"])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))

    log = []
    t0 = time.time()
    for pg in opt.param_groups:
        pg["lr"] = LR
    model.train()
    for step in range(1, STEPS + 1):
        cur_lr = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * step / STEPS)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        x, y = make_batches(train_ids, BATCH)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
            loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == 1:
            bpc = eval_bpc(model, val_ids, cpt)
            chars_seen = step * BATCH * SEQ * cpt
            el = time.time() - t0
            print(f"  {name} s{step}/{STEPS} train_loss={loss.item():.3f} val_BPC={bpc:.4f} chars={chars_seen/1e6:.1f}M ({el:.0f}s)", flush=True)
            log.append(dict(step=step, train_loss=loss.item(), val_bpc=bpc, chars_seen=chars_seen))
    final_bpc = eval_bpc(model, val_ids, cpt)
    with open(os.path.join(OUT_DIR, f"log_{name}.json"), "w") as f:
        json.dump(dict(name=name, vocab=V, hidden=H, cpt=cpt, alpha=meta["alpha"],
                        params=nparams, final_bpc=final_bpc, log=log), f, indent=2)
    del model, train_ids
    torch.cuda.empty_cache()
    print(f"  {name} FINAL BPC={final_bpc:.4f}", flush=True)
    return final_bpc


def main():
    with open(os.path.join(CACHE_DIR, "meta.json")) as f:
        meta = json.load(f)
    order = ["char", "bpe500", "bpe1500", "bpe3000", "bpe6000"] if LANG == "en" \
        else ["byte", "char", "bpe8000", "bpe12000", "bpe16000"]
    finals = {}
    for name in order:
        if name not in meta:
            continue
        finals[name] = train_one(name, meta[name])
    with open(os.path.join(OUT_DIR, "finals.json"), "w") as f:
        json.dump(finals, f, indent=2)
    print("\n==== FINAL val BPC ====")
    for k, v in sorted(finals.items(), key=lambda kv: kv[1]):
        print(f"  {k:9s} BPC={v:.4f}  alpha={meta[k]['alpha']:.3f}")


if __name__ == "__main__":
    main()
