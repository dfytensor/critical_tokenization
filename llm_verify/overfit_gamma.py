"""③ gamma drift under real training-stage degradation (RLHF-overfit proxy).

Load bpe1500, measure gamma at T=0.8 (baseline). Then OVERFIT-train it on a tiny
subset (mode-collapse proxy), re-measure gamma. If gamma moves with overfitting
(beyond what temperature alone shows), gamma tracks real training degradation.
"""
import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
import common as C
from frsmash_v36 import FRSMASHv36
import gen_monitor as G

CACHE = os.path.join(C.WORK, "caches_en")
OUT = os.path.join(C.WORK, "overfit_gamma")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
LAYERS = 6; HEADS = 4; H = 256
V = None


def overfit_steps(model, ids, n_steps, subset_len):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    sub = ids[:subset_len].to(DEV)
    SEQ = G.SEQ
    model.train()
    for step in range(n_steps):
        st = torch.randint(0, len(sub) - SEQ - 1, (G.GEN_BATCH,))
        sq = torch.stack([sub[s:s + SEQ + 1] for s in st])
        x, y = sq[:, :-1].long(), sq[:, 1:].long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
            loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"    overfit s{n_steps} loss={loss.item():.3f}", flush=True)


def gen_gamma(model, seed, tag):
    out = {}
    for T in [0.0, 0.8]:
        gen = G.generate(model, seed, 1024, T, V)
        g, _ = G.compute_gamma(gen)
        lm = G.local_metrics(gen)
        label = "greedy" if T == 0 else "T0.8"
        out[label] = dict(gamma=g, **lm)
        print(f"    {tag} {label}: gamma={g:.3f} d2={lm['distinct2']:.3f} rep4={lm['rep4']:.3f} ent={lm['entropy']:.2f}", flush=True)
    return out


def main():
    global V
    meta = json.load(open(os.path.join(CACHE, "meta.json")))["bpe1500"]
    V = meta["vocab"]
    data = torch.load(os.path.join(CACHE, "bpe1500.pt"), weights_only=False)
    model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=4).to(DEV)
    model.load_state_dict(torch.load(os.path.join(C.WORK, "gen_monitor", "bpe1500.pth"), map_location=DEV, weights_only=True))
    print("loaded bpe1500 ckpt", flush=True)

    torch.manual_seed(0)
    starts = torch.randint(0, len(data["train"]) - 3000, (G.GEN_BATCH,))
    seed = torch.stack([data["train"][s:s + G.SEED_LEN + 1] for s in starts]).long().to(DEV)

    print("\n== baseline (pre-overfit) ==", flush=True)
    base = gen_gamma(model, seed, "baseline")

    print("\n== overfit-training on tiny subset (mode-collapse proxy) ==", flush=True)
    overfit_steps(model, data["train"], 300, 40000)

    print("\n== post-overfit ==", flush=True)
    post = gen_gamma(model, seed, "post-overfit")

    res = dict(baseline=base, post_overfit=post)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print("\n==== gamma drift (T0.8) ====")
    print(f"  baseline gamma={base['T0.8']['gamma']:.3f}  ->  post-overfit gamma={post['T0.8']['gamma']:.3f}")
    print(f"  baseline rep4={base['T0.8']['rep4']:.3f}   ->  post-overfit rep4={post['T0.8']['rep4']:.3f}")
    dg = post['T0.8']['gamma'] - base['T0.8']['gamma']
    print(f"\nverdict: gamma drift = {dg:+.3f}  (nonzero => gamma tracks real overfit degradation, beyond temperature)")


if __name__ == "__main__":
    main()
