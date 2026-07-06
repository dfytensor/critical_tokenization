"""Multi-seed error bars on the V-N exponent (4-32M segment).

For each seed: train 5 tokenizers x {4M,12M,32M} (1000 steps, batch 32), find
V_opt(N) per seed, fit V_opt ~ N^alpha. Report alpha per seed + mean/std.
Resumable (skips done runs). alloc-conf to avoid fragmentation.
"""
import os
import sys
import json
import time
import math
import glob
import numpy as np
import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
import common as C
from frsmash_v36 import FRSMASHv36

CACHE = os.path.join(C.WORK, "caches_en")
OUT = os.path.join(C.WORK, "n_sweep_multiseed")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
SEQ = 512
BATCH = 32
STEPS = 1000
LAYERS = 6
HEADS = 4
LR = 5e-4
LOG2E = math.log2(math.e)
H_GRID = [128, 160, 192, 224, 256, 288, 320, 352]
N_TARGETS = [4e6, 12e6, 32e6]
TOKS = ["char", "bpe500", "bpe1500", "bpe3000", "bpe6000"]
V_OF = {"char": 155, "bpe500": 501, "bpe1500": 1501, "bpe3000": 3001, "bpe6000": 6001}
SEEDS = [0, 1, 2]


def pick_hidden(V, target):
    best_h, best_d, best_n = None, 1e18, 0
    for h in H_GRID:
        if h % HEADS != 0:
            continue
        m = FRSMASHv36(V, h, HEADS, LAYERS, n_slots=4)
        n = sum(p.numel() for p in m.parameters())
        del m
        d = abs(n - target)
        if d < best_d:
            best_d, best_h, best_n = d, h, n
    return best_h, best_n


def eval_bpc(model, val_ids, cpt, V):
    model.eval()
    n = len(val_ids)
    starts = list(range(0, n - SEQ - 1, SEQ))[:384]
    tl = tt = 0
    for i in range(0, len(starts), 16):
        idxs = starts[i:i + 16]
        seqs = torch.stack([val_ids[s:s + SEQ + 1] for s in idxs])
        x = seqs[:, :-1].long().to(DEV)
        y = seqs[:, 1:].long().to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
        tl += F.cross_entropy(o.reshape(-1, V), y.reshape(-1), reduction="sum").item()
        tt += y.numel()
    model.train()
    return (tl / max(tt, 1)) * LOG2E / cpt


def run_one(tag, V, H, cpt, train_cpu, val_cpu, seed):
    path = os.path.join(OUT, f"{tag}.json")
    if os.path.exists(path):
        return json.load(open(path))["bpc"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=4).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    ids = train_cpu.to(DEV)
    t0 = time.time()
    last = None
    for step in range(1, STEPS + 1):
        cur = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * step / STEPS)))
        for pg in opt.param_groups:
            pg["lr"] = cur
        st = torch.randint(0, len(ids) - SEQ - 1, (BATCH,))
        sq = torch.stack([ids[s:s + SEQ + 1] for s in st])
        x, y = sq[:, :-1].long(), sq[:, 1:].long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
            loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == STEPS:
            last = eval_bpc(model, val_cpu, cpt, V)
    json.dump(dict(tag=tag, seed=seed, V=V, H=H, cpt=cpt, bpc=last), open(path, "w"), indent=2)
    del model, ids
    torch.cuda.empty_cache()
    print(f"    {tag} bpc={last:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return last


def main():
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    data = {t: torch.load(os.path.join(CACHE, f"{t}.pt"), weights_only=False) for t in TOKS}
    summary = {}
    for seed in SEEDS:
        print(f"\n===== SEED {seed} =====", flush=True)
        vopt_per_N = {}
        for N in N_TARGETS:
            nM = int(N / 1e6)
            row = {}
            for t in TOKS:
                V = meta[t]["vocab"]
                cpt = meta[t]["cpt"]
                H, _ = pick_hidden(V, N)
                tag = f"s{seed}_N{nM}M_{t}"
                bpc = run_one(tag, V, H, cpt, data[t]["train"], data[t]["val"], seed)
                row[t] = bpc
            vopt = min(row, key=row.get)
            vopt_per_N[nM] = V_OF[vopt]
            print(f"  seed{seed} N={nM}M V_opt={vopt} (V={V_OF[vopt]})  " + " ".join(f"{k}:{row[k]:.3f}" for k in TOKS), flush=True)
            summary.setdefault(str(seed), {})[str(nM)] = dict(vopt=vopt, V=V_OF[vopt], row=row)
            json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
        Ns = np.array(sorted(vopt_per_N.keys()), dtype=float) * 1e6
        Vs = np.array([vopt_per_N[int(n / 1e6)] for n in Ns], dtype=float)
        exp = float(np.polyfit(np.log(Ns), np.log(Vs), 1)[0])
        summary[str(seed)]["exponent"] = exp
        print(f"  >> seed{seed} exponent = {exp:.3f}", flush=True)
        json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    exps = [summary[str(s)]["exponent"] for s in SEEDS if "exponent" in summary[str(s)]]
    if exps:
        print(f"\n==== EXPONENT across {len(exps)} seeds: {np.mean(exps):.3f} +/- {np.std(exps):.3f}  (values: {[round(e,3) for e in exps]})")


if __name__ == "__main__":
    main()
