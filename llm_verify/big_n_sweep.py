"""Extend V-N law to 50M/100M/150M. Resumable (skips done runs), saves each run
immediately so partial results survive a timeout.

Must extend the V grid upward (at 32M, V_opt already hit the 6000 ceiling), so we
train a bigger master BPE (~16k) and test V in {1500,6000,10000,bpe_max}.
"""
import os
import sys
import json
import time
import math
import glob
import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
import common as C
from frsmash_v36 import FRSMASHv36

OUT = os.path.join(C.WORK, "big_n_sweep")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
SEQ = 512
MICRO = 4
ACCUM = 8
BATCH = MICRO * ACCUM
STEPS = 600
EVAL_EVERY = 300
LAYERS = 8
HEADS = 8
LR = 5e-4
LOG2E = math.log2(math.e)
H_GRID = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]

N_TARGETS = [50e6, 100e6, 150e6]


def pick_hidden(V, target):
    best_h, best_d, best_n = None, 1e18, 0
    for h in H_GRID:
        if h % HEADS != 0 or h % 4 != 0:
            continue
        m = FRSMASHv36(V, h, HEADS, LAYERS, n_slots=4)
        n = sum(p.numel() for p in m.parameters())
        del m
        emb = 2 * V * h
        if emb / n > 0.30:
            continue
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
        loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1), reduction="sum")
        tl += loss.item(); tt += y.numel()
    model.train()
    return (tl / max(tt, 1)) * LOG2E / cpt


def run_one(tag, V, H, nparams, cpt, train_cpu, val_cpu):
    path = os.path.join(OUT, f"{tag}.json")
    if os.path.exists(path):
        return json.load(open(path))["final_bpc"]
    print(f"  [{tag}] V={V} H={H} P={nparams/1e6:.1f}M cpt={cpt:.3f} ...", flush=True)
    model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=4).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    ids = train_cpu.to(DEV)
    t0 = time.time()
    last_bpc = None
    for step in range(1, STEPS + 1):
        cur = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * step / STEPS)))
        for pg in opt.param_groups:
            pg["lr"] = cur
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            starts = torch.randint(0, len(ids) - SEQ - 1, (MICRO,))
            seqs = torch.stack([ids[s:s + SEQ + 1] for s in starts])
            x, y = seqs[:, :-1].long(), seqs[:, 1:].long()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = model(x)
                loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1)) / ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            torch.cuda.empty_cache()
        if step % EVAL_EVERY == 0 or step == STEPS:
            torch.cuda.empty_cache()
            last_bpc = eval_bpc(model, val_cpu, cpt, V)
            print(f"    {tag} s{step}/{STEPS} loss={loss.item()*ACCUM:.3f} bpc={last_bpc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    rec = dict(tag=tag, V=V, H=H, params=nparams, cpt=cpt, final_bpc=last_bpc)
    json.dump(rec, open(path, "w"), indent=2)
    del model, ids
    torch.cuda.empty_cache()
    return last_bpc


def main():
    raw = C.load_en_text(19_000_000)
    train_text, val_text = raw[:18_000_000], raw[18_000_000:19_000_000]
    mpath = os.path.join(OUT, "master_bpe.json")
    master = C.base_tok.BpeTokenizer(train_text, 16000)
    M = len(master.merges)
    print(f"master BPE: {M} merges (V={master.actual_vocab_size})", flush=True)
    levels = {"bpe1500": 1500, "bpe6000": 6000, "bpe10000": 10000}
    if M > 11000:
        levels[f"bpe{M}"] = M
    enc = {k: (master if v >= M else master.restrict_to(range(v))) for k, v in levels.items()}

    caches = {}
    for k, enc_ in enc.items():
        tr = torch.tensor(enc_.encode(train_text), dtype=torch.int32)
        va = torch.tensor(enc_.encode(val_text), dtype=torch.int32)
        cpt = len(val_text) / max(len(va), 1)
        caches[k] = (tr, va, cpt, enc_.actual_vocab_size)
        print(f"  cache {k}: train_tok={len(tr):,} cpt={cpt:.3f} V={caches[k][3]}", flush=True)

    summary = {}
    for N in N_TARGETS:
        nM = int(N / 1e6)
        print(f"\n========= N_target = {nM}M =========", flush=True)
        row = {}
        for k in levels:
            if N >= 100e6 and k == "bpe1500":
                continue
            tr, va, cpt, V = caches[k]
            H, nparams = pick_hidden(V, N)
            if H is None:
                print(f"  [{k}] no valid H (embedding too big), skip", flush=True)
                continue
            tag = f"N{nM}M_{k}"
            bpc = run_one(tag, V, H, nparams, cpt, tr, va)
            row[k] = (V, bpc)
            summary[f"N{nM}M"] = {kk: {"V": vv[0], "bpc": vv[1]} for kk, vv in row.items()}
            json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
        if row:
            vopt = min(row, key=lambda k: row[k][1])
            print(f"  >> N={nM}M  V_opt={vopt} (V={row[vopt][0]})  " + " ".join(f"{k}:{row[k][1]:.3f}" for k in row), flush=True)
    print("\n==== V_opt(N) [new big-scale points] ====")
    for nk, rd in summary.items():
        if rd:
            vopt = min(rd, key=lambda k: rd[k]["bpc"])
            print(f"  {nk}: V_opt={vopt} V={rd[vopt]['V']}  " + " ".join(f"{k}:{rd[k]['bpc']:.3f}" for k in rd))


if __name__ == "__main__":
    main()
