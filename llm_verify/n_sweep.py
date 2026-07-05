"""N-sweep: does the BPC-optimal vocabulary V_opt drift up with model size N?

Reuses caches_en (char/bpe500/bpe1500/bpe3000/bpe6000). For each N target and
each tokenizer, pick hidden H (layers=6 fixed) to hit ~N params, train 1000
steps, record final val BPC. argmin_V BPC = V_opt(N).

Prediction from the embedding-tax argument: V_opt(N) rises SUBLINEARLY with N
(anchored at V* for small N, ceiling ~ fN/2d). If V_opt is ~flat -> anchor
dominates; if ~N^(2/3) -> embedding tax dominates.
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

CACHE = os.path.join(C.WORK, "caches_en")
OUT = os.path.join(C.WORK, "n_sweep")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
SEQ = 512
BATCH = 32
STEPS = 1000
EVAL_EVERY = 250
LAYERS = 6
HEADS = 4
LR = 5e-4
LOG2E = math.log2(math.e)
H_GRID = [128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640]

N_TARGETS = [4e6, 12e6, 32e6]
TOKS = ["char", "bpe500", "bpe1500", "bpe3000", "bpe6000"]


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
    starts = list(range(0, n - SEQ - 1, SEQ))[:512]
    tl = 0.0
    tt = 0
    for i in range(0, len(starts), 64):
        idxs = starts[i:i + 64]
        seqs = torch.stack([val_ids[s:s + SEQ + 1] for s in idxs])
        x = seqs[:, :-1].long().to(DEV)
        y = seqs[:, 1:].long().to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
        loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1), reduction="sum")
        tl += loss.item()
        tt += y.numel()
    model.train()
    return (tl / max(tt, 1)) * LOG2E / cpt


def run(N_target, tok_name, meta, train_cpu, val_cpu):
    V = meta["vocab"]
    cpt = meta["cpt"]
    H, nparams = pick_hidden(V, N_target)
    tag = f"N{int(N_target/1e6)}M_{tok_name}"
    model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=4).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    train_ids = train_cpu.to(DEV)
    t0 = time.time()
    log = []
    for step in range(1, STEPS + 1):
        cur_lr = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * step / STEPS)))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        starts = torch.randint(0, len(train_ids) - SEQ - 1, (BATCH,))
        seqs = torch.stack([train_ids[s:s + SEQ + 1] for s in starts])
        x, y = seqs[:, :-1].long(), seqs[:, 1:].long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
            loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0 or step == STEPS:
            bpc = eval_bpc(model, val_cpu, cpt, V)
            log.append(dict(step=step, train_loss=loss.item(), bpc=bpc))
            print(f"  {tag} H={H} P={nparams/1e6:.2f}M s{step}/{STEPS} bpc={bpc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    final = log[-1]["bpc"]
    rec = dict(N_target=N_target, N_actual=nparams, tok=tok_name, V=V, alpha=meta["alpha"],
               cpt=cpt, hidden=H, final_bpc=final, log=log)
    with open(os.path.join(OUT, f"{tag}.json"), "w") as f:
        json.dump(rec, f, indent=2)
    del model, train_ids
    torch.cuda.empty_cache()
    return rec


def main():
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    data = {t: torch.load(os.path.join(CACHE, f"{t}.pt"), weights_only=False) for t in TOKS}
    results = {}
    for N_target in N_TARGETS:
        print(f"\n========= N_target = {N_target/1e6:.0f}M =========", flush=True)
        row = {}
        for t in TOKS:
            rec = run(N_target, t, meta[t], data[t]["train"], data[t]["val"])
            row[t] = rec["final_bpc"]
            results.setdefault("runs", []).append(rec)
        v_opt = min(row, key=row.get)
        print(f"  >> N={N_target/1e6:.0f}M  V_opt={v_opt} (V={meta[v_opt]['vocab']}, alpha={meta[v_opt]['alpha']:.3f})", flush=True)
        results.setdefault("v_opt", []).append(dict(N=N_target, v_opt=v_opt, V=meta[v_opt]["vocab"], row=row))
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(results, f, indent=2)
    print("\n==== V_opt(N) ====")
    for r in results["v_opt"]:
        print(f"  N={r['N']/1e6:>4.0f}M  V_opt={r['v_opt']:8s} V={r['V']:5d}  row=" + " ".join(f"{k}:{v:.3f}" for k, v in r["row"].items()))


if __name__ == "__main__":
    main()
