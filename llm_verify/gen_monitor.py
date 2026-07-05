"""Verify gamma-based generation monitoring.

Train bpe1500 (V*), generate at a temperature spectrum (greedy->incoherent) on a
fixed seed batch, compute gamma (MI power-law decay) + local diversity metrics,
and compare against human / shuffled / uniform baselines.

Key test: at high T, local distinct-n stays HIGH (looks diverse) but text is
incoherent -- if gamma reveals the lost long-range structure, it catches what
local metrics miss.
"""
import os
import sys
import json
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
import common as C
from frsmash_v36 import FRSMASHv36

CACHE = os.path.join(C.WORK, "caches_en")
OUT = os.path.join(C.WORK, "gen_monitor")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
SEQ = 512
BATCH = 32
STEPS = 1000
LAYERS = 6
HEADS = 4
H = 256
LR = 5e-4
LOG2E = math.log2(math.e)
TEMPS = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
GEN_BATCH = 64
SEED_LEN = 32
GEN_LEN = 2048


def train_and_save(V, train_cpu, path):
    model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=4).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    ids = train_cpu.to(DEV)
    t0 = time.time()
    for step in range(1, STEPS + 1):
        cur = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * step / STEPS)))
        for pg in opt.param_groups:
            pg["lr"] = cur
        starts = torch.randint(0, len(ids) - SEQ - 1, (BATCH,))
        seqs = torch.stack([ids[s:s + SEQ + 1] for s in starts])
        x, y = seqs[:, :-1].long(), seqs[:, 1:].long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
            loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"  train s{step}/{STEPS} loss={loss.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), path)
    print(f"  saved {path}", flush=True)
    return model


@torch.no_grad()
def generate(model, seed, n_steps, temperature, V):
    B = seed.size(0)
    dt = model.head.weight.dtype
    model.eval()
    states = [None] * model.num_ssm
    h_slow = torch.zeros(B, model.D, device=DEV, dtype=dt)
    recall_state = None
    pos = 0
    tok = seed[:, 0:1].long()
    for i in range(SEED_LEN):
        logits, states, h_slow, recall_state, pos = model.generate_step(
            tok, states, h_slow, recall_state, pos)
        tok = seed[:, i + 1:i + 2].long()
    out = []
    for _ in range(n_steps):
        logits, states, h_slow, recall_state, pos = model.generate_step(
            tok, states, h_slow, recall_state, pos)
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits.float() / max(temperature, 1e-3), -1)
            nxt = torch.multinomial(probs, 1)
        out.append(nxt)
        tok = nxt
    return torch.cat(out, 1).cpu().numpy().reshape(-1)


def compute_gamma(arr, max_lag=16):
    arr = np.asarray(arr, dtype=np.int64)
    N = len(arr)
    V = int(arr.max()) + 1
    uniq, ucnt = np.unique(arr, return_counts=True)
    p1 = np.ones(V + 2) * 1e-12
    p1[uniq] = ucnt / N
    mis = []
    for d in range(1, max_lag + 1):
        a = arr[: N - d]
        b = arr[d:]
        key = a * (V + 2) + b
        kk, cc = np.unique(key, return_counts=True)
        denom = N - d
        pab = cc / denom
        idx_a = kk // (V + 2)
        idx_b = kk % (V + 2)
        ratio = pab / (p1[idx_a] * p1[idx_b])
        mi = float(np.sum(pab * np.log2(ratio)))
        mis.append(max(mi, 1e-9))
    mis = np.array(mis)
    lags = np.arange(1, max_lag + 1, dtype=float)
    mask = mis > 1e-8
    gamma = float(-np.polyfit(np.log(lags[mask]), np.log(mis[mask]), 1)[0]) if mask.sum() >= 3 else float("nan")
    return gamma, mis.tolist()


def local_metrics(arr):
    arr = list(arr)
    n = len(arr)
    def distinct(k):
        grams = [tuple(arr[i:i+k]) for i in range(n-k+1)]
        return len(set(grams)) / max(len(grams), 1)
    u, c = np.unique(arr, return_counts=True)
    p = c / c.sum()
    H = float(-(p * np.log2(p)).sum())
    grams4 = [tuple(arr[i:i+4]) for i in range(n-3)]
    rep4 = 1 - len(set(grams4)) / max(len(grams4), 1)
    return dict(distinct1=distinct(1), distinct2=distinct(2), entropy=H, rep4=rep4)


def main():
    meta = json.load(open(os.path.join(CACHE, "meta.json")))["bpe1500"]
    V = meta["vocab"]
    data = torch.load(os.path.join(CACHE, "bpe1500.pt"), weights_only=False)
    train_cpu, val_cpu = data["train"], data["val"]
    ckpt = os.path.join(OUT, "bpe1500.pth")
    if os.path.exists(ckpt):
        model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=4).to(DEV)
        model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True))
        print("loaded ckpt", flush=True)
    else:
        print("training bpe1500...", flush=True)
        model = train_and_save(V, train_cpu, ckpt)

    val_ids = val_cpu.numpy().astype(np.int64)
    torch.manual_seed(0)
    starts = torch.randint(0, len(train_cpu) - SEED_LEN - GEN_LEN - 2, (GEN_BATCH,))
    seed = torch.stack([train_cpu[s:s + SEED_LEN + 1] for s in starts]).long().to(DEV)

    results = {}
    print("\n== baselines ==", flush=True)
    g_h, _ = compute_gamma(val_ids)
    lm_h = local_metrics(val_ids)
    rng = np.random.default_rng(0)
    shuf = val_ids.copy(); rng.shuffle(shuf)
    g_s, _ = compute_gamma(shuf)
    uni = rng.integers(0, V, len(val_ids))
    g_u, _ = compute_gamma(uni)
    results["human"] = dict(gamma=g_h, **lm_h)
    results["shuffled"] = dict(gamma=g_s, **local_metrics(shuf))
    results["uniform"] = dict(gamma=g_u, **local_metrics(uni))
    print(f"  human    gamma={g_h:.3f} d1={lm_h['distinct1']:.3f} d2={lm_h['distinct2']:.3f} ent={lm_h['entropy']:.2f} rep4={lm_h['rep4']:.3f}", flush=True)
    print(f"  shuffled gamma={g_s:.3f}  uniform gamma={g_u:.3f}", flush=True)

    print("\n== generation @ various T ==", flush=True)
    for T in TEMPS:
        t0 = time.time()
        gen = generate(model, seed, GEN_LEN, T, V)
        g, mis = compute_gamma(gen)
        lm = local_metrics(gen)
        label = "greedy" if T == 0 else f"T{T}"
        results[label] = dict(T=T, gamma=g, n=len(gen), **lm)
        print(f"  {label:7s} gamma={g:.3f} d1={lm['distinct1']:.3f} d2={lm['distinct2']:.3f} ent={lm['entropy']:.2f} rep4={lm['rep4']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwritten", os.path.join(OUT, "results.json"))
    print("\nKEY TEST -- high T: does gamma drop while distinct stays high?")
    for lab in ["T1.5", "T2.0"]:
        if lab in results:
            r = results[lab]
            print(f"  {lab}: gamma={r['gamma']:.3f} (human {g_h:.3f})  d1={r['distinct1']:.3f} (human {lm_h['distinct1']:.3f})  rep4={r['rep4']:.3f}")


if __name__ == "__main__":
    main()
