"""Verify the hybrid paradigm: CE + MI-curve-matching REINFORCE.

Blueprint test: can a structure-matching term (full I(d) curve, not just gamma),
anchored by CE, CONTROLLABLY drive gamma toward a target while keeping coherence?
- reward magnitude from I(d) curve is ~1-10 (vs ~0.001 for gamma-only) -> far
  stronger signal than the failed gamma-REINFORCE.
- CE anchor keeps the model in a coherent regime where gamma is meaningful.

3 conditions from the same SFT baseline:
  control : CE only (alpha=0)
  low     : CE + MI-REINFORCE, target I(d) = 1.5 x human  (more critical, lower gamma)
  high    : CE + MI-REINFORCE, target I(d) = 0.5 x human  (degraded, higher gamma)
If low -> gamma lower than control and high -> gamma higher than control, with rep4
stable => hybrid paradigm's premise (structure term controllable + coherent) VALIDATED.
"""
import os
import sys
import json
import math
import copy
import numpy as np
import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
sys.path.insert(0, r"F:\OpenASH2605")
os.chdir(r"F:\OpenASH2605")
import common as C
from frsmash_v36 import FRSMASHv36
import gen_monitor as G
import dpo_gamma as D
from open_ash_voc import OpenASHVoc

DEV = "cuda"
OUT = os.path.join(C.WORK, "critical_hybrid")
os.makedirs(OUT, exist_ok=True)
B = 8
L = 160
PL = 32
NSTEPS = 100
ALPHA = 0.5
MAXLAG = 8
CKPT = D.CKPT


def I_curve(arr, max_lag=MAXLAG):
    arr = np.asarray(arr, dtype=np.int64)
    N = len(arr)
    V = int(arr.max()) + 1
    uniq, ucnt = np.unique(arr, return_counts=True)
    p1 = np.ones(V + 2) * 1e-12
    p1[uniq] = ucnt / N
    mis = []
    for d in range(1, max_lag + 1):
        a = arr[:N - d]; b = arr[d:]
        kk, cc = np.unique(a * (V + 2) + b, return_counts=True)
        pab = cc / (N - d)
        ia = kk // (V + 2); ib = kk % (V + 2)
        mi = float(np.sum(pab * np.log2(pab / (p1[ia] * p1[ib]))))
        mis.append(max(mi, 1e-4))
    return mis


@torch.no_grad()
def rollout(model, prompts):
    dt = model.head.weight.dtype
    model.eval()
    states = [None] * model.num_ssm
    h_slow = torch.zeros(B, model.D, device=DEV, dtype=dt)
    recall = None
    pos = 0
    seeds = torch.tensor(prompts, device=DEV, dtype=torch.long)
    tok = seeds[:, 0:1]
    for i in range(1, PL):
        _, states, h_slow, recall, pos = model.generate_step(tok, states, h_slow, recall, pos)
        tok = seeds[:, i:i + 1]
    sampled = []
    for _ in range(L):
        logits, states, h_slow, recall, pos = model.generate_step(tok, states, h_slow, recall, pos)
        p = torch.softmax(logits.float(), -1)
        sampled.append(torch.multinomial(p, 1))
        tok = sampled[-1]
    return torch.cat(sampled, 1)


def teacher_logp(model, prompts, sampled):
    seeds = torch.tensor(prompts, device=DEV, dtype=torch.long)
    full = torch.cat([seeds, sampled], 1)
    model.train()
    logits = model(full)
    logp = F.log_softmax(logits.float(), -1)
    full_s = full[:, 1:]
    tok_lp = logp[:, :-1].gather(2, full_s.unsqueeze(-1)).squeeze(-1)
    return tok_lp[:, PL - 1:].sum(1)


def ce_batch(model, prompts_tok, chosens_tok):
    seqs = [prompts_tok[i] + chosens_tok[i] for i in range(len(prompts_tok))]
    m = max(len(s) for s in seqs)
    arr = torch.zeros(len(seqs), m, dtype=torch.long)
    lens = []
    for i, s in enumerate(seqs):
        s = s[:200]
        arr[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        lens.append(len(s))
    arr = arr.to(DEV)
    logits = model(arr)
    logp = F.log_softmax(logits.float(), -1)
    nxt = arr[:, 1:]
    lp = logp[:, :-1].gather(2, nxt.unsqueeze(-1)).squeeze(-1)
    pos = torch.arange(m - 1, device=DEV).unsqueeze(0)
    mask = (pos < (torch.tensor(lens, device=DEV) - 1).unsqueeze(1)).float()
    return -(lp * mask).sum() / mask.sum()


@torch.no_grad()
def clean_measure(model, gen_prompts):
    model.eval()
    allg = []
    for p in gen_prompts[:12]:
        allg += D.gen_from_prompt(model, p, 200, 0.8)
    g, _ = G.compute_gamma(np.array(allg))
    lm = G.local_metrics(np.array(allg))
    return g, lm["distinct2"], lm["rep4"]


def run(cond, target_mis, ce_pairs, prompts, gen_prompts):
    torch.manual_seed(0)
    model = FRSMASHv36(D.VOCAB, D.HD, D.HEADS, D.LAYERS, n_slots=4).to(DEV)
    model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)["model"])
    with torch.no_grad():
        _ = model(torch.randint(0, D.VOCAB, (1, 32), device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    baseline = None
    traj = []
    print(f"\n===== {cond} =====", flush=True)
    g0, d0, r0 = clean_measure(model, gen_prompts)
    print(f"  init: gamma={g0:.3f} d2={d0:.3f} rep4={r0:.3f}", flush=True)
    traj.append((0, g0, r0))
    rng = np.random.default_rng(0)
    for step in range(NSTEPS):
        idx = rng.integers(0, len(ce_pairs), B)
        pb = [ce_pairs[i][0][:PL] for i in idx]
        cb = [ce_pairs[i][1] for i in idx]
        loss_ce = ce_batch(model, pb, cb)
        loss_struct = torch.tensor(0.0, device=DEV)
        if target_mis is not None:
            with torch.no_grad():
                sampled = rollout(model, prompts)
            seq_lp = teacher_logp(model, prompts, sampled)
            mis = I_curve(sampled.cpu().numpy().reshape(-1))
            reward = -float(np.sum((np.array(mis) - np.array(target_mis)) ** 2))
            baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
            adv = reward - baseline
            loss_struct = -(adv * seq_lp).mean() * ALPHA
        loss = loss_ce + loss_struct
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            g, d2, r4 = clean_measure(model, gen_prompts)
            traj.append((step + 1, g, r4))
            print(f"  s{step+1:3d} gamma={g:.3f} rep4={r4:.3f} ce={loss_ce.item():.3f}", flush=True)
    return traj


def main():
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    sp = D._sp(voc)
    pairs, gen_prompts = D.load_pairs(voc, sp, n_train=600, n_gen=24)
    ce_pairs = [(p, c) for p, c, _ in pairs]
    prompts = [p[:PL] for p in gen_prompts if len(p) >= PL][:B]
    cache = torch.load(r"F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt", weights_only=False)
    human_tok = torch.cat([c.long() for c in cache[:400]])[:40000].numpy()
    human_mis = I_curve(human_tok)
    print(f"human I(d) curve (d=1..{MAXLAG}): {[round(x,2) for x in human_mis]}", flush=True)
    targets = {
        "control": None,
        "low_gamma": [x * 1.5 for x in human_mis],
        "high_gamma": [x * 0.5 for x in human_mis],
    }
    res = {}
    for cond, tm in targets.items():
        res[cond] = run(cond, tm, ce_pairs, prompts, gen_prompts)
        json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
    print("\n==== FINAL (gamma, rep4) per condition ====")
    for cond, tr in res.items():
        print(f"  {cond:11s}: gamma {tr[0][1]:.3f} -> {tr[-1][1]:.3f}   rep4 {tr[0][2]:.3f} -> {tr[-1][2]:.3f}")
    print("\nverdict: low_gamma<control<high_gamma in final gamma, with rep4 stable => hybrid controllable+coherent")


if __name__ == "__main__":
    main()
