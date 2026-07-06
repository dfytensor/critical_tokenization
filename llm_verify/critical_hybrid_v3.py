"""Closed-loop #2: does log-domain (multi-scale-shape) matching reduce the asymmetry?

Theory (RG saddle): breaking structure (high gamma) is easy, building it (low gamma)
is hard -> asymmetric control. Predicted fix: match in log domain (weights the decay
tail / slope, not the d=1 level). Test: low/high targets x linear/log matching.
If log_low moves gamma more than linear_low (toward 'more critical'), asymmetry reduced.
"""
import os
import sys
import json
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
sys.path.insert(0, r"F:\OpenASH2605")
os.chdir(r"F:\OpenASH2605")
import common as C
from frsmash_v36 import FRSMASHv36
import gen_monitor as G
import dpo_gamma as D
import critical_hybrid as CH
from open_ash_voc import OpenASHVoc

DEV = "cuda"
OUT = os.path.join(C.WORK, "critical_hybrid_v3")
os.makedirs(OUT, exist_ok=True)
NSTEPS = 80
EPS = 1e-3


def run(cond, target_scale, log_domain, ce_pairs, prompts, gen_prompts, I_data):
    torch.manual_seed(0)
    model = FRSMASHv36(D.VOCAB, D.HD, D.HEADS, D.LAYERS, n_slots=4).to(DEV)
    model.load_state_dict(torch.load(D.CKPT, map_location=DEV, weights_only=True)["model"])
    with torch.no_grad():
        _ = model(torch.randint(0, D.VOCAB, (1, 32), device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    baseline = None
    I_target = [x * target_scale for x in I_data]
    traj = []
    g0, _, r0 = CH.clean_measure(model, gen_prompts)
    traj.append((0, g0))
    print(f"\n===== {cond} =====  init gamma={g0:.3f}", flush=True)
    rng = np.random.default_rng(0)
    for step in range(NSTEPS):
        idx = rng.integers(0, len(ce_pairs), CH.B)
        pb = [ce_pairs[i][0][:CH.PL] for i in idx]
        cb = [ce_pairs[i][1] for i in idx]
        loss_ce = CH.ce_batch(model, pb, cb)
        with torch.no_grad():
            sampled = CH.rollout(model, prompts)
        seq_lp = CH.teacher_logp(model, prompts, sampled)
        mis = np.array(CH.I_curve(sampled.cpu().numpy().reshape(-1)))
        if log_domain:
            reward = -float(np.sum((np.log(np.maximum(mis, EPS)) - np.log(np.maximum(I_target, EPS))) ** 2))
        else:
            reward = -float(np.sum((mis - np.array(I_target)) ** 2))
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        adv = reward - baseline
        loss = loss_ce + (-(adv * seq_lp).mean() * CH.ALPHA)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 40 == 0 or step == NSTEPS - 1:
            g, _, _ = CH.clean_measure(model, gen_prompts)
            traj.append((step + 1, g))
            print(f"  s{step+1:3d} gamma={g:.3f} reward={reward:.3f}", flush=True)
    return traj


def main():
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    sp = D._sp(voc)
    pairs, gen_prompts = D.load_pairs(voc, sp, n_train=600, n_gen=24)
    ce_pairs = [(p, c) for p, c, _ in pairs]
    prompts = [p[:CH.PL] for p in gen_prompts if len(p) >= CH.PL][:CH.B]
    cache = torch.load(r"F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt", weights_only=False)
    I_data = CH.I_curve(torch.cat([c.long() for c in cache[:400]])[:40000].numpy())
    print(f"I_data={[round(x,2) for x in I_data]}", flush=True)
    conds = [
        ("linear_low", 1.5, False),
        ("linear_high", 0.5, False),
        ("log_low", 1.5, True),
        ("log_high", 0.5, True),
    ]
    res = {}
    for cond, scale, logd in conds:
        res[cond] = run(cond, scale, logd, ce_pairs, prompts, gen_prompts, I_data)
        json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
    print("\n==== FINAL gamma per condition (init ~0.047, control~0.028) ====")
    for cond, tr in res.items():
        print(f"  {cond:12s}: {tr[0][1]:.3f} -> {tr[-1][1]:.3f}")
    lin_low, lin_high = res["linear_low"][-1][1], res["linear_high"][-1][1]
    log_low, log_high = res["log_low"][-1][1], res["log_high"][-1][1]
    print(f"\nlinear: low={lin_low:.3f} high={lin_high:.3f}  (asymmetry |high-low|={abs(lin_high-lin_low):.3f})")
    print(f"log:    low={log_low:.3f} high={log_high:.3f}  (asymmetry |high-low|={abs(log_high-log_low):.3f})")
    print(f"\nlog_low < linear_low ? ({log_low:.3f} vs {lin_low:.3f}) => log-domain makes 'build criticality' direction work better => asymmetry reduced?")


if __name__ == "__main__":
    main()
