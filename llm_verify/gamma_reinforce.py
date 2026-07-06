"""Task 1: gamma as a REINFORCE reward (gamma-as-loss feasibility test).

Score-function REINFORCE: rollout (no_grad) -> teacher-forced re-forward (grad)
   -> logp of sampled tokens -> reward = -(gamma - gamma_target)^2 -> policy grad.
Tests whether gamma is CONTROLLABLE via a reward (i.e., usable as a loss/regularizer).
Two targets: 0.08 (more critical) and 0.25 (degraded). If gamma moves toward target,
gamma-as-reward is viable; coherence check shows if it's a blunt instrument.
"""
import os
import sys
import json
import math
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
OUT = os.path.join(C.WORK, "gamma_reinforce")
os.makedirs(OUT, exist_ok=True)
B = 8
L = 160
NSTEPS = 120


@torch.no_grad()
def rollout(model, prompts):
    dt = model.head.weight.dtype
    model.eval()
    states = [None] * model.num_ssm
    h_slow = torch.zeros(B, model.D, device=DEV, dtype=dt)
    recall = None
    pos = 0
    seeds = torch.tensor(prompts, device=DEV, dtype=torch.long)
    plen = seeds.size(1)
    tok = seeds[:, 0:1]
    for i in range(1, plen):
        _, states, h_slow, recall, pos = model.generate_step(tok, states, h_slow, recall, pos)
        tok = seeds[:, i:i + 1]
    sampled = []
    for _ in range(L):
        logits, states, h_slow, recall, pos = model.generate_step(tok, states, h_slow, recall, pos)
        probs = torch.softmax(logits.float(), -1)
        nxt = torch.multinomial(probs, 1)
        sampled.append(nxt)
        tok = nxt
    return torch.cat(sampled, 1)  # (B, L)


def teacher_logp(model, prompts, sampled):
    """teacher-forced re-forward WITH grad; return sum logp of sampled tokens per seq."""
    seeds = torch.tensor(prompts, device=DEV, dtype=torch.long)
    full = torch.cat([seeds, sampled], 1)  # (B, plen+L)
    model.train()
    logits = model(full)
    logp = F.log_softmax(logits.float(), -1)
    full_s = full[:, 1:]
    logp_s = logp[:, :-1]
    tok_lp = logp_s.gather(2, full_s.unsqueeze(-1)).squeeze(-1)  # (B, plen+L-1)
    plen = seeds.size(1)
    resp_lp = tok_lp[:, plen - 1:]  # logp of sampled positions
    return resp_lp.sum(1)  # (B,)


def run_target(gamma_target):
    torch.manual_seed(0)
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    sp = D._sp(voc)
    _, prompts_all = D.load_pairs(voc, sp, n_train=1, n_gen=64)
    PL = 24
    prompts = [p[:PL] for p in prompts_all if len(p) >= PL][:B]
    model = FRSMASHv36(D.VOCAB, D.HD, D.HEADS, D.LAYERS, n_slots=4).to(DEV)
    model.load_state_dict(torch.load(D.CKPT, map_location=DEV, weights_only=True)["model"])
    with torch.no_grad():
        _ = model(torch.randint(0, D.VOCAB, (1, 32), device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    baseline = None
    traj = []
    print(f"\n===== gamma_target = {gamma_target} =====", flush=True)
    for step in range(NSTEPS):
        with torch.no_grad():
            sampled = rollout(model, prompts)
        seq_logp = teacher_logp(model, prompts, sampled)
        flat = sampled.cpu().numpy().reshape(-1)
        g, _ = G.compute_gamma(flat)
        reward = -(g - gamma_target) ** 2
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        adv = reward - baseline
        loss = -(adv * seq_logp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == NSTEPS - 1:
            traj.append((step, g, float(reward)))
            print(f"  s{step:3d} gamma={g:.3f} reward={reward:.4f} adv={adv:.4f} seqlogp={seq_logp.mean().item():.1f}", flush=True)
    return traj


def main():
    res = {}
    for gt in [0.08, 0.25]:
        res[f"target_{gt}"] = run_target(gt)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
    print("\n==== controllability ====")
    for k, tr in res.items():
        g0, g1 = tr[0][1], tr[-1][1]
        print(f"  {k}: gamma {g0:.3f} -> {g1:.3f}  (target {k.split('_')[1]})")
    print("\nverdict: gamma moves toward target => gamma-as-reward is controllable (a usable loss/regularizer);")
    print("         but check coherence (pure REINFORCE likely blunts text => needs CE anchor in practice).")


if __name__ == "__main__":
    main()
