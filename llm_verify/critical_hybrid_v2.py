"""Theory-guided fix: does a higher-order stat (block entropy H(l)) suppress the
repetition-degeneracy (rep4 up) that the basic MI-matching suffered?

v0: CE + L2(I(d), target I_data x0.5)         -> expect gamma up AND rep4 up (degeneracy)
v1: CE + L2(I(d), x0.5) + L2(H(l), target H_data)  -> if rep4 stays down while gamma still
                                                       controllable => higher-order term cures
                                                       the degeneracy (theory prediction validated)
Target direction = high_gamma (I x0.5), the one that worked but damaged coherence.
"""
import os
import sys
import json
import numpy as np
import torch
from collections import Counter

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
OUT = os.path.join(C.WORK, "critical_hybrid_v2")
os.makedirs(OUT, exist_ok=True)
MAXLAG = 8
MAXL = 6
NSTEPS = 100


def H_curve(arr, max_l=MAXL):
    arr = list(arr); n = len(arr)
    out = []
    for l in range(1, max_l + 1):
        grams = [tuple(arr[i:i + l]) for i in range(n - l + 1)]
        c = Counter(grams)
        tot = sum(c.values())
        p = np.array(list(c.values())) / tot
        out.append(float(-(p * np.log2(p)).sum()))
    return out


def run(cond, use_H, ce_pairs, prompts, gen_prompts, I_data, H_data, I_scale, H_scale):
    torch.manual_seed(0)
    model = FRSMASHv36(D.VOCAB, D.HD, D.HEADS, D.LAYERS, n_slots=4).to(DEV)
    model.load_state_dict(torch.load(D.CKPT, map_location=DEV, weights_only=True)["model"])
    with torch.no_grad():
        _ = model(torch.randint(0, D.VOCAB, (1, 32), device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    baseline = None
    I_target = [x * 0.5 for x in I_data]
    traj = []
    g0, _, r0 = CH.clean_measure(model, gen_prompts)
    print(f"\n===== {cond} (use_H={use_H}) =====  init gamma={g0:.3f} rep4={r0:.3f}", flush=True)
    traj.append((0, g0, r0))
    rng = np.random.default_rng(0)
    for step in range(NSTEPS):
        idx = rng.integers(0, len(ce_pairs), CH.B)
        pb = [ce_pairs[i][0][:CH.PL] for i in idx]
        cb = [ce_pairs[i][1] for i in idx]
        loss_ce = CH.ce_batch(model, pb, cb)
        with torch.no_grad():
            sampled = CH.rollout(model, prompts)
        seq_lp = CH.teacher_logp(model, prompts, sampled)
        flat = sampled.cpu().numpy().reshape(-1)
        mis = CH.I_curve(flat)
        i_term = float(np.sum((np.array(mis) - np.array(I_target)) ** 2)) / I_scale
        h_term = 0.0
        if use_H:
            hs = H_curve(flat)
            h_term = float(np.sum((np.array(hs) - np.array(H_data)) ** 2)) / H_scale
        reward = -(i_term + h_term)
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        adv = reward - baseline
        loss_struct = -(adv * seq_lp).mean() * CH.ALPHA
        loss = loss_ce + loss_struct
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            g, _, r4 = CH.clean_measure(model, gen_prompts)
            traj.append((step + 1, g, r4))
            print(f"  s{step+1:3d} gamma={g:.3f} rep4={r4:.3f}  i_term={i_term:.2f} h_term={h_term:.2f}", flush=True)
    return traj


def main():
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    sp = D._sp(voc)
    pairs, gen_prompts = D.load_pairs(voc, sp, n_train=600, n_gen=24)
    ce_pairs = [(p, c) for p, c, _ in pairs]
    prompts = [p[:CH.PL] for p in gen_prompts if len(p) >= CH.PL][:CH.B]
    cache = torch.load(r"F:\OpenASH2605\minimind_data\pretrain_cached_30000_384.pt", weights_only=False)
    human_tok = torch.cat([c.long() for c in cache[:400]])[:40000].numpy()
    I_data = CH.I_curve(human_tok)
    H_data = H_curve(human_tok)
    I_scale = float(np.mean(np.array(I_data) ** 2))
    H_scale = float(np.mean(np.array(H_data) ** 2))
    print(f"I_data={[round(x,2) for x in I_data]}  H_data={[round(x,2) for x in H_data]}", flush=True)
    print(f"I_scale={I_scale:.2f} H_scale={H_scale:.2f}", flush=True)
    res = {}
    for cond, use_H in [("v0_no_H", False), ("v1_with_H", True)]:
        res[cond] = run(cond, use_H, ce_pairs, prompts, gen_prompts, I_data, H_data, I_scale, H_scale)
        json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
    print("\n==== FINAL (gamma, rep4) ====")
    for cond, tr in res.items():
        print(f"  {cond:11s}: gamma {tr[0][1]:.3f}->{tr[-1][1]:.3f}   rep4 {tr[0][2]:.3f}->{tr[-1][2]:.3f}")
    dg0, dg1 = res["v0_no_H"][-1][1], res["v1_with_H"][-1][1]
    dr0, dr1 = res["v0_no_H"][-1][2], res["v1_with_H"][-1][2]
    print(f"\nverdict: v1 rep4 ({dr1:.3f}) < v0 rep4 ({dr0:.3f}) while gamma both ~controlled ({dg0:.3f}/{dg1:.3f})?")
    print("         => higher-order H(l) term suppresses the repetition-degeneracy (theory pred validated)?")


if __name__ == "__main__":
    main()
