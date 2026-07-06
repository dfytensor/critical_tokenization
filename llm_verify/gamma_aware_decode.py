"""Task 2: gamma-aware decoding (closed-loop T control to hold gamma near target).

Unlike REINFORCE (training), decoding control measures gamma on the actual long
generated stream directly, so the population-statistic noise is much smaller.
A 'gamma thermostat': every W tokens, measure running gamma, adjust sampling T
(gamma decreases with T, so raise T to lower gamma, lower T to raise gamma).
Tests whether gamma is controllable at inference (where it failed in training).
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
from open_ash_voc import OpenASHVoc

DEV = "cuda"
OUT = os.path.join(C.WORK, "gamma_aware_decode")
os.makedirs(OUT, exist_ok=True)
GEN = 2048
WINDOW = 1024
W = 32


@torch.no_grad()
def gen_adaptive(model, prompt, target_gamma, adaptive):
    dt = model.head.weight.dtype
    model.eval()
    states = [None] * model.num_ssm
    h_slow = torch.zeros(1, model.D, device=DEV, dtype=dt)
    recall = None
    pos = 0
    tok = torch.tensor([[prompt[0]]], device=DEV, dtype=torch.long)
    for pid in prompt[1:]:
        _, states, h_slow, recall, pos = model.generate_step(tok, states, h_slow, recall, pos)
        tok = torch.tensor([[pid]], device=DEV, dtype=torch.long)
    out = []
    T = 0.8
    T_traj = []
    g_traj = []
    for i in range(GEN):
        logits, states, h_slow, recall, pos = model.generate_step(tok, states, h_slow, recall, pos)
        p = torch.softmax(logits.float() / max(T, 1e-3), -1)
        nxt = int(torch.multinomial(p, 1).item())
        out.append(nxt)
        tok = torch.tensor([[nxt]], device=DEV, dtype=torch.long)
        if adaptive and (i + 1) % W == 0 and len(out) >= 200:
            window = np.array(out[-WINDOW:])
            g, _ = G.compute_gamma(window)
            g_traj.append((i + 1, g, T))
            if g > target_gamma:
                T = min(1.6, T + 0.12)
            else:
                T = max(0.3, T - 0.12)
            T_traj.append(T)
    return out, g_traj


def gamma_curve(out, chunk=128):
    pts = []
    for i in range(chunk, len(out), chunk):
        g, _ = G.compute_gamma(np.array(out[max(0, i - 1024):i]))
        pts.append((i, g))
    return pts


def main():
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    sp = D._sp(voc)
    _, prompts = D.load_pairs(voc, sp, n_train=1, n_gen=8)
    prompt = [p for p in prompts if len(p) >= 24][0][:40]
    model = FRSMASHv36(D.VOCAB, D.HD, D.HEADS, D.LAYERS, n_slots=4).to(DEV)
    model.load_state_dict(torch.load(D.CKPT, map_location=DEV, weights_only=True)["model"])
    with torch.no_grad():
        _ = model(torch.randint(0, D.VOCAB, (1, 32), device=DEV))
    target = 0.13
    print(f"target gamma = {target}", flush=True)

    print("\nfixed T=0.8 ...", flush=True)
    out_fixed, _ = gen_adaptive(model, prompt, target, adaptive=False)
    gf = gamma_curve(out_fixed)
    gf_arr = np.array([g for _, g in gf])
    print(f"  fixed: gamma mean={gf_arr.mean():.3f} std={gf_arr.std():.3f} range=[{gf_arr.min():.3f},{gf_arr.max():.3f}]", flush=True)

    print("\ngamma-aware (adaptive T) ...", flush=True)
    out_adp, g_traj = gen_adaptive(model, prompt, target, adaptive=True)
    ga = gamma_curve(out_adp)
    ga_arr = np.array([g for _, g in ga])
    print(f"  aware: gamma mean={ga_arr.mean():.3f} std={ga_arr.std():.3f} range=[{ga_arr.min():.3f},{ga_arr.max():.3f}]", flush=True)
    print(f"  T trajectory (sampled): {[round(t,2) for _,_,t in g_traj[:12]]}", flush=True)

    lmf = G.local_metrics(np.array(out_fixed))
    lma = G.local_metrics(np.array(out_adp))
    print(f"\n  fixed: distinct2={lmf['distinct2']:.3f} rep4={lmf['rep4']:.3f} ent={lmf['entropy']:.2f}")
    print(f"  aware: distinct2={lma['distinct2']:.3f} rep4={lma['rep4']:.3f} ent={lma['entropy']:.2f}")

    json.dump(dict(target=target,
                   fixed=dict(gamma_mean=float(gf_arr.mean()), gamma_std=float(gf_arr.std()), curve=gf, **lmf),
                   aware=dict(gamma_mean=float(ga_arr.mean()), gamma_std=float(ga_arr.std()), curve=ga, T_traj=[(i, round(g, 3), round(t, 2)) for i, g, t in g_traj], **lma)),
              open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
    print("\n==== verdict ====")
    print(f"  |mean gamma - target|:  fixed={abs(gf_arr.mean()-target):.3f}   aware={abs(ga_arr.mean()-target):.3f}")
    print(f"  gamma std:             fixed={gf_arr.std():.3f}   aware={ga_arr.std():.3f}")
    print("  aware keeps gamma closer to target &/or tighter? => gamma controllable at decoding")


if __name__ == "__main__":
    main()
