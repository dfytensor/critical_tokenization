"""gamma-monitor test on a COHERENT 60M FRSMASH (pretrain+sft, Chinese).

The 8M toy generated garbage (gamma baseline meaningless). This 60M model
generates coherent text, so gamma has a real baseline. Tests:
  - does coherent-model gamma approach HUMAN (real-token) gamma?
  - does gamma drop after overfit-degradation (mode-collapse proxy)?
  => whether gamma usefully tracks quality on a real model.
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
import gen_monitor as G  # reuse compute_gamma, local_metrics

DEV = "cuda"
VOCAB = 23005
HD = 432
HEADS = 8
LAYERS = 8
CKPT = r"F:\rwkv\frsmash_v36\out\v36_sft_final.pth"
CACHE = r"F:\OpenASH2605\train_60m\cache\pt_cache_512.pt"
OUT = os.path.join(C.WORK, "gamma_monitor_60m")
os.makedirs(OUT, exist_ok=True)
SEED_LEN = 48
GEN_LEN = 2048
GB = 48


@torch.no_grad()
def generate(model, seed, n_steps, temperature):
    B = seed.size(0)
    dt = model.head.weight.dtype
    model.eval()
    states = [None] * model.num_ssm
    h_slow = torch.zeros(B, model.D, device=DEV, dtype=dt)
    recall_state = None
    pos = 0
    tok = seed[:, 0:1].long()
    for i in range(SEED_LEN):
        logits, states, h_slow, recall_state, pos = model.generate_step(tok, states, h_slow, recall_state, pos)
        tok = seed[:, i + 1:i + 2].long()
    out = []
    for _ in range(n_steps):
        logits, states, h_slow, recall_state, pos = model.generate_step(tok, states, h_slow, recall_state, pos)
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits.float() / max(temperature, 1e-3), -1)
            nxt = torch.multinomial(probs, 1)
        out.append(nxt)
        tok = nxt
    return torch.cat(out, 1).cpu().numpy().reshape(-1)


def overfit(model, tokens, n_steps):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    ids = tokens.to(DEV)
    SEQ = 256
    model.train()
    t0 = time.time()
    for step in range(n_steps):
        st = torch.randint(0, len(ids) - SEQ - 1, (16,))
        sq = torch.stack([ids[s:s + SEQ + 1] for s in st])
        x, y = sq[:, :-1].long(), sq[:, 1:].long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
            loss = F.cross_entropy(o.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"    overfit s{step} loss={loss.item():.3f}", flush=True)
    print(f"    overfit done ({time.time()-t0:.0f}s)", flush=True)


def measure(model, seed, tag):
    res = {}
    for T in [0.0, 0.8]:
        gen = generate(model, seed, GEN_LEN, T)
        g, _ = G.compute_gamma(gen)
        lm = G.local_metrics(gen)
        lab = "greedy" if T == 0 else "T0.8"
        res[lab] = dict(gamma=g, **lm)
        print(f"  {tag} {lab}: gamma={g:.3f} d2={lm['distinct2']:.3f} rep4={lm['rep4']:.3f} ent={lm['entropy']:.2f}", flush=True)
    return res


def main():
    model = FRSMASHv36(VOCAB, HD, HEADS, LAYERS, n_slots=4).to(DEV)
    model.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)["model"])
    print("loaded 60M sft model", flush=True)

    cache = torch.load(CACHE, weights_only=False)
    flat = torch.cat([c.long() for c in cache[:4000]])  # real tokens for human baseline + seed
    human_ids = flat[:300000].cpu().numpy()
    g_h, _ = G.compute_gamma(human_ids)
    lm_h = G.local_metrics(human_ids)
    print(f"\nHUMAN (real minimind tokens): gamma={g_h:.3f} d2={lm_h['distinct2']:.3f} rep4={lm_h['rep4']:.3f} ent={lm_h['entropy']:.2f}", flush=True)
    rng = np.random.default_rng(0)
    shuf = human_ids.copy(); rng.shuffle(shuf)
    g_s, _ = G.compute_gamma(shuf)
    print(f"SHUFFLED: gamma={g_s:.4f}", flush=True)

    torch.manual_seed(0)
    starts = torch.randint(0, len(flat) - SEED_LEN - 2, (GB,))
    seed = torch.stack([flat[s:s + SEED_LEN + 1] for s in starts]).to(DEV)

    print("\n== coherent model (baseline) ==", flush=True)
    base = measure(model, seed, "baseline")

    print("\n== overfit-degrade (mode-collapse proxy, 250 steps on 60k tokens) ==", flush=True)
    overfit(model, flat[:60000], 250)

    print("\n== post-degradation ==", flush=True)
    post = measure(model, seed, "degraded")

    res = dict(human=dict(gamma=g_h, **lm_h), shuffled_gamma=g_s, baseline=base, degraded=post)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print("\n==== VERDICT ====")
    print(f"  human gamma = {g_h:.3f}  |  shuffled = {g_s:.3f}")
    print(f"  coherent T0.8 gamma = {base['T0.8']['gamma']:.3f}  (vs 8M toy was ~0.13)  -> close to human?")
    print(f"  degraded  T0.8 gamma = {post['T0.8']['gamma']:.3f}   drift = {post['T0.8']['gamma']-base['T0.8']['gamma']:+.3f}")
    print(f"  coherent greedy gamma = {base['greedy']['gamma']:.3f} -> degraded greedy = {post['greedy']['gamma']:.3f}")


if __name__ == "__main__":
    main()
