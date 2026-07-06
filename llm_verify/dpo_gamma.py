"""Real DPO on the 60M FRSMASH v3.6 + gamma monitoring across over-optimization.

Validates gamma-monitor on REAL reward over-optimization (not overfit-proxy):
SFT baseline -> light DPO -> heavy DPO, measuring gamma at each. If gamma drifts
monotonically with DPO strength (and diverges from local metrics), gamma-RLHF
monitoring is confirmed.
"""
import os
import sys
import json
import math
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\wdlm_verification")
import common as C
from frsmash_v36 import FRSMASHv36
import gen_monitor as G
os.chdir(r"F:\OpenASH2605")
from open_ash_voc import OpenASHVoc


def _sp(tokenizer):
    t = tokenizer.token_to_id
    return {"pad": t.get("<|pad|>", 0), "im_start": t.get("<|im_start|>", 1),
            "im_end": t.get("<|im_end|>", 2), "think_s": t.get("<|think|>", 3),
            "think_e": t.get("<|end_think|>", 4), "user": 5,
            "agent": t.get("<|agent|>", 6), "system": 7}


def build_user_prompt(tokenizer, user_text, system_text=None):
    sp = _sp(tokenizer)
    ids = []
    if system_text:
        ids += [sp["im_start"], sp["system"]] + tokenizer.encode(system_text) + [sp["im_end"]]
    ids += [sp["im_start"], sp["user"]] + tokenizer.encode(user_text) + [sp["im_end"]]
    ids += [sp["im_start"], sp["agent"]]
    return ids

DEV = "cuda"
VOCAB = 23005
HD = 432
HEADS = 8
LAYERS = 8
CKPT = r"F:\rwkv\frsmash_v36\out\v36_sft_final.pth"
DPO = r"F:\OpenASH2605\minimind_data\dpo.jsonl"
OUT = os.path.join(C.WORK, "dpo_gamma")
os.makedirs(OUT, exist_ok=True)
MAXLEN = 320
BETA = 0.1


def load_pairs(voc, sp, n_train=1500, n_gen=48):
    pairs = []
    gen_prompts = []
    with open(DPO, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            ch, rj = o["chosen"], o["rejected"]
            if not ch or not rj or ch[-1]["role"] != "assistant" or rj[-1]["role"] != "assistant":
                continue
            user_msgs = [m for m in ch if m["role"] == "user"]
            if not user_msgs:
                continue
            user_text = "\n".join(m["content"] for m in user_msgs)
            prompt_ids = build_user_prompt(voc, user_text)
            ch_resp = voc.encode(ch[-1]["content"]) + [sp["im_end"]]
            rj_resp = voc.encode(rj[-1]["content"]) + [sp["im_end"]]
            if len(prompt_ids) < 5 or len(ch_resp) < 2 or len(rj_resp) < 2:
                continue
            ps = min(len(prompt_ids), 120)
            prompt_ids = prompt_ids[:ps]
            ch_resp = ch_resp[: MAXLEN - ps]
            rj_resp = rj_resp[: MAXLEN - ps]
            if len(gen_prompts) < n_gen:
                gen_prompts.append(prompt_ids)
            else:
                pairs.append((prompt_ids, ch_resp, rj_resp))
            if len(pairs) >= n_train and len(gen_prompts) >= n_gen:
                break
    print(f"loaded {len(pairs)} train pairs, {len(gen_prompts)} gen prompts", flush=True)
    return pairs, gen_prompts


def pad_batch(seqs, pad=0):
    m = max(len(s) for s in seqs)
    arr = torch.full((len(seqs), m), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return arr


def seq_logprob(model, ids):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
    logp = F.log_softmax(logits.float(), -1)
    ids_s = ids[:, 1:]
    logp_s = logp[:, :-1]
    tok = logp_s.gather(2, ids_s.unsqueeze(-1)).squeeze(-1)
    return tok  # (B, L-1) logp of token t by logits[t-1]


def resp_logprob(model, prompt_lens, full_ids):
    tok = seq_logprob(model, full_ids)  # (B,L-1)
    B, Lm = tok.shape
    pos = torch.arange(Lm, device=tok.device).unsqueeze(0)  # (1, Lm)
    pl = prompt_lens.to(tok.device).unsqueeze(1)            # (B, 1)
    mask = (pos >= (pl - 1)).float()                        # (B, Lm)
    return (tok * mask).sum(1)


def dpo_step(policy, ref, batch, opt):
    prompts, chs, rjs = batch
    B = len(prompts)
    pl = [len(p) for p in prompts]
    ch_full = [prompts[i] + chs[i] for i in range(B)]
    rj_full = [prompts[i] + rjs[i] for i in range(B)]
    ch_ids = pad_batch(ch_full).to(DEV)
    rj_ids = pad_batch(rj_full).to(DEV)
    pl_t = torch.tensor(pl)
    pol_ch = resp_logprob(policy, pl_t, ch_ids)
    pol_rj = resp_logprob(policy, pl_t, rj_ids)
    with torch.no_grad():
        ref_ch = resp_logprob(ref, pl_t, ch_ids)
        ref_rj = resp_logprob(ref, pl_t, rj_ids)
    cr = BETA * (pol_ch - ref_ch)
    rr = BETA * (pol_rj - ref_rj)
    loss = -F.logsigmoid(cr - rr).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    acc = (cr > rr).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def gen_from_prompt(model, prompt_ids, n_new, T):
    dt = model.head.weight.dtype
    model.eval()
    states = [None] * model.num_ssm
    h_slow = torch.zeros(1, model.D, device=DEV, dtype=dt)
    recall = None
    pos = 0
    tok = torch.tensor([[int(prompt_ids[0])]], device=DEV, dtype=torch.long)
    for pid in prompt_ids[1:]:
        logits, states, h_slow, recall, pos = model.generate_step(tok, states, h_slow, recall, pos)
        tok = torch.tensor([[int(pid)]], device=DEV, dtype=torch.long)
    out = []
    for _ in range(n_new):
        logits, states, h_slow, recall, pos = model.generate_step(tok, states, h_slow, recall, pos)
        if T <= 0:
            nxt = int(logits.argmax(-1).item())
        else:
            p = torch.softmax(logits.float() / T, -1)
            nxt = int(torch.multinomial(p, 1).item())
        out.append(nxt)
        tok = torch.tensor([[nxt]], device=DEV, dtype=torch.long)
    model.train()
    return out


def measure_gamma(model, gen_prompts, tag):
    results = {}
    for T in [0.0, 0.8]:
        allg = []
        for p in gen_prompts[:16]:
            allg += gen_from_prompt(model, p, 300, T)
        g, _ = G.compute_gamma(np.array(allg))
        lm = G.local_metrics(np.array(allg))
        lab = "greedy" if T == 0 else "T0.8"
        results[lab] = dict(gamma=g, **lm)
        print(f"  {tag} {lab}: gamma={g:.3f} d2={lm['distinct2']:.3f} rep4={lm['rep4']:.3f} ent={lm['entropy']:.2f}", flush=True)
    return results


def main():
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    sp = _sp(voc)
    pairs, gen_prompts = load_pairs(voc, sp)

    policy = FRSMASHv36(VOCAB, HD, HEADS, LAYERS, n_slots=4).to(DEV)
    policy.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)["model"])
    ref = FRSMASHv36(VOCAB, HD, HEADS, LAYERS, n_slots=4).to(DEV)
    ref.load_state_dict(torch.load(CKPT, map_location=DEV, weights_only=True)["model"])
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95))
    with torch.no_grad():
        _ = policy(torch.randint(0, VOCAB, (1, 32), device=DEV))  # warm up fla kernels
    print("loaded SFT model + reference", flush=True)

    log = {}
    print("\n== baseline (SFT, no DPO) ==", flush=True)
    log["baseline"] = measure_gamma(policy, gen_prompts, "baseline")

    schedule = [("light", 150), ("heavy", 350)]
    gs = 0
    for name, target in schedule:
        print(f"\n== DPO train to {target} total steps ({name}) ==", flush=True)
        t0 = time.time()
        rng = np.random.default_rng(0)
        while gs < target:
            idx = rng.integers(0, len(pairs), 8)
            batch = ([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], [pairs[i][2] for i in idx])
            loss, acc = dpo_step(policy, ref, batch, opt)
            gs += 1
            if gs % 50 == 0:
                print(f"    dpo s{gs} loss={loss:.3f} acc={acc:.2f} ({time.time()-t0:.0f}s)", flush=True)
        log[name] = measure_gamma(policy, gen_prompts, name)
        json.dump(log, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    print("\n==== gamma across DPO over-optimization ====")
    for k in ["baseline", "light", "heavy"]:
        if k in log:
            print(f"  {k:9s} T0.8 gamma={log[k]['T0.8']['gamma']:.3f} d2={log[k]['T0.8']['distinct2']:.3f} rep4={log[k]['T0.8']['rep4']:.3f}  | greedy gamma={log[k]['greedy']['gamma']:.3f}")
    dg = log["heavy"]["T0.8"]["gamma"] - log["baseline"]["T0.8"]["gamma"]
    print(f"\nverdict: gamma drift baseline->heavy (T0.8) = {dg:+.3f}  (monotone track of DPO over-opt? +local-metrics divergence?)")


if __name__ == "__main__":
    main()
