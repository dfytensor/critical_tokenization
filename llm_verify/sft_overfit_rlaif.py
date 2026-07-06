"""gamma-monitor on a SECOND degradation mode: SFT over-training on rlaif
conversations (vs the DPO over-optimization already confirmed). If gamma rises
here too, gamma generalizes across alignment methods (DPO + SFT-overfit), not
just one.
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
RLAIF = r"F:\OpenASH2605\minimind_data\rlaif.jsonl"
OUT = os.path.join(C.WORK, "sft_overfit_rlaif")
os.makedirs(OUT, exist_ok=True)


def conv_to_ids(voc, sp, conv):
    ids = []
    for m in conv:
        role = m.get("role")
        c = m.get("content", "")
        if role == "user":
            ids += [sp["im_start"], sp["user"]] + voc.encode(c) + [sp["im_end"]]
        elif role == "assistant":
            ids += [sp["im_start"], sp["agent"]] + voc.encode(c) + [sp["im_end"]]
    return ids


def main():
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    sp = D._sp(voc)
    seqs = []
    gen_prompts = []
    with open(RLAIF, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line)
            except Exception:
                continue
            conv = o.get("conversations", [])
            if not conv:
                continue
            ids = conv_to_ids(voc, sp, conv)
            if len(ids) > 20:
                seqs.append(ids[:200])
            user_msgs = [m for m in conv if m.get("role") == "user"]
            if user_msgs and len(gen_prompts) < 24:
                up = D.build_user_prompt(voc, user_msgs[0]["content"])
                gen_prompts.append(up[:80])
            if len(seqs) >= 3000 and len(gen_prompts) >= 24:
                break
    flat = []
    for s in seqs:
        flat += s
    flat = flat[:100000]
    print(f"rlaif: {len(seqs)} convs, {len(flat)} train tokens, {len(gen_prompts)} gen prompts", flush=True)

    model = FRSMASHv36(D.VOCAB, D.HD, D.HEADS, D.LAYERS, n_slots=4).to(DEV)
    model.load_state_dict(torch.load(D.CKPT, map_location=DEV, weights_only=True)["model"])
    with torch.no_grad():
        _ = model(torch.randint(0, D.VOCAB, (1, 32), device=DEV))
    print("loaded SFT model", flush=True)

    def measure(tag):
        res = {}
        for T in [0.0, 0.8]:
            allg = []
            for p in gen_prompts[:16]:
                allg += D.gen_from_prompt(model, p, 300, T)
            g, _ = G.compute_gamma(np.array(allg))
            lm = G.local_metrics(np.array(allg))
            lab = "greedy" if T == 0 else "T0.8"
            res[lab] = dict(gamma=g, **lm)
            print(f"  {tag} {lab}: gamma={g:.3f} d2={lm['distinct2']:.3f} rep4={lm['rep4']:.3f}", flush=True)
        return res

    print("\n== baseline (SFT) ==", flush=True)
    log = {"baseline": measure("baseline")}

    print("\n== SFT over-train on rlaif (300 steps, lr 5e-4) ==", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    ids_t = torch.tensor(flat, dtype=torch.long, device=DEV)
    SEQ = 256
    model.train()
    for step in range(300):
        st = torch.randint(0, len(ids_t) - SEQ - 1, (8,))
        sq = torch.stack([ids_t[s:s + SEQ + 1] for s in st])
        x, y = sq[:, :-1].long(), sq[:, 1:].long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
            loss = F.cross_entropy(o.reshape(-1, D.VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"    sft s{step} loss={loss.item():.3f}", flush=True)

    print("\n== post over-train ==", flush=True)
    log["overtrain"] = measure("overtrain")
    json.dump(log, open(os.path.join(OUT, "results.json"), "w"), indent=2)

    print("\n==== gamma: baseline -> rlaif-SFT-overtrain ====")
    for lab in ["greedy", "T0.8"]:
        b, a = log["baseline"][lab]["gamma"], log["overtrain"][lab]["gamma"]
        print(f"  {lab}: {b:.3f} -> {a:.3f} (drift {a-b:+.3f})  d2 {log['baseline'][lab]['distinct2']:.3f}->{log['overtrain'][lab]['distinct2']:.3f}  rep4 {log['baseline'][lab]['rep4']:.3f}->{log['overtrain'][lab]['rep4']:.3f}")
    print("\nverdict: gamma rises on SFT-overtrain too => generalizes across degradation modes (DPO + SFT-overfit)?")


if __name__ == "__main__":
    main()
