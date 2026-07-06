"""⑤ Gradual vocab expansion as a tractable proxy for differentiable tokenization.

The discrete dynamic experiment (abrupt m1500->full switch) was WORSE than static
(switch disruption). A differentiable/soft tokenizer would avoid abrupt switches.
This tests the cheap proxy: GRADUAL expansion (add merges smoothly over B-phase)
vs abrupt vs oracle. If gradual beats abrupt and approaches oracle, the
'smoothness' hypothesis (and thus the differentiable direction) is supported.
"""
import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
import common as C
from frsmash_v36 import FRSMASHv36

OUT = os.path.join(C.WORK, "gradual_test")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
SEQ = 512
MICRO = 8
ACCUM = 4
BATCH = MICRO * ACCUM
LAYERS = 6
HEADS = 4
H = 256
LR = 5e-4
LOG2E = math.log2(math.e)
NA, NB = 300, 200  # A-phase, B-phase steps (smaller for speed)


def eval_bpc(model, val_ids, cpt, V):
    model.eval()
    n = len(val_ids)
    starts = list(range(0, n - SEQ - 1, SEQ))[:256]
    tl = tt = 0
    for i in range(0, len(starts), 16):
        idxs = starts[i:i + 16]
        seqs = torch.stack([val_ids[s:s + SEQ + 1] for s in idxs])
        x = seqs[:, :-1].long().to(DEV)
        y = seqs[:, 1:].long().to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
        tl += F.cross_entropy(o.reshape(-1, V), y.reshape(-1), reduction="sum").item()
        tt += y.numel()
    model.train()
    return (tl / max(tt, 1)) * LOG2E / cpt


def run(name, V, phases, b_val_ids, b_cpt):
    model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=4).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    total = sum(p[0] for p in phases)
    gs = 0
    t0 = time.time()
    print(f"\n##### {name} ({total} steps)", flush=True)
    for n_steps, train_cpu, cpt, tag in phases:
        ids = train_cpu.to(DEV)
        print(f"  [{name}] {n_steps} steps @ {tag}", flush=True)
        for _ in range(n_steps):
            gs += 1
            cur = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * gs / total)))
            for pg in opt.param_groups:
                pg["lr"] = cur
            opt.zero_grad(set_to_none=True)
            for _ in range(ACCUM):
                st = torch.randint(0, len(ids) - SEQ - 1, (MICRO,))
                sq = torch.stack([ids[s:s + SEQ + 1] for s in st])
                x, y = sq[:, :-1].long(), sq[:, 1:].long()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    o = model(x)
                    loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1)) / ACCUM
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        del ids
        torch.cuda.empty_cache()
    bpc = eval_bpc(model, b_val_ids, b_cpt, V)
    print(f"  {name} final B-val BPC={bpc:.4f} ({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return bpc


def main():
    text = C.load_en_text(8_000_000)
    A = text[:4_000_000]
    Aval = text[4_000_000:4_500_000]
    B = text[5_000_000:7_000_000]
    Bval = text[7_000_000:7_500_000]
    print("building master BPE...", flush=True)
    master = C.base_tok.BpeTokenizer(A, 6000)
    M = len(master.merges)
    V = master.actual_vocab_size
    print(f"  master merges={M} V={V}", flush=True)
    levels = {"m1500": 1500, "m3000": 3000, "m4500": 4500, "full": M}

    def enc_split(tr, va, key):
        nm = levels[key]
        e = master if nm >= M else master.restrict_to(range(nm))
        tr_t = torch.tensor(e.encode(tr), dtype=torch.int32)
        va_t = torch.tensor(e.encode(va), dtype=torch.int32)
        return tr_t, va_t, len(va) / max(len(va_t), 1)

    Atr = {k: enc_split(A, Aval, k) for k in ["m1500", "full"]}
    Btr = {k: enc_split(B, Bval, k) for k in levels}
    b_val = Btr["m1500"][1]
    b_cpt_1500 = Btr["m1500"][2]
    b_val_full = Btr["full"][1]
    b_cpt_full = Btr["full"][2]

    results = {}
    results["frozen"] = run("frozen", V,
        [(NA, Atr["m1500"][0], Atr["m1500"][2], "A_m1500"),
         (NB, Btr["m1500"][0], Btr["m1500"][2], "B_m1500")], b_val, b_cpt_1500)
    results["abrupt"] = run("abrupt", V,
        [(NA, Atr["m1500"][0], Atr["m1500"][2], "A_m1500"),
         (NB, Btr["full"][0], Btr["full"][2], "B_full")], b_val_full, b_cpt_full)
    results["oracle"] = run("oracle", V,
        [(NA, Atr["full"][0], Atr["full"][2], "A_full"),
         (NB, Btr["full"][0], Btr["full"][2], "B_full")], b_val_full, b_cpt_full)
    seg = NB // len(levels)
    grad_phases = [(NA, Atr["m1500"][0], Atr["m1500"][2], "A_m1500")]
    for k in levels:
        grad_phases.append((seg, Btr[k][0], Btr[k][2], f"B_{k}"))
    results["gradual"] = run("gradual", V, grad_phases, b_val_full, b_cpt_full)

    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print("\n==== B-val BPC (lower better) ====")
    for k in ["frozen", "abrupt", "gradual", "oracle"]:
        print(f"  {k:8s} {results[k]:.4f}")
    print("\nverdict: gradual < abrupt ?  gradual ~ oracle ?  => smooth-expansion helps?")


if __name__ == "__main__":
    main()
