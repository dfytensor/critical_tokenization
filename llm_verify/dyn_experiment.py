"""Verify 'dynamic tokenizer / joint training' (the framework's unverified next step).

Two falsifiable tests, all at MATCHED compute and a fixed V_MAX embedding
(so changing the active tokenizer mid-training needs NO embedding surgery):

TEST 1 (static data): does a vocabulary CURRICULUM beat fixed vocab?
  - static_500 / static_1500(V*) / static_full   (fixed active merges)
  - curriculum: 500 -> 1500 -> 3000 -> full over 4 phases

TEST 2 (domain shift A->B): does ADAPTIVE re-merging beat a frozen tokenizer?
  Train 600 steps on domain A, 400 on domain B (disjoint book sets).
  - frozen:  1500 merges throughout A->B  (tokenizer sized for A, frozen into B)
  - oracle:  full merges throughout        (upper bound, sees B's vocab early)
  - dynamic: 1500 on A, EXPAND to full at the A->B boundary  (adapt to B)

If dynamic/curriculum beat their static counterparts at equal compute =>
supports the framework. If not on static data but yes under shift => the value
of dynamic tokenization is specifically for distribution shift (a refinement).
"""
import os
import sys
import json
import time
import math
from collections import Counter
import torch
import torch.nn.functional as F

sys.path.insert(0, r"F:\OpenASH2605\critical_tokenization\llm_verify")
import common as C
from frsmash_v36 import FRSMASHv36

DEV = "cuda"
SEQ = 512
BATCH = 32
EVAL_EVERY = 50
LR = 5e-4
HEADS = 4
LAYERS = 4
N_SLOTS = 4
H = 224
LOG2E = math.log2(math.e)
WORK = C.WORK
OUT = os.path.join(WORK, "dyn_runs")
os.makedirs(OUT, exist_ok=True)


def encode(tok, text):
    return torch.tensor(tok.encode(text), dtype=torch.int32)


def cpt_of(tok, val_text):
    ids = tok.encode(val_text)
    return len(val_text) / max(len(ids), 1)


@torch.no_grad()
def eval_bpc(model, val_ids, cpt, batch=64, max_windows=512):
    model.eval()
    n = len(val_ids)
    starts = list(range(0, n - SEQ - 1, SEQ))[:max_windows]
    tot_l = 0.0
    tot_t = 0
    V = model.em.num_embeddings
    for i in range(0, len(starts), batch):
        idxs = starts[i:i + batch]
        seqs = torch.stack([val_ids[s:s + SEQ + 1] for s in idxs])
        x = seqs[:, :-1].long().to(DEV)
        y = seqs[:, 1:].long().to(DEV)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
        loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1), reduction="sum")
        tot_l += loss.item()
        tot_t += y.numel()
    model.train()
    return (tot_l / max(tot_t, 1)) * LOG2E / cpt


def sample_batch(train_ids):
    starts = torch.randint(0, len(train_ids) - SEQ - 1, (BATCH,))
    seqs = torch.stack([train_ids[s:s + SEQ + 1] for s in starts])
    return seqs[:, :-1].long(), seqs[:, 1:].long()


def run_condition(name, V, phases, val_sets):
    """phases: list of (n_steps, train_ids_tensor, cpt, level_tag).
    val_sets: list of (tag, val_ids_tensor, cpt) to eval at every checkpoint."""
    model = FRSMASHv36(V, H, HEADS, LAYERS, n_slots=N_SLOTS).to(DEV)
    np_ = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    log = []
    gs = 0
    total_steps = sum(p[0] for p in phases)
    t0 = time.time()
    print(f"\n##### {name}  (V={V}, {np_/1e6:.2f}M params, {total_steps} steps)", flush=True)
    for pi, (n_steps, train_cpu, cpt, tag) in enumerate(phases):
        train_ids = train_cpu.to(DEV)
        print(f"  [{name}] phase {pi}: {n_steps} steps @ active={tag} cpt={cpt:.3f}", flush=True)
        for _ in range(n_steps):
            gs += 1
            cur_lr = LR * (0.1 + 0.45 * (1 + math.cos(math.pi * gs / total_steps)))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
            x, y = sample_batch(train_ids)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = model(x)
                loss = F.cross_entropy(o.reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if gs % EVAL_EVERY == 0 or gs == total_steps:
                row = dict(step=gs, phase=tag, train_loss=loss.item())
                for vtag, vids, vcpt in val_sets:
                    row[f"bpc_{vtag}"] = eval_bpc(model, vids, vcpt)
                log.append(row)
                bpcstr = " ".join(f"{k}={v:.3f}" for k, v in row.items() if k.startswith("bpc_"))
                print(f"    {name} s{gs}/{total_steps} [{tag}] loss={loss.item():.3f} {bpcstr} ({time.time()-t0:.0f}s)", flush=True)
        del train_ids
        torch.cuda.empty_cache()
    with open(os.path.join(OUT, f"log_{name}.json"), "w") as f:
        json.dump(dict(name=name, params=np_, V=V, total_steps=total_steps, log=log), f, indent=2)
    del model
    torch.cuda.empty_cache()
    return log


def main():
    text = C.load_en_text(19_000_000)
    EN_TRAIN, EN_VAL = text[0:18_000_000], text[18_000_000:19_000_000]
    A_TRAIN, A_VAL = text[0:10_000_000], text[10_000_000:10_500_000]
    B_TRAIN, B_VAL = text[11_000_000:17_000_000], text[17_000_000:17_500_000]
    print("building master BPE on EN_TRAIN...", flush=True)
    t0 = time.time()
    master = C.base_tok.BpeTokenizer(EN_TRAIN[:2_000_000], 6000)
    M = master.n_active_merges() if hasattr(master, "n_active_merges") else len(master.merges)
    M = len(master.merges)
    V_MAX = master.actual_vocab_size
    print(f"  master merges={M} V_MAX={V_MAX} ({time.time()-t0:.0f}s)", flush=True)

    LV = {"m500": 500, "m1500": 1500, "m3000": 3000, "full": M}
    enc = {k: (master if v >= M else master.restrict_to(range(v))) for k, v in LV.items()}

    print("encoding splits...", flush=True)
    def enc_split(train_t, val_t, level_key):
        te = encode(enc[level_key], train_t)
        va = encode(enc[level_key], val_t)
        c = len(val_t) / max(len(va), 1)
        return te, va, c

    EN = {}
    for k in LV:
        EN[k] = enc_split(EN_TRAIN, EN_VAL, k)
        print(f"  EN[{k}] train_tok={len(EN[k][0]):,} cpt={EN[k][2]:.3f}", flush=True)
    AB = {}
    for dom, trt, vlt in [("A", A_TRAIN, A_VAL), ("B", B_TRAIN, B_VAL)]:
        for k in ("m1500", "full"):
            AB[(dom, k)] = enc_split(trt, vlt, k)
            print(f"  {dom}[{k}] train_tok={len(AB[(dom,k)][0]):,} cpt={AB[(dom,k)][2]:.3f}", flush=True)

    results = {}

    # ---- TEST 1: static data, 1000 steps, eval on EN_VAL at current level ----
    print("\n========= TEST 1: static-data curriculum =========", flush=True)
    T1 = 1000
    for name, lvl in [("t1_static_500", "m500"), ("t1_static_1500", "m1500"), ("t1_static_full", "full")]:
        tr, va, cpt = EN[lvl]
        log = run_condition(name, V_MAX, [(T1, tr, cpt, lvl)], [("cur", va, cpt)])
        results[name] = log[-1][f"bpc_cur"]
    q = T1 // 4
    cur_log = run_condition("t1_curriculum", V_MAX, [
        (q, EN["m500"][0], EN["m500"][2], "m500"),
        (q, EN["m1500"][0], EN["m1500"][2], "m1500"),
        (q, EN["m3000"][0], EN["m3000"][2], "m3000"),
        (q, EN["full"][0], EN["full"][2], "full"),
    ], [("m500", EN["m500"][1], EN["m500"][2]),
        ("m1500", EN["m1500"][1], EN["m1500"][2]),
        ("full", EN["full"][1], EN["full"][2])])
    results["t1_curriculum"] = cur_log[-1]["bpc_full"]

    # ---- TEST 2: domain shift A(600)->B(400), final eval on B_VAL ----
    print("\n========= TEST 2: domain-shift adaptation =========", flush=True)
    NA, NB = 600, 400
    tA_tr1500, tA_va1500, cA1500 = AB[("A", "m1500")]
    tA_trfull, tA_vafull, cAfull = AB[("A", "full")]
    tB_tr1500, tB_va1500, cB1500 = AB[("B", "m1500")]
    tB_trfull, tB_vafull, cBfull = AB[("B", "full")]
    bvals = [("B1500", tB_va1500, cB1500), ("Bfull", tB_vafull, cBfull)]

    log = run_condition("t2_frozen", V_MAX,
        [(NA, tA_tr1500, cA1500, "m1500"), (NB, tB_tr1500, cB1500, "m1500")], bvals)
    results["t2_frozen"] = log[-1]["bpc_B1500"]
    log = run_condition("t2_oracle", V_MAX,
        [(NA, tA_trfull, cAfull, "full"), (NB, tB_trfull, cBfull, "full")], bvals)
    results["t2_oracle"] = log[-1]["bpc_Bfull"]
    log = run_condition("t2_dynamic", V_MAX,
        [(NA, tA_tr1500, cA1500, "m1500"), (NB, tB_trfull, cBfull, "full")], bvals)
    results["t2_dynamic"] = log[-1]["bpc_Bfull"]

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n==== SUMMARY (final val BPC) ====")
    print("-- TEST 1 (static data, 1000 steps, lower=better) --")
    for k in ["t1_static_500", "t1_static_1500", "t1_static_full", "t1_curriculum"]:
        print(f"  {k:18s} {results[k]:.4f}")
    print("-- TEST 2 (A->B shift, final B-val BPC) --")
    for k in ["t2_frozen", "t2_dynamic", "t2_oracle"]:
        print(f"  {k:18s} {results[k]:.4f}")
    print("\ninterpretation keys:")
    print("  T1: curriculum < best static ?  => scheduling helps on static data")
    print("  T2: dynamic < frozen ? => adaptation helps under shift ;  dynamic vs oracle => gap to upper bound")


if __name__ == "__main__":
    main()
