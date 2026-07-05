"""Plot the N-sweep: V_opt(N) drift and the N^{2/3} embedding-tax ceiling."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = r"F:\OpenASH2605\critical_tokenization\llm_verify"
RUN = os.path.join(WORK, "n_sweep")
summ = json.load(open(os.path.join(RUN, "summary.json")))
vopt = summ["v_opt"]

TOK_ORDER = ["char", "bpe500", "bpe1500", "bpe3000", "bpe6000"]
COL = {"char": "#1f77b4", "bpe500": "#ff7f0e", "bpe1500": "#2ca02c",
       "bpe3000": "#d62728", "bpe6000": "#9467bd"}

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Left: BPC vs tokenizer (V) for each N
for r in vopt:
    N = r["N"] / 1e6
    xs = [f"{t}\nV={summ_v}" for t in TOK_ORDER for summ_v in [r["row"][t]]]
    xs = TOK_ORDER
    ys = [r["row"][t] for t in TOK_ORDER]
    ax[0].plot(range(len(TOK_ORDER)), ys, "o-", label=f"N={N:.0f}M (V_opt={r['v_opt']})", lw=2)
    bi = int(np.argmin(ys))
    ax[0].plot(bi, ys[bi], "s", ms=12, mfc="none", mec="black", mew=2)
ax[0].set_xticks(range(len(TOK_ORDER)))
ax[0].set_xticklabels([f"{t}\n(V*)" if t == "bpe1500" else t for t in TOK_ORDER])
ax[0].set_ylabel("final val BPC (lower=better)")
ax[0].set_title("BPC vs tokenizer at each model size N\n(square = argmin = V_opt)")
ax[0].legend(); ax[0].grid(alpha=0.3)

# Right: V_opt vs N log-log with N^{2/3} reference
Ns = np.array([r["N"] for r in vopt], dtype=float)
Vs = np.array([r["V"] for r in vopt], dtype=float)
ax[1].loglog(Ns, Vs, "ko-", lw=2, ms=10, label="measured V_opt(N)")
for r in vopt:
    ax[1].annotate(f"  {r['v_opt']}", (r["N"], r["V"]), fontsize=9)
ref = Ns ** (2.0 / 3.0)
ax[1].loglog(Ns, Vs[0] * ref / ref[0], "r--", lw=1.5, label=r"reference $N^{2/3}$")
exp = np.polyfit(np.log(Ns), np.log(Vs), 1)[0]
ax[1].set_title(f"V_opt vs N  (log-log)\nfitted exponent = {exp:.3f}  (~2/3 = embedding-tax ceiling)")
ax[1].set_xlabel("model params N"); ax[1].set_ylabel("optimal vocab V_opt")
ax[1].legend(); ax[1].grid(alpha=0.3, which="both")

fig.suptitle("Vocabulary-size vs model-scale relationship (FRSMASH, English, 1000 steps)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(RUN, "n_sweep.png"), dpi=120)
plt.close(fig)

print("fitted V_opt ~ N^%.3f" % exp)
print("\nV_opt(N) table:")
for r in vopt:
    frac = {"char": 0.10, "bpe500": 0.11, "bpe1500": 0.12, "bpe3000": 0.13, "bpe6000": 0.15}.get(r["v_opt"], 0)
    print(f"  N={r['N']/1e6:>4.0f}M  V_opt={r['v_opt']:8s} V={r['V']:5d}  emb_frac~{frac:.0%}")
print("\nsaved", os.path.join(RUN, "n_sweep.png"))
