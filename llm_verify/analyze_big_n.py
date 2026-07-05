"""Plot extended V-N curve: 4/12/32M (pinned, 1000-step) + 50/100/150M (>=16k ceiling, 250-step)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"F:\OpenASH2605\critical_tokenization\llm_verify\big_n_sweep"
big = json.load(open(os.path.join(OUT, "summary.json")))

# pinned small-N points (old n_sweep, 1000-step, V_grid<=6k)
pinned = [(4e6, 1501), (12e6, 3001), (32e6, 6001)]
# new large-N points: V_opt at >=16k ceiling (250-step)
ceiling_V = 16001
ceil_n = []
for nk in ["N50M", "N100M", "N150M"]:
    rd = big[nk]
    N = float(int(nk[1:-1]) * 1e6)
    vopt = min(rd, key=lambda k: rd[k]["bpc"])
    ceil_n.append((N, vopt, rd[vopt]["bpc"], rd))

fig, ax = plt.subplots(figsize=(9, 6))
pe = np.array(pinned)
ax.loglog(pe[:, 0], pe[:, 1], "ko-", lw=2, ms=10, label="pinned V_opt (4-32M, 1000-step)")
for n, v in pinned:
    ax.annotate(f"  {v}", (n, v), fontsize=9)
ref_x = pe[:, 0]
ax.loglog(ref_x, pe[0, 1] * (ref_x / ref_x[0]) ** (2/3), "r--", lw=1.3, label=r"$N^{2/3}$ reference (fitted 0.666 on 4-32M)")

for N, vopt, bpc, rd in ceil_n:
    ax.loglog([N], [ceiling_V], "r^", ms=14, mfc="none", mec="red", mew=2)
    ax.annotate(f"  >=16k\n  ({vopt})", (N, ceiling_V), fontsize=8, color="red")
ax.loglog([c[0] for c in ceil_n], [ceiling_V] * len(ceil_n), "r--", lw=1, alpha=0.5,
          label="V_opt at measurement ceiling (50-150M, 250-step)")

exp = np.polyfit(np.log(pe[:, 0]), np.log(pe[:, 1]), 1)[0]
ax.set_title(f"V_opt vs N (extended)\npinned exponent (4-32M) = {exp:.3f};  50-150M saturate the >=16k ceiling")
ax.set_xlabel("model params N"); ax.set_ylabel("optimal vocab V_opt")
ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "big_n_sweep.png"), dpi=120)
plt.close(fig)

print(f"pinned exponent (4-32M) = {exp:.3f}")
print("\nlarge-N rows (250-step, V_opt at >=16k ceiling):")
for N, vopt, bpc, rd in ceil_n:
    print(f"  N={N/1e6:.0f}M: V_opt={vopt} (bpc={bpc:.3f})  " + " ".join(f"{k}:{rd[k]['bpc']:.3f}" for k in rd))
print("\nNOTE: 4-32M are 1000-step (old); 50-150M are 250-step (new) -> budgets differ;")
print("ceiling (master BPE saturated 15846 merges on 18M chars) prevents pinning V_opt at >=50M.")
