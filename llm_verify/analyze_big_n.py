"""Plot extended V-N: 4/12/32M (clean, 1000-step) + 50/100/150M (600-step, unstable).

Honest framing: the 50-150M V_opt points are single-seed undertrained and
non-monotonic (0.3+ BPC swings) -> they do NOT cleanly extend the V~N^(2/3) law.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"F:\OpenASH2605\critical_tokenization\llm_verify\big_n_sweep"
big = json.load(open(os.path.join(OUT, "summary.json")))
V_OF = {"bpe1500": 1500, "bpe6000": 6000, "bpe10000": 10000, "bpe15846": 16001}

pinned = [(4e6, 1501), (12e6, 3001), (32e6, 6001)]
big_pts = []
for nk in ["N50M", "N100M", "N150M"]:
    rd = big[nk]
    N = float(int(nk[1:-1]) * 1e6)
    vopt = min(rd, key=lambda k: rd[k]["bpc"])
    big_pts.append((N, V_OF[vopt], vopt, rd))

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Left: BPC vs V for the three big-N rows (shows instability)
for N, vopt_v, vopt_k, rd in big_pts:
    xs = [V_OF[k] for k in rd]
    ys = [rd[k]["bpc"] for k in rd]
    ax[0].plot(xs, ys, "o-", lw=2, label=f"N={N/1e6:.0f}M (V_opt={vopt_v})")
ax[0].set_xscale("log"); ax[0].set_xlabel("vocab V"); ax[0].set_ylabel("val BPC")
ax[0].set_title("BPC vs V at 50/100/150M (600-step, single seed)\nnon-monotone => undertraining instability")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

# Right: V_opt vs N log-log
pe = np.array(pinned)
ax[1].loglog(pe[:, 0], pe[:, 1], "ko-", lw=2, ms=10, label="pinned V_opt (4-32M, 1000-step)")
ref_x = np.array([3e6, 200e6])
ax[1].loglog(ref_x, pe[0, 1] * (ref_x / pe[0, 0]) ** (2/3), "r--", lw=1.3, label=r"$N^{2/3}$ (fitted 0.666 on 4-32M)")
for N, vopt_v, vopt_k, rd in big_pts:
    ax[1].loglog([N], [vopt_v], "^", ms=14, color="orange", mfc="none", mec="orange", mew=2)
    ax[1].annotate(f"  {vopt_v}", (N, vopt_v), fontsize=8, color="orange")
ax[1].loglog([p[0] for p in big_pts], [p[1] for p in big_pts], "--", color="orange", lw=1, alpha=0.5,
             label="50-150M V_opt (UNSTABLE, single-seed)")
exp = np.polyfit(np.log(pe[:, 0]), np.log(pe[:, 1]), 1)[0]
ax[1].set_title(f"V_opt vs N\npinned exponent (4-32M) = {exp:.3f};  50-150M unstable (undertrained)")
ax[1].set_xlabel("model params N"); ax[1].set_ylabel("V_opt")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

fig.suptitle("V-N law extension to 50/100/150M: undertraining blocks a clean measurement", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "big_n_sweep.png"), dpi=120)
plt.close(fig)

print(f"pinned exponent (4-32M) = {exp:.3f}")
print("\nbig-N V_opt (600-step, single-seed, UNSTABLE):")
for N, vopt_v, vopt_k, rd in big_pts:
    print(f"  N={N/1e6:.0f}M: V_opt={vopt_v}  " + " ".join(f"{k}:{rd[k]['bpc']:.3f}" for k in rd))
print("\nConclusion: 50-150M do NOT cleanly extend V~N^(2/3); need Chinchilla-token training + multi-seed.")
