"""Plot hybrid verification: gamma trajectory per condition (control / low / high)."""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"F:\OpenASH2605\critical_tokenization\llm_verify\critical_hybrid"
R = json.load(open(os.path.join(OUT, "results.json")))
COL = {"control": "#888888", "low_gamma": "#2ca02c", "high_gamma": "#d62728"}
LAB = {"control": "CE only (no structure term)", "low_gamma": "CE + MI-match, target I x1.5 (more critical)",
       "high_gamma": "CE + MI-match, target I x0.5 (degraded)"}

fig, ax = plt.subplots(figsize=(9, 6))
for cond in ["control", "low_gamma", "high_gamma"]:
    tr = R[cond]
    steps = [t[0] for t in tr]
    g = [t[1] for t in tr]
    ax.plot(steps, g, "o-", color=COL[cond], lw=2.5, ms=8, label=f"{LAB[cond]}  (final gamma={g[-1]:.3f})")
ax.set_xlabel("training step"); ax.set_ylabel("generated gamma (T0.8)")
ax.set_title("Hybrid CE + MI-curve-matching: gamma IS directionally controllable\n(high-gamma target raises gamma vs CE-only control)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "critical_hybrid.png"), dpi=120)
plt.close(fig)
print("saved critical_hybrid.png")
print("\nfinal gamma:  control=%.3f  low=%.3f  high=%.3f" % (
    R["control"][-1][1], R["low_gamma"][-1][1], R["high_gamma"][-1][1]))
print("final rep4:   control=%.3f  low=%.3f  high=%.3f" % (
    R["control"][-1][2], R["low_gamma"][-1][2], R["high_gamma"][-1][2]))
