"""Plot gamma vs DPO over-optimization, with local metrics diverging."""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"F:\OpenASH2605\critical_tokenization\llm_verify\dpo_gamma"
R = json.load(open(os.path.join(OUT, "results.json")))
stages = ["baseline", "light", "heavy"]
g_g = [R[s]["greedy"]["gamma"] for s in stages]
g_t = [R[s]["T0.8"]["gamma"] for s in stages]
d2 = [R[s]["T0.8"]["distinct2"] for s in stages]
rep4 = [R[s]["T0.8"]["rep4"] for s in stages]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(stages, g_g, "o-", color="#d62728", lw=2.5, label="gamma (greedy)")
ax.plot(stages, g_t, "s-", color="#ff7f0e", lw=2.5, label="gamma (T0.8)")
ax.axhline(0.147, color="green", ls="--", lw=1, label="human gamma ~0.147")
ax.set_ylabel("gamma (long-range structure; UP = degraded)", color="#d62728")
ax.tick_params(axis="y", labelcolor="#d62728")
ax.set_xlabel("DPO over-optimization stage")
ax2 = ax.twinx()
ax2.plot(stages, d2, "^--", color="#1f77b4", lw=1.5, label="distinct-2 (T0.8)")
ax2.plot(stages, rep4, "v--", color="#9467bd", lw=1.5, label="rep4 (T0.8)")
ax2.set_ylabel("local metrics (distinct-2 / rep4)")
ax.set_title("gamma tracks DPO over-optimization while local metrics diverge\n(gamma UP = degraded; distinct-2 UP & rep4 DOWN misleadingly suggest 'better')")
l1, la1 = ax.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, la1 + la2, fontsize=8, loc="center left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dpo_gamma.png"), dpi=120)
plt.close(fig)
print("saved dpo_gamma.png")
print(f"greedy gamma: {g_g}  (drift {g_g[-1]-g_g[0]:+.3f})")
print(f"T0.8 gamma:   {g_t}  (drift {g_t[-1]-g_t[0]:+.3f})")
print(f"T0.8 distinct-2: {d2}  | rep4: {[round(r,3) for r in rep4]}  (local metrics say 'better' while gamma says 'worse')")
