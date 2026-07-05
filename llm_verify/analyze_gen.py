"""Plot gamma-monitoring verification: gamma tracks generation degradation AND
catches high-T incoherence that local diversity metrics miss."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"F:\OpenASH2605\critical_tokenization\llm_verify\gen_monitor"
R = json.load(open(os.path.join(OUT, "results.json")))

gen_order = ["greedy", "T0.5", "T0.8", "T1.0", "T1.5", "T2.0"]
temps = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]
g_human = R["human"]["gamma"]
d2_human = R["human"]["distinct2"]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Left: gamma across all regimes
all_order = gen_order + ["human", "shuffled", "uniform"]
gammas = [R[k]["gamma"] for k in all_order]
cols = ["#888888"] * len(gen_order) + ["#2ca02c", "#d62728", "#d62728"]
bars = ax[0].bar(range(len(all_order)), gammas, color=cols)
ax[0].axhline(g_human, color="#2ca02c", ls="--", lw=1, label=f"human gamma={g_human:.3f}")
ax[0].set_xticks(range(len(all_order)))
ax[0].set_xticklabels(all_order, rotation=30)
ax[0].set_ylabel("gamma (MI power-law decay exponent)")
ax[0].set_title("gamma across regimes\n(generation << human; shuffled/uniform -> 0)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
for b, v in zip(bars, gammas):
    ax[0].text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)

# Right: the divergence -- gamma vs distinct-2 across temperature
g_gen = [R[k]["gamma"] for k in gen_order]
d2_gen = [R[k]["distinct2"] for k in gen_order]
ax[1].plot(temps, g_gen, "o-", color="#d62728", lw=2, label="gamma (long-range structure)")
ax[1].plot(temps, [g_human]*len(temps), "--", color="#d62728", alpha=0.4, label=f"human gamma={g_human:.3f}")
ax[1].set_xlabel("sampling temperature")
ax[1].set_ylabel("gamma", color="#d62728")
ax[1].tick_params(axis="y", labelcolor="#d62728")
ax2 = ax[1].twinx()
ax2.plot(temps, d2_gen, "s-", color="#1f77b4", lw=2, label="distinct-2 (local diversity)")
ax2.plot(temps, [d2_human]*len(temps), "--", color="#1f77b4", alpha=0.4, label=f"human d2={d2_human:.3f}")
ax2.set_ylabel("distinct-2", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax[1].set_title("HIGH-T DIVERGENCE:\ndistinct-2 rises (false 'diverse') while gamma collapses (true degradation)")
ax[1].set_xticks(temps)
l1, la1 = ax[1].get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax[1].legend(l1 + l2, la1 + la2, fontsize=7, loc="center left")
ax[1].grid(alpha=0.3)

fig.suptitle("gamma-based generation monitoring (FRSMASH bpe1500, 1000-step model)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gen_monitor.png"), dpi=120)
plt.close(fig)
print("saved", os.path.join(OUT, "gen_monitor.png"))
print("\nVERDICT:")
print(f"  human gamma={g_human:.3f}  vs  best generation (greedy)={R['greedy']['gamma']:.3f}  -> gamma separates human from model output")
print(f"  T2.0: distinct-2={R['T2.0']['distinct2']:.3f} (>{d2_human:.3f} human, looks diverse) BUT gamma={R['T2.0']['gamma']:.3f} (<<{g_human:.3f}, reveals incoherence)")
