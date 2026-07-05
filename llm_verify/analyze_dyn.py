"""Plot the dynamic-tokenization experiment results (Test1 curriculum + Test2 shift)."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = r"F:\OpenASH2605\critical_tokenization\llm_verify"
RUN = os.path.join(WORK, "dyn_runs")


def load(name):
    with open(os.path.join(RUN, f"log_{name}.json")) as f:
        return json.load(f)


def main():
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Test 1
    c = {"t1_static_500": "#1f77b4", "t1_static_1500": "#2ca02c",
         "t1_static_full": "#d62728", "t1_curriculum": "#9467bd"}
    for n, col in c.items():
        d = load(n)
        steps = [r["step"] for r in d["log"]]
        key = "bpc_cur" if n != "t1_curriculum" else "bpc_full"
        bpc = [r[key] for r in d["log"]]
        ax[0].plot(steps, bpc, "-", color=col, label=f"{n} (final {bpc[-1]:.3f})", lw=2)
    ax[0].axvline(250, color="gray", ls=":", alpha=0.5); ax[0].axvline(500, color="gray", ls=":", alpha=0.5)
    ax[0].axvline(750, color="gray", ls=":", alpha=0.5)
    ax[0].set_title("TEST 1: static-data vocabulary curriculum\n(curriculum = WORST: budget-split hurts)")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("val BPC"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # Test 2
    c2 = {"t2_frozen": "#1f77b4", "t2_dynamic": "#ff7f0e", "t2_oracle": "#d62728"}
    for n, col in c2.items():
        d = load(n)
        steps = [r["step"] for r in d["log"]]
        key = "bpc_Bfull" if n != "t2_frozen" else "bpc_B1500"
        bpc = [r[key] for r in d["log"]]
        ax[1].plot(steps, bpc, "-o", color=col, label=f"{n} (final {bpc[-1]:.3f})", lw=2, ms=3)
    ax[1].axvline(600, color="black", ls="--", alpha=0.6)
    ax[1].text(610, 2.55, "A->B boundary\n(tokenizer switch)", fontsize=8)
    ax[1].set_title("TEST 2: domain-shift A->B\ndynamic = WORST: switch disruption > adaptation")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("val BPC on B"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    fig.suptitle("Dynamic/Joint Tokenization Verification (FRSMASH ~8M, matched compute)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RUN, "dyn_results.png"), dpi=120)
    plt.close(fig)
    print("saved", os.path.join(RUN, "dyn_results.png"))

    summ = json.load(open(os.path.join(RUN, "summary.json")))
    print("\nTEST 1 (static):", {k: round(v, 4) for k, v in summ.items() if k.startswith("t1")})
    print("TEST 2 (shift): ", {k: round(v, 4) for k, v in summ.items() if k.startswith("t2")})


if __name__ == "__main__":
    main()
