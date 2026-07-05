"""Analyze the LLM tokenizer-comparison runs: plot BPC vs steps, BPC vs chars-seen,
and final-BPC vs alpha. Honest dual-axis framing (equal-compute vs equal-info)."""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
WORK = r"F:\OpenASH2605\critical_tokenization\llm_verify"
RUN = os.path.join(WORK, f"runs_{LANG}")
CACHE = os.path.join(WORK, f"caches_{LANG}")
OUT = RUN

with open(os.path.join(CACHE, "meta.json")) as f:
    META = json.load(f)

ORDER = ["char", "bpe500", "bpe1500", "bpe3000", "bpe6000"] if LANG == "en" \
    else ["byte", "char", "bpe8000", "bpe10000", "bpe12000"]
COLORS = {"char": "#1f77b4", "byte": "#17becf", "bpe500": "#ff7f0e",
          "bpe1500": "#2ca02c", "bpe3000": "#d62728", "bpe6000": "#9467bd",
          "bpe8000": "#ff7f0e", "bpe10000": "#d62728", "bpe12000": "#9467bd"}


def load(name):
    with open(os.path.join(RUN, f"log_{name}.json")) as f:
        return json.load(f)


def main():
    data = {n: load(n) for n in ORDER if os.path.exists(os.path.join(RUN, f"log_{n}.json"))}

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for n, d in data.items():
        steps = [r["step"] for r in d["log"]]
        bpc = [r["val_bpc"] for r in d["log"]]
        chars = [r["chars_seen"] / 1e6 for r in d["log"]]
        lbl = f"{n} (a={META[n]['alpha']:.2f})"
        ax[0].plot(steps, bpc, "o-", color=COLORS.get(n, "gray"), label=lbl, ms=3)
        ax[1].plot(chars, bpc, "o-", color=COLORS.get(n, "gray"), label=lbl, ms=3)
    ax[0].set_title("val BPC vs optimizer steps  [EQUAL COMPUTE / equal tokens]")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("val bits-per-char")
    ax[1].set_title("val BPC vs chars-seen  [EQUAL INFORMATION]")
    ax[1].set_xlabel("characters seen (millions)"); ax[1].set_ylabel("val bits-per-char")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3); a.set_ylim(bottom=1.7 if LANG == "en" else 4.5)
    fig.suptitle(f"FRSMASH v3.6 (~8M params, 1000 steps) tokenizer comparison [{LANG.upper()}]", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"bpc_curves_{LANG}.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    alphas = [META[n]["alpha"] for n in data]
    bpcs = [data[n]["final_bpc"] for n in data]
    names = list(data.keys())
    ax.plot(alphas, bpcs, "o-", color="#222")
    for a, b, nm in zip(alphas, bpcs, names):
        ax.annotate(nm, (a, b), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axvline(1.0, color="green", ls="--", label="alpha=1 (critical)")
    ax.set_xlabel("tokenizer Zipf exponent alpha"); ax.set_ylabel("final val BPC")
    ax.set_title(f"Final BPC vs alpha [{LANG.upper()}]  (U-shape => critical optimum at a=1)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"bpc_vs_alpha_{LANG}.png"), dpi=120)
    plt.close(fig)

    print("\n== final BPC (equal 1000 steps / equal tokens) ==")
    for n in sorted(data, key=lambda k: data[k]["final_bpc"]):
        print(f"  {n:9s} a={META[n]['alpha']:.3f} cpt={META[n]['cpt']:.2f} P={META[n]['params']/1e6:.2f}M  BPC={data[n]['final_bpc']:.4f}")

    print("\n== BPC @ equal chars-seen = 15M (information efficiency) ==")
    target = 15.0
    for n in sorted(data, key=lambda k: -1):
        log = data[n]["log"]
        bpcs = np.array([r["val_bpc"] for r in log])
        chars = np.array([r["chars_seen"] / 1e6 for r in log])
        if (chars <= target).any():
            i = np.where(chars <= target)[0][-1]
            print(f"  {n:9s} BPC@15Mchars={bpcs[i]:.4f} (step {log[i]['step']})")
        else:
            print(f"  {n:9s} never reached 15M chars in log")


if __name__ == "__main__":
    main()
