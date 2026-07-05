"""Critical-tokenization verification experiments.

Each experiment returns a plain-dict result and optionally writes a plot.
Honesty policy: we report what the data shows, including results that only
PARTIALLY support the framework (e.g. Higuchi D is weakly discriminating for
symbolic sequences; fixed-capacity n-gram BPC is monotonic in vocab).
"""
from __future__ import annotations
import os
import json
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import corpus as corp
import tokenizers as tok
import metrics as mt

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

_CTX = {}

LANG_CONFIG = {
    "en": dict(loader=lambda n: corp.load_english(n), total=1_100_000,
               bpe_train=700_000, ng_train=200_000, test=100_000,
               vocabs=[200, 400, 700, 1000, 1500, 2000, 3000, 4000, 6000, 8000],
               bpe_max=12000),
    "cn": dict(loader=lambda n: corp.load_chinese(n), total=700_000,
               bpe_train=320_000, ng_train=180_000, test=120_000,
               vocabs=[3239, 3600, 4000, 4500, 5000, 5600, 6200, 6800],
               bpe_max=7000),
}


def _slices(cfg, text):
    a = cfg["bpe_train"]
    b = a + cfg["ng_train"]
    c = b + cfg["test"]
    return text[:a], text[a:b], text[b:c]


def _get_context(lang):
    if lang in _CTX:
        return _CTX[lang]
    cfg = LANG_CONFIG[lang]
    text = corp.normalize(cfg["loader"](cfg["total"]))
    bpe_train, ng_text, te_text = _slices(cfg, text)
    full = tok.BpeTokenizer(bpe_train, cfg["bpe_max"])
    ctx = dict(cfg=cfg, text=text, bpe_train=bpe_train, ng_text=ng_text,
               te_text=te_text, full=full, base_chars=sorted(set(bpe_train)))
    _CTX[lang] = ctx
    return ctx


def evaluate_tokenizer(t, ng_text, te_text):
    ids_ng = t.encode(ng_text)
    ids_te = t.encode(te_text)
    c_te = Counter(ids_te)
    c_ng = Counter(ids_ng)
    z = mt.zipf_alpha(c_te)
    mi = mt.mutual_information(ids_te, 16, vocab_size=t.actual_vocab_size)
    D = mt.higuchi_fd(mt.freq_rank_sequence(ids_te, c_ng))
    cpt = len(te_text) / max(len(ids_te), 1)
    uni_bpc = mt.unigram_bpc(ids_ng, ids_te, vocab_size=t.actual_vocab_size) / cpt
    tri = mt.ngram_bpc(ids_ng, ids_te, vocab_size=t.actual_vocab_size)
    tri_bpc = tri["bits_per_token"] / cpt
    return dict(
        vocab=t.actual_vocab_size,
        alpha=z["alpha"], alpha_r2=z["r2"], n_types=z["n_types"],
        gamma=mi["gamma"], mi=mi["mi"].tolist(), lags=mi["lags"].tolist(),
        higuchi_D=D, cpt=cpt,
        uni_bpc=uni_bpc, tri_bpt=tri["bits_per_token"], tri_bpc=tri_bpc,
        ids_te=ids_te, ids_ng=ids_ng,
    )


def exp_vocab_sweep(lang="en"):
    ctx = _get_context(lang)
    cfg, ng_text, te_text, full, base_chars = (
        ctx["cfg"], ctx["ng_text"], ctx["te_text"], ctx["full"], ctx["base_chars"])
    rows = []
    for vs in cfg["vocabs"]:
        b = full.restrict(vs)
        r = evaluate_tokenizer(b, ng_text, te_text)
        rows.append({k: v for k, v in r.items() if k not in ("ids_te", "ids_ng", "mi", "lags")})
    cap = full.actual_vocab_size
    v_star_row = min(rows, key=lambda r: abs(r["alpha"] - 1.0))
    result = dict(
        lang=lang, base_vocab=len(base_chars), bpe_cap=cap,
        vocabs=[r["vocab"] for r in rows],
        alpha=[r["alpha"] for r in rows],
        gamma=[r["gamma"] for r in rows],
        cpt=[r["cpt"] for r in rows],
        uni_bpc=[r["uni_bpc"] for r in rows],
        tri_bpc=[r["tri_bpc"] for r in rows],
        v_star=v_star_row["vocab"],
        alpha_at_v_star=v_star_row["alpha"],
        rows=rows,
    )
    with open(os.path.join(RESULTS, f"sweep_{lang}.json"), "w") as f:
        json.dump(result, f, indent=2)
    _plot_sweep(result, lang)
    return result


def _plot_sweep(res, lang):
    V = res["vocabs"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(V, res["alpha"], "o-", color="#d62728")
    ax[0, 0].axhline(1.0, color="gray", ls="--", label="alpha=1 (Zipf/critical)")
    ax[0, 0].axvline(res["v_star"], color="green", ls=":", label=f"V*={res['v_star']}")
    ax[0, 0].set_title("Zipf exponent alpha vs vocab"); ax[0, 0].set_xlabel("vocab"); ax[0, 0].set_ylabel("alpha")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(V, res["gamma"], "o-", color="#2ca02c")
    ax[0, 1].set_title("MI power-law decay gamma (small=critical)"); ax[0, 1].set_xlabel("vocab"); ax[0, 1].set_ylabel("gamma")
    ax[1, 0].plot(V, res["uni_bpc"], "o-", color="#1f77b4", label="unigram BPC (intrinsic)")
    ax[1, 0].plot(V, res["tri_bpc"], "s-", color="#ff7f0e", label="trigram BPC (capacity-limited)")
    ax[1, 0].axvline(res["v_star"], color="green", ls=":", label=f"V*={res['v_star']}")
    ax[1, 0].set_title("Bits-per-char vs vocab"); ax[1, 0].set_xlabel("vocab"); ax[1, 0].set_ylabel("bits/char")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(V, res["cpt"], "o-", color="#9467bd")
    ax[1, 1].set_title("Compression (chars/token) vs vocab"); ax[1, 1].set_xlabel("vocab"); ax[1, 1].set_ylabel("chars/token")
    fig.suptitle(f"Critical statistics vs vocabulary size [{lang.upper()}]", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"sweep_{lang}.png"), dpi=120)
    plt.close(fig)


def exp_criticality_test(lang="en", v_target=2000):
    ctx = _get_context(lang)
    ng_text, te_text, full = ctx["ng_text"], ctx["te_text"], ctx["full"]
    b = full.restrict(min(v_target, full.actual_vocab_size))
    r = evaluate_tokenizer(b, ng_text, te_text)
    lags = np.array(r["lags"])
    mi = np.array(r["mi"])
    mask = mi > 1e-8
    g_pl = -np.polyfit(np.log(lags[mask]), np.log(mi[mask]), 1)[0]
    pred_pl = np.exp(np.polyval(np.polyfit(np.log(lags[mask]), np.log(mi[mask]), 1), np.log(lags)))
    try:
        a_exp = -np.polyfit(lags[mask], np.log(mi[mask]), 1)[0]
        pred_exp = np.exp(np.polyval(np.polyfit(lags[mask], np.log(mi[mask]), 1), lags))
        ss_res_exp = np.sum((np.log(mi[mask]) - np.log(pred_exp[mask])) ** 2)
        ss_tot = np.sum((np.log(mi[mask]) - np.log(mi[mask]).mean()) ** 2)
        r2_exp = 1 - ss_res_exp / ss_tot
    except Exception:
        a_exp, r2_exp = float("nan"), float("nan")
    ss_res_pl = np.sum((np.log(mi[mask]) - np.log(pred_pl[mask])) ** 2)
    ss_tot = np.sum((np.log(mi[mask]) - np.log(mi[mask]).mean()) ** 2)
    r2_pl = 1 - ss_res_pl / ss_tot
    ids_sh = np.array(r["ids_te"])
    rng = np.random.default_rng(0)
    rng.shuffle(ids_sh)
    mi_sh = mt.mutual_information(ids_sh.tolist(), 16, vocab_size=b.actual_vocab_size)
    result = dict(
        lang=lang, vocab=b.actual_vocab_size,
        gamma_power_law=float(g_pl), r2_power_law=float(r2_pl),
        decay_exp_rate=float(a_exp), r2_exponential=float(r2_exp),
        gamma_shuffled=float(mi_sh["gamma"]),
        lags=lags.tolist(), mi=mi.tolist(), mi_shuffled=mi_sh["mi"].tolist(),
        alpha=r["alpha"],
    )
    with open(os.path.join(RESULTS, f"criticality_{lang}.json"), "w") as f:
        json.dump(result, f, indent=2)
    _plot_criticality(result, lang)
    return result


def _plot_criticality(res, lang):
    lags = np.array(res["lags"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(lags, res["mi"], "o-", color="#d62728", label=f"original (gamma={res['gamma_power_law']:.3f})")
    ax.loglog(lags, res["mi_shuffled"], "s-", color="#7f7f7f", label=f"shuffled (gamma={res['gamma_shuffled']:.3f})")
    ax.set_xlabel("lag d (tokens)"); ax.set_ylabel("Mutual information I(d) [bits]")
    ax.set_title(f"Long-range correlation / criticality test [{lang.upper()}, V={res['vocab']}]\n"
                 f"power-law R2={res['r2_power_law']:.3f} vs exponential R2={res['r2_exponential']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"criticality_{lang}.png"), dpi=120)
    plt.close(fig)


def _knee(xs, ys, frac=0.7):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 3:
        return int(x[0])
    span = y[0] - y[-1]
    if abs(span) < 1e-9:
        return int(x[-1])
    for i in range(len(x)):
        if (y[0] - y[i]) / span >= frac:
            return int(x[i])
    return int(x[-1])


def exp_mdl_tradeoff(lang="en"):
    sweep = exp_vocab_sweep(lang)
    V = np.array(sweep["vocabs"], dtype=float)
    uni = np.array(sweep["uni_bpc"])
    n_chars = LANG_CONFIG[lang]["test"]
    v_base = sweep["base_vocab"]
    vocab_cost_per_char = V * np.log2(max(v_base, 2)) / n_chars
    knee_v = _knee(V, uni)
    marginal = np.diff(uni)
    result_extra = dict(knee_v=knee_v, marginal_gain=marginal.tolist(),
                        v_star=sweep["v_star"], alpha_at_v_star=sweep["alpha_at_v_star"])
    betas = [0.02, 0.05, 0.1, 0.2]
    curves = {}
    optima = {}
    for beta in betas:
        total = uni + beta * vocab_cost_per_char
        curves[str(beta)] = total.tolist()
        i = int(np.argmin(total))
        optima[str(beta)] = dict(vocab=int(V[i]), total=float(total[i]), bpc=float(uni[i]))
    result = dict(
        lang=lang, vocabs=V.tolist(), uni_bpc=uni.tolist(),
        vocab_cost_per_char=vocab_cost_per_char.tolist(),
        betas=betas, curves=curves, optima=optima, **result_extra,
    )
    with open(os.path.join(RESULTS, f"mdl_{lang}.json"), "w") as f:
        json.dump(result, f, indent=2)
    _plot_mdl(result, lang)
    return result


def _plot_mdl(res, lang):
    V = res["vocabs"]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(V, res["uni_bpc"], "k--", label="L(z): unigram BPC (data cost)")
    for beta in res["betas"]:
        ax.plot(V, res["curves"][str(beta)], "o-", label=f"total, beta={beta}, argmin V={res['optima'][str(beta)]['vocab']}")
    ax.axvline(res["v_star"], color="green", ls=":", label=f"V* (alpha=1)={res['v_star']}")
    ax.axvline(res["knee_v"], color="purple", ls="-.", label=f"compression knee={res['knee_v']}")
    ax.set_xlabel("vocab size"); ax.set_ylabel("bits per char")
    ax.set_title(f"MDL tradeoff: data cost + beta*vocab cost [{lang.upper()}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"mdl_{lang}.png"), dpi=120)
    plt.close(fig)


def exp_tokenizer_compare(lang="en"):
    ctx = _get_context(lang)
    cfg, bpe_train, ng_text, te_text, full = (
        ctx["cfg"], ctx["bpe_train"], ctx["ng_text"], ctx["te_text"], ctx["full"])
    char_tok = tok.CharTokenizer(bpe_train)
    word_tok = tok.WordTokenizer(bpe_train)
    v_star = None
    best_d = 9
    for vs in cfg["vocabs"]:
        b = full.restrict(vs)
        z = mt.zipf_alpha(Counter(b.encode(te_text)))
        d = abs(z["alpha"] - 1.0)
        if d < best_d:
            best_d = d
            v_star = b.actual_vocab_size
    bpe_star = full.restrict(v_star if v_star else 1500)
    out = []
    for name, t in [("char", char_tok), ("word", word_tok), ("bpe(V*)", bpe_star)]:
        r = evaluate_tokenizer(t, ng_text, te_text)
        out.append(dict(name=name, vocab=r["vocab"], alpha=r["alpha"],
                        gamma=r["gamma"], higuchi_D=r["higuchi_D"],
                        cpt=r["cpt"], uni_bpc=r["uni_bpc"], tri_bpc=r["tri_bpc"]))
    result = dict(lang=lang, rows=out)
    with open(os.path.join(RESULTS, f"tokcompare_{lang}.json"), "w") as f:
        json.dump(result, f, indent=2)
    _plot_compare(result, lang)
    return result


def _plot_compare(res, lang):
    names = [r["name"] for r in res["rows"]]
    alpha = [r["alpha"] for r in res["rows"]]
    gamma = [r["gamma"] for r in res["rows"]]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    bars = ax[0].bar(names, alpha, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax[0].axhline(1.0, color="gray", ls="--")
    ax[0].set_title("Zipf alpha (target = 1.0)"); ax[0].set_ylabel("alpha")
    for b, v in zip(bars, alpha):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    bars = ax[1].bar(names, gamma, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax[1].set_title("MI decay gamma (smaller = more critical)"); ax[1].set_ylabel("gamma")
    for b, v in zip(bars, gamma):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle(f"Tokenizer type comparison [{lang.upper()}]")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, f"tokcompare_{lang}.png"), dpi=120)
    plt.close(fig)
