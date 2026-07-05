"""Orchestrate all critical-tokenization verification experiments."""
import json
import os
import time

import experiments as ex

RESULTS = ex.RESULTS


def main():
    summary = {}
    for lang in ["en", "cn"]:
        print(f"\n===== {lang.upper()} =====")
        t = time.time()
        sweep = ex.exp_vocab_sweep(lang)
        print(f"[sweep] V* (alpha~1) = {sweep['v_star']}  alpha={sweep['alpha_at_v_star']:.3f}  "
              f"bpe_cap={sweep['bpe_cap']}  ({time.time()-t:.0f}s)")
        t = time.time()
        crit = ex.exp_criticality_test(lang, v_target=min(sweep["v_star"], 2000))
        print(f"[criticality] gamma_power={crit['gamma_power_law']:.3f} (R2={crit['r2_power_law']:.3f}) "
              f"vs exp R2={crit['r2_exponential']:.3f}  gamma_shuffled={crit['gamma_shuffled']:.3f}  ({time.time()-t:.0f}s)")
        t = time.time()
        mdl = ex.exp_mdl_tradeoff(lang)
        for beta, opt in mdl["optima"].items():
            print(f"[mdl] beta={beta}: argmin V={opt['vocab']}  (V*={mdl['v_star']})")
        t = time.time()
        cmpres = ex.exp_tokenizer_compare(lang)
        for row in cmpres["rows"]:
            print(f"[compare] {row['name']:9s} V={row['vocab']:6d} alpha={row['alpha']:.3f} "
                  f"gamma={row['gamma']:.3f} uni_bpc={row['uni_bpc']:.3f} tri_bpc={row['tri_bpc']:.3f}")
        summary[lang] = dict(
            v_star=sweep["v_star"], alpha_at_v_star=sweep["alpha_at_v_star"],
            gamma_at_v_star=dict(zip(sweep["vocabs"], sweep["gamma"])).get(sweep["v_star"]),
            gamma_shuffled=crit["gamma_shuffled"],
            r2_power_law=crit["r2_power_law"], r2_exponential=crit["r2_exponential"],
            mdl_optima=mdl["optima"],
        )

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nResults written to", RESULTS)
    for f in sorted(os.listdir(RESULTS)):
        print("  ", f)


if __name__ == "__main__":
    main()
