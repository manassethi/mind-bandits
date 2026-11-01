"""Q3 summary across SAE layers: results/q3/summary.md and figures/*.png.

    python -m scripts.analyze_q3
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.run_q3 import OUT, RETRAIN_KS

NAMES = {"g12b_sae_L41": "Gemma-3-12B, layer 41 (main)", "g12b_sae_L31": "Gemma-3-12B, layer 31",
         "g1b_sae_L17": "Gemma-3-1B, layer 17"}
TOP, RAND, INK, MUTED, GRID = "#2a78d6", "#8a8984", "#1f1e1d", "#5f5e5a", "#ecebe6"


def style(ax, xlabel, ylabel, title):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)


def label(x: dict) -> str:
    return x["label"] or f"(unlabeled f{x['feature_id']})"


def main() -> None:
    res = json.loads((OUT / "results.json").read_text())
    md = ["# Q3 results: reading and testing the SAE-feature policy\n",
          "LinUCB on SAE features (d = 32) in the Q2 main environment. Feature importance is ranked on "
          "train-day runs only; everything else is measured on the 30 test-day runs. Importance = how much "
          "a feature's score contribution differs across the live articles of a decision (a contribution "
          "shared by all articles cannot change the choice).\n",
          "## Overview\n",
          "| Layer | Test regret | Top-10 labeled | Importance rank: train vs test (Spearman) | "
          "Split-half stability (test) | Post-hoc: top-4 removal changes | Random-4 changes | "
          "Retrain: top-4 − random-4 regret |",
          "|---|---|---|---|---|---|---|---|"]
    for rep, r in res.items():
        p4, t4 = r["posthoc"]["4"], r["retrain"]["4"]["top_minus_random"]
        md.append(f"| {NAMES[rep]} | {r['test_regret'][0]:,.0f} ± {r['test_regret'][1]:,.0f} | {r['top10_labeled']}/10 | "
                  f"{r['rank_agreement_train_vs_test_spearman']:.2f} | {r['split_half_spearman']:.2f} | "
                  f"{p4['top_change']:.0%} of decisions | {p4['random_change'][0]:.0%} | "
                  f"{t4[0]:+.0f} ± {t4[1]:.0f} clicks |")

    for rep, r in res.items():
        md += [f"\n## {NAMES[rep]}\n", "### Top 10 features (train-ranked)\n",
               "w_article: weight on the feature for everyone; w_personal: extra weight when the user's "
               "history has the feature. ± = 95% CI over test runs.\n",
               "| Rank | Feature | Neuronpedia label | w_article | w_personal |", "|---|---|---|---|---|"]
        for x in r["top10"]:
            md.append(f"| {x['rank_train']} | {x['feature_id']} | {label(x)} | {x['w_article']:+.3f} ± {x['w_article_ci']:.3f} | "
                      f"{x['w_personal']:+.3f} ± {x['w_personal_ci']:.3f} |")
        md += ["\n### Post-hoc ablation (final test policies, recorded decisions)\n",
               "| k removed | Top-k: decisions changed | Random-k: decisions changed | Top-k: clicks lost / 1k | "
               "Random-k: clicks lost / 1k | Top-k beats random draws |", "|---|---|---|---|---|---|"]
        for k, v in r["posthoc"].items():
            md.append(f"| {k} | {v['top_change']:.0%} | {v['random_change'][0]:.0%} ± {v['random_change'][1]:.0%} | "
                      f"{v['top_clicks_lost_per_1k']:.1f} | {v['random_clicks_lost_per_1k'][0]:.1f} ± "
                      f"{v['random_clicks_lost_per_1k'][1]:.1f} | {v['top_change_percentile_vs_random']:.0%} |")
        md += ["\n### Retrain ablation (features removed before learning; paired regret increase)\n",
               "| k removed | Top-k | Random-k (mean of 5 draws) | Top − random |", "|---|---|---|---|"]
        for k, v in r["retrain"].items():
            md.append(f"| {k} | {v['top_regret_increase'][0]:+.0f} ± {v['top_regret_increase'][1]:.0f} | "
                      f"{v['random_regret_increase'][0]:+.0f} ± {v['random_regret_increase'][1]:.0f} | "
                      f"{v['top_minus_random'][0]:+.0f} ± {v['top_minus_random'][1]:.0f} |")
        md += ["\n### What the policy favors, by reader group\n",
               "Everyone: highest effective weights. Groups (users' most-read category, ≥5% of users): "
               "concepts weighted more for that group than for everyone.\n",
               "| Group | Share | Top concepts |", "|---|---|---|"]
        for g, v in r["groups"].items():
            md.append(f"| {g.strip('_')} | {v['share']:.0%} | {'; '.join(label(t) for t in v['top'])} |")
        md.append(f"\nHand-check sheet: `results/q3/{rep}/handcheck.csv`. Full weights: `results/q3/{rep}/weights.csv`.")

    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    plot(res)
    print((OUT / "summary.md").read_text())


def plot(res: dict) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.8), dpi=150, sharey=True)
    for ax, (rep, r) in zip(axes, res.items()):
        ks = [int(k) for k in r["posthoc"]]
        top = [r["posthoc"][str(k)]["top_change"] for k in ks]
        rm = np.array([r["posthoc"][str(k)]["random_change"][0] for k in ks])
        rh = np.array([r["posthoc"][str(k)]["random_change"][1] for k in ks])
        ax.plot(ks, top, color=TOP, lw=2, marker="o", ms=5, label="Top-k features removed")
        ax.plot(ks, rm, color=RAND, lw=2, ls="--", marker="o", ms=5, label="Random k removed")
        ax.fill_between(ks, rm - rh, rm + rh, color=RAND, alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        style(ax, "Features removed (k of 32)", "Decisions changed" if ax is axes[0] else "", NAMES[rep])
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "posthoc_ablation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    width = 0.38
    x = np.arange(len(res) * len(RETRAIN_KS))
    names, tops, rands, th, rh = [], [], [], [], []
    for rep, r in res.items():
        for k in RETRAIN_KS:
            v = r["retrain"][str(k)]
            short = NAMES[rep].split(" (")[0].replace("Gemma-3-", "").replace(", layer ", " L")
            names.append(f"{short}\nk={k}")
            tops.append(v["top_regret_increase"][0]); th.append(v["top_regret_increase"][1])
            rands.append(v["random_regret_increase"][0]); rh.append(v["random_regret_increase"][1])
    ax.bar(x - width / 2, tops, width, yerr=th, color=TOP, label="Top-k removed", error_kw={"ecolor": INK, "lw": 1})
    ax.bar(x + width / 2, rands, width, yerr=rh, color=RAND, label="Random k removed", error_kw={"ecolor": INK, "lw": 1})
    ax.set_xticks(x, names, fontsize=7.5)
    ax.axhline(0, color="#c3c2b7", lw=1)
    style(ax, "", "Extra regret vs full policy (clicks)", "Extra regret when features are removed before learning (paired, 95% CI)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "retrain_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
