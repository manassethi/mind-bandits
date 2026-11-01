"""Q1 analysis: tune each policy family on the train split, report it on the test split.

For every condition and family, the variant with the lowest mean final regret on
train-day runs is chosen; its test-day runs are reported with 95% confidence
intervals, plus paired differences against Thompson sampling (same pools, users
and click draws within a run).

    python -m scripts.analyze_q1
Writes results/q1/summary.md, all_variants.csv, tuned.json and figures/*.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from scripts.run_q1 import CONDITIONS, OUT, load_env

FAMILIES = {  # display name -> variant-name prefix; order fixes colour slots
    "A/B test (tuned length)": "ab_frac_",
    "A/B test (power analysis)": "ab_power_",
    "ε-greedy": "egreedy_",
    "UCB1": "ucb1_",
    "Thompson sampling": "thompson",
    "Discounted Thompson": "dthompson_",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
UNIFORM_COLOR = "#8a8984"
REFERENCE = "Thompson sampling"


def load(split: str, cond: str) -> dict:
    out: dict = {}
    with np.load(OUT / split / f"{cond}.npz") as f:
        for key in f.files:
            name, field = key.split("/")
            out.setdefault(name, {})[field] = f[key]
    return out


def ci(x: np.ndarray) -> tuple[float, float]:
    """Mean and 95% t-interval half-width over runs."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def family_of(variant: str) -> str | None:
    for fam, prefix in FAMILIES.items():
        if variant == prefix or (prefix.endswith("_") and variant.startswith(prefix)):
            return fam
    return None


def tune(train: dict) -> dict:
    best = {}
    for name, r in train.items():
        fam = family_of(name)
        if fam is None:
            continue
        score = r["regret"][:, -1].mean()
        if fam not in best or score < best[fam][1]:
            best[fam] = (name, score)
    return {fam: name for fam, (name, _) in best.items()}


def fmt(m: float, h: float, digits: int = 0) -> str:
    return f"{m:,.{digits}f} ± {h:,.{digits}f}"


def main() -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tuned, summary, rows, md = {}, {}, [], []

    for cond, (K, T, drift) in CONDITIONS.items():
        train, test = load("train", cond), load("test", cond)
        choice = tune(train)
        tuned[cond] = choice
        env = load_env("test", cond)
        best_ctr = env.mu0.max(1)
        ref = test[choice[REFERENCE]]["regret"][:, -1]
        cond_sum = {"K": K, "T": T, "drift": drift,
                    "best_arm_ctr": ci(best_ctr), "personalized_oracle_ctr": ci(env.personalized_oracle_ctr())}
        for name, r in test.items():
            reg = r["regret"][:, -1]
            row = {"condition": cond, "variant": name, "family": family_of(name) or name,
                   "tuned_pick": choice.get(family_of(name)) == name,
                   "regret": ci(reg), "ctr": ci(r["exp_clicks"][:, -1] / T),
                   "best_share_final_window": ci(r["best_share"][:, -1]),
                   "regret_minus_thompson": ci(reg - ref)}
            if "committed_correct" in r:
                committed = r["commit_time"] >= 0
                row["share_committed"] = float(committed.mean())
                row["committed_correct"] = float(r["committed_correct"].mean())
            rows.append(row)
        cond_sum["families"] = {fam: next(x for x in rows if x["condition"] == cond and x["variant"] == v)
                                for fam, v in choice.items()}
        cond_sum["uniform"] = next(x for x in rows if x["condition"] == cond and x["variant"] == "uniform")
        summary[cond] = cond_sum

    (OUT / "tuned.json").write_text(json.dumps(tuned, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "all_variants.csv", "w") as f:
        f.write("condition,variant,family,tuned_pick,regret,regret_ci,ctr,ctr_ci,"
                "regret_minus_thompson,diff_ci,best_share_final,committed_correct\n")
        for r in rows:
            f.write(",".join(map(str, [r["condition"], r["variant"], r["family"], r["tuned_pick"],
                                       *np.round(r["regret"], 2), *np.round(r["ctr"], 5),
                                       *np.round(r["regret_minus_thompson"], 2),
                                       round(r["best_share_final_window"][0], 3),
                                       r.get("committed_correct", "")])) + "\n")

    # Markdown summary: one table per condition, tuned variant per family.
    for cond, s in summary.items():
        md.append(f"### {cond}: K={s['K']}, T={s['T']:,}, drift={s['drift']}\n")
        md.append(f"Best arm CTR {s['best_arm_ctr'][0]:.4f}; personalized oracle CTR "
                  f"{s['personalized_oracle_ctr'][0]:.4f} (contextual headroom).\n")
        md.append("| Policy | Tuned variant | Regret (clicks lost) | CTR | Regret − Thompson | "
                  "Plays on best arm, last window | A/B picked the best arm |")
        md.append("|---|---|---|---|---|---|---|")
        for fam, r in list(s["families"].items()) + [("Uniform (never-ending A/B)", s["uniform"])]:
            pick = "" if "committed_correct" not in r else (
                f"{r['committed_correct']:.0%}" + (f" (only {r['share_committed']:.0%} finished)"
                                                   if r["share_committed"] < 1 else ""))
            md.append(f"| {fam} | `{r['variant']}` | {fmt(*r['regret'])} | {r['ctr'][0]:.4f} | "
                      f"{fmt(*r['regret_minus_thompson'])} | {r['best_share_final_window'][0]:.0%} | {pick} |")
        md.append("")
    header = ("# Q1 results (test-day pools; settings tuned on train-day pools)\n\n"
              "Regret = expected clicks lost vs always showing the best arm, summed over the run. "
              "± is a 95% t-interval over runs; Regret − Thompson is paired within runs.\n")
    (OUT / "summary.md").write_text(header + "\n".join(md))

    plot_curves(summary, tuned, fig_dir)
    plot_by_k(summary, fig_dir)
    print((OUT / "summary.md").read_text())


def style(ax, xlabel: str, ylabel: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#5f5e5a", labelsize=9)
    ax.grid(axis="y", color="#ecebe6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color="#5f5e5a")
    ax.set_ylabel(ylabel, color="#5f5e5a")


def plot_curves(summary: dict, tuned: dict, fig_dir: Path) -> None:
    """Cumulative regret over time for the tuned variant of each family."""
    for cond in ("K10", "K50", "K10_abrupt", "K10_gradual"):
        test = load("test", cond)
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
        # Uniform is left out: its regret is ~10x the others and would flatten every curve.
        series = [(fam, tuned[cond][fam], c) for fam, c in zip(FAMILIES, COLORS)]
        for fam, variant, color in series:
            r = test[variant]
            mean = r["regret"].mean(0)
            half = stats.t.ppf(0.975, len(r["regret"]) - 1) * r["regret"].std(0, ddof=1) / np.sqrt(len(r["regret"]))
            ax.plot(r["marks"], mean, color=color, lw=2, label=f"{fam} ({variant})")
            ax.fill_between(r["marks"], mean - half, mean + half, color=color, alpha=0.12, lw=0)
        ax.set_xscale("log")
        if summary[cond]["drift"] == "abrupt":
            ax.axvline(summary[cond]["T"] // 2, color="#8a8984", lw=1, ls=":")
            ax.text(summary[cond]["T"] // 2, ax.get_ylim()[1] * 0.97, " preferences switch",
                    color="#5f5e5a", fontsize=8, va="top")
        style(ax, "Rounds (log scale)", "Cumulative regret (clicks lost)")
        s = summary[cond]
        ax.set_title(f"Cumulative regret, {s['K']} arms" + ("" if s["drift"] == "none" else f", {s['drift']} drift")
                     + f"  (uniform split: {s['uniform']['regret'][0]:,.0f}, off-chart)",
                     loc="left", fontsize=11, color="#1f1e1d")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(fig_dir / f"regret_curve_{cond}.png")
        plt.close(fig)


def plot_by_k(summary: dict, fig_dir: Path) -> None:
    """Final regret against number of arms, tuned variant per family (T = 100k, no drift)."""
    conds = ["K2", "K5", "K10", "K20", "K50"]
    ks = [summary[c]["K"] for c in conds]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    for fam, color in zip(FAMILIES, COLORS):
        m = np.array([summary[c]["families"][fam]["regret"][0] for c in conds])
        h = np.array([summary[c]["families"][fam]["regret"][1] for c in conds])
        ax.plot(ks, m, color=color, lw=2, marker="o", ms=5, label=fam)
        ax.fill_between(ks, np.maximum(m - h, 1), m + h, color=color, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax, "Number of articles (arms)", "Regret after 100k rounds (clicks lost, log scale)")
    ax.set_title("Regret grows with the number of arms", loc="left", fontsize=11, color="#1f1e1d")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "regret_by_arms.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
