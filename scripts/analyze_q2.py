"""Q2 analysis: tables and figures from results/q2.

Each representation is reported with its main policy (whichever of tuned LinUCB
and LinTS had the lower train-split regret). Regret is against the per-user
oracle; +- are 95% t-intervals over the 30 test runs; differences are paired
within runs (same users, articles and click draws).

    python -m scripts.analyze_q2
Writes results/q2/summary.md, summary.json and figures/*.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from scripts.run_q2 import DIMS, LIFETIMES, MAIN_LIFE, MODELS, OUT, task_name, tune_results

NICE = {"none": "No features (per-article TS)", "category": "Category one-hot", "bgem3": "bge-m3 (XLM-R family)",
        "qwen3": "Qwen3 embeddings (simulator's encoder: ceiling)"}
INK, MUTED, GRID = "#1f1e1d", "#5f5e5a", "#ecebe6"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE_GRAY = "#8a8984"


def color_of(rep: str) -> str:
    """One fixed colour per representation kind, so every figure agrees."""
    if rep in ("none", "category"):
        return REFERENCE_GRAY
    order = ["bgem3", "qwen3", "g1b_L", "g1b_sae", "g12b_L", "g12b_sae"]
    return COLORS[next(i for i, k in enumerate(order) if rep.startswith(k))]


def nice(rep: str) -> str:
    if rep in NICE:
        return NICE[rep]
    m, rest = rep.split("_", 1)
    size = "Gemma-3-1B" if m == "g1b" else "Gemma-3-12B"
    return f"{size} SAE (layer {rest.split('L')[1]})" if rest.startswith("sae") else f"{size} dense (layer {rest[1:]})"


def load(stage: str, task: dict) -> dict | None:
    f = OUT / stage / f"{task_name(task)}.npz"
    if not f.exists():
        return None
    with np.load(f) as z:
        return {k: z[k] for k in z.files}


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main_policy(rep: str, tuned: dict) -> str:
    from scripts.run_q2 import tuned_family
    fam = tuned[tuned_family(rep)]
    lin, ts = fam["linucb"], fam["lints"]
    return lin if fam["linucb_scores"][lin] <= fam["lints_scores"][ts] else ts


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
    ax.set_title(title, loc="left", fontsize=11, color=INK)


def main() -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tuned = tune_results()
    layers = json.loads((OUT / "chosen_layers.json").read_text())
    reps = ["category", "bgem3", "qwen3", *layers.values()]
    base = {"split": "test", "life": MAIN_LIFE}
    T = None
    md, summary = [], {"tuned": tuned, "chosen_layers": layers}

    ref = load("test", {**base, "rep": "none", "d": 0, "policy": "beta_ts"})
    T = int(ref["marks"][-1])

    # --- Main table: d = 32 (category: 18), all policies.
    md.append(f"## Main comparison (article lifetime {MAIN_LIFE}, 20 live articles, {T:,} rounds, d = 32)\n")
    md.append("Regret is vs the per-user oracle. Cold-start regret: rounds where the user's best live "
              "article is in its first 10% of life. ± = 95% CI over 30 runs; Δ columns are paired.\n")
    md.append("| Representation | Policy | Regret | Cold-start regret / round | CTR | Δ vs no features |")
    md.append("|---|---|---|---|---|---|")
    rows = [("none", "beta_ts", 0)]
    for r in reps:
        d = 18 if r == "category" else 32
        for pol in ["greedy", tuned[_fam(r)]["linucb"], tuned[_fam(r)]["lints"]]:
            rows.append((r, pol, d))
    main_rows = {}
    for r, pol, d in rows:
        res = load("test", {**base, "rep": r, "d": d, "policy": pol})
        if res is None:
            continue
        reg = res["regret"][:, -1]
        cold = res["regret_new"][:, -1] / np.maximum(res["rounds_new"][:, -1], 1)
        delta = ci(reg - ref["regret"][:, -1])
        main_rows[(r, pol)] = {"regret": ci(reg), "cold_per_round": ci(cold), "ctr": ci(res["clicks"][:, -1] / T),
                               "delta_vs_none": delta}
        star = " **(main)**" if r != "none" and pol == main_policy(r, tuned) else ""
        md.append(f"| {nice(r)} | `{pol}`{star} | {ci(reg)[0]:,.0f} ± {ci(reg)[1]:,.0f} | "
                  f"{ci(cold)[0]:.4f} ± {ci(cold)[1]:.4f} | {ci(res['clicks'][:, -1] / T)[0]:.4f} | "
                  f"{delta[0]:,.0f} ± {delta[1]:,.0f} |")
    summary["main"] = {f"{r}|{p}": v for (r, p), v in main_rows.items()}

    # --- Interpretability cost: SAE minus dense (same model), paired, main policy, every d.
    md.append("\n## Cost of interpretability: SAE − dense regret (same model, paired)\n")
    md.append("| Model | d | Dense regret | SAE regret | SAE − dense |")
    md.append("|---|---|---|---|---|")
    summary["sae_cost"] = {}
    for m in MODELS:
        dense, sae = layers[f"{m}_dense"], layers[f"{m}_sae"]
        for d in DIMS:
            a = load("test", {**base, "rep": dense, "d": d, "policy": main_policy(dense, tuned)})
            b = load("test", {**base, "rep": sae, "d": d, "policy": main_policy(sae, tuned)})
            if a is None or b is None:
                continue
            diff = ci(b["regret"][:, -1] - a["regret"][:, -1])
            summary["sae_cost"][f"{m}|{d}"] = diff
            md.append(f"| {m} | {d} | {ci(a['regret'][:, -1])[0]:,.0f} | {ci(b['regret'][:, -1])[0]:,.0f} | "
                      f"{diff[0]:,.0f} ± {diff[1]:,.0f} |")

    # --- Figures.
    plot_dims(reps, tuned, base, ref, fig_dir)
    plot_depth(tuned, layers, ref, fig_dir)
    plot_churn(reps, tuned, fig_dir, md, summary)

    header = "# Q2 results (test-day pools; settings and layers chosen on train-day pools)\n\n"
    (OUT / "summary.md").write_text(header + "\n".join(md) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print((OUT / "summary.md").read_text())


def _fam(rep):
    from scripts.run_q2 import tuned_family
    return tuned_family(rep)


def plot_dims(reps, tuned, base, ref, fig_dir):
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)
    for r in [r for r in reps if r != "category"]:
        pol = main_policy(r, tuned)
        pts = [(d, load("test", {**base, "rep": r, "d": d, "policy": pol})) for d in DIMS]
        pts = [(d, x["regret"][:, -1]) for d, x in pts if x is not None]
        if not pts:
            continue
        ds = [d for d, _ in pts]
        m = np.array([ci(x)[0] for _, x in pts])
        h = np.array([ci(x)[1] for _, x in pts])
        ax.plot(ds, m, color=color_of(r), lw=2, marker="o", ms=5, label=nice(r))
        ax.fill_between(ds, m - h, m + h, color=color_of(r), alpha=0.12, lw=0)
    cat = load("test", {**base, "rep": "category", "d": 18, "policy": main_policy("category", tuned)})
    if cat is not None:
        ax.axhline(cat["regret"][:, -1].mean(), color="#8a8984", lw=1.5, ls="--", label="Category one-hot")
    ax.axhline(ref["regret"][:, -1].mean(), color="#8a8984", lw=1.5, ls=":", label="No features (per-article TS)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS, [str(d) for d in DIMS])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax, "Feature dimension d", "Regret vs per-user oracle (clicks lost)", "Regret by representation and dimension")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "regret_by_dimension.png", bbox_inches="tight")
    plt.close(fig)


def plot_depth(tuned, layers, ref, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, sharey=True)
    for ax, (m, c) in zip(axes, MODELS.items()):
        for split, color in (("test", COLORS[0]), ("train", "#86b6ef")):
            ys, xs = [], []
            for L in range(c["layers"]):
                rep = f"{m}_L{L}"
                vals = [load("layers", {"split": split, "life": MAIN_LIFE, "rep": rep, "d": 32, "policy": p})
                        for p in (tuned[_fam(rep)]["linucb"], tuned[_fam(rep)]["lints"])]
                vals = [v["regret"][:, -1].mean() for v in vals if v is not None]
                if vals:
                    xs.append(L / c["layers"])
                    ys.append(min(vals))
            if split == "test":
                ax.plot(xs, ys, color=color_of(f"{m}_L0"), lw=2, label="Dense (every layer)")
            sx, sy = [], []
            for L in c["sae"]:
                rep = f"{m}_sae_L{L}"
                vals = [load("layers", {"split": split, "life": MAIN_LIFE, "rep": rep, "d": 32, "policy": p})
                        for p in (tuned[_fam(rep)]["linucb"], tuned[_fam(rep)]["lints"])]
                vals = [v["regret"][:, -1].mean() for v in vals if v is not None]
                if vals:
                    sx.append(L / c["layers"])
                    sy.append(min(vals))
            if split == "test":
                ax.plot(sx, sy, color=color_of(f"{m}_sae"), lw=0, marker="D", ms=7, label="SAE (labeled layers)")
        ax.axhline(ref["regret"][:, -1].mean(), color="#8a8984", lw=1.2, ls=":", label="No features")
        style(ax, "Layer depth (fraction of model)", "Regret (d = 32, best of LinUCB/LinTS)" if m == "g1b" else "",
              "Gemma-3-1B" if m == "g1b" else "Gemma-3-12B")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "regret_by_layer_depth.png")
    plt.close(fig)


def plot_churn(reps, tuned, fig_dir, md, summary):
    md.append("\n## Article churn: regret by article lifetime (d = 32, main policy)\n")
    md.append("| Representation | " + " | ".join(f"life {l}" for l in LIFETIMES) + " |")
    md.append("|---|" + "---|" * len(LIFETIMES))
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)
    summary["churn"] = {}
    for i, r in enumerate(["none", *reps]):
        pol = "beta_ts" if r == "none" else main_policy(r, tuned)
        d = 0 if r == "none" else 18 if r == "category" else 32
        cells, xs, ys = [], [], []
        for life in LIFETIMES:
            stage = "test" if life == MAIN_LIFE else "churn"
            x = load(stage, {"split": "test", "life": life, "rep": r, "d": d, "policy": pol})
            if x is None:
                cells.append("")
                continue
            m, h = ci(x["regret"][:, -1])
            cells.append(f"{m:,.0f} ± {h:,.0f}")
            xs.append(life)
            ys.append(m)
        summary["churn"][r] = dict(zip(map(str, LIFETIMES), cells))
        md.append(f"| {nice(r)} | " + " | ".join(cells) + " |")
        if xs:
            ls = ":" if r == "none" else "--" if r == "category" else "-"
            ax.plot(xs, ys, color=color_of(r), lw=2, marker="o", ms=5, ls=ls, label=nice(r))
    ax.set_xscale("log")
    ax.set_xticks(LIFETIMES, [f"{l:,}" for l in LIFETIMES])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style(ax, "Article lifetime (rounds); shorter = faster churn", "Regret vs per-user oracle",
          "Content features matter more when articles churn faster")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "regret_by_churn.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
