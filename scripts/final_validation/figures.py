"""Render scientific diagnostics from completed, saved experiment outputs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.final_validation.common import OUT, file_hash, interval, write_json

COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
GROUPS = {"g1b17": "Gemma 1B · layer 17", "g12b12": "Gemma 12B · layer 12",
          "g12b41": "Gemma 12B · layer 41"}


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200})
    directory = OUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    manifest, sources = {}, {}

    def load(relative):
        path = OUT / relative
        if not path.exists():
            return None
        sources[relative] = file_hash(path)
        return json.loads(path.read_text())

    def save(fig, name, caption):
        for suffix in ("pdf", "png"):
            fig.savefig(directory / f"{name}.{suffix}", bbox_inches="tight", facecolor="white")
        manifest[name] = {"caption": caption, "source_hashes": dict(sources)}
        plt.close(fig)

    temporal = load("temporal_simulator/summary.json")
    if temporal:
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), layout="constrained")
        kinds = ("logistic", "bilinear", "mlp")
        names = ["Logistic", "Bilinear", "MLP"]
        for ax, metric, title in zip(axes, ("group_auc", "log_loss", "ece"),
                                     ("Within-impression AUC ↑", "Log loss ↓", "Calibration error ↓")):
            for offset, (method, label) in enumerate((("identity", "Uncalibrated"), ("intercept", "Intercept"), ("platt", "Platt"))):
                values = [temporal["results"][kind]["methods"][method][metric] for kind in kinds]
                ax.bar(np.arange(3) + (offset - 1) * .24, values, .23, color=COLORS[offset], label=label)
            ax.set_xticks(range(3), names)
            ax.set_title(title)
            ax.grid(axis="y", alpha=.15)
        axes[0].set_ylim(.5, .72)
        axes[1].set_ylim(.15, .17)
        axes[2].legend(frameon=False, fontsize=8)
        save(fig, "temporal_model_validation", "All three locked model families and calibrators on the previously inspected evaluation day. Ranking and probability quality are separate diagnostics; calibration does not establish the simulator as the true user-response process. Bars have no uncertainty intervals.")

    allocation = load("q1_allocation/K10_T10k_T10000/summary.json")
    retune = load("q1_batch_retune/summary.json")
    if allocation and retune:
        fig, ax = plt.subplots(figsize=(5.6, 3.2), layout="constrained")
        for offset, (label, data, key, color) in enumerate((("Frozen coefficient", allocation["results"], "ucb_tuned", COLORS[0]),
                                                      ("Train-retuned coefficient", retune["results"]["K10_T10k_T10000"], "ucb", COLORS[1]))):
            values = [data[str(batch)][key]["paired_minus_ts"] for batch in (1, 100, 1000)]
            x = np.arange(3) + (offset - .5) * .13
            ax.errorbar(x, [v["mean"] for v in values], yerr=[v["half_width_95"] for v in values],
                        color=color, fmt="o-", capsize=3, label=label)
        ax.axhline(0, color=".4", linestyle="--", linewidth=.8)
        ax.set_xticks(range(3), ("1", "100", "1,000"))
        ax.set_xlabel("Feedback batch size")
        ax.set_ylabel("UCB − Thompson sampling regret")
        ax.set_title("Delayed feedback and train-only retuning")
        ax.legend(frameon=False, fontsize=8)
        save(fig, "feedback_retuning", "K=10, T=10,000. Negative differences favor UCB. Points and bars show paired mean differences and 95% Student-t intervals across 30 article pools. Retuning uses training pools only and is an exploratory follow-up to the frozen-setting reversal.")

    q2 = load("q2_controls/summary.json")
    if q2:
        wide = load("q2_wide_controls/summary.json")
        primary = load("q2_controls/environment_manifest.json")["primary_family"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained", sharey=True)
        controls = {"Dense PCA": ("dense",), "Raw neurons": ("neurons",),
                    "Random projection": tuple(f"rp_s{s}" for s in (11, 23, 47)),
                    "Random threshold": tuple(f"threshold_s{s}" for s in (11, 23, 47))}
        if wide:
            controls.update({"Wide random projection": tuple(f"wide_rp_s{s}" for s in (11, 23, 47)),
                             "Wide random threshold": tuple(f"wide_threshold_s{s}" for s in (11, 23, 47))})
        for ax, (group, title) in zip(axes, GROUPS.items()):
            def values(kind):
                rep = f"{group}_{kind}"
                data = wide if kind.startswith("wide_") else q2
                policy = data["chosen"][f"{rep}_d32"]
                return data["results"][f"test_{primary}__{rep}__d32__{policy}"]["regret"]["runs"]
            sae = np.array(values("sae"))
            for i, (label, kinds) in enumerate(controls.items()):
                difference = interval(sae - np.mean([values(k) for k in kinds], axis=0))
                ax.errorbar(difference["mean"], i, xerr=difference["half_width_95"], fmt="s" if i >= 4 else "o", color=COLORS[i % 5], capsize=3)
            ax.axvline(0, color=".4", linestyle="--", linewidth=.8)
            ax.set_title(title)
            ax.set_xlabel("SAE − control regret")
            ax.grid(axis="x", alpha=.15)
        axes[0].set_yticks(range(len(controls)), list(controls))
        axes[0].invert_yaxis()
        save(fig, "matched_feature_controls", "D=32 at matched layers, using separately train-selected exploration coefficients. Negative differences favor SAE features. Random controls average three direction seeds within each of the same 30 pools; bars are paired 95% Student-t intervals. Small dictionaries have 128 directions. Wide dictionaries select 128 training-variance coordinates from 16,384 directions. Article-level thresholded directions control activation frequency and are not token-level random SAEs. Full dimension and common-coefficient grids accompany this diagnostic.")

    q3 = load("q3_interventions/summary.json")
    if q3:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
        for color, (group, title) in zip(COLORS, GROUPS.items()):
            r = q3["results"][group]
            for ax, field, key in ((axes[0], "posthoc", "paired_top_minus_random_loss"),
                                   (axes[1], "retrain", "paired_top_minus_random")):
                vals = [r[field][str(k)][key] for k in (1, 4, 8, 16)]
                ax.errorbar(range(4), [v["mean"] for v in vals], yerr=[v["half_width_95"] for v in vals], fmt="o-", color=color, capsize=2, label=title)
        for ax in axes:
            ax.axhline(0, color=".4", linestyle="--", linewidth=.8)
            ax.set_xticks(range(4), ("1", "4", "8", "16"))
            ax.set_xlabel("Removed coordinates")
        axes[0].set_title("Immediate UCB score intervention")
        axes[0].set_ylabel("Top − random loss / 1,000 decisions")
        axes[1].set_title("Retraining after removal")
        axes[1].set_ylabel("Top − random cumulative regret")
        axes[1].legend(frameon=False, fontsize=7)
        save(fig, "feature_interventions", "Train-ranked top features versus equal-sized random masks. Left: local interventions in actual pre-action UCB states, with mean and uncertainty scores recomputed. Right: policies retrained from the first round without those coordinates. Random masks are averaged within pools; intervals use 30 independent pools. These are algorithmic feature interventions under the simulator, not causal claims about real users.")

    transferred = load("q3_teacher_transfer/summary.json")
    if q3 and transferred:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), layout="constrained", sharey=True)
        for ax, (group, title) in zip(axes, GROUPS.items()):
            primary = q3["results"][group]["retrain"]
            alternate = transferred["results"][group]["counts"]
            for offset, (label, data, color) in enumerate((("Qwen simulator", primary, COLORS[0]),
                                                         ("BGE simulator", alternate, COLORS[1]))):
                values = [data[str(k)]["paired_top_minus_random"] for k in (1, 4, 8, 16)]
                ax.errorbar(np.arange(4) + (offset - .5) * .12, [v["mean"] for v in values],
                            yerr=[v["half_width_95"] for v in values], fmt="o-", color=color, capsize=2, label=label)
            ax.axhline(0, color=".4", linestyle="--", linewidth=.8)
            ax.set_xticks(range(4), ("1", "4", "8", "16"))
            ax.set_xlabel("Removed coordinates")
            ax.set_title(title)
        axes[0].set_ylabel("Top − random cumulative regret")
        axes[0].legend(frameon=False, fontsize=7)
        save(fig, "feature_necessity_across_simulators", "Retraining comparisons use the same primary training-derived ranks, top/bottom/random masks, exploration coefficients, and article/user pools under both response simulators. Positive differences indicate greater regret after top-ranked removal than after equal-sized random removal. Five random masks are averaged within each pool; intervals use 30 paired pools. The two simulator fits are sensitivity conditions, not independent replications or real-user causal evidence.")

    semantic = load("semantic_comparison/summary.json")
    if semantic:
        kinds = ("sae", "dense", "wide_rp_s11", "wide_threshold_s11", "shuffled_sae")
        labels = ("SAE", "Dense\nPCA", "Random\nprojection\n(16k)", "Random\nthreshold\n(16k)", "Shuffled\nnoise")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained", sharey=True)
        for ax, (group, title) in zip(axes, GROUPS.items()):
            values = [semantic["results"][group][k]["matched_auc"]["all_coordinates_unscored_at_chance_mean"] for k in kinds]
            ax.bar(range(5), values, color=COLORS)
            ax.axhline(.5, color=".35", linestyle="--", linewidth=.8)
            ax.set_xticks(range(5), labels, fontsize=7)
            ax.set_title(title)
            ax.set_ylim(0, 1)
            for i, kind in enumerate(kinds):
                r = semantic["results"][group][kind]
                if r["matched_auc"]["n_features"] < 32:
                    ax.text(i, values[i] + .025, f"{r['matched_auc']['n_features']}/32", ha="center", fontsize=7)
        axes[0].set_ylabel("Category-matched retrieval AUROC")
        save(fig, "automated_semantic_controls", "One blinded Gemma 12B captioner describes all first 32 coordinates from earlier-day examples. Wide controls select training-variance coordinates from 16,384 directions; a prespecified direction seed (11) is used for semantic comparisons. BGE scores caption agreement on strictly unseen news with category-matched negatives. Bars average all 32 coordinates, assigning chance to captioner abstention or insufficient held-out support; annotations indicate incomplete coverage when present. Scores conditional on adequate support are also retained. This automated proxy is not human semantic validation. Coordinates within a layer are dependent, so no independent-feature confidence bars are shown. Small-dictionary semantic controls are also retained in the complete results.")

    statistics = load("q4_statistics/summary.json")
    if statistics:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained", sharey=True)
        labels = {"tool_mean": "Posterior mean", "tool_upper": "Posterior upper bound",
                  "tool_pbest": "Probability of best", "ts": "Thompson sampling",
                  "llm_tool": "Historical LLM + tool", "llm_notool": "Historical LLM"}
        for ax, K in zip(axes, (5, 10)):
            policies = statistics["results"][f"K{K}"]["policies"]
            reference = np.array(policies["tool_mean"]["regret_runs"])
            for i, key in enumerate(labels):
                value = interval(np.array(policies[key]["regret_runs"]) - reference)
                ax.errorbar(value["mean"], i, xerr=value["half_width_95"], color=COLORS[i % 5], fmt="o", capsize=3)
            ax.axvline(0, color=".4", linestyle="--", linewidth=.8)
            ax.set_xlabel("Regret − direct posterior-mean policy")
            ax.set_title(f"{K} candidate articles")
        axes[0].set_yticks(range(len(labels)), list(labels.values()))
        axes[0].invert_yaxis()
        save(fig, "tool_only_baselines", "Historical LLM decisions and direct policies on identical fixed environments, K=5/10 and T=500. Negative differences favor the listed policy. Intervals are paired 95% Student-t intervals across 30 pools. These historical LLM runs precede the new matched ablation harness and remain explicitly labeled.")

    matched = load("q4_comparisons/summary.json")
    if matched:
        agents = {}
        for model in ("gemma-3-12b-it", "Qwen3-8B"):
            data = load(f"q4_ablation/{model}/original/summary.json")
            agents[model] = {r["tag"]: np.asarray(r["regret_runs"]) for r in data["results"]}
        priors = load("q4_prior_controls/summary.json")
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), layout="constrained", sharey=True)
        labels = ("Gemma 12B", "Gemma 12B + tool", "Qwen 8B", "Qwen 8B + tool",
                  "Thompson sampling", "Empirical-Bayes TS", "Direct posterior mean")
        for ax, K in zip(axes, (5, 10)):
            policies = statistics["results"][f"K{K}"]["policies"]
            reference = np.asarray(policies["tool_mean"]["regret_runs"])
            vectors = [agents[model][f"K{K}_{tag}"]
                       for model in ("gemma-3-12b-it", "Qwen3-8B") for tag in ("notool", "tool")]
            vectors += [policies["ts"]["regret_runs"], priors["results"][f"K{K}"]["empirical_bayes"]["regret_runs"], reference]
            for i, vector in enumerate(vectors):
                value = interval(np.asarray(vector) - reference)
                ax.errorbar(value["mean"], i, xerr=value["half_width_95"], fmt="o", capsize=3,
                            color=COLORS[(i // 2) if i < 4 else 2 + (i - 4) % 3])
            ax.axvline(0, color=".4", linestyle="--", linewidth=.8)
            ax.set_title(f"{K} candidate articles")
            ax.set_xlabel("Regret − direct posterior-mean policy")
            ax.grid(axis="x", alpha=.15)
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].invert_yaxis()
        save(fig, "matched_agent_baselines", "Fresh matched Gemma and Qwen agents on the same frozen legacy response bank, 30 pools and 500 rounds. Negative differences favor the listed agent/policy over direct posterior-mean choice. Bars are pointwise paired 95% Student-t intervals; the complete eight-baseline comparison family and Holm-adjusted p-values are retained in the tables. Headline-only prior probes are separate from online decisions. These simulations do not establish real-user effects.")

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), layout="constrained", sharey=True)
        variants = (("temperature0", "Temperature 0"), ("shuffle", "Shuffled article order"),
                    ("anonymous", "Anonymous articles"), ("warmup", "Initial arm coverage"))
        labels = [f"{label} · {mode}" for _, label in variants for mode in ("no tool", "tool")]
        labels.append("Statistics always displayed")
        for ax, K in zip(axes, (5, 10)):
            rows = [r for r in matched["contrasts"] if r["family"] == "ablation" and f"/K{K}_" in r["left"]]
            order = [(variant, mode) for variant, _ in variants for mode in ("notool", "tool")]
            order.append(("always_stats", "tool"))
            for i, (variant, mode) in enumerate(order):
                row = next(r for r in rows if r["left"].endswith(f"/{variant}/K{K}_{mode}"))
                ax.errorbar(row["difference"], i, xerr=row["ci95"], fmt="o", capsize=3,
                            color=COLORS[(i // 2) % len(COLORS)])
            ax.axvline(0, color=".4", linestyle="--", linewidth=.8)
            ax.set_title(f"{K} candidate articles")
            ax.set_xlabel("Ablated − original Gemma regret")
            ax.grid(axis="x", alpha=.15)
        axes[0].set_yticks(range(len(labels)), labels)
        axes[0].invert_yaxis()
        save(fig, "matched_agent_ablations", "Every Gemma ablation at K=5 and K=10, paired to its corresponding original tool or no-tool condition. Negative differences favor the ablation. Always-displayed statistics are compared with the original optional-tool agent. Bars are pointwise paired 95% Student-t intervals across 30 pools; all 18 contrasts and their family-wise Holm-adjusted p-values are retained. Generation uses one fixed seed per condition, so intervals do not estimate repeated-generation variability.")

    write_json(directory / "manifest.json", {"figures": manifest,
        "status": "Scientific diagnostics only; report writing remains deferred.",
        "source": str(Path(__file__)), "source_sha256": file_hash(Path(__file__))})
    print("Rendered", len(manifest), "figures")


if __name__ == "__main__":
    main()
