"""Export the complete matched-agent contrast families after all runs finish."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.final_validation.common import OUT, fingerprint, interval, write_json

GEMMA = "gemma-3-12b-it"
QWEN = "Qwen3-8B"
CONDITIONS = ("original", "temperature0", "shuffle", "anonymous", "warmup", "always_stats")
NUMERICAL = ("tool_mean", "tool_upper", "tool_pbest", "ts", "flat_10", "flat_100", "flat_1000", "empirical_bayes")


def required_paths(root):
    return [root / f"q4_ablation/{model}/{condition}/summary.json"
            for model, conditions in ((GEMMA, CONDITIONS), (QWEN, ("original",)))
            for condition in conditions] + [root / "q4_statistics/summary.json", root / "q4_prior_controls/summary.json"]


def export(root=OUT):
    paths = required_paths(root)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Matched-agent comparisons await completed runs:\n" + "\n".join(missing))
    agents, summaries, rows, conditions = {}, {}, [], []
    for model, variants in ((GEMMA, CONDITIONS), (QWEN, ("original",))):
        for variant in variants:
            data = json.loads((root / f"q4_ablation/{model}/{variant}/summary.json").read_text())
            summaries[model, variant] = data
            decisions = sum(len(r["regret_runs"]) * 500 for r in data["results"])
            for r in data["results"]:
                v = np.asarray(r["regret_runs"], float)
                if v.shape != (30,) or not np.isfinite(v).all():
                    raise ValueError(f"Incomplete pool vector: {model}/{variant}/{r['tag']}")
                agents[model, variant, r["tag"]] = v
                conditions.append({"model": model, "condition": variant, "configuration": r["tag"],
                    "regret": interval(v)["mean"], "regret_ci95": interval(v)["half_width_95"],
                    "statistics_exposure_fraction": r["tool_call_rate"], "parse_failure_fraction": r["parse_failure_rate"],
                    "tokens_per_1000": r["tokens_per_1k"],
                    "amortized_decision_seconds_per_1000": data["seconds_decisions"] / decisions * 1000,
                    "model_revision": data["model_revision"], "gpu": data["hardware"]["gpu"],
                    "resumed_at_round": data["resumed_at_round"]})
    numerical = json.loads((root / "q4_statistics/summary.json").read_text())["results"]
    priors = json.loads((root / "q4_prior_controls/summary.json").read_text())["results"]

    def contrast(family, left, right, a, b):
        delta = np.asarray(a, float) - np.asarray(b, float)
        if delta.shape != (30,) or not np.isfinite(delta).all():
            raise ValueError(f"Invalid paired comparison: {left} / {right}")
        value = interval(delta)
        pvalue = (float(stats.ttest_1samp(delta, 0).pvalue) if delta.std() > 0
                  else (1.0 if np.all(delta == 0) else 0.0))
        rows.append({"family": family, "left": left, "right": right,
                     "difference": value["mean"], "ci95": value["half_width_95"],
                     "n_pools": value["n"], "p_two_sided": pvalue})

    for (model, variant, tag), values in agents.items():
        left = f"{model}/{variant}/{tag}"
        K = tag.split("_")[0]
        if tag.endswith("_tool") and (model, variant, f"{K}_notool") in agents:
            contrast("optional_tool", left, f"{model}/{variant}/{K}_notool", values,
                     agents[model, variant, f"{K}_notool"])
        if variant != "original":
            contrast("ablation", left, f"{model}/original/{tag}", values, agents[model, "original", tag])
        if model == QWEN:
            contrast("model", left, f"{GEMMA}/original/{tag}", values, agents[GEMMA, "original", tag])
        for policy in NUMERICAL:
            if policy.startswith("flat_"):
                reference = priors[K][policy.split("_")[1]]["regret_runs"]
            elif policy == "empirical_bayes":
                reference = priors[K][policy]["regret_runs"]
            else:
                reference = numerical[K]["policies"][policy]["regret_runs"]
            contrast("numerical_baseline", left, f"{K}/{policy}", values, reference)
    # The four complete families are fixed before inspecting the new agent results.
    expected = {"optional_tool": 12, "ablation": 18, "model": 4, "numerical_baseline": 208}
    for family, count in expected.items():
        members = [r for r in rows if r["family"] == family]
        if len(members) != count:
            raise ValueError(f"Incomplete contrast family {family}: {len(members)} != {count}")
        order = sorted(members, key=lambda r: r["p_two_sided"])
        running = 0.0
        for i, row in enumerate(order):
            running = max(running, min(1.0, (count - i) * row["p_two_sided"]))
            row["p_holm_within_family"] = running
    for name, table in (("q4_matched_conditions", conditions), ("q4_matched_contrasts", rows)):
        path = root / f"tables/{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    result = {"provenance": fingerprint({"contrasts_per_family": expected, "pools": 30,
                 "horizon": 500, "interval": "pointwise paired 95% Student-t",
                 "multiplicity": "Holm within each complete listed family"}, [Path(__file__), *paths]),
              "conditions": conditions, "contrasts": rows,
              "notes": ["Negative differences favor the left policy. All runs use the frozen legacy response bank.",
                        "In always_stats, the saved tool_called flag records statistics exposure on every round; there is no separate model-generated tool request.",
                        "Pools are paired across conditions; model and direction seeds are not additional independent samples.",
                        "Intervals are pointwise; Holm adjustments apply to p-values, not confidence intervals.",
                        "One generation seed is used per condition. This does not estimate repeated-generation uncertainty.",
                        "Decision-loop timing is amortized over all concurrent agents, excludes model startup and fresh prior probes, and is not single-request latency.",
                        "Headline-ranking prior controls in the earlier numerical study use historical rankings, not the new agent's fresh probes."]}
    write_json(root / "q4_comparisons/summary.json", result)
    return result


if __name__ == "__main__":
    result = export()
    print(f"Exported {len(result['conditions'])} conditions and {len(result['contrasts'])} complete paired contrasts")
