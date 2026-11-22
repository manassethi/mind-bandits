"""Export complete experiment tables and an explicit evidence/readiness audit."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.final_validation.common import OUT, file_hash, interval, write_json
from scripts.final_validation.q1_allocation import CONDITIONS
from scripts.final_validation.q2_controls import MATCHED


def uniform_contextual_regret(path):
    import torch
    with np.load(path) as data:
        P = data["P"]
        seed, horizon = int(data["seed"]), int(data["horizon"])
        K, lifetime = int(data["n_live"]), int(data["lifetime"])
    S, U, M = P.shape
    generator = torch.Generator(device="cpu").manual_seed(seed + 7)
    users = torch.randint(U, (S, horizon), generator=generator).numpy()
    live, next_article = np.arange(K), K
    result = np.zeros(S)
    for t in range(horizon):
        if t > 0 and t % max(1, lifetime // K) == 0 and next_article < M:
            live[(t // max(1, lifetime // K) - 1) % K] = next_article
            next_article += 1
        probabilities = P[np.arange(S)[:, None], users[:, t, None], live[None, :]]
        result += probabilities.max(1).astype(np.float64) - probabilities.mean(1, dtype=np.float64)
    return result


def paired(left, right):
    a, b = np.asarray(left, float), np.asarray(right, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) != 30:
        raise ValueError("Expected paired results for the same 30 evaluation pools")
    return interval(a - b)


def export_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def adjust_holm(rows):
    """Adjust the complete comparison family, retaining all reported cells."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_two_sided"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p_two_sided"]))
        rows[index]["p_holm"] = running


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strict", action="store_true", help="fail until every required result and integrity check passes")
    args = p.parse_args()
    missing, checks, findings = [], [], {}

    def load(relative):
        path = OUT / relative
        if not path.exists():
            missing.append(relative)
            return None
        value = json.loads(path.read_text())
        if "provenance" not in value and relative != "q3_recheck/results.json":
            checks.append({"check": f"{relative} has provenance", "passed": False})
        return value

    def check_vector(name, actual, reported):
        actual = np.asarray(actual, float)
        checks.append({"check": name,
            "passed": actual.shape == (30,) and bool(np.isfinite(actual).all()) and
                      bool(np.allclose(actual, reported["runs"], rtol=0, atol=1e-9))})

    temporal = load("temporal_simulator/summary.json")
    if temporal:
        protocol = temporal["protocol"]
        checks += [{"check": "fit/selection/refit/calibration dates are chronologically disjoint where required",
                    "passed": max(protocol["fit_dates"]) < min(protocol["selection_dates"]) and
                              max(protocol["refit_dates"]) < min(protocol["calibration_dates"])},
                   {"check": "evaluation-day development history is disclosed",
                    "passed": "Previously inspected" in protocol["evaluation_status"]}]
        findings["temporal_simulator"] = temporal["results"]
        export_csv(OUT / "tables/temporal_simulator.csv", [{"family": kind, "calibration": method,
            **{key: result[key] for key in ("auc", "group_auc", "log_loss", "brier", "ece", "mean_pred", "observed_ctr")}}
            for kind, family in temporal["results"].items() for method, result in family["methods"].items()])
    q1_rows = []
    for condition, horizon in CONDITIONS:
        name = f"{condition}_T{horizon}"
        data = load(f"q1_allocation/{name}/summary.json")
        if not data:
            continue
        for batch, results in data["results"].items():
            for label, result in results.items():
                run_path = OUT / f"q1_allocation/{name}/B{batch}_{label}.npz"
                with np.load(run_path) as saved:
                    checks.append({"check": f"{name}/B{batch}/{label} summary reproduces stored regret",
                                   "passed": bool(np.allclose(saved["regret"], result["regret"]["runs"], rtol=0, atol=1e-9))})
                checks.append({"check": f"{name}/B{batch}/{label} has 30 finite regret observations",
                    "passed": result["regret"]["n"] == 30 and np.isfinite(result["regret"]["runs"]).all().item()})
                q1_rows.append({"condition": name, "feedback_batch": batch, "policy": label,
                    "regret": result["regret"]["mean"], "regret_ci95": result["regret"]["half_width_95"],
                    "terminal_correct": result["terminal_correct"]["mean"],
                    "terminal_simple_regret": result["terminal_simple_regret"]["mean"],
                    "paired_minus_ts": result["paired_minus_ts"]["mean"],
                    "paired_minus_ts_ci95": result["paired_minus_ts"]["half_width_95"]})
    export_csv(OUT / "tables/q1_allocation.csv", q1_rows)
    for track in ("q1_simulator_family", "q1_batch_retune", "q4_statistics", "q4_prior_controls", "q4_tool_audit", "statistics_runtime"):
        data = load(f"{track}/summary.json")
        if data:
            findings[track] = data["results"]
            if track in ("q4_statistics", "q4_prior_controls"):
                checks.append({"check": f"{track} uses the same immutable response bank as the agents",
                    "passed": data["provenance"]["inputs_sha256"].get("results/q4/envs.npz") == file_hash(Path("results/q4/envs.npz"))})
                with np.load("results/q4/envs.npz") as environment, np.load(OUT / track / "choices.npz") as saved:
                    for K in (5, 10):
                        records = data["results"][f"K{K}"]
                        if track == "q4_statistics":
                            records = records["policies"]
                        for policy, result in records.items():
                            key = (f"K{K}/{policy}/choice" if track == "q4_statistics" else
                                   f"K{K}/empirical_bayes" if policy == "empirical_bayes" else
                                   f"K{K}/strength{policy}")
                            choices = saved[key]
                            valid = choices.shape == (30, 500) and bool(((choices >= 0) & (choices < K)).all())
                            checks.append({"check": f"{track}/K{K}/{policy} retains complete valid choices", "passed": valid})
                            if valid:
                                mu = environment[f"K{K}_mu"]
                                regret = (mu.max(1)[:, None] - mu[np.arange(30)[:, None], choices]).cumsum(1)[:, -1]
                                checks.append({"check": f"{track}/K{K}/{policy} regret reproduces saved choices",
                                    "passed": bool(np.allclose(regret, result["regret_runs"], rtol=0, atol=1e-6))})
    q4_rows = []
    baseline, q4_vectors = {}, {}
    for model, conditions in [("gemma-3-12b-it", ("original", "temperature0", "shuffle", "anonymous", "warmup", "always_stats")),
                               ("Qwen3-8B", ("original",))]:
        for condition in conditions:
            data = load(f"q4_ablation/{model}/{condition}/summary.json")
            if data:
                for source, expected_hash in data["provenance"]["inputs_sha256"].items():
                    local = Path(source)
                    if local.is_absolute():
                        # Cluster fingerprints retain their original absolute
                        # source path; audit the corresponding checkout file.
                        for anchor in ("scripts", "src", "results"):
                            marker = f"/{anchor}/"
                            if marker in source:
                                local = Path(anchor) / source.split(marker, 1)[1]
                                break
                    checks.append({"check": f"{model}/{condition} input hash matches {local}",
                                   "passed": local.is_file() and file_hash(local) == expected_hash})
                    if local == Path("scripts/final_validation/q4_ablation.py") and not checks[-1]["passed"]:
                        # Completed conditions retain the source they executed;
                        # the later always-stats prompt fix has a separate hash.
                        snapshot = local.parent / "source_versions" / f"q4_ablation_{expected_hash}.py"
                        checks[-1] = {"check": f"{model}/{condition} executed source matches immutable snapshot {snapshot}",
                                      "passed": snapshot.is_file() and file_hash(snapshot) == expected_hash}
                revision = data.get("model_revision")
                checks.append({"check": f"{model}/{condition} uses a recorded immutable model revision",
                    "passed": isinstance(revision, str) and len(revision) == 40 and
                              all(c in "0123456789abcdef" for c in revision) and
                              revision == data["provenance"]["config"].get("revision")})
                if model == "gemma-3-12b-it":
                    checks.append({"check": f"{model}/{condition} uses the same locked model weights",
                        "passed": revision == "96b6f1eccf38110c56df3a15bffe176da04bfd80"})
                else:
                    checks.append({"check": f"{model}/{condition} uses locked Qwen weights with thinking disabled",
                        "passed": revision == "b968826d9c46dd6066d109eabc6255188de91218" and
                                  data["provenance"]["config"].get("chat_template_kwargs") == {"enable_thinking": False}})
                with np.load("results/q4/envs.npz") as environment, np.load(OUT / f"q4_ablation/{model}/{condition}/decisions.npz") as decisions:
                    for result in data["results"]:
                        tag = result["tag"]
                        K = int(tag.split("_")[0][1:])
                        choices = decisions[f"{tag}/choice"]
                        valid = choices.shape == (30, 500) and bool(((choices >= 0) & (choices < K)).all())
                        checks.append({"check": f"{model}/{condition}/{tag} has complete valid decisions", "passed": valid})
                        if valid:
                            mu = environment[f"K{K}_mu"]
                            chosen_p = mu[np.arange(30)[:, None], choices]
                            regret = (mu.max(1)[:, None] - chosen_p).cumsum(1)[:, -1]
                            rewards = environment[f"K{K}_uniforms"][:, :500] < chosen_p
                            checks.append({"check": f"{model}/{condition}/{tag} regret reproduces actual choices",
                                "passed": bool(np.allclose(regret, result["regret_runs"], rtol=0, atol=1e-6))})
                            checks.append({"check": f"{model}/{condition}/{tag} every saved reward matches the paired draw",
                                "passed": bool(np.array_equal(rewards, decisions[f"{tag}/reward"]))})
                            for metric, log_key in (("parse_failure_rate", "parse_failed"), ("tool_call_rate", "tool_called")):
                                checks.append({"check": f"{model}/{condition}/{tag} {metric} reproduces the decision log",
                                    "passed": bool(np.isclose(decisions[f"{tag}/{log_key}"].mean(), result[metric], rtol=0, atol=1e-12))})
                            token_count = decisions[f"{tag}/prompt_tokens"].sum() + decisions[f"{tag}/completion_tokens"].sum()
                            checks.append({"check": f"{model}/{condition}/{tag} token usage reproduces the decision log",
                                "passed": bool(np.isclose(token_count / choices.size * 1000, result["tokens_per_1k"], rtol=0, atol=1e-6))})
                            if tag.endswith("_notool"):
                                checks.append({"check": f"{model}/{condition}/{tag} contains no tool invocations",
                                    "passed": not bool(decisions[f"{tag}/tool_called"].any())})
                            if condition == "warmup":
                                checks.append({"check": f"{model}/{condition}/{tag} initially displays every arm once",
                                    "passed": bool(np.array_equal(choices[:, :K], np.broadcast_to(np.arange(K), (30, K))))})
                                checks.append({"check": f"{model}/{condition}/{tag} forced initial coverage uses no model tokens or tools",
                                    "passed": not any(bool(decisions[f"{tag}/{key}"][:, :K].any())
                                                      for key in ("tool_called", "prompt_tokens", "completion_tokens"))})
                            if condition == "always_stats":
                                checks.append({"check": f"{model}/{condition}/{tag} records statistics exposure on every round",
                                    "passed": bool(decisions[f"{tag}/tool_called"].all())})
                for result in data["results"]:
                    checks.append({"check": f"{model}/{condition}/{result['tag']} has 30 pools",
                                   "passed": len(result["regret_runs"]) == 30})
                    if condition == "original":
                        baseline[(model, result["tag"])] = result["regret_runs"]
                    q4_vectors[(model, condition, result["tag"])] = result["regret_runs"]
                    q4_rows.append({"model": model, "condition": condition, "configuration": result["tag"],
                        "regret": result["regret"][0], "regret_ci95": result["regret"][1],
                        "stuck": result["stuck"], "tokens_per_1k": result["tokens_per_1k"],
                        "parse_failure_fraction": result["parse_failure_rate"]})
    for row in q4_rows:
        key = (row["model"], row["configuration"])
        if row["condition"] != "original" and key in baseline:
            difference = paired(q4_vectors[(row["model"], row["condition"], row["configuration"])], baseline[key])
            row["paired_minus_matched_baseline"] = difference["mean"]
            row["paired_minus_matched_baseline_ci95"] = difference["half_width_95"]
    export_csv(OUT / "tables/q4_ablation.csv", q4_rows)
    from scripts.final_validation.q4_comparisons import export as export_q4, required_paths
    if all(path.exists() for path in required_paths(OUT)):
        matched_agents = export_q4(OUT)
        checks.append({"check": "matched-agent exports retain all 26 conditions and 242 paired contrasts",
                       "passed": len(matched_agents["conditions"]) == 26 and len(matched_agents["contrasts"]) == 242})
    q2 = load("q2_controls/summary.json")
    if q2:
        rows, comparisons = [], {}
        uniform = {}
        for key, result in q2["results"].items():
            task = result["task"]
            if task["split"] != "test":
                continue
            with np.load(OUT / f"q2_controls/runs/{key}.npz") as saved:
                checks.append({"check": f"{key} summary reproduces stored regret",
                               "passed": bool(np.allclose(saved["regret"][:, -1], result["regret"]["runs"], rtol=0, atol=1e-9))})
            environment = task["environment"]
            if environment not in uniform:
                uniform[environment] = uniform_contextual_regret(OUT / f"q2_controls/envs/{environment}.npz")
            denominator = uniform[environment]
            normalized = interval(np.array(result["regret"]["runs"]) / denominator) if (denominator > 1e-8).all() else None
            rows.append({**task, "regret": result["regret"]["mean"],
                         "regret_ci95": result["regret"]["half_width_95"],
                         "expected_ctr": result["expected_ctr"]["mean"],
                         "cold_start_regret": result["cold_start_regret_per_eligible_round"]["mean"],
                         "regret_fraction_of_uniform": normalized["mean"] if normalized else None,
                         "regret_fraction_of_uniform_ci95": normalized["half_width_95"] if normalized else None})
            checks.append({"check": f"{key} has 30 finite pool observations",
                           "passed": result["regret"]["n"] == 30 and np.isfinite(result["regret"]["runs"]).all().item()})
            if environment == "test_gap0":
                checks.append({"check": f"{key} zero-gap cumulative regret is zero",
                               "passed": bool(np.allclose(result["regret"]["runs"], 0, rtol=0, atol=1e-6))})
        primary = json.loads((OUT / "q2_controls/environment_manifest.json").read_text())["primary_family"]
        for group in MATCHED:
            comparisons[group] = {}
            for d in (16, 32, 64):
                values = {}
                for kind in ("dense", "sae"):
                    rep = f"{group}_{kind}"
                    policy = q2["chosen"][f"{rep}_d{d}"]
                    key = f"test_{primary}__{rep}__d{d}__{policy}"
                    values[kind] = q2["results"][key]["regret"]["runs"]
                comparisons[group][f"d{d}_train_tuned"] = paired(values["sae"], values["dense"])
            dense_key = f"test_{primary}__{group}_dense__d32__linucb_0.1"
            sae_key = f"test_{primary}__{group}_sae__d32__linucb_0.1"
            comparisons[group]["d32_common_alpha"] = paired(q2["results"][sae_key]["regret"]["runs"],
                                                              q2["results"][dense_key]["regret"]["runs"])
        findings["q2_sae_minus_dense"] = comparisons
        export_csv(OUT / "tables/q2_controls.csv", rows)
        controls = {"dense": ("dense",), "raw_neurons": ("neurons",),
                    "random_projection_mean": tuple(f"rp_s{s}" for s in (11, 23, 47)),
                    "random_threshold_mean": tuple(f"threshold_s{s}" for s in (11, 23, 47))}
        contrast_rows = []
        for group in MATCHED:
            for dimension in (16, 32, 64):
                for tuning in ("train_tuned", "common_alpha"):
                    if tuning == "common_alpha" and dimension != 32:
                        continue
                    def observations(kind):
                        rep = f"{group}_{kind}"
                        policy = q2["chosen"][f"{rep}_d{dimension}"] if tuning == "train_tuned" else "linucb_0.1"
                        return np.array(q2["results"][f"test_{primary}__{rep}__d{dimension}__{policy}"]["regret"]["runs"])
                    sae_values = observations("sae")
                    for name, kinds in controls.items():
                        # Average controls inside each pool; seeds do not create
                        # additional independent evaluation environments.
                        control_values = np.mean([observations(k) for k in kinds], axis=0)
                        difference = paired(sae_values, control_values)
                        p_value = stats.ttest_rel(sae_values, control_values).pvalue
                        contrast_rows.append({"group": group, "dimension": dimension, "tuning": tuning,
                            "control": name, "sae_minus_control": difference["mean"],
                            "ci95": difference["half_width_95"], "n_pools": 30,
                            "control_seeds": len(kinds), "p_two_sided": float(p_value) if np.isfinite(p_value) else 1.0})
        adjust_holm(contrast_rows)
        findings["q2_all_matched_control_contrasts"] = contrast_rows
        export_csv(OUT / "tables/q2_matched_contrasts.csv", contrast_rows)
        manifest = json.loads((OUT / "q2_controls/environment_manifest.json").read_text())
        earlier = np.load(OUT / "temporal_simulator/earlier_article_rows.npz")["article_rows"]
        with np.load(OUT / "q2_controls/envs/test_strict.npz") as strict:
            checks.append({"check": "strict streams exclude every earlier candidate and history article ID",
                           "passed": not np.isin(strict["articles"], earlier).any().item()})
            fit_rows = np.load(OUT / "q2_controls/fit_article_rows.npy")
            checks.append({"check": "strict streams excluded from feature-reduction and normalization fitting rows",
                           "passed": not np.isin(strict["articles"], fit_rows).any().item()})
        for key, path in manifest["environments"].items():
            if not key.startswith("test_"):
                continue
            local = OUT / "q2_controls/envs" / Path(path).name
            with np.load(local) as env:
                P = env["P"]
                checks.append({"check": f"{key} probabilities finite and bounded",
                    "passed": np.isfinite(P).all().item() and float(P.min()) >= 0 and float(P.max()) <= 1})
    wide = load("q2_wide_controls/summary.json")
    if q2 and wide:
        for group in MATCHED:
            for dimension in (16, 32, 64):
                for tuning in ("train_tuned", "common_alpha"):
                    if tuning == "common_alpha" and dimension != 32:
                        continue
                    rep = f"{group}_sae"
                    policy = q2["chosen"][f"{rep}_d{dimension}"] if tuning == "train_tuned" else "linucb_0.1"
                    sae = q2["results"][f"test_{primary}__{rep}__d{dimension}__{policy}"]["regret"]["runs"]
                    for kind in ("wide_rp", "wide_threshold"):
                        controls = []
                        for seed in (11, 23, 47):
                            rep = f"{group}_{kind}_s{seed}"
                            policy = wide["chosen"][f"{rep}_d{dimension}"] if tuning == "train_tuned" else "linucb_0.1"
                            controls.append(wide["results"][f"test_{primary}__{rep}__d{dimension}__{policy}"]["regret"]["runs"])
                        control = np.mean(controls, axis=0)
                        difference = paired(sae, control)
                        p_value = stats.ttest_rel(sae, control).pvalue
                        contrast_rows.append({"group": group, "dimension": dimension, "tuning": tuning,
                            "control": kind + "_mean", "sae_minus_control": difference["mean"],
                            "ci95": difference["half_width_95"], "n_pools": 30, "control_seeds": 3,
                            "p_two_sided": float(p_value) if np.isfinite(p_value) else 1.0})
        strict_controls = {"dense": ("dense",), "raw_neurons": ("neurons",),
            **{name: tuple(f"{prefix}_s{s}" for s in (11, 23, 47)) for name, prefix in
               (("random_projection_mean", "rp"), ("random_threshold_mean", "threshold"),
                ("wide_rp_mean", "wide_rp"), ("wide_threshold_mean", "wide_threshold"))}}
        for group in MATCHED:
            for tuning in ("train_tuned", "common_alpha"):
                def strict_values(kind):
                    rep = f"{group}_{kind}"
                    policy = wide["chosen"][f"{rep}_d32"] if tuning == "train_tuned" else "linucb_0.1"
                    return wide["results"][f"test_strict__{rep}__d32__{policy}"]["regret"]["runs"]
                sae = strict_values("sae")
                for name, kinds in strict_controls.items():
                    control = np.mean([strict_values(k) for k in kinds], axis=0)
                    difference = paired(sae, control)
                    p_value = stats.ttest_rel(sae, control).pvalue
                    contrast_rows.append({"group": group, "dimension": 32, "tuning": tuning,
                        "environment": "strict", "control": name, "sae_minus_control": difference["mean"],
                        "ci95": difference["half_width_95"], "n_pools": 30, "control_seeds": len(kinds),
                        "p_two_sided": float(p_value) if np.isfinite(p_value) else 1.0})
        # A single multiplicity family covers every displayed main/strict contrast.
        for row in contrast_rows:
            row.setdefault("environment", "main")
        adjust_holm(contrast_rows)
        export_csv(OUT / "tables/q2_matched_contrasts.csv", contrast_rows)
        for key, result in wide["results"].items():
            with np.load(OUT / f"q2_wide_controls/runs/{key}.npz") as saved:
                checks.append({"check": f"{key} wide-control summary reproduces stored regret",
                    "passed": bool(np.allclose(saved["regret"][:, -1], result["regret"]["runs"], rtol=0, atol=1e-9))})
            checks.append({"check": f"{key} wide control has 30 finite pool observations",
                "passed": result["regret"]["n"] == 30 and bool(np.isfinite(result["regret"]["runs"]).all())})
        findings["q2_wide_controls"] = wide["results"]
    geometry = []
    fit_path = OUT / "q2_controls/fit_article_rows.npy"
    for root in (OUT / "q2_controls/features", OUT / "q2_wide_controls/features"):
        if root.exists():
            for path in sorted(root.glob("*.npy")):
                features = np.load(path, mmap_mode="r")
                checks.append({"check": f"{path.parent.parent.name}/{path.stem} feature matrix is finite",
                               "passed": bool(np.isfinite(features).all())})
                if fit_path.exists():
                    X = np.asarray(features[np.load(fit_path), :32], np.float64)
                    X -= X.mean(0)
                    scale = X.std(0)
                    active = scale > 1e-8
                    X = X[:, active] / scale[active]
                    if X.shape[1]:
                        correlation = X.T @ X / len(X)
                        eigenvalues = np.linalg.eigvalsh(correlation).clip(0)
                        effective_rank = float(eigenvalues.sum()**2 / (eigenvalues**2).sum())
                        off_diagonal = np.abs(correlation)[~np.eye(len(correlation), dtype=bool)]
                    else:
                        effective_rank, off_diagonal = None, np.array([])
                    geometry.append({"source": path.parent.parent.name, "representation": path.stem,
                        "dimensions": min(32, features.shape[1]), "active_columns": int(active.sum()),
                        "training_correlation_participation_rank": effective_rank,
                        "training_mean_absolute_off_diagonal_correlation": float(off_diagonal.mean()) if len(off_diagonal) else None,
                        "scope": "D32 article-feature geometry on fitting rows; diagnostic, not a policy-covariance or causal estimate."})
    export_csv(OUT / "tables/feature_geometry.csv", geometry)
    seeds = load("simulator_seeds/summary.json")
    seed_models = load("simulator_seeds/model_summary.json")
    if seeds:
        findings["simulator_seed_effects"] = {}
        for seed in (0, 11):
            for case in ("main", "strict"):
                reference = seeds["results"][f"{case}_seed{seed}__none"]["regret"]["runs"]
                findings["simulator_seed_effects"][f"{case}_seed{seed}"] = {
                    rep: paired(result["regret"]["runs"], reference)
                    for key, result in seeds["results"].items()
                    if key.startswith(f"{case}_seed{seed}__") and (rep := key.split("__")[1]) != "none"}
    if seed_models:
        findings["simulator_seed_metrics"] = seed_models["results"]
    alternate = load("alternate_encoder/summary.json")
    alternate_model = load("alternate_encoder/model_summary.json")
    if alternate:
        findings["teacher_encoder_sae_minus_dense"] = {}
        for teacher in ("qwen", "bge"):
            findings["teacher_encoder_sae_minus_dense"][teacher] = {
                group: paired(alternate["results"][f"{teacher}_main__{group}_sae__linucb_0.1"]["regret"]["runs"],
                              alternate["results"][f"{teacher}_main__{group}_dense__linucb_0.1"]["regret"]["runs"])
                for group in MATCHED}
    if alternate_model:
        findings["alternate_teacher_predictive_metrics"] = alternate_model["metrics"]
    q3 = load("q3_interventions/summary.json")
    if q3:
        findings["full_ucb_interventions"] = q3["results"]
        for name, result in q3["results"].items():
            checks.append({"check": f"{name} recorded UCB decisions reconstruct within numeric tolerance",
                           "passed": result["reconstruction_max_score_gap"] <= 1e-7})
            with np.load(OUT / "q3_interventions/baselines.npz") as baseline:
                base = baseline[f"{name}/base_regret"]
            for count in (1, 4, 8, 16):
                runs = {}
                for kind in ("top", "bottom", *[f"random{i}" for i in range(5)]):
                    with np.load(OUT / f"q3_interventions/retrain/{name}__k{count}_{kind}.npz") as saved:
                        runs[kind] = saved["regret"][:, -1] - base
                random = np.mean([r for k, r in runs.items() if k.startswith("random")], axis=0)
                record = result["retrain"][str(count)]
                for kind, values in (("top", runs["top"]), ("bottom", runs["bottom"]), ("random", random),
                                     ("paired_top_minus_random", runs["top"] - random)):
                    check_vector(f"{name}/k{count}/{kind} retraining summary reproduces stored runs", values, record[kind])
    redundancy = load("q3_redundancy/summary.json")
    if redundancy:
        findings["redundancy_and_subset_sufficiency"] = redundancy["results"]
        tasks = json.loads((OUT / "q3_redundancy/tasks.json").read_text())
        correlations = json.loads((OUT / "q3_redundancy/train_correlations.json").read_text())
        checks.append({"check": "redundancy experiment retains all 126 matched intervention runs",
                       "passed": len(tasks) == 126 and len({(t["group"], t["name"]) for t in tasks}) == 126})
        for group, experiments in redundancy["results"].items():
            for experiment, result in experiments.items():
                selected = [t for t in tasks if t["group"] == group and t["experiment"] == experiment]
                vectors = {}
                for task in selected:
                    with np.load(OUT / f"q3_redundancy/runs/{group}__{task['name']}.npz") as saved:
                        vectors[task["kind"]] = saved["regret"][:, -1]
                    drop = task["drop"]
                    checks.append({"check": f"{group}/{task['name']} uses unique valid feature IDs",
                                   "passed": len(set(drop)) == len(drop) and all(0 <= d < 32 for d in drop)})
                top_task = next(t for t in selected if t["kind"] == "top")
                if experiment.startswith("closure_top"):
                    count = int(experiment.removeprefix("closure_top"))
                    corr = np.asarray(correlations[group]["correlation"])
                    rank = q3["results"][group]["train_rank"]
                    expected_drop = np.flatnonzero((np.abs(corr[:, rank[:count]]) >= .7).any(1))
                    checks.append({"check": f"{group}/{experiment} is the specified one-hop training correlation neighborhood",
                                   "passed": np.array_equal(sorted(top_task["drop"]), expected_drop)})
                else:
                    count = int(experiment.removeprefix("keep"))
                    expected_drop = np.setdiff1d(np.arange(32), q3["results"][group]["train_rank"][:count])
                    checks.append({"check": f"{group}/{experiment} retains only the specified train-ranked subset",
                                   "passed": np.array_equal(sorted(top_task["drop"]), expected_drop)})
                checks.append({"check": f"{group}/{experiment} matches removal counts for top, bottom and every random mask",
                               "passed": len(selected) == 7 and len({len(t["drop"]) for t in selected}) == 1})
                random = np.mean([vectors[f"random{i}"] for i in range(5)], axis=0)
                for kind, values in (("top", vectors["top"]), ("bottom", vectors["bottom"]),
                                     ("random", random), ("paired_top_minus_random", vectors["top"] - random)):
                    check_vector(f"{group}/{experiment}/{kind} redundancy summary reproduces stored runs", values, result[kind])
    transferred = load("q3_teacher_transfer/summary.json")
    if transferred:
        findings["feature_necessity_teacher_transfer"] = transferred["results"]
        for group, result in transferred["results"].items():
            with np.load(OUT / f"q3_teacher_transfer/runs/{group}__unmasked.npz") as saved:
                base = saved["regret"][:, -1]
            check_vector(f"{group} teacher-transfer baseline reproduces stored regret", base, result["baseline"])
            for count in (1, 4, 8, 16):
                runs = {}
                for kind in ("top", "bottom", *[f"random{i}" for i in range(5)]):
                    with np.load(OUT / f"q3_teacher_transfer/runs/{group}__k{count}_{kind}.npz") as saved:
                        runs[kind] = saved["regret"][:, -1]
                random = np.mean([r for k, r in runs.items() if k.startswith("random")], axis=0)
                record = result["counts"][str(count)]
                for kind, values in (("top_change", runs["top"] - base), ("bottom_change", runs["bottom"] - base),
                                     ("random_change", random - base), ("paired_top_minus_random", runs["top"] - random)):
                    check_vector(f"{group}/k{count}/{kind} transfer summary reproduces stored runs", values, record[kind])
        necessity = []
        if q3:
            for teacher, results in (("qwen", q3["results"]), ("bge", transferred["results"])):
                for group, result in results.items():
                    values = result["retrain"] if teacher == "qwen" else result["counts"]
                    for count, record in values.items():
                        effect = record["paired_top_minus_random"]
                        p_value = stats.ttest_1samp(effect["runs"], 0).pvalue
                        necessity.append({"teacher": teacher, "group": group, "removed_coordinates": int(count),
                            "top_minus_random": effect["mean"], "ci95": effect["half_width_95"], "n_pools": 30,
                            "p_two_sided": float(p_value) if np.isfinite(p_value) else 1.0})
            adjust_holm(necessity)
            export_csv(OUT / "tables/q3_necessity_contrasts.csv", necessity)
    corrected = load("q3_recheck/results.json")
    if corrected:
        findings["corrected_original_q3"] = {name: {"test_regret": r["test_regret"],
             "rank_agreement_train_vs_test": r["rank_agreement_train_vs_test_spearman"],
             "top8_retraining_minus_random": r["retrain"]["8"]["top_minus_random"]} for name, r in corrected.items()}
    for track in ("synthetic", "semantic_checks", "semantic_comparison"):
        data = load(f"{track}/summary.json")
        if data:
            findings[track] = data.get("results", data.get("contextual"))
            if track == "semantic_comparison":
                checks.append({"check": "Fresh semantic-label encoder matches cached article embedding space",
                    "passed": data["embedding_space_check"]["passed"] and data["embedding_space_check"]["min_cosine"] >= .999})
                coordinates = [(r["group"], r["kind"], r["coordinate"]) for r in data["features"]]
                checks.append({"check": "All 672 prespecified semantic coordinates retained without duplicates",
                    "passed": len(coordinates) == len(set(coordinates)) == 672})
    semantic_path = OUT / "semantic_comparison/summary.json"
    if q3 and semantic_path.exists():
        semantic = json.loads(semantic_path.read_text())
        alignment, alignment_rows = {}, []
        for group in MATCHED:
            importance = q3["results"][group]["train_importance"]
            features = [r for r in semantic["features"] if r["group"] == group and r["kind"] == "sae"
                        and r["matched_auc"] is not None]
            correlation = stats.spearmanr([importance[r["coordinate"]] for r in features],
                                         [r["matched_auc"] for r in features]).statistic if len(features) >= 3 else np.nan
            alignment[group] = {"n_scored_coordinates": len(features),
                "semantic_auc_vs_train_importance_spearman": float(correlation) if np.isfinite(correlation) else None,
                "scope": "Exploratory within-layer association. Coordinates are dependent; no independent-feature significance claim."}
            alignment_rows += [{**r, "train_importance": importance[r["coordinate"]]} for r in features]
        findings["semantic_quality_vs_policy_use"] = alignment
        export_csv(OUT / "tables/semantic_policy_alignment.csv", alignment_rows)
    failures = [check for check in checks if not check["passed"]]
    audit = {"all_required_outputs_present": not missing, "missing": missing, "checks": checks,
        "failed_checks": failures, "experiment_packet_ready": not missing and not failures,
        "scope": "Computational evidence audit. Final claims, novelty assessment, and report writing require interpretation of the complete results.",
        "findings": findings}
    write_json(OUT / "readiness.json", audit)
    print("Missing", len(missing), "Failed checks", len(failures), "Passed checks", len(checks) - len(failures))
    for item in missing:
        print("PENDING", item)
    if args.strict and (missing or failures):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
