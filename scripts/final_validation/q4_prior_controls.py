"""Separate information about the base CTR from the LLM's headline ordering."""

import json
from pathlib import Path

import numpy as np

from scripts.analyze_q4 import metrics
from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q4_statistics import BASE_CTR, simulate


def empirical_bayes(mu, uniforms, alpha, beta, seed):
    rng = np.random.default_rng(seed)
    S, K = mu.shape
    shows, clicks = np.zeros((S, K)), np.zeros((S, K))
    rows = np.arange(S)
    choices = np.empty(uniforms.shape, np.int16)
    for t in range(uniforms.shape[1]):
        arm = rng.beta(alpha + clicks, beta + shows - clicks).argmax(1)
        reward = uniforms[:, t] < mu[rows, arm]
        shows[rows, arm] += 1
        clicks[rows, arm] += reward
        choices[:, t] = arm
    return choices


def main():
    root = OUT / "q4_prior_controls"
    training_files = [Path(f"results/q1/envs/train_K{k}.npz") for k in (2, 5, 10, 20, 50)]
    provenance = fingerprint({"strengths": [10, 100, 1000], "prior_ctr": BASE_CTR, "seed": SEED,
                              "status": "exploratory control isolating base-rate knowledge from rank information"},
                             [Path(__file__), Path("scripts/final_validation/q4_statistics.py"),
                              OUT / "q4_statistics/summary.json", Path("results/q4/envs.npz"), *training_files])
    if is_complete(root / "summary.json", provenance):
        return
    env = np.load("results/q4/envs.npz")
    original = json.loads((OUT / "q4_statistics/summary.json").read_text())["results"]
    prior_rates = []
    for path in training_files:
        with np.load(path) as training:
            prior_rates.extend(training["P"].mean(1).ravel().tolist())
    mean, variance = float(np.mean(prior_rates)), float(np.var(prior_rates))
    concentration = mean * (1 - mean) / max(variance, 1e-10) - 1
    if concentration <= 0:
        raise ValueError("Training moments do not define a Beta prior")
    alpha, beta = mean * concentration, (1 - mean) * concentration
    results, arrays = {}, {}
    for K in (5, 10):
        mu, uniforms = env[f"K{K}_mu"], env[f"K{K}_uniforms"]
        result = {}
        for strength in (10, 100, 1000):
            choice = simulate(mu, uniforms, "ts", SEED + 10 * K,
                              prior=np.full(mu.shape, BASE_CTR), strength=strength)
            metrics_ = metrics(choice, mu)
            regret = np.array(metrics_["regret_runs"])
            existing = original[f"K{K}"]["policies"]
            metrics_["paired_minus_uninformative_ts"] = interval(regret - np.array(existing["ts"]["regret_runs"]))
            metrics_["paired_llm_prior_minus_flat"] = interval(np.array(existing[f"prior_llm_{strength}"]["regret_runs"]) - regret)
            metrics_["paired_shuffled_prior_minus_flat"] = interval(np.array(existing[f"prior_shuffled_{strength}"]["regret_runs"]) - regret)
            result[str(strength)] = metrics_
            arrays[f"K{K}/strength{strength}"] = choice
            print(K, strength, metrics_["regret"], flush=True)
        choice = empirical_bayes(mu, uniforms, alpha, beta, SEED + 10 * K)
        result["empirical_bayes"] = metrics(choice, mu)
        regret = np.array(result["empirical_bayes"]["regret_runs"])
        result["empirical_bayes"]["paired_llm_tool_minus_empirical_bayes"] = interval(
            np.array(original[f"K{K}"]["policies"]["llm_tool"]["regret_runs"]) - regret)
        arrays[f"K{K}/empirical_bayes"] = choice
        results[f"K{K}"] = result
    save_npz(root / "choices.npz", **arrays)
    write_json(root / "summary.json", {"provenance": provenance, "results": results,
        "empirical_prior": {"alpha": alpha, "beta": beta, "training_mean": mean,
                            "training_variance": variance, "training_article_rate_samples": len(prior_rates)}})


if __name__ == "__main__":
    main()
