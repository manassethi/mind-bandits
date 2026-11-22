"""Warm, batched decision-compute timings for the numerical Q4 baselines."""

import json
from pathlib import Path
import platform
import time

import numpy as np

from scripts.final_validation.common import OUT, SEED, fingerprint, is_complete, write_json
from scripts.final_validation.q4_prior_controls import empirical_bayes
from scripts.final_validation.q4_statistics import simulate


def main():
    path = OUT / "statistics_runtime/summary.json"
    prior_path = OUT / "q4_prior_controls/summary.json"
    provenance = fingerprint({"seed": SEED, "repeats": 3, "draws_pbest": 4000, "pools": 30, "rounds": 500},
                             [Path(__file__), Path("scripts/final_validation/q4_statistics.py"), prior_path,
                              Path("results/q4/envs.npz")])
    if is_complete(path, provenance):
        return
    env = np.load("results/q4/envs.npz")
    prior = json.loads(prior_path.read_text())["empirical_prior"]
    results = {}
    for K in (5, 10):
        mu, uniforms = env[f"K{K}_mu"], env[f"K{K}_uniforms"]
        results[f"K{K}"] = {}
        for rule in ("mean", "upper", "pbest", "ts", "empirical_bayes"):
            timings, cpu = [], []
            for repeat in range(3):
                before_wall, before_cpu = time.perf_counter(), time.process_time()
                if rule == "empirical_bayes":
                    choices = empirical_bayes(mu, uniforms, prior["alpha"], prior["beta"], SEED)
                else:
                    choices = simulate(mu, uniforms, rule, SEED, draws=4000)
                timings.append(time.perf_counter() - before_wall)
                cpu.append(time.process_time() - before_cpu)
                assert choices.shape == uniforms.shape
            n = uniforms.size
            results[f"K{K}"][rule] = {"wall_seconds": timings, "cpu_seconds": cpu,
                "median_wall_seconds_per_1k_decisions": float(np.median(timings) * 1000 / n),
                "decisions_per_repeat": n}
            print(K, rule, results[f"K{K}"][rule]["median_wall_seconds_per_1k_decisions"], flush=True)
    write_json(path, {"provenance": provenance, "results": results, "hostname": platform.node(),
        "scope": "Warm-process decision computation, batched across 30 pools. Excludes imports, data loading, and model fitting. Compare with LLM amortized decision-loop time, not single-request latency."})


if __name__ == "__main__":
    main()
