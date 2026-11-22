"""Repeat headline allocation comparisons with independently fitted model families."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q1_allocation import make_policy, optimized_test_size, simulate
from scripts.final_validation.q2_controls import ROOT as Q2_ROOT, load_env
from src.bandits.environment import BanditEnv

ROOT = OUT / "q1_simulator_family"


def task_run(task):
    kind, horizon = task
    source = Q2_ROOT / "population_envs"
    train, test = load_env(source / f"train_{kind}.npz"), load_env(source / f"test_{kind}.npz")
    env = BanditEnv(test["P"], test["articles"], horizon, seed=test["seed"])
    n, sizing = optimized_test_size(train["P"].mean(1), horizon, SEED)
    results, arrays = {}, {}
    tuned = json.loads(Path("results/q1/tuned.json").read_text())["K10_T10k" if horizon == 10000 else "K10"]
    fraction = float(tuned["A/B test (tuned length)"].split("_")[-1])
    sizes = {"etc_original_train_tuned": max(1, int(fraction * horizon / env.K)), "etc_reward_optimized": n}
    for label in ("etc_original_train_tuned", "etc_reward_optimized", "ucb_tuned", "ts", "kl_ucb", "uniform"):
        path = ROOT / f"{kind}_T{horizon}" / f"{label}.npz"
        if path.exists():
            with np.load(path) as f:
                result = {k: f[k] for k in f.files}
        else:
            result = simulate(make_policy(label, env, 1, sizes.get(label, n)), env, 1)
            save_npz(path, **result)
        arrays.update({f"{label}/{k}": v for k, v in result.items()})
        results[label] = result
    reference = results["ts"]["regret"]
    summary = {label: {"regret": interval(result["regret"]), "paired_minus_ts": interval(result["regret"] - reference),
                      "terminal_correct": interval(result["terminal_correct"]),
                      "terminal_simple_regret": interval(result["terminal_simple_regret"])} for label, result in results.items()}
    print(kind, horizon, {k: round(v["regret"]["mean"], 2) for k, v in summary.items()}, flush=True)
    return f"{kind}_T{horizon}", {"policies": summary, "sizing": sizing}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    provenance = fingerprint({"seed": SEED, "horizons": [10000, 100000]},
                             [Path(__file__), Q2_ROOT / "environment_manifest.json"])
    if is_complete(ROOT / "summary.json", provenance):
        return
    marker = ROOT / "provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial family comparison differs in settings")
    write_json(marker, provenance)
    tasks = [(kind, horizon) for kind in ("logistic", "bilinear", "mlp") for horizon in (10000, 100000)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = dict(pool.imap_unordered(task_run, tasks))
    write_json(ROOT / "summary.json", {"provenance": provenance, "results": results})


if __name__ == "__main__":
    main()
