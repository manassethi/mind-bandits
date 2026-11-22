"""Explore redundancy and subset sufficiency after the corrected removal result."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q2_controls import ROOT as Q2_ROOT, ENVS, MATCHED, load_env, policy_features
from src.bandits.contextual import run, user_features
from src.data.mind import load_processed

ROOT = OUT / "q3_redundancy"


def one_task(task):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    path = ROOT / "runs" / f"{task['group']}__{task['name']}.npz"
    if not path.exists():
        env = load_env(ENVS / f"test_{task['family']}.npz")
        X = policy_features(f"{task['group']}_sae", 32)
        X[:, task["drop"]] = 0
        imps = load_processed("test")
        a = X[env["articles"]]
        u = user_features(X, env["users"], imps.hist_ptr, imps.hist_idx)
        result = run(env, a, u, task["policy"], "cpu", seed=SEED)
        save_npz(path, **result)
    with np.load(path) as data:
        regret = data["regret"][:, -1]
    print("Redundancy check", task["group"], task["name"], flush=True)
    return task, regret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    inputs = [Path(__file__), OUT / "q3_interventions/summary.json", Q2_ROOT / "feature_manifest.json"]
    provenance = fingerprint({"seed": SEED, "correlation_threshold": .7, "closure_top_k": [1, 4, 8],
                              "keep_top_k": [4, 8, 16], "random_masks": 5,
                              "status": "exploratory follow-up to corrected inconclusive necessity tests"}, inputs)
    if is_complete(ROOT / "summary.json", provenance):
        return
    marker = ROOT / "provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial redundancy experiment has different settings")
    write_json(marker, provenance)
    prior = json.loads(inputs[1].read_text())["results"]
    family = json.loads((Q2_ROOT / "environment_manifest.json").read_text())["primary_family"]
    fit_rows = np.load(Q2_ROOT / "fit_article_rows.npy")
    tasks, clusters = [], {}
    for group in MATCHED:
        X = policy_features(f"{group}_sae", 32)
        corr = np.nan_to_num(np.corrcoef(X[fit_rows].T))
        np.fill_diagonal(corr, 1)
        order = np.array(prior[group]["train_rank"])
        policy = prior[group]["policy"]
        clusters[group] = {"correlation": corr.tolist(), "removals": {}}
        for count in (1, 4, 8):
            closure = np.flatnonzero((np.abs(corr[:, order[:count]]) >= .7).any(1))
            label = f"closure_top{count}"
            clusters[group]["removals"][label] = closure.tolist()
            rng = np.random.default_rng(SEED + count)
            masks = {"top": closure.tolist(), "bottom": order[-len(closure):].tolist()}
            masks.update({f"random{i}": rng.choice(32, len(closure), replace=False).tolist() for i in range(5)})
            for kind, drop in masks.items():
                tasks.append({"group": group, "name": f"{label}_{kind}", "drop": drop,
                              "experiment": label, "kind": kind, "family": family, "policy": policy})
        for count in (4, 8, 16):
            label = f"keep{count}"
            rng = np.random.default_rng(SEED + 100 + count)
            kept = {"top": order[:count], "bottom": order[-count:]}
            kept.update({f"random{i}": rng.choice(32, count, replace=False) for i in range(5)})
            for kind, keep in kept.items():
                tasks.append({"group": group, "name": f"{label}_{kind}", "drop": np.setdiff1d(np.arange(32), keep).tolist(),
                              "experiment": label, "kind": kind, "family": family, "policy": policy})
    write_json(ROOT / "tasks.json", tasks)
    write_json(ROOT / "train_correlations.json", clusters)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        completed = list(pool.imap_unordered(one_task, tasks))
    output = {}
    for group in MATCHED:
        output[group] = {}
        for experiment in sorted({t["experiment"] for t in tasks}):
            runs = {t["kind"]: r for t, r in completed if t["group"] == group and t["experiment"] == experiment}
            random = np.mean([r for key, r in runs.items() if key.startswith("random")], 0)
            output[group][experiment] = {"top": interval(runs["top"]), "bottom": interval(runs["bottom"]),
                "random": interval(random), "paired_top_minus_random": interval(runs["top"] - random)}
    write_json(ROOT / "summary.json", {"provenance": provenance, "results": output,
        "scope": "Necessity tests remove the train-correlated closure of important coordinates and match the random removal count. Sufficiency tests retrain on only the retained coordinates. The exploration coefficient stays fixed across matched interventions."})


if __name__ == "__main__":
    main()
