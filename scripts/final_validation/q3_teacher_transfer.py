"""Transfer frozen feature-removal masks to the alternate click-model teacher."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q2_controls import MATCHED, load_env, policy_features
from src.bandits.contextual import run, user_features
from src.data.mind import load_processed

ROOT = OUT / "q3_teacher_transfer"


def one_task(task):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    path = ROOT / f"runs/{task['group']}__{task['name']}.npz"
    if not path.exists():
        env = load_env(OUT / "alternate_encoder/envs/main.npz")
        X = policy_features(f"{task['group']}_sae", 32)
        X[:, task["drop"]] = 0
        imps = load_processed("test")
        result = run(env, X[env["articles"]], user_features(X, env["users"], imps.hist_ptr, imps.hist_idx),
                     task["policy"], "cpu", seed=SEED)
        save_npz(path, **result)
    with np.load(path) as data:
        regret = data["regret"][:, -1]
    print("Transferred mask", task["group"], task["name"], flush=True)
    return task, regret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    inputs = [Path(__file__), OUT / "q3_interventions/summary.json", OUT / "q3_interventions/retrain_tasks.json",
              OUT / "alternate_encoder/model_summary.json", OUT / "alternate_encoder/envs/main.npz",
              Path("src/bandits/contextual.py")]
    provenance = fingerprint({"seed": SEED, "teacher": "BGE-M3 MLP", "counts": [1, 4, 8, 16],
        "rank_and_masks": "frozen from Qwen-teacher training states; no BGE-evaluation reranking",
        "exploration": "freeze each primary train-selected coefficient",
        "status": "exploratory transfer of conditional feature-necessity effects"}, inputs)
    if is_complete(ROOT / "summary.json", provenance):
        return
    marker = ROOT / "provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial teacher-transfer results have different inputs")
    write_json(marker, provenance)
    original = json.loads(inputs[1].read_text())["results"]
    tasks = json.loads(inputs[2].read_text())
    tasks += [{"group": group, "name": "unmasked", "drop": [], "policy": original[group]["policy"],
               "count": 0, "kind": "unmasked"} for group in MATCHED]
    write_json(ROOT / "tasks.json", tasks)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        completed = list(pool.imap_unordered(one_task, tasks))
    results = {}
    for group in MATCHED:
        baseline = next(r for t, r in completed if t["group"] == group and t["kind"] == "unmasked")
        results[group] = {"baseline": interval(baseline), "counts": {}}
        for count in (1, 4, 8, 16):
            runs = {t["kind"]: r for t, r in completed if t["group"] == group and t["count"] == count}
            random = np.mean([r for k, r in runs.items() if k.startswith("random")], axis=0)
            results[group]["counts"][str(count)] = {
                "top_change": interval(runs["top"] - baseline),
                "bottom_change": interval(runs["bottom"] - baseline),
                "random_change": interval(random - baseline),
                "paired_top_minus_random": interval(runs["top"] - random)}
    write_json(ROOT / "summary.json", {"provenance": provenance, "results": results,
        "scope": "Identical pools, frozen training-derived ranks, masks, and exploration settings, with user-response probabilities replaced by the BGE-based simulator. All four removal sizes and three layers retained. This tests transfer between two learned simulators, not causality for real readers."})


if __name__ == "__main__":
    main()
