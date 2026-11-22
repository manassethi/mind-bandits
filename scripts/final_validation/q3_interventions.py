"""Feature removal using the actual pre-action UCB state and train-only rankings."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q2_controls import ENVS, FEATURES, MATCHED, ROOT as Q2_ROOT, load_env, policy_features
from src.bandits.contextual import article_features, run, user_features
from src.data.mind import load_processed

ROOT = OUT / "q3_interventions"
D = 32
COUNTS = (1, 4, 8, 16)


def context_data(rep, split, environment, policy):
    path = Q2_ROOT / "runs" / f"{environment}__{rep}__d32__{policy}.npz"
    with np.load(path) as f:
        state = {k: f[k] for k in f.files}
    env = load_env(ENVS / f"{environment}.npz")
    X = policy_features(rep, D)
    imps = load_processed(split)
    article = X[env["articles"]]
    users = user_features(X, env["users"], imps.hist_ptr, imps.hist_idx)
    S = len(article)
    a = article[np.arange(S)[:, None, None], state["ctx_live"]]
    u = users[np.arange(S)[:, None], state["ctx_users"]][:, :, None, :]
    phi = np.concatenate([np.ones(a.shape[:-1] + (1,), np.float32), a, a * u], -1).astype(np.float64)
    probabilities = env["P"][np.arange(S)[:, None, None], state["ctx_users"][:, :, None], state["ctx_live"]]
    return state, phi, probabilities


def rank_importance(state, phi):
    theta = state["ctx_theta"]
    contribution = phi[..., 1:1 + D] * theta[:, :, None, 1:1 + D]
    contribution += phi[..., 1 + D:] * theta[:, :, None, 1 + D:]
    return np.abs(contribution - contribution.mean(2, keepdims=True)).mean((0, 1, 2))


def scores(phi, state, alpha):
    mean = np.einsum("snkd,snd->snk", phi, state["ctx_theta"], optimize=True)
    variance = np.einsum("snkd,snde,snke->snk", phi, state["ctx_A_inv"], phi, optimize=True)
    return mean + alpha * np.sqrt(np.maximum(variance, 0))


def removal(state, phi, probabilities, alpha, drop):
    altered = phi.copy()
    columns = np.r_[1 + np.array(drop), 1 + D + np.array(drop)].astype(int)
    altered[..., columns] = 0
    action = scores(altered, state, alpha).argmax(2)
    original = state["ctx_choice"]
    old_p = np.take_along_axis(probabilities, original[..., None], 2)[..., 0]
    new_p = np.take_along_axis(probabilities, action[..., None], 2)[..., 0]
    return (action != original).mean(1), (old_p - new_p).mean(1) * 1000


def retrain(task):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    path = ROOT / "retrain" / f"{task['group']}__{task['name']}.npz"
    if path.exists():
        return task, str(path)
    env = load_env(ENVS / f"{task['environment']}.npz")
    X = policy_features(f"{task['group']}_sae", D)
    X[:, task["drop"]] = 0
    imps = load_processed("test")
    a = X[env["articles"]]
    u = user_features(X, env["users"], imps.hist_ptr, imps.hist_idx)
    result = run(env, a, u, task["policy"], "cpu", seed=SEED)
    save_npz(path, **result)
    return task, str(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--random-posthoc", type=int, default=50)
    p.add_argument("--random-retrain", type=int, default=5)
    args = p.parse_args()
    provenance = fingerprint({"seed": SEED, "counts": COUNTS, "random_posthoc": args.random_posthoc,
                              "random_retrain": args.random_retrain},
                             [Path(__file__), Q2_ROOT / "summary.json", Path("src/bandits/contextual.py")])
    if is_complete(ROOT / "summary.json", provenance):
        return
    marker = ROOT / "provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial intervention results differ in protocol")
    write_json(marker, provenance)
    primary = json.loads((Q2_ROOT / "environment_manifest.json").read_text())["primary_family"]
    chosen = json.loads((Q2_ROOT / "chosen.json").read_text())
    output, tasks, arrays = {}, [], {}
    for group in MATCHED:
        rep = f"{group}_sae"
        policy = chosen[f"{rep}_d32"]
        alpha = float(policy.split("_")[1])
        train, train_phi, _ = context_data(rep, "train", f"train_{primary}_importance30", policy)
        importance = rank_importance(train, train_phi)
        order = np.argsort(-importance)
        state, phi, probabilities = context_data(rep, "test", f"test_{primary}", policy)
        score = scores(phi, state, alpha)
        observed_score = np.take_along_axis(score, state["ctx_choice"][..., None], 2)[..., 0]
        discrepancy = score.max(2) - observed_score
        if discrepancy.max() > 1e-7:
            raise RuntimeError(f"UCB reconstruction failed: {group}, gap {discrepancy.max()}")
        record = {"policy": policy, "train_importance": importance.tolist(), "train_rank": order.tolist(),
                  "reconstruction_argmax_match": float((score.argmax(2) == state["ctx_choice"]).mean()),
                  "reconstruction_max_score_gap": float(discrepancy.max()), "posthoc": {}}
        for count in COUNTS:
            rng = np.random.default_rng(SEED + count)
            masks = {"top": order[:count].tolist(), "bottom": order[-count:].tolist()}
            masks.update({f"random{i}": rng.choice(D, count, replace=False).tolist() for i in range(args.random_posthoc)})
            evaluated = {name: removal(state, phi, probabilities, alpha, drop) for name, drop in masks.items()}
            random_change = np.mean([value[0] for name, value in evaluated.items() if name.startswith("random")], 0)
            random_loss = np.mean([value[1] for name, value in evaluated.items() if name.startswith("random")], 0)
            record["posthoc"][str(count)] = {
                "top_change": interval(evaluated["top"][0]), "bottom_change": interval(evaluated["bottom"][0]),
                "random_change": interval(random_change), "top_loss_per_1k": interval(evaluated["top"][1]),
                "bottom_loss_per_1k": interval(evaluated["bottom"][1]), "random_loss_per_1k": interval(random_loss),
                "paired_top_minus_random_loss": interval(evaluated["top"][1] - random_loss)}
            for name in ["top", "bottom", *[f"random{i}" for i in range(args.random_retrain)]]:
                tasks.append({"group": group, "name": f"k{count}_{name}", "drop": masks[name],
                              "environment": f"test_{primary}", "policy": policy, "count": count, "kind": name})
        arrays[f"{group}/base_regret"] = state["regret"][:, -1]
        output[group] = record
        write_json(ROOT / "posthoc_progress.json", output)
        print("Full-UCB post-hoc completed", group, flush=True)
    write_json(ROOT / "retrain_tasks.json", tasks)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for task, path in pool.imap_unordered(retrain, tasks):
            print("Retrained", task["group"], task["name"], flush=True)
    for group in MATCHED:
        output[group]["retrain"] = {}
        for count in COUNTS:
            regress = {}
            for task in tasks:
                if task["group"] == group and task["count"] == count:
                    path = ROOT / "retrain" / f"{group}__{task['name']}.npz"
                    with np.load(path) as data:
                        regress[task["kind"]] = data["regret"][:, -1] - arrays[f"{group}/base_regret"]
            random_mean = np.mean([v for k, v in regress.items() if k.startswith("random")], 0)
            output[group]["retrain"][str(count)] = {
                "top": interval(regress["top"]), "bottom": interval(regress["bottom"]), "random": interval(random_mean),
                "paired_top_minus_random": interval(regress["top"] - random_mean)}
    save_npz(ROOT / "baselines.npz", **arrays)
    write_json(ROOT / "summary.json", {"provenance": provenance, "results": output,
        "scope": "Post-hoc interventions use pre-action theta and covariance at actual recorded online decisions. Retraining removes the coordinates before the first round."})


if __name__ == "__main__":
    main()
