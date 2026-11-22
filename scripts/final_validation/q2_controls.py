"""Matched feature controls and clean-simulator contextual-policy robustness."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit
from scipy.optimize import brentq

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from src.bandits.contextual import article_features, run, user_features
from src.data.mind import load_processed, load_processed_news

ROOT = OUT / "q2_controls"
FEATURES = ROOT / "features"
ENVS = ROOT / "envs"
MATCHED = {"g1b17": ("gemma-3-1b", 17, "g1b_sae_L17", "g1b_L17"),
           "g12b12": ("gemma-3-12b", 12, "g12b_sae_L12", "g12b_L12"),
           "g12b41": ("gemma-3-12b", 41, "g12b_sae_L41", "g12b_L41")}
CONTROL_SEEDS = (11, 23, 47)
DIMENSIONS = (16, 32, 64)
ALPHAS = (.01, .1, .3)


def load_env(path):
    with np.load(path) as f:
        return {k: f[k] if f[k].ndim else f[k].item() for k in f.files}


def policy_features(representation, dimension):
    X = np.load(FEATURES / f"{representation}.npy")[:, :dimension].astype(np.float32)
    fit_rows = np.load(ROOT / "fit_article_rows.npy")
    scale = np.sqrt((X[fit_rows]**2).sum(1).mean(dtype=np.float64))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Degenerate training representation: {representation}")
    return (X / scale).astype(np.float32)


def prepare():
    source = Path("dataset/processed")
    raw = source / "policy_raw"
    inputs = [Path(__file__), source / "policy_features/meta.json", source / "train_impressions.npz"]
    inputs += [raw / model / f"dense_layer_{layer}.npy" for model, layer, _, _ in MATCHED.values()]
    provenance = fingerprint({"control_seeds": CONTROL_SEEDS, "dim_max": 128,
                              "thresholding": "article-level random directions matched to selected SAE training activation frequencies"}, inputs)
    if is_complete(ROOT / "feature_manifest.json", provenance):
        return
    FEATURES.mkdir(parents=True, exist_ok=True)
    train = load_processed("train")
    rows = np.unique(np.r_[train.cand_idx, train.hist_idx])
    np.save(ROOT / "fit_article_rows.npy", rows)
    del train
    meta = json.loads((source / "policy_features/meta.json").read_text())
    representations = {}
    for group, (model, layer, sae, dense) in MATCHED.items():
        X = np.load(raw / model / f"dense_layer_{layer}.npy", mmap_mode="r")
        fitted = X[rows]
        mean = fitted.mean(0, dtype=np.float64).astype(np.float32)
        std = fitted.std(0, dtype=np.float64).clip(1e-8).astype(np.float32)
        top = np.argsort(-std)[:128]
        del fitted
        for label, rep in (("dense", dense), ("sae", sae)):
            path = FEATURES / f"{group}_{label}.npy"
            original = (source / "policy_features" / f"{rep}.npy").resolve()
            if not path.exists():
                path.symlink_to(original)
            representations[f"{group}_{label}"] = {"group": group, "type": label, "source": rep}
        neurons = np.asarray(X[:, top], np.float32)
        rms = np.sqrt((neurons[rows].astype(np.float64)**2).mean(0)).clip(1e-8)
        np.save(FEATURES / f"{group}_neurons.npy", (neurons / rms).astype(np.float16))
        representations[f"{group}_neurons"] = {"group": group, "type": "top_variance_raw_neurons", "columns": top.tolist()}
        del neurons
        for seed in CONTROL_SEEDS:
            rng = np.random.default_rng(seed)
            projection = rng.normal(size=(X.shape[1], 128)).astype(np.float32) / np.sqrt(X.shape[1])
            Y = np.empty((len(X), 128), np.float32)
            for start in range(0, len(X), 4096):
                Y[start:start + 4096] = ((X[start:start + 4096] - mean) / std) @ projection
            rms = np.sqrt((Y[rows].astype(np.float64)**2).mean(0)).clip(1e-8)
            np.save(FEATURES / f"{group}_rp_s{seed}.npy", (Y / rms).astype(np.float16))
            freq = np.asarray(meta[sae]["train_freq"], float)
            thresholds = np.array([np.quantile(Y[rows, j], 1 - freq[j]) for j in range(128)])
            thresholds[freq >= 1] -= 1e-6
            Z = np.maximum(Y - thresholds, 0)
            rms = np.sqrt((Z[rows].astype(np.float64)**2).mean(0)).clip(1e-8)
            np.save(FEATURES / f"{group}_threshold_s{seed}.npy", (Z / rms).astype(np.float16))
            representations[f"{group}_rp_s{seed}"] = {"group": group, "type": "Gaussian_projection", "seed": seed}
            representations[f"{group}_threshold_s{seed}"] = {
                "group": group, "type": "article_level_random_thresholded", "seed": seed,
                "target_frequency": freq.tolist(), "actual_train_frequency": (Z[rows] > 0).mean(0).tolist()}
        print("Prepared matched controls", group, flush=True)
    for rep in ("bgem3", "g1b_L23", "g1b_sae_L17", "g12b_L43", "g12b_sae_L12"):
        path = FEATURES / f"{rep}.npy"
        if not path.exists():
            path.symlink_to((source / "policy_features" / f"{rep}.npy").resolve())
    write_json(ROOT / "feature_manifest.json", {"provenance": provenance, "representations": representations,
        "fit_article_count": len(rows), "note": "Matched article-level controls. Thresholded projections are not token-level random SAE replicas."})


def build_envs(device):
    from src.simulator import click_model as cm
    from src.simulator.encode import load_embeddings
    from src.simulator.features import ArticleTable, FeatureStore, UserTable
    from src.simulator.simulator import ClickSimulator
    from src.bandits.contextual import build_churn_env, ChurnSchedule
    temporal = OUT / "temporal_simulator"
    protocol = json.loads((temporal / "protocol.json").read_text())
    summary = json.loads((temporal / "summary.json").read_text())
    primary = protocol["primary_family"]
    earlier = np.load(temporal / "earlier_article_rows.npz")["article_rows"]
    inputs = [Path(__file__), temporal / "summary.json"]
    inputs += [temporal / kind / "intercept.pt" for kind in cm.KINDS]
    inputs += [Path(f"results/q1/envs/{split}_K10.npz") for split in ("train", "test")]
    provenance = fingerprint({"primary": primary, "strict_exclusion": "all Nov9-14 candidates and histories",
                              "gap_scales": [0, .25, .5], "ctrs": [.02, .08], "temperatures": [.5, 2]}, inputs)
    if is_complete(ROOT / "environment_manifest.json", provenance):
        return
    ENVS.mkdir(parents=True, exist_ok=True)
    articles = ArticleTable.from_news(load_processed_news(), load_embeddings())
    records = {}
    for split in ("train", "test"):
        imps = load_processed(split)
        users = UserTable.from_impressions(imps, articles)
        store = FeatureStore(users, articles, device)
        old = load_env(Path(f"results/q2/envs/{split}_life2000.npz"))
        for kind in cm.KINDS:
            path = ENVS / f"{split}_{kind}.npz"
            if not path.exists():
                model = cm.load(temporal / kind / "intercept.pt", device)
                sim = ClickSimulator(model, store, np.flatnonzero(users.hist_len >= 1), np.unique(imps.cand_idx))
                P = np.stack([sim.prob_matrix(u, a) for u, a in zip(old["users"], old["articles"])])
                save_npz(path, **{**old, "P": P})
                del model
            records[f"{split}_{kind}"] = str(path)
            population_path = ROOT / "population_envs" / f"{split}_{kind}.npz"
            if not population_path.exists():
                model = cm.load(temporal / kind / "intercept.pt", device)
                sim = ClickSimulator(model, store, np.flatnonzero(users.hist_len >= 1), np.unique(imps.cand_idx))
                old_pop = load_env(Path(f"results/q1/envs/{split}_K10.npz"))
                rng = np.random.default_rng((1000000 if split == "train" else 2000000) + 10000)
                population_users = []
                for expected in old_pop["articles"]:
                    drawn = sim.sample_articles(10, rng)
                    if not np.array_equal(drawn, expected):
                        raise RuntimeError("Cannot reconstruct the original Q1 sampling stream")
                    population_users.append(sim.sample_users(old_pop["P"].shape[1], rng))
                P = np.stack([sim.prob_matrix(u, a) for u, a in zip(population_users, old_pop["articles"])])
                save_npz(population_path, **{**old_pop, "P": P, "user_rows": np.array(population_users)})
                del model
        if split == "train":
            model = cm.load(temporal / primary / "intercept.pt", device)
            sim = ClickSimulator(model, store, np.flatnonzero(users.hist_len >= 1), np.unique(imps.cand_idx))
            path = ENVS / f"train_{primary}_importance30.npz"
            if not path.exists():
                importance_env = build_churn_env(sim, 30, 2000, ChurnSchedule(20, 2000, 20000), 5002000)
                save_npz(path, **importance_env)
            records[f"train_{primary}_importance30"] = str(path)
            del model
        if split == "test":
            model = cm.load(temporal / primary / "intercept.pt", device)
            sim = ClickSimulator(model, store, np.flatnonzero(users.hist_len >= 1), np.unique(imps.cand_idx))
            pool = np.setdiff1d(np.unique(imps.cand_idx), earlier)
            if len(pool) < old["P"].shape[2]:
                raise RuntimeError(f"Only {len(pool)} strict holdout articles for stream size {old['P'].shape[2]}")
            path = ENVS / "test_strict.npz"
            if not path.exists():
                rng = np.random.default_rng(SEED)
                stream = np.stack([rng.choice(pool, old["P"].shape[2], replace=False) for _ in range(len(old["P"]))])
                P = np.stack([sim.prob_matrix(u, a) for u, a in zip(old["users"], stream)])
                assert not np.isin(stream, earlier).any()
                save_npz(path, **{**old, "P": P, "articles": stream})
            records["test_strict"] = str(path)
            records["strict_pool_articles"] = int(len(pool))
            del model
        del store, users, imps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    base = load_env(ENVS / f"test_{primary}.npz")
    P = base["P"].astype(np.float64)
    for gap in (0, .25, .5):
        mean = P.mean(2, keepdims=True)
        transformed = mean + gap * (P - mean)
        assert np.allclose(transformed.mean(2), mean[:, :, 0], atol=1e-12)
        path = ENVS / f"test_gap{gap}.npz"
        save_npz(path, **{**base, "P": transformed.astype(np.float32)})
        records[f"test_gap{gap}"] = str(path)
    z = logit(np.clip(P, 1e-7, 1 - 1e-7))
    calibrators = json.loads((temporal / primary / "calibrators.json").read_text())
    intercept = calibrators["intercept"][1]
    for method in ("identity", "platt"):
        scale, bias = calibrators[method]
        transformed = expit(scale * (z - intercept) + bias).astype(np.float32)
        path = ENVS / f"test_calibration_{method}.npz"
        save_npz(path, **{**base, "P": transformed})
        records[f"test_calibration_{method}"] = str(path)
    for label, temperature, target in [("ctr02", 1, .02), ("ctr08", 1, .08),
                                       ("temperature05", .5, float(P.mean())), ("temperature2", 2, float(P.mean()))]:
        bias = brentq(lambda b: expit(z / temperature + b).mean() - target, -50, 50)
        transformed = expit(z / temperature + bias).astype(np.float32)
        assert np.isclose(transformed.mean(), target, atol=1e-6)
        path = ENVS / f"test_{label}.npz"
        save_npz(path, **{**base, "P": transformed})
        records[f"test_{label}"] = str(path)
    write_json(ROOT / "environment_manifest.json", {"provenance": provenance, "primary_family": primary,
        "environments": records, "temporal_evaluation": summary["protocol"]})


def task_name(task):
    return f"{task['environment']}__{task['representation']}__d{task['dimension']}__{task['policy']}"


def execute_task(task):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    path = ROOT / "runs" / f"{task_name(task)}.npz"
    if path.exists():
        return task, str(path)
    env = load_env(ENVS / f"{task['environment']}.npz")
    X = art_x = user_x = None
    if task["representation"] != "none":
        X = policy_features(task["representation"], task["dimension"])
        imps = load_processed(task["split"])
        art_x = X[env["articles"]]
        user_x = user_features(X, env["users"], imps.hist_ptr, imps.hist_idx)
    record = task.get("record_state", False)
    result = run(env, art_x, user_x, task["policy"], "cpu", seed=SEED,
                 n_contexts=200 if record else 0, record_state=record)
    if task["environment"] == "test_gap0":
        assert np.allclose(result["regret"], 0, atol=1e-6)
    save_npz(path, **result)
    return task, str(path)


def execute(tasks, workers):
    with mp.get_context("spawn").Pool(workers) as pool:
        for task, path in pool.imap_unordered(execute_task, tasks):
            print("Completed", task_name(task), flush=True)


def experiments(workers):
    feature_manifest = json.loads((ROOT / "feature_manifest.json").read_text())
    env_manifest = json.loads((ROOT / "environment_manifest.json").read_text())
    primary = env_manifest["primary_family"]
    reps = list(feature_manifest["representations"])
    provenance = fingerprint({"seed": SEED, "dimensions": DIMENSIONS, "alphas": ALPHAS},
                             [Path(__file__), Path("src/bandits/contextual.py"), ROOT / "feature_manifest.json",
                              ROOT / "environment_manifest.json"])
    if is_complete(ROOT / "summary.json", provenance):
        return
    marker = ROOT / "run_provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Existing Q2 tasks use different input/settings")
    write_json(marker, provenance)
    tuning = [{"environment": f"train_{primary}", "split": "train", "representation": rep,
               "dimension": d, "policy": f"linucb_{alpha}"}
              for rep in reps for d in DIMENSIONS for alpha in ALPHAS]
    execute(tuning, workers)
    chosen = {}
    for rep in reps:
        for d in DIMENSIONS:
            candidates = [t for t in tuning if t["representation"] == rep and t["dimension"] == d]
            regrets = [float(np.load(ROOT / "runs" / f"{task_name(t)}.npz")["regret"][:, -1].mean()) for t in candidates]
            chosen[f"{rep}_d{d}"] = candidates[int(np.argmin(regrets))]["policy"]
    write_json(ROOT / "chosen.json", chosen)
    tests = [{"environment": f"test_{primary}", "split": "test", "representation": rep,
              "dimension": d, "policy": chosen[f"{rep}_d{d}"],
              "record_state": rep.endswith("_sae") and d == 32}
             for rep in reps for d in DIMENSIONS]
    # In addition to separately tuned policies, isolate representation changes at one common coefficient.
    tests += [{"environment": f"test_{primary}", "split": "test", "representation": rep,
               "dimension": 32, "policy": "linucb_0.1"} for rep in reps]
    # Freeze the existing train-chosen policies for independent-encoder sensitivity.
    fixed = [("none", 0, "beta_ts"), ("bgem3", 32, "linucb_0.3"), ("g1b_L23", 32, "linucb_0.3"),
             ("g1b_sae_L17", 32, "linucb_0.1"), ("g12b_L43", 32, "linucb_0.3"), ("g12b_sae_L12", 32, "lints_0.1")]
    env_names = [k for k in env_manifest["environments"] if k.startswith("test_")]
    for name in env_names:
        tests += [{"environment": name, "split": "test", "representation": rep, "dimension": d, "policy": policy}
                  for rep, d, policy in fixed]
    # Record train contexts to rank the three matched SAE policies without evaluation leakage.
    tests += [{"environment": f"train_{primary}_importance30", "split": "train", "representation": f"{group}_sae",
               "dimension": 32, "policy": chosen[f"{group}_sae_d32"], "record_state": True} for group in MATCHED]
    unique = {}
    for task in tests:
        key = task_name(task)
        if key in unique:
            unique[key]["record_state"] = unique[key].get("record_state", False) or task.get("record_state", False)
        else:
            unique[key] = task
    tests = list(unique.values())
    execute(tests, workers)
    output = {}
    for task in tests:
        path = ROOT / "runs" / f"{task_name(task)}.npz"
        with np.load(path) as res:
            output[task_name(task)] = {"task": task, "regret": interval(res["regret"][:, -1]),
                "expected_ctr": interval(res["clicks"][:, -1] / float(res["marks"][-1])),
                "cold_start_regret_per_eligible_round": interval(res["regret_new"][:, -1] / np.maximum(res["rounds_new"][:, -1], 1))}
    write_json(ROOT / "summary.json", {"provenance": provenance, "chosen": chosen, "results": output})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "envs", "experiments"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    {"prepare": prepare, "envs": lambda: build_envs(args.device), "experiments": lambda: experiments(args.workers)}[args.stage]()


if __name__ == "__main__":
    main()
