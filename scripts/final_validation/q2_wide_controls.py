"""Width/selection-matched article-level random dictionary controls."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q2_controls import MATCHED, ROOT as Q2, load_env
from src.bandits.contextual import run, user_features
from src.data.mind import load_processed

ROOT = OUT / "q2_wide_controls"
SEEDS = (11, 23, 47)
WIDTH = 16384
KEEP = 128


def retain(best, W, variance, indices, thresholds, frequencies):
    candidates = {"weights": W, "variance": variance, "indices": indices,
                  "thresholds": thresholds, "target_frequencies": frequencies}
    if best is not None:
        candidates = {k: np.concatenate([best[k], v], axis=1 if k == "weights" else 0)
                      for k, v in candidates.items()}
    order = np.argsort(-candidates["variance"], kind="stable")[:KEEP]
    return {k: v[:, order] if k == "weights" else v[order] for k, v in candidates.items()}


def prepare(device):
    inputs = [Path(__file__), Q2 / "fit_article_rows.npy"]
    for model, layer, _, _ in MATCHED.values():
        inputs += [Path(f"dataset/processed/policy_raw/{model}/dense_layer_{layer}.npy"),
                   Path(f"dataset/processed/policy_raw/{model}/sae_layer_{layer}.npz")]
    provenance = fingerprint({"width": WIDTH, "keep": KEEP, "seeds": SEEDS,
        "selection": "descending training variance before per-column RMS scaling",
        "threshold_frequencies": "permutation of the complete source SAE training article activation frequencies",
        "scope": "article-level controls, not token-level random SAEs", "projection_dtype": "float32; TF32 disabled"}, inputs)
    if is_complete(ROOT / "feature_manifest.json", provenance):
        return
    marker = ROOT / "prepare_provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial wide-control preparation has different settings")
    write_json(marker, provenance)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = np.load(Q2 / "fit_article_rows.npy")
    features = ROOT / "features"
    features.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for group, (model, layer, _, _) in MATCHED.items():
        raw = np.load(f"dataset/processed/policy_raw/{model}/dense_layer_{layer}.npy", mmap_mode="r")
        fit = raw[rows]
        mean = fit.mean(0, dtype=np.float64).astype(np.float32)
        std = fit.std(0, dtype=np.float64).clip(1e-8).astype(np.float32)
        train_X = torch.from_numpy((fit - mean) / std).to(device)
        del fit
        source = sp.load_npz(f"dataset/processed/policy_raw/{model}/sae_layer_{layer}.npz").tocsr()
        counts = np.asarray((source[rows] > 0).sum(axis=0), dtype=np.int64).ravel()
        frequencies = counts / np.float64(len(rows))
        if not np.isfinite(frequencies).all() or (frequencies < 0).any() or (frequencies > 1).any():
            raise ValueError("Invalid source activation counts")
        if len(frequencies) != WIDTH:
            raise ValueError("Source SAE width differs from the matched random dictionary")
        del source
        for seed in SEEDS:
            selection_path = ROOT / f"selection/{group}_s{seed}.npz"
            complete = selection_path.exists() and all((features / f"{group}_{kind}_s{seed}.npy").exists()
                                                        for kind in ("wide_rp", "wide_threshold"))
            if complete:
                metadata[f"{group}_s{seed}"] = str(selection_path)
                continue
            rng = np.random.default_rng(seed)
            freq = np.random.default_rng(seed + 1000000).permutation(frequencies)
            best_dense = best_threshold = None
            for start in range(0, WIDTH, 128):
                W = rng.normal(size=(raw.shape[1], 128)).astype(np.float32) / np.float32(np.sqrt(raw.shape[1]))
                Y = (train_X @ torch.from_numpy(W).to(device)).cpu().numpy()
                target = freq[start:start + 128]
                thresholds = np.array([np.quantile(Y[:, j], 1 - target[j]) for j in range(128)], np.float32)
                thresholds[target >= 1] -= 1e-6
                Z = np.maximum(Y - thresholds, 0)
                indices = np.arange(start, start + 128)
                best_dense = retain(best_dense, W, Y.var(0, dtype=np.float64), indices,
                                    np.zeros(128, np.float32), target)
                best_threshold = retain(best_threshold, W, Z.var(0, dtype=np.float64), indices,
                                        thresholds, target)
                if (start + 128) % 4096 == 0:
                    print("Width selection", group, seed, start + 128, flush=True)
            arrays = {"source_train_frequency": frequencies, "fit_mean": mean, "fit_std": std}
            # Only selected directions are projected on evaluation articles.
            for kind, best in (("wide_rp", best_dense), ("wide_threshold", best_threshold)):
                values = np.empty((len(raw), KEEP), np.float32)
                W = torch.from_numpy(best["weights"].astype(np.float32)).to(device)
                for start in range(0, len(raw), 4096):
                    chunk = torch.from_numpy((np.asarray(raw[start:start + 4096]) - mean) / std).to(device)
                    Y = (chunk @ W).cpu().numpy()
                    if kind == "wide_threshold":
                        Y = np.maximum(Y - best["thresholds"], 0)
                    values[start:start + 4096] = Y
                scale = np.sqrt((values[rows].astype(np.float64)**2).mean(0)).clip(1e-8)
                np.save(features / f"{group}_{kind}_s{seed}.npy", (values / scale).astype(np.float16))
                for key, value in best.items():
                    arrays[f"{kind}/{key}"] = value
                arrays[f"{kind}/rms"] = scale
                arrays[f"{kind}/actual_training_frequency"] = (values[rows] > 0).mean(0)
                del values, W
            save_npz(selection_path, **arrays)
            metadata[f"{group}_s{seed}"] = str(selection_path)
        del train_X, raw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(ROOT / "feature_manifest.json", {"provenance": provenance, "selections": metadata,
        "scope": "Both controls use 16,384 candidates and the 128 highest-variance training coordinates. Thresholded projections use the source SAE's full training activation-frequency distribution, assigned independently of random directions. This controls dictionary width and selection but does not replicate token-level SAE encoding."})


def one_task(task):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    environment, rep, dimension, policy = task
    key = f"{environment}__{rep}__d{dimension}__{policy}"
    path = ROOT / f"runs/{key}.npz"
    if not path.exists():
        source = ROOT if "_wide_" in rep else Q2
        X = np.load(source / f"features/{rep}.npy")[:, :dimension].astype(np.float32)
        rows = np.load(Q2 / "fit_article_rows.npy")
        X /= np.sqrt((X[rows]**2).sum(1).mean(dtype=np.float64))
        split = environment.split("_")[0]
        env = load_env(Q2 / f"envs/{environment}.npz")
        imps = load_processed(split)
        result = run(env, X[env["articles"]], user_features(X, env["users"], imps.hist_ptr, imps.hist_idx),
                     policy, "cpu", seed=SEED)
        save_npz(path, **result)
    with np.load(path) as data:
        regret = data["regret"][:, -1]
    print("Wide control policy", key, flush=True)
    return key, {"task": {"environment": environment, "representation": rep, "dimension": dimension, "policy": policy},
                 "regret": interval(regret)}


def evaluate(workers):
    provenance = fingerprint({"seed": SEED, "dimensions": [16, 32, 64], "alphas": [.01, .1, .3],
                             "strict_article_controls": "all matched representations at D32; train-selected and common alpha0.1"},
                             [Path(__file__), ROOT / "feature_manifest.json", Q2 / "summary.json",
                              Path("src/bandits/contextual.py")])
    if is_complete(ROOT / "summary.json", provenance):
        return
    marker = ROOT / "run_provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial wide-control evaluation has different settings")
    write_json(marker, provenance)
    family = json.loads((Q2 / "environment_manifest.json").read_text())["primary_family"]
    reps = [f"{group}_{kind}_s{seed}" for group in MATCHED for kind in ("wide_rp", "wide_threshold") for seed in SEEDS]
    tasks = [(f"train_{family}", rep, dimension, f"linucb_{alpha}")
             for rep in reps for dimension in (16, 32, 64) for alpha in (.01, .1, .3)]
    with mp.get_context("spawn").Pool(workers) as pool:
        train = dict(pool.imap_unordered(one_task, tasks))
    chosen = dict(json.loads((Q2 / "summary.json").read_text())["chosen"])
    for rep in reps:
        for dimension in (16, 32, 64):
            chosen[f"{rep}_d{dimension}"] = min((f"linucb_{a}" for a in (.01, .1, .3)),
                key=lambda policy: train[f"train_{family}__{rep}__d{dimension}__{policy}"]["regret"]["mean"])
    write_json(ROOT / "chosen.json", chosen)
    tasks = [(f"test_{family}", rep, dimension, chosen[f"{rep}_d{dimension}"])
             for rep in reps for dimension in (16, 32, 64)]
    tasks += [(f"test_{family}", rep, 32, "linucb_0.1") for rep in reps]
    small = [f"{group}_{kind}" for group in MATCHED for kind in
             ("sae", "dense", "neurons", "rp_s11", "rp_s23", "rp_s47",
              "threshold_s11", "threshold_s23", "threshold_s47")]
    tasks += [("test_strict", rep, 32, policy) for rep in small + reps
              for policy in (chosen[f"{rep}_d32"], "linucb_0.1")]
    tasks = list(dict.fromkeys(tasks))
    with mp.get_context("spawn").Pool(workers) as pool:
        results = dict(pool.imap_unordered(one_task, tasks))
    write_json(ROOT / "summary.json", {"provenance": provenance, "chosen": chosen, "results": results,
        "scope": "Exploratory width/selection control triggered by the small random dictionaries' lower performance. All dimensions and direction seeds retained; evaluation pools are paired with the primary SAE experiments. Matched small and wide controls are also evaluated at D32 on strictly unseen article streams, with frozen train-selected and common exploration coefficients."})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare", "evaluate"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.stage == "prepare":
        prepare(args.device)
    else:
        evaluate(args.workers)


if __name__ == "__main__":
    main()
