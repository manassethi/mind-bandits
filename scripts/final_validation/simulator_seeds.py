"""Separate neural initialization variability from the clean temporal protocol."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from scripts.calibrate_click_model import fit_calibration
from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q2_controls import ROOT as Q2_ROOT, load_env, policy_features
from scripts.train_click_model import score
from src.bandits.contextual import run, user_features
from src.data.mind import load_processed, load_processed_news
from src.simulator import click_model as cm
from src.simulator.encode import load_embeddings
from src.simulator.features import ArticleTable, FeatureStore, UserTable
from src.simulator.simulator import ClickSimulator

ROOT = OUT / "simulator_seeds"
SEEDS = (0, 11)
POLICIES = (("none", "beta_ts"), ("bgem3", "linucb_0.3"), ("g1b_L23", "linucb_0.3"),
            ("g1b_sae_L17", "linucb_0.1"), ("g12b_L43", "linucb_0.3"), ("g12b_sae_L12", "lints_0.1"))


def fit(device):
    temporal = OUT / "temporal_simulator"
    original = json.loads((temporal / "summary.json").read_text())
    family = original["protocol"]["primary_family"]
    epoch = original["results"][family]["selected_epoch"]
    provenance = fingerprint({"seeds": SEEDS, "family": family, "epochs": epoch,
                              "purpose": "fixed-configuration initialization replication; no evaluation-driven selection"},
                             [Path(__file__), temporal / "summary.json", Q2_ROOT / "environment_manifest.json"])
    if is_complete(ROOT / "model_summary.json", provenance):
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    marker = ROOT / "model_provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial simulator-seed fit has different settings")
    write_json(marker, provenance)
    train = load_processed("train")
    day = train.imp_time // 86400
    cutoff = day.max()
    refit, calibration = train.subset(day < cutoff), train.subset(day == cutoff)
    assert refit.imp_time.max() < calibration.imp_time.min()
    articles = ArticleTable.from_news(load_processed_news(), load_embeddings())
    tables = {"train": UserTable.from_impressions(train, articles)}
    store = FeatureStore(tables["train"], articles, device)
    cu, ca, _ = cm.candidate_tensors(calibration, store.device)
    for seed in SEEDS:
        dest = ROOT / f"seed{seed}"
        dest.mkdir(exist_ok=True)
        path = dest / "intercept.pt"
        if not path.exists():
            torch.manual_seed(seed)
            model = cm.build(family, store)
            history = cm.train(model, store, refit, epochs=epoch, schedule_epochs=6, batch_size=8192, seed=seed)
            logits = cm.predict(model, store, cu, ca, logits=True).astype(np.float64)
            _, bias, _ = fit_calibration(logits, calibration.cand_label, platt=False)
            with torch.no_grad():
                model.calib_bias.fill_(bias)
            cm.save(model, path, seed=seed, epochs=epoch, provenance=provenance,
                    calibration={"method": "intercept", "bias": bias}, history=history)
            del model
    del store, cu, ca, refit, calibration
    result = {}
    test = load_processed("test")
    tables["test"] = UserTable.from_impressions(test, articles)
    test_store = FeatureStore(tables["test"], articles, device)
    tu, ta, _ = cm.candidate_tensors(test, test_store.device)
    for seed in SEEDS:
        model = cm.load(ROOT / f"seed{seed}/intercept.pt", device)
        prediction = cm.predict(model, test_store, tu, ta)
        result[str(seed)] = score(prediction, test, ranking=True)
        sim = ClickSimulator(model, test_store, np.flatnonzero(tables["test"].hist_len >= 1), np.unique(test.cand_idx))
        for case, base_path in [("main", Q2_ROOT / f"envs/test_{family}.npz"), ("strict", Q2_ROOT / "envs/test_strict.npz")]:
            path = ROOT / f"envs/{case}_seed{seed}.npz"
            if not path.exists():
                base = load_env(base_path)
                P = np.stack([sim.prob_matrix(u, a) for u, a in zip(base["users"], base["articles"])])
                save_npz(path, **{**base, "P": P})
        print("Seed", seed, {k: result[str(seed)][k] for k in ("auc", "group_auc", "log_loss", "ece")}, flush=True)
        del model
    write_json(ROOT / "model_summary.json", {"provenance": provenance, "results": result,
        "primary_seed": SEED, "primary_metrics": original["results"][family]["methods"]["intercept"]})


def one_task(task):
    seed, case, representation, policy = task
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    path = ROOT / f"runs/{case}_seed{seed}__{representation}.npz"
    if not path.exists():
        env = load_env(ROOT / f"envs/{case}_seed{seed}.npz")
        a = u = None
        if representation != "none":
            X = policy_features(representation, 32)
            imps = load_processed("test")
            a = X[env["articles"]]
            u = user_features(X, env["users"], imps.hist_ptr, imps.hist_idx)
        result = run(env, a, u, policy, "cpu", seed=SEED)
        save_npz(path, **result)
    with np.load(path) as f:
        regret = f["regret"][:, -1]
    print("Seed comparison", seed, case, representation, flush=True)
    return f"{case}_seed{seed}__{representation}", {"regret": interval(regret), "policy": policy}


def evaluate(workers):
    provenance = fingerprint({"seeds": SEEDS, "policies": POLICIES},
                             [Path(__file__), ROOT / "model_summary.json", Q2_ROOT / "feature_manifest.json"])
    if is_complete(ROOT / "summary.json", provenance):
        return
    tasks = [(seed, case, rep, policy) for seed in SEEDS for case in ("main", "strict") for rep, policy in POLICIES]
    with mp.get_context("spawn").Pool(workers) as pool:
        result = dict(pool.imap_unordered(one_task, tasks))
    write_json(ROOT / "summary.json", {"provenance": provenance, "results": result,
        "analysis_note": "The same evaluation pools recur across initialization seeds. Analyze seed-specific paired effects or average across seeds within each pool; do not treat them as 90 independent pools."})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("fit", "evaluate"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    {"fit": lambda: fit(args.device), "evaluate": lambda: evaluate(args.workers)}[args.stage]()


if __name__ == "__main__":
    main()
