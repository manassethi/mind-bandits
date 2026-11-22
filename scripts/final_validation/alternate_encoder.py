"""Test teacher-encoder dependence with an independently encoded click simulator."""

import argparse
from argparse import Namespace
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from scripts.calibrate_click_model import fit_calibration
from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q2_controls import ROOT as Q2_ROOT, FEATURES, MATCHED, load_env, policy_features
from scripts.train_click_model import score, select
from src.bandits.contextual import run, user_features
from src.data.mind import load_processed, load_processed_news
from src.simulator import click_model as cm
from src.simulator.features import ArticleTable, FeatureStore, UserTable
from src.simulator.simulator import ClickSimulator

ROOT = OUT / "alternate_encoder"


def fit(device):
    embedding_path = Path("dataset/processed/policy_raw/bge-m3/dense.npy")
    provenance = fingerprint({"seed": SEED, "encoder": "BAAI/bge-m3", "family": "mlp", "selection_epochs": 6,
                              "fit": "Nov9-12", "select": "Nov13", "refit": "Nov9-13", "calibrate": "Nov14", "evaluate": "Nov15"},
                             [Path(__file__), embedding_path, Q2_ROOT / "environment_manifest.json"])
    if is_complete(ROOT / "model_summary.json", provenance):
        return
    marker = ROOT / "model_provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial alternative encoder model has different inputs")
    write_json(marker, provenance)
    train, test = load_processed("train"), load_processed("test")
    day = train.imp_time // 86400
    validation_day, calibration_day = np.unique(day)[-2:]
    fit_data, validation = train.subset(day < validation_day), train.subset(day == validation_day)
    refit, calibration = train.subset(day < calibration_day), train.subset(day == calibration_day)
    embedding = np.load(embedding_path).astype(np.float32)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True).clip(1e-8)
    articles = ArticleTable.from_news(load_processed_news(), embedding)
    store = FeatureStore(UserTable.from_impressions(train, articles), articles, device)
    selection_path = ROOT / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        selection = select(["mlp"], store, fit_data, validation,
                           Namespace(seed=SEED, max_epochs=6, batch_size=8192, lr=1e-3))
        write_json(selection_path, selection)
    checkpoint = ROOT / "intercept.pt"
    if not checkpoint.exists():
        torch.manual_seed(SEED)
        model = cm.build("mlp", store)
        history = cm.train(model, store, refit, epochs=selection["mlp"]["best_epoch"], schedule_epochs=6, seed=SEED)
        u, a, _ = cm.candidate_tensors(calibration, store.device)
        logits = cm.predict(model, store, u, a, logits=True).astype(np.float64)
        _, bias, _ = fit_calibration(logits, calibration.cand_label, platt=False)
        with torch.no_grad():
            model.calib_bias.fill_(bias)
        cm.save(model, checkpoint, provenance=provenance, history=history,
                epochs=selection["mlp"]["best_epoch"], calibration={"method": "intercept", "bias": bias})
        del model, u, a, logits
    population = load_env(Q2_ROOT / "population_envs/train_mlp.npz")
    train_population_path = ROOT / "population_envs/train.npz"
    if not train_population_path.exists():
        model = cm.load(checkpoint, device)
        sim = ClickSimulator(model, store, np.arange(train.n_users), np.unique(train.cand_idx))
        P = np.stack([sim.prob_matrix(u, a) for u, a in zip(population["user_rows"], population["articles"])])
        save_npz(train_population_path, **{**population, "P": P})
        del model
    del fit_data, validation, refit, calibration, store
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    users = UserTable.from_impressions(test, articles)
    store = FeatureStore(users, articles, device)
    model = cm.load(checkpoint, device)
    u, a, _ = cm.candidate_tensors(test, store.device)
    probabilities = cm.predict(model, store, u, a)
    metrics = score(probabilities, test, ranking=True)
    sim = ClickSimulator(model, store, np.flatnonzero(users.hist_len >= 1), np.unique(test.cand_idx))
    original = json.loads((Q2_ROOT / "environment_manifest.json").read_text())["primary_family"]
    for case, path in [("main", Q2_ROOT / f"envs/test_{original}.npz"), ("strict", Q2_ROOT / "envs/test_strict.npz")]:
        dest = ROOT / f"envs/{case}.npz"
        if not dest.exists():
            base = load_env(path)
            P = np.stack([sim.prob_matrix(u, a) for u, a in zip(base["users"], base["articles"])])
            save_npz(dest, **{**base, "P": P})
    population = load_env(Q2_ROOT / "population_envs/test_mlp.npz")
    P = np.stack([sim.prob_matrix(u, a) for u, a in zip(population["user_rows"], population["articles"])])
    save_npz(ROOT / "population_envs/test.npz", **{**population, "P": P})
    print("Alternative teacher", {k: metrics[k] for k in ("auc", "group_auc", "log_loss", "ece")}, flush=True)
    write_json(ROOT / "model_summary.json", {"provenance": provenance, "metrics": metrics,
        "selected_epoch": selection["mlp"]["best_epoch"],
        "scope": "BGE-M3 policy now shares the teacher encoder. Gemma policies remain independent of both teachers."})


def one_task(task):
    teacher, case, rep, policy = task
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    path = ROOT / f"runs/{teacher}_{case}__{rep}__{policy}.npz"
    if not path.exists():
        if teacher == "bge":
            env_path = ROOT / f"envs/{case}.npz"
        else:
            primary = json.loads((Q2_ROOT / "environment_manifest.json").read_text())["primary_family"]
            env_path = Q2_ROOT / f"envs/test_{'strict' if case == 'strict' else primary}.npz"
        env = load_env(env_path)
        a = u = None
        if rep != "none":
            X = policy_features(rep, 32)
            imps = load_processed("test")
            a = X[env["articles"]]
            u = user_features(X, env["users"], imps.hist_ptr, imps.hist_idx)
        result = run(env, a, u, policy, "cpu", seed=SEED)
        save_npz(path, **result)
    with np.load(path) as data:
        regret = data["regret"][:, -1]
    print("Teacher check", teacher, case, rep, flush=True)
    return f"{teacher}_{case}__{rep}__{policy}", {"regret": interval(regret)}


def evaluate(workers):
    provenance = fingerprint({"seed": SEED, "teachers": ["qwen", "bge"], "dimension": 32},
                             [Path(__file__), ROOT / "model_summary.json", Q2_ROOT / "feature_manifest.json"])
    if is_complete(ROOT / "summary.json", provenance):
        return
    qwen = FEATURES / "qwen3.npy"
    if not qwen.exists():
        qwen.symlink_to(Path("dataset/processed/policy_features/qwen3.npy").resolve())
    fixed = [("none", "beta_ts"), ("bgem3", "linucb_0.3"), ("qwen3", "linucb_0.3"),
             ("g1b_L23", "linucb_0.3"), ("g1b_sae_L17", "linucb_0.1"),
             ("g12b_L43", "linucb_0.3"), ("g12b_sae_L12", "lints_0.1")]
    matched = [(f"{group}_{kind}", "linucb_0.1") for group in MATCHED for kind in ("dense", "sae")]
    tasks = [(teacher, case, rep, policy) for teacher in ("qwen", "bge") for case in ("main", "strict")
             for rep, policy in fixed + (matched if case == "main" else [])]
    with mp.get_context("spawn").Pool(workers) as pool:
        results = dict(pool.imap_unordered(one_task, tasks))
    from scripts.final_validation.q1_allocation import make_policy, optimized_test_size, simulate
    from src.bandits.environment import BanditEnv
    train_pop, test_pop = load_env(ROOT / "population_envs/train.npz"), load_env(ROOT / "population_envs/test.npz")
    population_results = {}
    for horizon in (10000, 100000):
        env = BanditEnv(test_pop["P"], test_pop["articles"], horizon, seed=test_pop["seed"])
        n, _ = optimized_test_size(train_pop["P"].mean(1), horizon, SEED)
        runs = {}
        for policy in ("etc_original_train_tuned", "etc_reward_optimized", "ucb_tuned", "ts", "kl_ucb", "uniform"):
            per_arm = int((.2 if horizon == 10000 else .05) * horizon / 10) if policy == "etc_original_train_tuned" else n
            result = simulate(make_policy(policy, env, 1, per_arm), env, 1)
            save_npz(ROOT / f"population_runs/T{horizon}_{policy}.npz", **result)
            runs[policy] = result["regret"]
        population_results[str(horizon)] = {policy: {"regret": interval(regret), "paired_minus_ts": interval(regret - runs["ts"])}
                                            for policy, regret in runs.items()}
    write_json(ROOT / "summary.json", {"provenance": provenance, "results": results, "population_results": population_results})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("fit", "evaluate"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    {"fit": lambda: fit(args.device), "evaluate": lambda: evaluate(args.workers)}[args.stage]()


if __name__ == "__main__":
    main()
