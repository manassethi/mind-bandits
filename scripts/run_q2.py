"""Q2: contextual bandits with different article representations on a churning pool.

Stages (run in order; each writes results/q2/<stage>/<task>.npz):

    envs    tabulate simulator click probabilities for train/test churn environments
    tune    train split: exploration grid for LinUCB / LinTS per representation (d=32)
    layers  train + test splits: every dense layer and every SAE layer (d=32), tuned
            settings; the train split picks one dense and one SAE layer per model
    test    test split: chosen representations x d in {16,32,64,128} x policies
    churn   test split: chosen representations at d=32 under short and long article lifetimes

    python -m scripts.run_q2 --stage envs
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

# Env overrides exist for smoke tests (spawned workers re-import this module).
OUT = Path(os.environ.get("Q2_OUT", "results/q2"))
FEATS = Path(os.environ.get("Q2_FEATS", "dataset/processed/policy_features"))
SMOKE = os.environ.get("Q2_SMOKE") == "1"
LIFETIMES = (500, 2000, 10_000)
MAIN_LIFE = 2000
N_LIVE, HORIZON, N_USERS = (20, 2_000, 200) if SMOKE else (20, 20_000, 2000)
RUNS = {"train": 3, "test": 4} if SMOKE else {"train": 10, "test": 30}
SPLIT_SEED = {"train": 3_000_000, "test": 4_000_000}
LINUCB_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
LINTS_GRID = (0.01, 0.03, 0.1, 0.3)
MODELS = {"g1b": {"mid": 13, "sae": (7, 13, 17, 22), "layers": 26},
          "g12b": {"mid": 24, "sae": (12, 24, 31, 41), "layers": 48}}
TUNE_REPS = ["category", "bgem3", "qwen3"] + [f"{m}_L{c['mid']}" for m, c in MODELS.items()] + \
            [f"{m}_sae_L{c['mid']}" for m, c in MODELS.items()]
DIMS = (16, 32, 64, 128)


def env_path(split: str, life: int) -> Path:
    return OUT / "envs" / f"{split}_life{life}.npz"


def build_envs(device: str) -> None:
    from src.bandits.contextual import ChurnSchedule, build_churn_env
    from src.simulator.simulator import ClickSimulator

    env_path("train", 0).parent.mkdir(parents=True, exist_ok=True)
    for split in RUNS:
        lives = [l for l in LIFETIMES if not env_path(split, l).exists() and (split == "test" or l == MAIN_LIFE)]
        if not lives:
            continue
        sim = ClickSimulator.load(split=split, device=device)
        for life in lives:
            env = build_churn_env(sim, RUNS[split], N_USERS, ChurnSchedule(N_LIVE, life, HORIZON),
                                  SPLIT_SEED[split] + life)
            np.savez(env_path(split, life), **env)
            print(f"built {split} life={life}: P {env['P'].shape}, mean {env['P'].mean():.4f}", flush=True)


# ---- workers ------------------------------------------------------------------------------

_CACHE: dict = {}


def _cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def _env(split, life):
    def load():
        with np.load(env_path(split, life)) as f:
            return {k: (f[k] if f[k].ndim else f[k].item()) for k in f.files}
    return _cached(("env", split, life), load)


def _hist(split):
    def load():
        from src.data.mind import load_processed
        imps = load_processed(split)
        return imps.hist_ptr, imps.hist_idx
    return _cached(("hist", split), load)


def run_task(task: dict) -> tuple[dict, dict, float]:
    import torch
    from src.bandits.contextual import article_features, run, user_features

    env = _env(task["split"], task["life"])
    art_x = user_x = None
    if task["rep"] != "none":
        feats = _cached(("feat", task["rep"]), lambda: np.load(FEATS / f"{task['rep']}.npy"))
        X = article_features(feats, task["d"])
        art_x = X[env["articles"]]
        user_x = user_features(X, env["users"], *_hist(task["split"]))
    t0 = time.time()
    res = run(env, art_x, user_x, task["policy"], "cuda" if torch.cuda.is_available() else "cpu",
              seed=task.get("seed", 0))
    return task, res, time.time() - t0


def _pin_gpu(queue) -> None:
    gpu = queue.get()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def task_name(t: dict) -> str:
    return f"{t['split']}_life{t['life']}__{t['rep']}_d{t['d']}__{t['policy']}"


def execute(stage: str, tasks: list[dict], workers: int) -> None:
    out_dir = OUT / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (out_dir / f"{task_name(t)}.npz").exists()]
    print(f"{stage}: {len(todo)}/{len(tasks)} tasks to run", flush=True)
    todo.sort(key=lambda t: (t["split"], t["life"], t["rep"]))  # keeps worker caches warm
    # Cluster GPUs run in exclusive-process mode: one worker per GPU, pinned before torch loads.
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    ctx = mp.get_context("spawn")
    queue = ctx.Manager().Queue()
    for i in range(workers):
        queue.put(gpus[i % len(gpus)] if gpus else None)
    with ctx.Pool(workers, initializer=_pin_gpu, initargs=(queue,)) as pool:
        for task, res, secs in pool.imap_unordered(run_task, todo):
            np.savez(out_dir / f"{task_name(task)}.npz", **res)
            print(f"{task_name(task)}: regret {res['regret'][:, -1].mean():.1f} ({secs:.0f}s)", flush=True)


def load_stage(stage: str) -> dict:
    out = {}
    for f in sorted((OUT / stage).glob("*.npz")):
        with np.load(f) as z:
            out[f.stem] = {k: z[k] for k in z.files}
    return out


def final_regret(stage: str, name: str) -> float:
    with np.load(OUT / stage / f"{name}.npz") as z:
        return float(z["regret"][:, -1].mean())


def policies_for(rep: str, tuned: dict) -> list[str]:
    if rep == "none":
        return ["beta_ts"]
    family = tuned_family(rep)
    return ["greedy", tuned[family]["linucb"], tuned[family]["lints"]]


def tuned_family(rep: str) -> str:
    """Representation whose tuned exploration settings ``rep`` uses."""
    for m, c in MODELS.items():
        if rep.startswith(f"{m}_sae_"):
            return f"{m}_sae_L{c['mid']}"
        if rep.startswith(f"{m}_L"):
            return f"{m}_L{c['mid']}"
    return rep


def tune_results() -> dict:
    tuned = {}
    for rep in TUNE_REPS:
        tuned[rep] = {}
        for kind, grid in (("linucb", LINUCB_GRID), ("lints", LINTS_GRID)):
            scores = {f"{kind}_{v}": final_regret("tune", task_name(
                {"split": "train", "life": MAIN_LIFE, "rep": rep, "d": 32, "policy": f"{kind}_{v}"}))
                for v in grid}
            tuned[rep][kind] = min(scores, key=scores.get)
            tuned[rep][f"{kind}_scores"] = scores
    (OUT / "tuned.json").write_text(json.dumps(tuned, indent=2))
    return tuned


def chosen_layers(tuned: dict) -> dict:
    """Per model, the dense and SAE layer with the lowest train regret (best of LinUCB/LinTS)."""
    choice = {}
    for m, c in MODELS.items():
        for kind, layers, fmt in (("dense", range(c["layers"]), f"{m}_L{{}}"), ("sae", c["sae"], f"{m}_sae_L{{}}")):
            def score(L):
                rep = fmt.format(L)
                return min(final_regret("layers", task_name({"split": "train", "life": MAIN_LIFE, "rep": rep,
                                                              "d": 32, "policy": p}))
                           for p in policies_for(rep, tuned)[1:])
            choice[f"{m}_{kind}"] = fmt.format(min(layers, key=score))
    (OUT / "chosen_layers.json").write_text(json.dumps(choice, indent=2))
    return choice


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("envs", "tune", "layers", "test", "churn", "all"), required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    stages = ["envs", "tune", "layers", "test", "churn"] if args.stage == "all" else [args.stage]

    for stage in stages:
        if stage == "envs":
            build_envs(args.device)
        elif stage == "tune":
            base = {"split": "train", "life": MAIN_LIFE, "d": 32}
            tasks = [{**base, "rep": "none", "policy": "beta_ts"}]
            tasks += [{**base, "rep": r, "policy": "greedy"} for r in TUNE_REPS]
            tasks += [{**base, "rep": r, "policy": f"linucb_{a}"} for r in TUNE_REPS for a in LINUCB_GRID]
            tasks += [{**base, "rep": r, "policy": f"lints_{v}"} for r in TUNE_REPS for v in LINTS_GRID]
            execute(stage, tasks, args.workers)
            tune_results()
        elif stage == "layers":
            tuned = tune_results()
            reps = [f"{m}_L{L}" for m, c in MODELS.items() for L in range(c["layers"])] + \
                   [f"{m}_sae_L{L}" for m, c in MODELS.items() for L in c["sae"]]
            tasks = [{"split": s, "life": MAIN_LIFE, "rep": r, "d": 32, "policy": pol}
                     for s in RUNS for r in reps for pol in policies_for(r, tuned)[1:]]
            execute(stage, tasks, args.workers)
            chosen_layers(tuned)
        elif stage in ("test", "churn"):
            tuned = tune_results()
            layers = json.loads((OUT / "chosen_layers.json").read_text())
            reps = ["category", "bgem3", "qwen3", *layers.values()]
            if stage == "test":
                tasks = [{"split": "test", "life": MAIN_LIFE, "rep": "none", "d": 0, "policy": "beta_ts"}]
                for r in reps:
                    for d in ([18] if r == "category" else DIMS):
                        tasks += [{"split": "test", "life": MAIN_LIFE, "rep": r, "d": d, "policy": pol}
                                  for pol in policies_for(r, tuned)]
            else:
                tasks = []
                for life in LIFETIMES:
                    if life == MAIN_LIFE:
                        continue
                    tasks.append({"split": "test", "life": life, "rep": "none", "d": 0, "policy": "beta_ts"})
                    for r in reps:
                        d = 18 if r == "category" else 32
                        tasks += [{"split": "test", "life": life, "rep": r, "d": d, "policy": pol}
                                  for pol in policies_for(r, tuned)]
            execute(stage, tasks, args.workers)


if __name__ == "__main__":
    main()
