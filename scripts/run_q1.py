"""Q1: run every non-contextual policy in every condition on one split.

    python -m scripts.run_q1 --split train   # tuning runs (train-day users and articles)
    python -m scripts.run_q1 --split test    # reported runs (test-day users and articles)

Environments are tabulated once with the click simulator and cached under
results/q1/envs; policy runs are pure numpy and run in parallel processes.
Output: results/q1/<split>/<condition>.npz with every policy's curves.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from src.bandits.environment import DRIFTS, BanditEnv, build_env
from src.bandits.policies import policy_suite
from src.bandits.runner import run

# name -> (arms K, horizon T, drift)
CONDITIONS = {
    "K2": (2, 100_000, "none"),
    "K5": (5, 100_000, "none"),
    "K10": (10, 100_000, "none"),
    "K20": (20, 100_000, "none"),
    "K50": (50, 100_000, "none"),
    "K10_T1k": (10, 1_000, "none"),
    "K10_T10k": (10, 10_000, "none"),
    "K10_abrupt": (10, 100_000, "abrupt"),
    "K10_gradual": (10, 100_000, "gradual"),
}
SPLIT_SEED = {"train": 1_000_000, "test": 2_000_000}
OUT = Path("results/q1")


def env_path(split: str, cond: str) -> Path:
    return OUT / "envs" / f"{split}_{cond}.npz"


def env_seed(split: str, K: int, drift: str) -> int:
    # Same K (and drift) -> same article pools across horizons, so horizon effects are paired.
    return SPLIT_SEED[split] + 1000 * K + DRIFTS.index(drift)


def build_all(split: str, n_runs: int, n_users: int, device: str) -> None:
    todo = [c for c in CONDITIONS if not env_path(split, c).exists()]
    if not todo:
        return
    from src.simulator.simulator import ClickSimulator

    sim = ClickSimulator.load(split=split, device=device)
    env_path(split, todo[0]).parent.mkdir(parents=True, exist_ok=True)
    for cond in todo:
        K, T, drift = CONDITIONS[cond]
        env = build_env(sim, n_runs, K, n_users, T, drift, env_seed(split, K, drift))
        np.savez(env_path(split, cond), P=env.P, articles=env.articles, horizon=T, drift=drift,
                 perm=env.perm if env.perm is not None else np.zeros(0), seed=env.seed)
        print(f"built {split}/{cond}: mean arm CTR {env.mu0.mean():.4f}, "
              f"best/median {np.median(env.mu0.max(1) / np.median(env.mu0, 1)):.2f}", flush=True)


def load_env(split: str, cond: str) -> BanditEnv:
    with np.load(env_path(split, cond)) as f:
        perm = f["perm"] if f["perm"].size else None
        return BanditEnv(f["P"], f["articles"], int(f["horizon"]), str(f["drift"]), perm, int(f["seed"]))


_ENVS: dict = {}


def run_task(task):
    split, cond, i = task
    if (split, cond) not in _ENVS:
        _ENVS.clear()
        _ENVS[(split, cond)] = load_env(split, cond)
    env = _ENVS[(split, cond)]
    policy = policy_suite(env.S, env.K, env.horizon, env.mu0.mean(1), seed=env.seed)[i]
    t0 = time.time()
    res = run(policy, env)
    return cond, policy.name, res, time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=SPLIT_SEED, required=True)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument("--runs", type=int, default=None, help="default: 20 for train, 30 for test")
    p.add_argument("--users", type=int, default=2000)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    n_runs = args.runs or (20 if args.split == "train" else 30)

    build_all(args.split, n_runs, args.users, args.device)
    n_policies = len(policy_suite(1, 2, 10, np.array([0.04]), 0))
    tasks = [(args.split, c, i) for c in args.conditions for i in range(n_policies)]
    tasks.sort(key=lambda t: -CONDITIONS[t[1]][1])  # longest horizons first

    results: dict = {c: {} for c in args.conditions}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for cond, name, res, secs in pool.imap_unordered(run_task, tasks):
            results[cond][name] = res
            print(f"{args.split}/{cond}/{name}: regret {res['regret'][:, -1].mean():.1f} ({secs:.0f}s)",
                  flush=True)
            if len(results[cond]) == n_policies:
                out = OUT / args.split / f"{cond}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, **{f"{n}/{k}": v for n, r in results[cond].items() for k, v in r.items()})


if __name__ == "__main__":
    main()
