"""Test whether the observed batch-effect reversal survives train-only retuning."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q1_allocation import make_policy, simulate
from scripts.run_q1 import load_env
from src.bandits.environment import BanditEnv
from src.bandits.policies import EpsilonGreedy

ROOT = OUT / "q1_batch_retune"
CONDITIONS = (("K2", 10000), ("K10_T10k", 10000), ("K10", 100000))


def candidate(family, value, env, batch):
    if family == "ucb":
        return make_policy("ucb_tuned", env, batch, 1, coefficient=value)
    if family == "discounted_ts":
        return make_policy("discounted_ts", env, batch, 1, discount=value)
    return EpsilonGreedy(env.S, env.K, np.random.default_rng(SEED), eps=value)


def one_condition(task):
    name, horizon = task
    old_train, old_test = load_env("train", name), load_env("test", name)
    train = BanditEnv(old_train.P, old_train.articles, horizon, seed=old_train.seed)
    test = BanditEnv(old_test.P, old_test.articles, horizon, seed=old_test.seed)
    results = {}
    for batch in (1, 100, 1000):
        reference = simulate(make_policy("ts", test, batch, 1), test, batch)["regret"]
        result = {}
        for family, grid in [("ucb", (.01, .03, .1, .3, 1.)),
                             ("discounted_ts", (.999, .9999, .99999)), ("epsilon", (.01, .05, .1))]:
            regrets = [simulate(candidate(family, v, train, batch), train, batch, diagnostics=False)["regret"] for v in grid]
            selected = int(np.argmin([v.mean() for v in regrets]))
            winner = grid[selected]
            evaluation = simulate(candidate(family, winner, test, batch), test, batch)
            result[family] = {"selected": winner, "train_grid": list(grid),
                "train_regret": [interval(v) for v in regrets], "test_regret": interval(evaluation["regret"]),
                "paired_minus_ts": interval(evaluation["regret"] - reference)}
            save_npz(ROOT / f"{name}_T{horizon}_B{batch}_{family}.npz", **evaluation)
        results[str(batch)] = result
        print(name, horizon, batch, {k: v["selected"] for k, v in result.items()}, flush=True)
    return f"{name}_T{horizon}", results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()
    provenance = fingerprint({"seed": SEED, "conditions": CONDITIONS, "batches": [1, 100, 1000],
                              "status": "exploratory follow-up to frozen-policy batch reversal"},
                             [Path(__file__), Path("scripts/final_validation/q1_allocation.py")])
    if is_complete(ROOT / "summary.json", provenance):
        return
    with mp.get_context("spawn").Pool(args.workers) as pool:
        result = dict(pool.imap_unordered(one_condition, CONDITIONS))
    write_json(ROOT / "summary.json", {"provenance": provenance, "results": result})


if __name__ == "__main__":
    main()
