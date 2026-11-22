"""Known-response controls for probability gaps, pairing, and contextual transfer."""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.final_validation.q1_allocation import make_policy, simulate
from src.bandits.contextual import run
from src.bandits.environment import BanditEnv
from src.simulator.synthetic import SyntheticClickSimulator

ROOT = OUT / "synthetic"


def population(task):
    ctr, gap, K = task
    means = np.full((30, K), ctr - gap / K)
    means[:, 0] += gap
    assert np.allclose(means.mean(1), ctr)
    P = np.repeat(means[:, None, :], 64, 1).astype(np.float32)
    env = BanditEnv(P, np.tile(np.arange(K), (30, 1)), 10000, seed=SEED)
    result = {}
    for name in ("etc_fixed10pct", "ucb_tuned", "ts", "kl_ucb", "uniform"):
        policy = make_policy(name, env, 1, 10000 // (10 * K))
        res = simulate(policy, env, 1)
        if gap == 0:
            assert np.all(res["regret"] == 0)
        assert np.allclose(res["expected_ctr"] * 10000 + res["regret"], env.mu0.max(1) * 10000, atol=.001)
        result[name] = {"regret": interval(res["regret"]), "terminal_correct": interval(res["terminal_correct"])}
    label = f"ctr{ctr}_gap{gap}_K{K}"
    print("Synthetic", label, flush=True)
    return label, result


def contextual():
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    S, U, M, D = 30, 200, 221, 8
    article = rng.choice([-1., 1.], (S, M, D)).astype(np.float32) / np.sqrt(D)
    user = rng.choice([-1., 1.], (S, U, D)).astype(np.float32) / np.sqrt(D)
    random_a = rng.choice([-1., 1.], article.shape).astype(np.float32) / np.sqrt(D)
    random_u = rng.choice([-1., 1.], user.shape).astype(np.float32) / np.sqrt(D)
    P = (.04 + .03 * np.einsum("sud,smd->sum", user, article)).astype(np.float32)
    assert P.min() >= 0 and P.max() <= 1
    env = {"P": P, "n_live": 20, "lifetime": 2000, "horizon": 20000, "seed": SEED}
    results, arrays = {}, {}
    for name, a, u, policy in [("known_content", article, user, "linucb_0.1"),
                               ("random_content", random_a, random_u, "linucb_0.1"),
                               ("no_content", None, None, "beta_ts")]:
        torch.manual_seed(SEED)
        res = run(env, a, u, policy, "cpu", seed=SEED)
        results[name] = {"regret": interval(res["regret"][:, -1])}
        arrays[name] = res["regret"][:, -1]
    results["paired_known_minus_random"] = interval(arrays["known_content"] - arrays["random_content"])
    results["paired_known_minus_none"] = interval(arrays["known_content"] - arrays["no_content"])
    save_npz(ROOT / "contextual_regret.npz", **arrays)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    provenance = fingerprint({"seed": SEED, "ctrs": [.02, .04, .08], "gaps": [0, .001, .005, .02], "arms": [2, 10]},
                             [Path(__file__), Path("src/simulator/synthetic.py"), Path("src/bandits/contextual.py")])
    if is_complete(ROOT / "summary.json", provenance):
        return
    mock = SyntheticClickSimulator(n_users=20, n_articles=40, base_ctr=.04, signal=0, seed=SEED)
    assert np.allclose(mock.prob_matrix(np.arange(20), np.arange(40)), .04)
    tasks = [(ctr, gap, K) for ctr in (.02, .04, .08) for gap in (0, .001, .005, .02) for K in (2, 10)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        population_results = dict(pool.imap_unordered(population, tasks))
    contextual_results = contextual()
    write_json(ROOT / "summary.json", {"provenance": provenance, "population": population_results,
        "contextual": contextual_results, "checks": ["probability bounds", "fixed mean CTR", "zero-gap regret", "oracle reward identity", "shared probability interface"],
        "contextual_truth": "P(click)=0.04+0.03*dot(user,article), with unit-norm Rademacher features"})


if __name__ == "__main__":
    main()
