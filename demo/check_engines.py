"""Check that the demo engines reproduce the paper's averages (needs the Git LFS inputs).

1. race() on the 30 stored Q1 test pools (K=10, T=10,000) with the tuned settings from
   results/q1/tuned.json. Expected (results/q1/summary.md): A/B 179 +- 56, eps-greedy 90 +- 29,
   UCB 47 +- 16, TS 80 +- 6, discounted TS 80 +- 5.
2. churn_run() on the 30 stored Q2 test pools. Expected (results/q2/summary.md, d = 32):
   per-article TS 1338 +- 28, Gemma-3-12B dense L43 LinUCB 754 +- 44.

The engines use their own random streams (the Q1 runner also samples users per round), so
means should land inside the paper's intervals, not match digit for digit.

    python -m demo.check_engines            # both checks
    python -m demo.check_engines --only q1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np
from scipy import stats


def ci(x) -> str:
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return f"{x.mean():7.1f} +- {h:5.1f}"


def check_q1() -> None:
    from demo.engines import race

    with np.load("results/q1/envs/test_K10_T10k.npz") as f:
        mu = f["P"].mean(1)                     # population click rate of each headline
    res = race(mu, 10_000, seed=0, ab_frac=0.2, eps_c=30.0, ucb_c=0.1, gamma=0.99999)
    print("Q1, K=10, T=10,000, 30 pools (paper: A/B 179+-56, eps 90+-29, UCB 47+-16, TS 80+-6, DTS 80+-5)")
    for name, r in res.items():
        if not name.startswith("_"):
            print(f"  {name:20s} {ci(r['regret'][:, -1])}")


def _churn_task(args):
    s, rep = args
    from demo.engines import churn_run
    from scripts.run_q2 import FEATS
    from src.bandits.contextual import article_features, user_features
    from src.data.mind import load_processed

    with np.load("results/q2/envs/test_life2000.npz") as f:
        env = {k: (f[k] if f[k].ndim else f[k].item()) for k in f.files}
    kw = dict(n_live=env["n_live"], lifetime=env["lifetime"], horizon=env["horizon"], seed=1000 + s)
    if rep is None:
        return churn_run(env["P"][s], None, None, 0.0, **kw)["regret"][-1]
    X = article_features(np.load(FEATS / f"{rep}.npy"), 32)
    imps = load_processed("test")
    ux = user_features(X, env["users"][s:s + 1], imps.hist_ptr, imps.hist_idx)[0]
    return churn_run(env["P"][s], X[env["articles"][s]], ux, 0.3, **kw)["regret"][-1]


def check_q2(workers: int) -> None:
    print("Q2, 20 live articles, T=20,000, 30 pools (paper: TS 1338+-28, dense L43 LinUCB 754+-44)")
    with mp.get_context("spawn").Pool(workers) as pool:
        for label, rep in (("per-article TS", None), ("LinUCB dense L43", "g12b_L43")):
            out = pool.map(_churn_task, [(s, rep) for s in range(30)])
            print(f"  {label:20s} {ci(out)}", flush=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # policy names contain ε (Windows consoles default to cp1252)
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["q1", "q2"])
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.only != "q2":
        check_q1()
    if a.only != "q1":
        check_q2(a.workers)
