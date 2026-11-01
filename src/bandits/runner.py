"""Run a policy against a BanditEnv and record regret, clicks and best-arm share."""

from __future__ import annotations

import numpy as np

from src.bandits.environment import BanditEnv
from src.bandits.policies import ExploreThenCommit, Policy


def checkpoints(horizon: int, n: int = 40) -> np.ndarray:
    """Log-spaced round counts in [10, horizon], always ending at horizon."""
    return np.unique(np.r_[np.geomspace(10, horizon, n).astype(np.int64), horizon])


def run(policy: Policy, env: BanditEnv, marks: np.ndarray | None = None) -> dict:
    """Play ``env.horizon`` rounds. Curves are cumulative totals at each mark (round counts).

    regret         sum_t max_k mu_t[k] - mu_t[arm_t]   (expected clicks lost to the best arm)
    exp_clicks     sum_t mu_t[arm_t]
    clicks         realized clicks
    best_share     share of rounds in (previous mark, mark] spent on the current best arm
    """
    marks = checkpoints(env.horizon) if marks is None else marks
    S, C, rows = env.S, len(marks), env.rows
    out = {k: np.zeros((S, C), np.float64) for k in ("regret", "exp_clicks", "clicks", "best_share")}
    regret = exp_clicks = clicks = np.zeros(S)
    on_best, window, c = np.zeros(S), 0, 0
    static = env.drift == "none"
    if static:
        mu, best, best_arm = env.mu0, env.mu0.max(1), env.mu0.argmax(1)
    for t in range(env.horizon):
        if not static:
            mu = env.mu(t)
            best, best_arm = mu.max(1), mu.argmax(1)
        arms = policy.select(t)
        r = env.uniforms[:, t] < env.click_prob(t, arms)
        policy.update(arms, r)
        m = mu[rows, arms]
        regret = regret + (best - m)
        exp_clicks = exp_clicks + m
        clicks = clicks + r
        on_best += arms == best_arm
        window += 1
        if t + 1 == marks[c]:
            out["regret"][:, c], out["exp_clicks"][:, c], out["clicks"][:, c] = regret, exp_clicks, clicks
            out["best_share"][:, c] = on_best / window
            on_best, window, c = np.zeros(S), 0, c + 1
    out["marks"] = marks
    if isinstance(policy, ExploreThenCommit):
        out["commit_time"] = policy.commit_time.copy()
        out["committed_correct"] = np.where(policy.choice >= 0, policy.choice == env.mu0.argmax(1), False)
    return out
