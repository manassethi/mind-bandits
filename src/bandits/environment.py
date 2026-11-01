"""Bandit environments drawn from the click simulator, vectorized over S runs.

Each run s has K articles (arms) and a population of U users. Every round one
user is drawn uniformly from that population, so arm k's expected reward is
its population click rate mu[s, k] = mean_u P[s, u, k].

Drift: arm click rates move from P to P[:, :, perm] (an arm permutation that
changes which arm is best), either abruptly at T/2 or linearly over the run.

All policies in a run see the same users and the same click draws (common
random numbers), so per-run differences between policies are paired.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DRIFTS = ("none", "abrupt", "gradual")


@dataclass
class BanditEnv:
    P: np.ndarray          # (S, U, K) true click probability per user and arm
    articles: np.ndarray   # (S, K) news rows of the arms
    horizon: int
    drift: str = "none"
    perm: np.ndarray | None = None  # (S, K) arm k takes on arm perm[s, k]'s click rates
    seed: int = 0

    def __post_init__(self):
        S, U, K = self.P.shape
        self.S, self.U, self.K = S, U, K
        self.mu0 = self.P.mean(1)
        self.mu1 = np.take_along_axis(self.mu0, self.perm, 1) if self.perm is not None else self.mu0
        rng = np.random.default_rng(self.seed)
        self.users = rng.integers(U, size=(S, self.horizon), dtype=np.int32)
        self.uniforms = rng.random((S, self.horizon), dtype=np.float32)
        self.rows = np.arange(S)

    def weight(self, t: int) -> float:
        if self.drift == "none":
            return 0.0
        if self.drift == "abrupt":
            return float(t >= self.horizon // 2)
        return t / max(self.horizon - 1, 1)

    def mu(self, t: int) -> np.ndarray:
        w = self.weight(t)
        return self.mu0 if w == 0 else (1 - w) * self.mu0 + w * self.mu1

    def click_prob(self, t: int, arms: np.ndarray) -> np.ndarray:
        u = self.users[:, t]
        p = self.P[self.rows, u, arms]
        w = self.weight(t)
        if w > 0:
            p = (1 - w) * p + w * self.P[self.rows, u, self.perm[self.rows, arms]]
        return p

    def personalized_oracle_ctr(self) -> np.ndarray:
        """(S,) click rate if every user saw their own best arm (no drift)."""
        return self.P.max(2).mean(1)


def drift_permutation(mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random arm permutation per run under which the best arm changes identity."""
    S, K = mu.shape
    perm = np.empty((S, K), dtype=np.int64)
    for s in range(S):
        best = mu[s].argmax()
        while True:
            p = rng.permutation(K)
            if p[best] != best:
                perm[s] = p
                break
    return perm


def build_env(sim, n_runs: int, n_arms: int, n_users: int, horizon: int, drift: str,
              seed: int) -> BanditEnv:
    """Draw S article pools and user populations from ``sim`` and tabulate their click probabilities."""
    assert drift in DRIFTS
    rng = np.random.default_rng(seed)
    P = np.empty((n_runs, n_users, n_arms), dtype=np.float32)
    articles = np.empty((n_runs, n_arms), dtype=np.int64)
    for s in range(n_runs):
        articles[s] = sim.sample_articles(n_arms, rng)
        P[s] = sim.prob_matrix(sim.sample_users(n_users, rng), articles[s])
    perm = drift_permutation(P.mean(1), rng) if drift != "none" else None
    return BanditEnv(P, articles, horizon, drift, perm, seed=seed + 1)
