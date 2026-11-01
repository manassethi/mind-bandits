"""Non-contextual Bernoulli bandit policies, vectorized over S independent runs.

Every policy keeps per-run state of shape (S, K) and exposes:

    select(t)          -> (S,) arm indices for round t (0-based)
    update(arms, r)    -> record (S,) rewards in {0, 1}

Ties are broken uniformly at random. Except for explore-then-commit (which
explores round-robin), the index policies first play every arm once.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


class Policy:
    name = "policy"

    def __init__(self, n_runs: int, n_arms: int, rng: np.random.Generator):
        self.S, self.K, self.rng = n_runs, n_arms, rng
        self.pulls = np.zeros((n_runs, n_arms))
        self.wins = np.zeros((n_runs, n_arms))
        self.rows = np.arange(n_runs)

    def argmax(self, score: np.ndarray) -> np.ndarray:
        return (score + 1e-12 * self.rng.random(score.shape)).argmax(1)

    def means(self) -> np.ndarray:
        return self.wins / np.maximum(self.pulls, 1)

    def untried_first(self, score: np.ndarray) -> np.ndarray:
        return np.where(self.pulls == 0, np.inf, score)

    def update(self, arms: np.ndarray, rewards: np.ndarray) -> None:
        self.pulls[self.rows, arms] += 1
        self.wins[self.rows, arms] += rewards


class Uniform(Policy):
    """Split traffic evenly forever: an A/B test that never ends."""

    name = "uniform"

    def select(self, t):
        return self.rng.integers(self.K, size=self.S)


class ExploreThenCommit(Policy):
    """A/B test: round-robin for m rounds per arm, then commit to the best observed arm."""

    def __init__(self, n_runs, n_arms, rng, per_arm: np.ndarray | int, name: str):
        super().__init__(n_runs, n_arms, rng)
        self.explore_rounds = np.broadcast_to(np.asarray(per_arm) * n_arms, (n_runs,)).astype(np.int64)
        self.choice = np.full(n_runs, -1)
        self.commit_time = np.full(n_runs, -1)
        self.name = name

    def select(self, t):
        exploring = t < self.explore_rounds
        newly = (~exploring) & (self.choice < 0)
        if newly.any():
            self.choice[newly] = self.argmax(self.means())[newly]
            self.commit_time[newly] = t
        return np.where(exploring, t % self.K, self.choice)


def ab_sample_size(p0: np.ndarray, mde: float, alpha: float = 0.05, power: float = 0.8) -> np.ndarray:
    """Per-arm sample size of a two-sided two-proportion z-test detecting a relative lift ``mde``."""
    p1 = p0 * (1 + mde)
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return np.ceil(z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2).astype(np.int64)


class EpsilonGreedy(Policy):
    """Exploit the best empirical mean; explore uniformly with probability eps (or eps_t = min(1, c K / t))."""

    def __init__(self, n_runs, n_arms, rng, eps: float | None = None, decay_c: float | None = None):
        super().__init__(n_runs, n_arms, rng)
        self.eps, self.decay_c = eps, decay_c
        self.name = f"egreedy_{eps}" if eps is not None else f"egreedy_decay_{decay_c}"

    def select(self, t):
        eps = self.eps if self.eps is not None else min(1.0, self.decay_c * self.K / max(t, 1))
        greedy = self.argmax(self.untried_first(self.means()))
        explore = self.rng.random(self.S) < eps
        return np.where(explore, self.rng.integers(self.K, size=self.S), greedy)


class UCB1(Policy):
    """mean + c * sqrt(2 ln t / n). c = 1 is Auer et al. (2002); smaller c suits low click rates."""

    def __init__(self, n_runs, n_arms, rng, c: float = 1.0):
        super().__init__(n_runs, n_arms, rng)
        self.c, self.name = c, f"ucb1_{c}"

    def select(self, t):
        bonus = self.c * np.sqrt(2 * np.log(max(t, 1)) / np.maximum(self.pulls, 1))
        return self.argmax(self.untried_first(self.means() + bonus))


class Thompson(Policy):
    """Beta-Bernoulli Thompson sampling with a Beta(1, 1) prior."""

    name = "thompson"

    def select(self, t):
        return self.argmax(self.rng.beta(1 + self.wins, 1 + self.pulls - self.wins))


class DiscountedThompson(Policy):
    """Thompson sampling whose counts decay by gamma every round, so old evidence fades (for drift)."""

    def __init__(self, n_runs, n_arms, rng, gamma: float):
        super().__init__(n_runs, n_arms, rng)
        self.gamma, self.name = gamma, f"dthompson_{gamma}"

    def select(self, t):
        return self.argmax(self.rng.beta(1 + self.wins, 1 + self.pulls - self.wins))

    def update(self, arms, rewards):
        self.pulls *= self.gamma
        self.wins *= self.gamma
        super().update(arms, rewards)


def policy_suite(n_runs: int, n_arms: int, horizon: int, pool_ctr: np.ndarray, seed: int) -> list[Policy]:
    """Every policy variant compared in Q1, each with its own random stream.

    ``pool_ctr`` (S,) is the average click rate of each run's pool: the site-wide
    CTR a team would plug into an A/B power calculation.
    """
    variants = []
    for frac in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        variants.append(lambda r, f=frac: ExploreThenCommit(
            n_runs, n_arms, r, max(1, int(f * horizon / n_arms)), f"ab_frac_{f}"))
    for mde in (0.2, 0.5):
        variants.append(lambda r, m=mde: ExploreThenCommit(
            n_runs, n_arms, r, ab_sample_size(pool_ctr, m), f"ab_power_mde_{m}"))
    for eps in (0.01, 0.05, 0.1):
        variants.append(lambda r, e=eps: EpsilonGreedy(n_runs, n_arms, r, eps=e))
    for c in (1.0, 10.0, 30.0, 100.0):
        variants.append(lambda r, c=c: EpsilonGreedy(n_runs, n_arms, r, decay_c=c))
    for c in (1.0, 0.3, 0.1, 0.03, 0.01):
        variants.append(lambda r, c=c: UCB1(n_runs, n_arms, r, c=c))
    variants.append(lambda r: Thompson(n_runs, n_arms, r))
    for g in (0.999, 0.9999, 0.99999):
        variants.append(lambda r, g=g: DiscountedThompson(n_runs, n_arms, r, gamma=g))
    variants.append(lambda r: Uniform(n_runs, n_arms, r))
    seeds = np.random.SeedSequence(seed).spawn(len(variants))
    return [make(np.random.default_rng(s)) for make, s in zip(variants, seeds)]
