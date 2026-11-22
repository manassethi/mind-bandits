"""Small known-response simulator implementing the shared probability interface."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.optimize import brentq


class SyntheticClickSimulator:
    def __init__(self, n_users=2000, n_articles=1000, dim=16, base_ctr=.04, signal=1.0, seed=0):
        if not 0 < base_ctr < 1 or signal < 0:
            raise ValueError("CTR must be in (0,1) and signal must be nonnegative")
        rng = np.random.default_rng(seed)
        self.user_features = rng.normal(size=(n_users, dim)).astype(np.float32)
        self.article_features = rng.normal(size=(n_articles, dim)).astype(np.float32)
        self.user_pool, self.article_pool = np.arange(n_users), np.arange(n_articles)
        score = signal * (self.user_features @ self.article_features.T) / np.sqrt(dim)
        bias = brentq(lambda b: expit(score + b).mean() - base_ctr, -50, 50)
        self.probabilities = expit(score + bias).astype(np.float32)

    def sample_users(self, n, rng):
        return rng.choice(self.user_pool, size=n, replace=True)

    def sample_articles(self, k, rng):
        return rng.choice(self.article_pool, size=k, replace=False)

    def prob_matrix(self, user_rows, article_rows, **kwargs):
        return self.probabilities[np.ix_(user_rows, article_rows)]
