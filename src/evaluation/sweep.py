"""Reproducible pools and confidence intervals over independent simulation seeds."""
import numpy as np
from scipy.stats import t


def make_pools(arms, pool_count, pool_size, seed):
    if pool_count < 1 or pool_size < 2 or pool_size > len(arms):
        raise ValueError('Need at least one pool and 2 <= pool size <= available articles')
    rng = np.random.default_rng(seed)
    return [rng.choice(len(arms), pool_size, replace=False).tolist() for _ in range(pool_count)]


def interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError('At least two seeds are needed for a confidence interval')
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean-half, mean+half
