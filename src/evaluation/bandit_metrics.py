"""Q4 decision metrics without model-training or plotting dependencies."""

import numpy as np
from scipy import stats


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def metrics(choices: np.ndarray, mu: np.ndarray) -> dict:
    S, T = choices.shape
    K = mu.shape[1]
    rows = np.arange(S)[:, None]
    best = mu.argmax(1)[:, None]
    inst = mu.max(1)[:, None] - mu[rows, choices]
    regret = inst.cumsum(1)
    half = choices[:, T // 2:]
    counts = np.stack([(choices == k).sum(1) for k in range(K)], 1) / T
    top_half = np.stack([(half == k).mean(1) for k in range(K)], 1)
    stuck = (top_half.max(1) >= 0.9) & (top_half.argmax(1) != best[:, 0])
    return {"regret": ci(regret[:, -1]), "regret_runs": regret[:, -1].tolist(), "curve": regret.mean(0).tolist(),
            "best_share": ci((choices == best).mean(1)), "best_share_last100": ci((choices[:, -100:] == best).mean(1)),
            "suffix_failure": float((half != best).all(1).mean()), "k_minfrac": ci(K * counts.min(1)),
            "stuck": float(stuck.mean())}
