"""Q4 environments: small non-contextual bandits whose arms are real test-day headlines.

Each run draws K test-day articles; an arm's reward is a click with probability
equal to the article's population click rate (the simulator's probability
averaged over 20,000 test users). Click draws are stored (common random numbers),
so the LLM agents and every baseline face the same coin flips.

    python -m scripts.build_q4_envs
Writes results/q4/envs.npz (mu, uniforms, article rows) and results/q4/envs.json (headlines).
"""

import json
from pathlib import Path

import numpy as np

from src.data.mind import load_processed_news
from src.simulator.simulator import ClickSimulator

OUT = Path("results/q4")
ARMS = (5, 10)
RUNS, HORIZON, POP_USERS, SEED = 30, 500, 20_000, 6_000_000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sim = ClickSimulator.load(split="test", device="mps")
    news = load_processed_news()
    arrays, meta = {}, {}
    for K in ARMS:
        rng = np.random.default_rng(SEED + K)
        users = sim.sample_users(POP_USERS, rng)
        rows = np.stack([sim.sample_articles(K, rng) for _ in range(RUNS)])
        mu = sim.prob_matrix(users, rows.ravel()).mean(0).reshape(RUNS, K)
        arrays[f"K{K}_rows"], arrays[f"K{K}_mu"] = rows, mu
        arrays[f"K{K}_uniforms"] = rng.random((RUNS, HORIZON))
        meta[f"K{K}"] = [[{"title": news["title"].iloc[r], "category": news["category"].iloc[r]} for r in run]
                         for run in rows]
        gap = np.sort(mu, 1)[:, -1] - np.sort(mu, 1)[:, -2]
        print(f"K={K}: mean CTR {mu.mean():.4f}, best {mu.max(1).mean():.4f}, "
              f"best-vs-second gap median {np.median(gap):.4f}", flush=True)
    np.savez(OUT / "envs.npz", horizon=HORIZON, **arrays)
    (OUT / "envs.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
