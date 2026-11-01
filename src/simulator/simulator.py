"""Semi-synthetic click simulator built on the fitted click model.

The simulator knows the true click probability of every (user, article) pair,
so any policy can be scored against an oracle. Users come from the test
split's histories and articles from the test split's candidate pool.

Typical use: draw a user population and an article pool, compute their
probability matrix once with ``prob_matrix``, then run policies against
that matrix (one row lookup per round instead of one model call).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.mind import PROCESSED_DIR, load_processed, load_processed_news
from src.simulator import click_model as cm
from src.simulator.encode import load_embeddings
from src.simulator.features import ArticleTable, FeatureStore, UserTable


def user_table_path(split: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / f"sim_users_{split}.npz"


class ClickSimulator:
    def __init__(self, model: cm.ClickModel, store: FeatureStore,
                 user_pool: np.ndarray, article_pool: np.ndarray):
        self.model = model
        self.store = store
        self.user_pool = user_pool        # user rows eligible to arrive
        self.article_pool = article_pool  # news rows eligible to be candidates

    @classmethod
    def load(cls, split: str = "test", min_history: int = 1, device: str = "cpu") -> "ClickSimulator":
        articles = ArticleTable.from_news(load_processed_news(), load_embeddings())
        imps = load_processed(split)
        if user_table_path(split).exists():
            with np.load(user_table_path(split)) as f:
                users = UserTable(**{k: f[k] for k in f.files})
        else:  # only test's table is cached; others take a few seconds to build
            users = UserTable.from_impressions(imps, articles)
        store = FeatureStore(users, articles, device)
        model = cm.load(device=device)
        user_pool = np.flatnonzero(users.hist_len >= min_history)
        return cls(model, store, user_pool, np.unique(imps.cand_idx))

    def sample_users(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.user_pool, size=n, replace=True)

    def sample_articles(self, k: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.article_pool, size=k, replace=False)

    @torch.no_grad()
    def prob_matrix(self, user_rows: np.ndarray, article_rows: np.ndarray,
                    batch_size: int = 65_536) -> np.ndarray:
        """True click probability for every user x article pair, shape (len(users), len(articles))."""
        u = np.repeat(user_rows, len(article_rows))
        a = np.tile(article_rows, len(user_rows))
        dev = self.store.device
        probs = cm.predict(self.model, self.store, torch.from_numpy(u).to(dev),
                           torch.from_numpy(a).to(dev), batch_size)
        return probs.reshape(len(user_rows), len(article_rows))

    def article_ctr(self, article_rows: np.ndarray, rng: np.random.Generator,
                    n_users: int = 20_000) -> np.ndarray:
        """Population click rate of each article (average over random users).

        This is the arm mean a non-contextual bandit is chasing.
        """
        return self.prob_matrix(self.sample_users(n_users, rng), article_rows).mean(0)

    @staticmethod
    def click(p: np.ndarray | float, rng: np.random.Generator):
        return rng.random(np.shape(p)) < p
