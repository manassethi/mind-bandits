"""User and article features for the click model.

A user is the L2-normalized mean simulator-encoder (Qwen3) embedding of their click history plus
category/subcategory click counts. A (user, article) pair is scored from:

    u * a      elementwise interaction (sums to cosine similarity)
    a          article content, i.e. how clickable it is for anyone
    u          user content, i.e. how much this user clicks in general
    scalars    cosine, category affinity, subcategory affinity,
               log(1 + history length), empty-history flag

Everything is content-only (no article ids or popularity counts), so the
model can score articles it never saw during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from src.data.mind import Impressions

N_SCALARS = 5


@dataclass
class ArticleTable:
    emb: np.ndarray        # (N, D) float32, L2-normalized
    category: np.ndarray   # (N,) int32
    subcategory: np.ndarray  # (N,) int32
    n_categories: int
    n_subcategories: int

    @classmethod
    def from_news(cls, news: pd.DataFrame, emb: np.ndarray) -> "ArticleTable":
        cat, cats = pd.factorize(news["category"])
        sub, subs = pd.factorize(news["subcategory"])
        return cls(emb.astype(np.float32), cat.astype(np.int32), sub.astype(np.int32),
                   len(cats), len(subs))


@dataclass
class UserTable:
    vec: np.ndarray          # (U, D) float16, L2-normalized mean history embedding
    cat_count: np.ndarray    # (U, n_categories) float16
    sub_count: np.ndarray    # (U, n_subcategories) float16
    hist_len: np.ndarray     # (U,) int32

    @classmethod
    def from_impressions(cls, imps: Impressions, articles: ArticleTable) -> "UserTable":
        hist_len = np.diff(imps.hist_ptr).astype(np.int32)
        n_users, n_articles = imps.n_users, len(articles.emb)
        hist = sp.csr_matrix((np.ones(len(imps.hist_idx), np.float32), imps.hist_idx, imps.hist_ptr),
                             shape=(n_users, n_articles))

        vec = np.empty((n_users, articles.emb.shape[1]), np.float16)
        for s in range(0, n_users, 50_000):
            v = hist[s:s + 50_000] @ articles.emb
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            vec[s:s + 50_000] = np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)

        def counts(codes: np.ndarray, n: int) -> np.ndarray:
            onehot = sp.csr_matrix((np.ones(n_articles, np.float32), (np.arange(n_articles), codes)),
                                   shape=(n_articles, n))
            return (hist @ onehot).toarray().astype(np.float16)

        return cls(vec, counts(articles.category, articles.n_categories),
                   counts(articles.subcategory, articles.n_subcategories), hist_len)


class FeatureStore:
    """Holds user and article tables on a torch device and builds pair features."""

    def __init__(self, users: UserTable, articles: ArticleTable, device: str | torch.device):
        self.device = torch.device(device)
        to = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        self.a_emb = to(articles.emb)
        self.a_cat = to(articles.category).long()
        self.a_sub = to(articles.subcategory).long()
        self.u_vec = to(users.vec)
        self.u_cat = to(users.cat_count)
        self.u_sub = to(users.sub_count)
        self.u_len = to(users.hist_len).float()
        self.dim = articles.emb.shape[1]

    @property
    def n_features(self) -> int:
        return 3 * self.dim + N_SCALARS

    def cosine(self, user_rows: torch.Tensor, article_rows: torch.Tensor) -> torch.Tensor:
        return (self.u_vec[user_rows].float() * self.a_emb[article_rows]).sum(1)

    def pair_features(self, user_rows: torch.Tensor, article_rows: torch.Tensor) -> torch.Tensor:
        u = self.u_vec[user_rows].float()
        a = self.a_emb[article_rows]
        n = self.u_len[user_rows]
        denom = n.clamp(min=1)
        cat_aff = self.u_cat[user_rows, self.a_cat[article_rows]].float() / denom
        sub_aff = self.u_sub[user_rows, self.a_sub[article_rows]].float() / denom
        ua = u * a
        scalars = torch.stack([ua.sum(1), cat_aff, sub_aff, torch.log1p(n), (n == 0).float()], dim=1)
        return torch.cat([ua, a, u, scalars], dim=1)
