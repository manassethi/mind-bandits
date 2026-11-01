"""Click-model metrics: global and MIND-style per-impression ranking metrics, calibration."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def impression_metrics(probs: np.ndarray, labels: np.ndarray, ptr: np.ndarray) -> dict:
    """Mean over impressions of AUC, MRR, nDCG@5, nDCG@10 (the MIND leaderboard metrics).

    Impressions with no clicks or no non-clicks are skipped, as in the official scorer.
    """
    auc, mrr, n5, n10 = [], [], [], []
    for s, e in zip(ptr[:-1], ptr[1:]):
        y, p = labels[s:e], probs[s:e]
        npos = int(y.sum())
        if npos == 0 or npos == len(y):
            continue
        r = _ranks_desc(p)
        pos_r = r[y == 1]
        # AUC via rank-sum on ascending ranks (Mann-Whitney U).
        asc = len(y) + 1 - r
        auc.append((asc[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))
        mrr.append(np.mean(1.0 / pos_r))
        for k, store in ((5, n5), (10, n10)):
            dcg = np.sum(1.0 / np.log2(pos_r[pos_r <= k] + 1))
            idcg = np.sum(1.0 / np.log2(np.arange(1, min(npos, k) + 1) + 1))
            store.append(dcg / idcg)
    return {"group_auc": float(np.mean(auc)), "mrr": float(np.mean(mrr)),
            "ndcg@5": float(np.mean(n5)), "ndcg@10": float(np.mean(n10)),
            "n_impressions_scored": len(auc)}


def calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """Equal-mass bins of predicted probability vs observed click rate."""
    order = np.argsort(probs)
    bins = np.array_split(order, n_bins)
    rows = [{"mean_pred": float(probs[b].mean()), "observed": float(labels[b].mean()), "n": len(b)}
            for b in bins]
    ece = sum(r["n"] * abs(r["mean_pred"] - r["observed"]) for r in rows) / len(probs)
    return {"ece": float(ece), "bins": rows}


def global_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    p = np.clip(probs.astype(np.float64), 1e-7, 1 - 1e-7)
    return {"auc": float(roc_auc_score(labels, p)), "log_loss": float(log_loss(labels, p)),
            "brier": float(np.mean((p - labels) ** 2)), "mean_pred": float(p.mean()),
            "observed_ctr": float(labels.mean()), "n": int(len(labels))}
