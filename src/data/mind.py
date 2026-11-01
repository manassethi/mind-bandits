"""Load MIND news and behavior logs into compact index arrays.

Article ids (N12345) map to row indices of a single news table that is the
union of every split, so embeddings and click-model inputs share one index.
Histories and candidate lists are stored CSR-style: ``ptr`` holds offsets into
a flat ``idx`` array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("dataset/raw/mind-large")
PROCESSED_DIR = Path("dataset/processed")
SPLITS = ("train", "test")

NEWS_COLUMNS = ["news_id", "category", "subcategory", "title", "abstract",
                "url", "title_entities", "abstract_entities"]


def load_news(raw_dir: Path = RAW_DIR, splits=SPLITS) -> pd.DataFrame:
    """Union of news.tsv across splits, one row per article, indexed 0..N-1."""
    frames = [
        pd.read_csv(raw_dir / s / "news.tsv", sep="\t", header=None,
                    names=NEWS_COLUMNS, quoting=3, keep_default_na=False)
        for s in splits
    ]
    news = pd.concat(frames).drop_duplicates("news_id").reset_index(drop=True)
    return news[["news_id", "category", "subcategory", "title", "abstract"]]


@dataclass
class Impressions:
    """One split of behaviors.tsv, encoded against the news table.

    Each impression points at a user row (``imp_user``). A user row is a
    unique (user id, history) pair; MIND keeps a user's history fixed within
    a split, so this deduplicates history vectors.
    """

    imp_id: np.ndarray        # (I,) int64 original impression id
    imp_user: np.ndarray      # (I,) int32 row into user arrays
    imp_time: np.ndarray      # (I,) int64 unix seconds
    cand_ptr: np.ndarray      # (I+1,) int64
    cand_idx: np.ndarray      # (C,) int32 news row
    cand_label: np.ndarray    # (C,) int8 1 = clicked
    user_ids: np.ndarray      # (U,) str original user id
    hist_ptr: np.ndarray      # (U+1,) int64
    hist_idx: np.ndarray      # (H,) int32 news row

    @property
    def n_impressions(self) -> int:
        return len(self.imp_id)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    def candidates(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.cand_ptr[i], self.cand_ptr[i + 1]
        return self.cand_idx[s:e], self.cand_label[s:e]

    def history(self, u: int) -> np.ndarray:
        return self.hist_idx[self.hist_ptr[u]:self.hist_ptr[u + 1]]

    def cand_imp(self) -> np.ndarray:
        """Impression index of every candidate row."""
        return np.repeat(np.arange(self.n_impressions, dtype=np.int32),
                         np.diff(self.cand_ptr))

    def subset(self, mask: np.ndarray) -> "Impressions":
        """Impressions where ``mask`` is true; user rows and histories are shared, not copied."""
        keep = np.flatnonzero(mask)
        sizes = np.diff(self.cand_ptr)[keep]
        cand_ptr = np.zeros(len(keep) + 1, dtype=np.int64)
        cand_ptr[1:] = np.cumsum(sizes)
        rows = np.repeat(self.cand_ptr[keep], sizes) + (np.arange(cand_ptr[-1]) - np.repeat(cand_ptr[:-1], sizes))
        return Impressions(self.imp_id[keep], self.imp_user[keep], self.imp_time[keep], cand_ptr,
                           self.cand_idx[rows], self.cand_label[rows],
                           self.user_ids, self.hist_ptr, self.hist_idx)

    def split_by_day(self, last_days: int = 1) -> tuple["Impressions", "Impressions"]:
        """(earlier days, final ``last_days`` days), split on calendar day of the impression."""
        day = self.imp_time // 86_400
        cutoff = day.max() - last_days + 1
        return self.subset(day < cutoff), self.subset(day >= cutoff)

    def save(self, path: Path) -> None:
        np.savez(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> "Impressions":
        with np.load(path, allow_pickle=False) as f:
            return cls(**{k: f[k] for k in f.files})


def parse_behaviors(path: Path, news_index: dict[str, int]) -> Impressions:
    beh = pd.read_csv(path, sep="\t", header=None, quoting=3,
                      names=["imp_id", "user", "time", "history", "impressions"],
                      keep_default_na=False)

    user_key = beh["user"] + "\t" + beh["history"]
    codes, uniques = pd.factorize(user_key)
    first = pd.Series(np.arange(len(beh))).groupby(codes).first().to_numpy()

    hist_lists = [h.split() if h else [] for h in beh["history"].to_numpy()[first]]
    hist_ptr = np.zeros(len(uniques) + 1, dtype=np.int64)
    hist_ptr[1:] = np.cumsum([len(h) for h in hist_lists])
    hist_idx = np.fromiter((news_index[n] for h in hist_lists for n in h),
                           dtype=np.int32, count=hist_ptr[-1])

    tokens = [s.split() for s in beh["impressions"]]
    cand_ptr = np.zeros(len(beh) + 1, dtype=np.int64)
    cand_ptr[1:] = np.cumsum([len(t) for t in tokens])
    flat = [tok for t in tokens for tok in t]
    cand_idx = np.fromiter((news_index[t[:-2]] for t in flat), dtype=np.int32, count=len(flat))
    cand_label = np.fromiter((t[-1] == "1" for t in flat), dtype=np.int8, count=len(flat))

    times = pd.to_datetime(beh["time"], format="%m/%d/%Y %I:%M:%S %p")
    return Impressions(
        imp_id=beh["imp_id"].to_numpy(np.int64),
        imp_user=codes.astype(np.int32),
        imp_time=((times - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy(np.int64),
        cand_ptr=cand_ptr,
        cand_idx=cand_idx,
        cand_label=cand_label,
        user_ids=beh["user"].to_numpy()[first].astype(str),
        hist_ptr=hist_ptr,
        hist_idx=hist_idx,
    )


def news_path(processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / "news.parquet"


def impressions_path(split: str, processed_dir: Path = PROCESSED_DIR) -> Path:
    return processed_dir / f"{split}_impressions.npz"


def load_processed(split: str, processed_dir: Path = PROCESSED_DIR) -> Impressions:
    return Impressions.load(impressions_path(split, processed_dir))


def load_processed_news(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    return pd.read_parquet(news_path(processed_dir))
