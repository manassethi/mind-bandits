"""Reduce raw policy representations to 128-dim article feature matrices.

Every reduction is fit on train-split articles only (articles shown or clicked
in train impressions), then applied to all articles, so test-day articles are
never seen by PCA or feature selection.

    dense (Gemma layers, bge-m3, Qwen3 ceiling)  PCA to 128 components (ordered by variance)
    SAE codes                                 the 128 highest-variance features on train
                                              articles, scaled to unit RMS (0 stays 0)
    category                                  one-hot of the 18 MIND categories

The d-dim representation used by a bandit is the first d columns, so one file
serves every d <= 128. Output: dataset/processed/policy_features/<rep>.npy
(N, <=128) float16 and meta.json.

    python -m scripts.build_policy_features
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from src.data.mind import load_processed, load_processed_news
from src.simulator.encode import EMBED_PATH
from scripts.extract_policy_features import GEMMA, RAW_DIR

OUT = Path("dataset/processed/policy_features")
D_MAX = 128
SHORT = {"gemma-3-1b": "g1b", "gemma-3-12b": "g12b"}


def train_articles() -> np.ndarray:
    tr = load_processed("train")
    return np.unique(np.r_[tr.cand_idx, tr.hist_idx])


def pca(X: np.ndarray, fit_rows: np.ndarray, k: int = D_MAX) -> tuple[np.ndarray, dict]:
    """Correlation PCA: each dimension is standardized on train articles first.

    Raw LLM residual streams have a few "massive activation" dimensions (|x| in the
    tens of thousands vs a median std near 10) that would otherwise own the top
    components. Exact eigendecomposition of the float64 correlation matrix.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not np.isfinite(X[fit_rows]).all():
        raise ValueError("non-finite values in representation")
    Xf = torch.from_numpy(np.asarray(X[fit_rows], np.float64)).to(dev)
    mean, std = Xf.mean(0), Xf.std(0).clamp(min=1e-8)
    Z = (Xf - mean) / std
    evals, evecs = torch.linalg.eigh(Z.T @ Z / (len(Z) - 1))
    order = torch.argsort(evals, descending=True)[:k]
    comps, var = evecs[:, order], evals[order]
    out = np.empty((len(X), k), np.float32)
    for s in range(0, len(X), 20_000):
        chunk = torch.from_numpy(np.asarray(X[s:s + 20_000], np.float64)).to(dev)
        out[s:s + 20_000] = (((chunk - mean) / std) @ comps).cpu().numpy()
    return out, {"explained_variance_ratio_cum": (var.cumsum(0) / Z.shape[1]).cpu().numpy()[[15, 31, 63, 127]].round(4).tolist()}


def sae_top(path: Path, fit_rows: np.ndarray, k: int = D_MAX) -> tuple[np.ndarray, dict]:
    M = sp.load_npz(path).tocsr()
    F = M[fit_rows]
    mean = np.asarray(F.mean(0)).ravel()
    sq = np.asarray(F.multiply(F).mean(0)).ravel()
    var = sq - mean**2
    freq = np.asarray((F > 0).mean(0)).ravel()
    top = np.argsort(-var)[:k]
    rms = np.sqrt(sq[top])
    out = (M[:, top].toarray() / rms).astype(np.float32)
    return out, {"feature_ids": top.tolist(), "train_freq": freq[top].round(4).tolist(),
                 "n_features_ever_active_train": int((freq > 0).sum())}


def save(name: str, X: np.ndarray, meta: dict, metas: dict) -> None:
    np.save(OUT / f"{name}.npy", X.astype(np.float16))
    metas[name] = {"dim": X.shape[1], **meta}
    print(name, X.shape, {k: v for k, v in meta.items() if k != "feature_ids" and k != "train_freq"}, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "meta.json"
    metas = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows = train_articles()
    print(f"fitting on {len(rows):,} train articles", flush=True)

    news = load_processed_news()
    cats, codes = np.unique(news["category"], return_inverse=True)
    save("category", np.eye(len(cats), dtype=np.float32)[codes], {"categories": cats.tolist()}, metas)
    save("qwen3", *pca(np.load(EMBED_PATH), rows), metas)
    save("bgem3", *pca(np.load(RAW_DIR / "bge-m3" / "dense.npy", mmap_mode="r"), rows), metas)

    for model, (_, _, sae_layers) in GEMMA.items():
        raw, short = RAW_DIR / model, SHORT[model]
        n_layers = json.loads((raw / "meta.json").read_text())["n_layers"]
        for L in range(n_layers):
            save(f"{short}_L{L}", *pca(np.load(raw / f"dense_layer_{L}.npy", mmap_mode="r"), rows), metas)
        for L in sae_layers:
            save(f"{short}_sae_L{L}", *sae_top(raw / f"sae_layer_{L}.npz", rows), metas)
        meta_path.write_text(json.dumps(metas, indent=1))
    meta_path.write_text(json.dumps(metas, indent=1))


if __name__ == "__main__":
    main()
