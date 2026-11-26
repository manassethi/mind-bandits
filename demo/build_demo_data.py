"""Build the small data bundle the Streamlit demo runs on.

The full experiments need about 4 GB of Git LFS files (simulator, impressions,
feature caches). The demo only needs a few cached probability tables and
feature matrices, so this script copies them into demo/data/ once:

    demo/data/headlines.npz   Q4 headline pools: population click rates mu (30 pools of K=5 and K=10)
    demo/data/headlines.json  their titles and categories (copied from results/q4/envs.json)
    demo/data/churn.npz       first N_POOLS pools of the Q2 test environment (20 live articles,
                              2,000-round lifetimes): click probabilities P, article and user
                              features for each representation, exactly as Q2/Q3 build them
    demo/data/churn.json      headlines of those articles, SAE coordinate labels, tuned settings

Nothing here is re-fitted: every number is read from the stored experiment inputs.

Run from the repository root after `git lfs pull` of dataset/processed and results/q2, results/q4:

    python -m demo.build_demo_data
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_q2 import FEATS, OUT as Q2_OUT, tuned_family
from src.bandits.contextual import article_features, user_features
from src.data.mind import load_processed

OUT = Path("demo/data")
N_POOLS = 6   # pools of the 30-pool Q2 test environment kept for the live demo
D = 32        # feature dimension used in Q2/Q3 (category one-hot has 18)

# Representations offered in the demo, keyed by the name shown in the app.
REPS = {
    "category": "category",          # 18-dimensional category one-hot
    "dense": "g12b_L43",             # Gemma-3-12B dense activations, PCA-reduced (Q2 selected layer)
    "sae": "g12b_sae_L41",           # Gemma-3-12B SAE features, layer 41 (the Q3 main representation)
}


def build_headlines() -> None:
    """Q4 pools: real test-day headlines with simulator population click rates."""
    with np.load("results/q4/envs.npz") as f:
        np.savez_compressed(OUT / "headlines.npz", K5_mu=f["K5_mu"], K10_mu=f["K10_mu"])
    shutil.copy("results/q4/envs.json", OUT / "headlines.json")


def build_churn() -> None:
    with np.load(Q2_OUT / "envs" / "test_life2000.npz") as f:
        env = {k: (f[k] if f[k].ndim else f[k].item()) for k in f.files}
    keep = slice(0, N_POOLS)
    users, articles = env["users"][keep], env["articles"][keep]
    imps = load_processed("test")

    arrays = {"P": env["P"][keep].astype(np.float16), "articles": articles}
    for name, rep in REPS.items():
        X = article_features(np.load(FEATS / f"{rep}.npy"), D)   # same scaling as Q2/Q3
        arrays[f"{name}_art"] = X[articles].astype(np.float32)
        arrays[f"{name}_user"] = user_features(X, users, imps.hist_ptr, imps.hist_idx).astype(np.float32)
    np.savez_compressed(OUT / "churn.npz", **arrays)

    news = pd.read_parquet("dataset/processed/news.parquet")
    tuned = json.loads((Q2_OUT / "tuned.json").read_text())
    sae_meta = json.loads((FEATS / "meta.json").read_text())[REPS["sae"]]
    sae_labels = json.loads(Path(f"results/q3/labels/{REPS['sae']}.json").read_text())["features"]
    cat_meta = json.loads((FEATS / "meta.json").read_text())["category"]
    meta = {
        "n_live": int(env["n_live"]), "lifetime": int(env["lifetime"]), "horizon": int(env["horizon"]),
        "seed": int(env["seed"]), "n_pools": N_POOLS,
        "titles": [[news["title"].iloc[r] for r in run] for run in articles],
        "categories": [[news["category"].iloc[r] for r in run] for run in articles],
        "linucb": {name: tuned[tuned_family(rep)]["linucb"] for name, rep in REPS.items()},
        "feature_labels": {
            "category": cat_meta["categories"],
            "dense": [f"PCA component {j + 1}" for j in range(D)],
            "sae": [(f["explanations"][0] if f["explanations"] else f"(unlabeled #{f['id']})")
                    for f in sae_labels[:D]],
        },
        "sae_feature_ids": sae_meta["feature_ids"][:D],
        # Last three clicked headlines of every reader, to show who the policy is choosing for.
        "reader_history": [[[news["title"].iloc[r] for r in imps.history(u)[-3:]] for u in run]
                           for run in users],
    }
    (OUT / "churn.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_headlines()
    build_churn()
    for p in sorted(OUT.iterdir()):
        print(f"{p}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
