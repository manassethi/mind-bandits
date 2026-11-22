"""Automated external-label retrieval on held-out news with label-permutation controls."""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from scripts.final_validation.common import OUT, SEED, fingerprint, is_complete, write_json
from scripts.final_validation.q2_controls import FEATURES, MATCHED, ROOT as Q2_ROOT
from src.data.mind import load_processed, load_processed_news

ROOT = OUT / "semantic_checks"


def examples(activation, categories, rng, matched):
    order = np.argsort(-activation)
    positive = order[activation[order] > 0][:min(100, max(10, len(order) // 10))]
    negative_pool = order[len(order) // 2:]
    if matched:
        positives, negatives = [], []
        for category in np.unique(categories[positive]):
            pos = positive[categories[positive] == category]
            same_category = order[::-1][categories[order[::-1]] == category]
            neg = np.setdiff1d(same_category[:max(1, len(same_category) // 2)], positive)
            count = min(len(pos), len(neg))
            if count:
                positives.extend(pos[:count])
                negatives.extend(rng.choice(neg, count, replace=False))
        positive, negative = np.array(positives, int), np.array(negatives, int)
    else:
        negative = rng.choice(negative_pool, len(positive), replace=False)
    return np.r_[positive, negative], np.r_[np.ones(len(positive)), np.zeros(len(negative))]


def main():
    inputs = [Path(__file__), Q2_ROOT / "feature_manifest.json", Path("dataset/processed/policy_features/meta.json")]
    inputs += [Path(f"results/q3/labels/{sae}.json") for _, _, sae, _ in MATCHED.values()]
    provenance = fingerprint({"seed": SEED, "dimensions": 32, "permutations": 1000,
                              "encoder": "BAAI/bge-m3", "scopes": ["unmatched", "category_matched"]}, inputs)
    if is_complete(ROOT / "summary.json", provenance):
        return
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)
    news = load_processed_news()
    imps = load_processed("test")
    prior = load_processed("train")
    earlier = np.unique(np.r_[prior.cand_idx, prior.hist_idx])
    rows = np.setdiff1d(np.unique(imps.cand_idx), earlier)
    if len(rows) < 200:
        raise RuntimeError("Insufficient strictly unseen news for external-label checks")
    del prior, imps
    categories = news["category"].to_numpy()[rows]
    article_emb = np.asarray(np.load("dataset/processed/policy_raw/bge-m3/dense.npy", mmap_mode="r")[rows], np.float32)
    article_emb /= np.linalg.norm(article_emb, axis=1, keepdims=True).clip(1e-8)
    meta = json.loads(Path("dataset/processed/policy_features/meta.json").read_text())
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    output = {}
    for group, (_, _, sae, _) in MATCHED.items():
        external = json.loads(Path(f"results/q3/labels/{sae}.json").read_text())
        by_id = {int(f["id"]): f for f in external["features"]}
        activations = np.asarray(np.load(FEATURES / f"{group}_sae.npy", mmap_mode="r")[rows], np.float32)
        labels, coordinates, ids = [], [], []
        insufficient_activity = []
        for j, fid in enumerate(meta[sae]["feature_ids"][:32]):
            descriptions = by_id.get(fid, {}).get("explanations", [])
            text = next((str(x).strip() for x in descriptions if x and str(x).strip()), "")
            if text:
                if np.count_nonzero(activations[:, j] > 0) < 20 or np.ptp(activations[:, j]) <= 1e-8:
                    insufficient_activity.append(int(fid))
                    continue
                labels.append(text)
                coordinates.append(j)
                ids.append(fid)
        if len(labels) < 3:
            raise RuntimeError(f"Too few externally labeled features in {group}")
        label_emb = model.encode(labels, batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
        similarity = article_emb @ label_emb.T
        record = {"labeled_count": len(labels), "eligible_features": 32, "feature_ids": ids,
                  "labels": labels, "coordinates": coordinates, "conditions": {},
                  "labeled_features_with_insufficient_holdout_activity": insufficient_activity}
        for matched in (False, True):
            matrix = np.full((len(labels), len(labels)), np.nan)
            counts = []
            for i, coordinate in enumerate(coordinates):
                idx, truth = examples(activations[:, coordinate], categories,
                                      np.random.default_rng(SEED + coordinate), matched)
                counts.append(len(idx))
                if len(idx) < 10 or len(np.unique(truth)) < 2:
                    continue
                for j in range(len(labels)):
                    matrix[i, j] = roc_auc_score(truth, similarity[idx, j])
            valid = np.flatnonzero(np.isfinite(matrix).all(1))
            observed = float(matrix[valid, valid].mean()) if len(valid) else None
            rng = np.random.default_rng(SEED)
            null = np.array([matrix[valid, rng.permutation(len(labels))[valid]].mean() for _ in range(1000)]) if len(valid) else np.array([])
            per_feature = []
            for i, coordinate in enumerate(coordinates):
                rho = stats.spearmanr(activations[:, coordinate], similarity[:, i]).statistic
                per_feature.append({"feature_id": ids[i], "coordinate": coordinate, "label": labels[i],
                    "detection_auc": float(matrix[i, i]) if i in valid else None,
                    "mean_other_label_auc": float(np.delete(matrix[i], i).mean()) if i in valid else None,
                    "all_holdout_articles_spearman": float(rho) if np.isfinite(rho) else None,
                    "example_count": counts[i]})
            record["conditions"]["category_matched" if matched else "unmatched"] = {
                "mean_detection_auc": observed, "permutation_null_mean": float(null.mean()) if len(valid) else None,
                "label_permutation_p": float((1 + (null >= observed).sum()) / (1 + len(null))) if len(valid) else None,
                "null_quantiles_025_975": np.quantile(null, [.025, .975]).tolist() if len(valid) else None,
                "valid_feature_count": len(valid), "features": per_feature,
                "label_auc_matrix": [[float(x) if np.isfinite(x) else None for x in row] for row in matrix]}
        output[group] = record
        print(group, {key: v["mean_detection_auc"] for key, v in record["conditions"].items()}, flush=True)
    write_json(ROOT / "summary.json", {"provenance": provenance, "strict_holdout_article_count": len(rows),
        "scope": "Automated agreement between external label embeddings and high-activation held-out headlines. These outcomes are not human judgments of semantic correctness.",
        "results": output})


if __name__ == "__main__":
    main()
