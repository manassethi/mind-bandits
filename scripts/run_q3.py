"""Q3: read and test the SAE-feature bandit policy.

For each SAE representation (main: Gemma-3-12B layer 41; replications: 12B
layer 31, 1B layer 17), with LinUCB at d = 32 on the Q2 main environment:

1. Importance: run on a 30-run train-day environment, then rank features by mean
   |attribution| = |theta_a,j * a_j + theta_ua,j * u_j * a_j| over recorded
   contexts. The ranking never uses test-day results.
2. Weights: run on the test-day environment (30 runs) and report each feature's
   weights with 95% CIs, its Neuronpedia label, split-half stability, and
   agreement with the train ranking.
3. Post-hoc ablation (faithfulness of the explanation): re-score the final test
   policies on recorded contexts with the top-k train-ranked features zeroed vs
   k random features zeroed; measure how often the choice changes and how many
   expected clicks are lost.
4. Retrain ablation: rerun the bandit with those features removed from the start;
   paired regret increase vs the full policy.
5. User groups: effective concept weights for readers of different categories.
6. Hand-check sheet: label plus the test-pool headlines that activate each top
   feature most, for a human to mark whether the label fits.

    python -m scripts.run_q3
Writes results/q3/<rep>/{weights.csv, handcheck.csv, result.json} and results/q3/summary.md.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.run_q2 import FEATS, OUT as Q2_OUT, tuned_family

OUT = Path("results/q3")
REPS = ["g12b_sae_L41", "g12b_sae_L31", "g1b_sae_L17"]
D = 32
KS = (1, 2, 4, 8, 16)
RETRAIN_KS = (1, 4, 8)
N_RANDOM_POSTHOC, N_RANDOM_RETRAIN = 50, 5
N_CONTEXTS = 500


def load_env(split: str) -> dict:
    # The importance ranking uses its own 30-run train-day environment (Q2's train
    # environment has 10 runs, too few for a stable ranking); test is Q2's main one.
    path = OUT / "envs" / "train30_life2000.npz" if split == "train" else Q2_OUT / "envs" / f"{split}_life2000.npz"
    with np.load(path) as f:
        return {k: (f[k] if f[k].ndim else f[k].item()) for k in f.files}


def features(rep: str, split: str, env: dict, drop: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    from src.bandits.contextual import article_features, user_features
    from src.data.mind import load_processed

    X = article_features(np.load(FEATS / f"{rep}.npy"), D)
    X[:, list(drop)] = 0.0  # removing a feature removes it from both a and u * a
    imps = load_processed(split)
    return X[env["articles"]], user_features(X, env["users"], imps.hist_ptr, imps.hist_idx)


def policy_spec(rep: str) -> str:
    return json.loads((Q2_OUT / "tuned.json").read_text())[tuned_family(rep)]["linucb"]


def run_task(task: dict) -> tuple[dict, dict]:
    import torch
    from src.bandits.contextual import run

    torch.set_num_threads(1)
    env = load_env(task["split"])
    art_x, user_x = features(task["rep"], task["split"], env, tuple(task["drop"]))
    res = run(env, art_x, user_x, policy_spec(task["rep"]), "cpu", n_contexts=task.get("contexts", 0))
    return task, res


def run_all(tasks: list[dict], workers: int) -> list[tuple[dict, dict]]:
    with mp.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap(run_task, tasks))


def attributions(theta: np.ndarray, art_x: np.ndarray, user_x: np.ndarray, users: np.ndarray,
                 live: np.ndarray) -> np.ndarray:
    """Decision importance of each feature -> (d,).

    The policy picks the arg-max article, so a contribution shared by every live
    article cannot change a choice. Importance is the mean absolute deviation of
    the feature's contribution across the live articles of a context, averaged
    over runs and contexts.
    """
    S = theta.shape[0]
    a = art_x[np.arange(S)[:, None, None], live]                 # (S, n, K, d)
    u = user_x[np.arange(S)[:, None], users][:, :, None, :]      # (S, n, 1, d)
    ta, tu = theta[:, 1:1 + D], theta[:, 1 + D:]
    contrib = ta[:, None, None, :] * a + tu[:, None, None, :] * u * a
    return np.abs(contrib - contrib.mean(2, keepdims=True)).mean((0, 1, 2))


def posthoc(theta, art_x, user_x, env, users, live, drop) -> tuple[float, float]:
    """(choice-change rate, expected clicks lost per 1,000 decisions) when ``drop`` is zeroed."""
    S = theta.shape[0]
    a = art_x[np.arange(S)[:, None, None], live]
    u = user_x[np.arange(S)[:, None], users][:, :, None, :]
    phi = np.concatenate([np.ones(a.shape[:-1] + (1,)), a, u * a], -1)
    base = np.einsum("snkd,sd->snk", phi, theta)
    keep = np.ones(theta.shape[1])
    for j in drop:
        keep[1 + j] = keep[1 + D + j] = 0
    abl = np.einsum("snkd,sd->snk", phi, theta * keep)
    c0, c1 = base.argmax(-1), abl.argmax(-1)
    P = env["P"]
    s_idx = np.arange(S)[:, None]
    p0 = P[s_idx, users, np.take_along_axis(live, c0[..., None], -1)[..., 0]]
    p1 = P[s_idx, users, np.take_along_axis(live, c1[..., None], -1)[..., 0]]
    return float((c0 != c1).mean()), float((p0 - p1).mean() * 1000)


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def labels(rep: str) -> list[str]:
    feats = json.loads((OUT / "labels" / f"{rep}.json").read_text())["features"]
    return [(f["explanations"][0] if f["explanations"] else "") for f in feats]


def analyze(rep: str, workers: int) -> dict:
    from src.data.mind import load_processed, load_processed_news

    out_dir = OUT / rep
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((FEATS / "meta.json").read_text())[rep]
    fids, labs = meta["feature_ids"][:D], labels(rep)[:D]
    env_tr, env_te = load_env("train"), load_env("test")

    # 1-2. Full policies on train (importance) and test (weights, post-hoc ablation).
    (_, tr), (_, te) = run_all([{"rep": rep, "split": "train", "drop": (), "contexts": N_CONTEXTS},
                                {"rep": rep, "split": "test", "drop": (), "contexts": N_CONTEXTS}], 2)
    ax_tr, ux_tr = features(rep, "train", env_tr)
    ax_te, ux_te = features(rep, "test", env_te)
    imp_tr = attributions(tr["theta"], ax_tr, ux_tr, tr["ctx_users"], tr["ctx_live"])
    imp_te = attributions(te["theta"], ax_te, ux_te, te["ctx_users"], te["ctx_live"])
    order = np.argsort(-imp_tr)
    half = len(te["theta"]) // 2
    imp_a = attributions(te["theta"][:half], ax_te[:half], ux_te[:half], te["ctx_users"][:half], te["ctx_live"][:half])
    imp_b = attributions(te["theta"][half:], ax_te[half:], ux_te[half:], te["ctx_users"][half:], te["ctx_live"][half:])

    th = te["theta"]
    rows = []
    for rank, j in enumerate(order):
        ma, ha = ci(th[:, 1 + j])
        mu, hu = ci(th[:, 1 + D + j])
        rows.append({"rank_train": rank + 1, "col": int(j), "feature_id": fids[j], "label": labs[j],
                     "importance_train": float(imp_tr[j]), "importance_test": float(imp_te[j]),
                     "w_article": ma, "w_article_ci": ha, "w_personal": mu, "w_personal_ci": hu})
    with open(out_dir / "weights.csv", "w") as f:
        f.write("rank_train,feature_id,label,importance_train,importance_test,w_article,w_article_ci,"
                "w_personal,w_personal_ci\n")
        for r in rows:
            f.write(f"{r['rank_train']},{r['feature_id']},\"{r['label']}\",{r['importance_train']:.5f},"
                    f"{r['importance_test']:.5f},{r['w_article']:.5f},{r['w_article_ci']:.5f},"
                    f"{r['w_personal']:.5f},{r['w_personal_ci']:.5f}\n")

    # 3. Post-hoc ablation: top-k (train ranking) vs random k.
    rng = np.random.default_rng(0)
    post = {}
    for k in KS:
        top = posthoc(th, ax_te, ux_te, env_te, te["ctx_users"], te["ctx_live"], order[:k])
        rand = [posthoc(th, ax_te, ux_te, env_te, te["ctx_users"], te["ctx_live"],
                        rng.choice(D, k, replace=False)) for _ in range(N_RANDOM_POSTHOC)]
        rc, rl = np.array(rand).T
        post[k] = {"top_change": top[0], "top_clicks_lost_per_1k": top[1],
                   "random_change": ci(rc), "random_clicks_lost_per_1k": ci(rl),
                   "top_change_percentile_vs_random": float((rc < top[0]).mean())}

    # 4. Retrain ablation: remove features before learning.
    tasks = [{"rep": rep, "split": "test", "drop": tuple(int(x) for x in order[:k]), "tag": f"top{k}"} for k in RETRAIN_KS]
    for k in RETRAIN_KS:
        for i in range(N_RANDOM_RETRAIN):
            tasks.append({"rep": rep, "split": "test", "drop": tuple(int(x) for x in rng.choice(D, k, replace=False)),
                          "tag": f"rand{k}_{i}"})
    base_reg = te["regret"][:, -1]
    retrain = {}
    for task, res in run_all(tasks, workers):
        retrain[task["tag"]] = res["regret"][:, -1] - base_reg
    retrain_sum = {}
    for k in RETRAIN_KS:
        rand = np.mean([retrain[f"rand{k}_{i}"] for i in range(N_RANDOM_RETRAIN)], 0)
        retrain_sum[k] = {"top_regret_increase": ci(retrain[f"top{k}"]), "random_regret_increase": ci(rand),
                          "top_minus_random": ci(retrain[f"top{k}"] - rand)}

    # 5. Reader groups: effective weight w_g = theta_a + theta_ua * mean u over the group.
    news = load_processed_news()
    imps = load_processed("test")
    cat = news["category"].to_numpy()
    flat = env_te["users"].ravel()
    dom = []
    for urow in flat:
        h = imps.hist_idx[imps.hist_ptr[urow]:imps.hist_ptr[urow + 1]]
        vals, cnt = np.unique(cat[h], return_counts=True)
        dom.append(vals[cnt.argmax()])
    dom = np.array(dom)
    U = ux_te.reshape(-1, D)
    ta, tu = th[:, 1:1 + D].mean(0), th[:, 1 + D:].mean(0)
    w_all = ta + tu * U.mean(0)
    groups = {"_everyone": {"share": 1.0, "top": [{"feature_id": fids[j], "label": labs[j], "weight": float(w_all[j])}
                                                  for j in np.argsort(-w_all)[:5]]}}
    for g, n in zip(*np.unique(dom, return_counts=True)):
        if n < 0.05 * len(dom):
            continue
        lift = tu * (U[dom == g].mean(0) - U.mean(0))  # how this group's weights differ from everyone's
        top = np.argsort(-lift)[:5]
        groups[str(g)] = {"share": float(n / len(dom)),
                          "top": [{"feature_id": fids[j], "label": labs[j], "lift": float(lift[j])} for j in top]}

    # 6. Hand-check sheet: top features' labels with their most-activating test-pool headlines.
    pool = np.unique(env_te["articles"])
    X = np.load(FEATS / f"{rep}.npy")[:, :D].astype(np.float32)
    with open(out_dir / "handcheck.csv", "w") as f:
        f.write("rank_train,feature_id,neuronpedia_label,top_headlines,label_matches_headlines(y/n/partial)\n")
        for r in rows[:10]:
            best = pool[np.argsort(-X[pool, r["col"]])[:5]]
            heads = " || ".join(news["title"].iloc[best].str.replace('"', "'"))
            f.write(f"{r['rank_train']},{r['feature_id']},\"{r['label']}\",\"{heads}\",\n")

    result = {
        "rep": rep, "policy": policy_spec(rep), "d": D,
        "test_regret": ci(base_reg),
        "top10": rows[:10],
        "top10_labeled": sum(bool(r["label"]) for r in rows[:10]),
        "rank_agreement_train_vs_test_spearman": float(stats.spearmanr(imp_tr, imp_te)[0]),
        "split_half_spearman": float(stats.spearmanr(imp_a, imp_b)[0]),
        "posthoc": post, "retrain": retrain_sum, "groups": groups,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reps", nargs="+", default=REPS)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = p.parse_args()
    results = {rep: analyze(rep, args.workers) for rep in args.reps}
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({r: {k: v for k, v in x.items() if k in ("test_regret", "top10_labeled",
                                                              "rank_agreement_train_vs_test_spearman",
                                                              "split_half_spearman")}
                      for r, x in results.items()}, indent=1))


if __name__ == "__main__":
    main()
