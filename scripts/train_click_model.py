"""Select, refit and evaluate the simulator's click model.

1. Selection: fit each model kind on train days 1-5 (Nov 9-13) and score the
   held-out last train day (Nov 14) after every epoch. Each kind keeps the
   epoch with the lowest validation log loss.
2. Choice: the simplest kind whose validation AUC is within --auc-margin of the
   best and whose log loss is within --ll-margin (relative) of the best.
3. Refit the chosen kind on all of train for its chosen number of epochs (same
   learning-rate schedule as in selection) and save it.
4. Evaluate once on test, plus how much click rates vary across articles and
   users (a simulator where every article has the same CTR gives bandits
   nothing to learn).

    python -m scripts.train_click_model
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from src.data.mind import load_processed, load_processed_news
from src.evaluation.metrics import calibration, global_metrics, impression_metrics
from src.simulator import click_model as cm
from src.simulator.encode import MODEL_NAME, load_embeddings
from src.simulator.features import ArticleTable, FeatureStore, UserTable
from src.simulator.simulator import ClickSimulator, user_table_path


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def score(probs: np.ndarray, imps, ranking: bool = False) -> dict:
    out = {**global_metrics(probs, imps.cand_label), "ece": calibration(probs, imps.cand_label)["ece"]}
    if ranking:
        out.update(impression_metrics(probs, imps.cand_label, imps.cand_ptr))
    return out


def select(kinds, store, fit, val, args) -> dict:
    users, arts, _ = cm.candidate_tensors(val, store.device)
    results = {}
    for kind in kinds:
        print(f"=== selection: {kind}", flush=True)
        torch.manual_seed(args.seed)
        model = cm.build(kind, store)
        best = {"log_loss": float("inf")}

        def on_epoch_end(epoch: int) -> dict:
            m = score(cm.predict(model, store, users, arts), val)
            if m["log_loss"] < best["log_loss"]:
                best.update(m, epoch=epoch, state=cm.snapshot(model))
            return {f"val_{k}": round(v, 5) for k, v in m.items() if k != "n"}

        history = cm.train(model, store, fit, epochs=args.max_epochs, batch_size=args.batch_size,
                           lr=args.lr, seed=args.seed, on_epoch_end=on_epoch_end)
        model.load_state_dict(best.pop("state"))
        ranking = impression_metrics(cm.predict(model, store, users, arts), val.cand_label, val.cand_ptr)
        results[kind] = {"best_epoch": best.pop("epoch"), "val": {**best, **ranking},
                         "config": cm.config(model), "history": history}
        del model
        gc.collect()
    return results


def choose(results: dict, auc_margin: float, ll_margin: float) -> str:
    best_auc = max(r["val"]["auc"] for r in results.values())
    best_ll = min(r["val"]["log_loss"] for r in results.values())
    for kind in cm.KINDS:  # simplest first
        if kind in results:
            v = results[kind]["val"]
            if v["auc"] >= best_auc - auc_margin and v["log_loss"] <= best_ll * (1 + ll_margin):
                return kind
    return min(results, key=lambda k: results[k]["val"]["log_loss"])


def spread(sim: ClickSimulator, n_users: int = 5000, n_articles: int = 100, seed: int = 0) -> dict:
    """How different are click rates across articles, and across users for the same article?"""
    rng = np.random.default_rng(seed)
    p = sim.prob_matrix(sim.sample_users(n_users, rng), sim.sample_articles(n_articles, rng))
    article_ctr = p.mean(0)
    pop_best = int(article_ctr.argmax())
    q = lambda x: {f"p{k}": float(np.percentile(x, k)) for k in (5, 25, 50, 75, 95)}
    return {
        "n_users": n_users, "n_articles": n_articles,
        "pair_prob_quantiles": q(p),
        "article_ctr_quantiles": q(article_ctr),
        "best_over_median_article_ctr": float(article_ctr.max() / np.median(article_ctr)),
        "share_users_whose_best_article_is_population_best": float((p.argmax(1) == pop_best).mean()),
        "mean_within_article_std_across_users": float(p.std(0).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kinds", nargs="+", default=list(cm.KINDS), choices=cm.KINDS)
    p.add_argument("--max-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--auc-margin", type=float, default=0.005)
    p.add_argument("--ll-margin", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=default_device())
    p.add_argument("--out-dir", type=Path, default=Path("results/click_model"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    articles = ArticleTable.from_news(load_processed_news(), load_embeddings())
    train = load_processed("train")
    train_articles = np.unique(train.cand_idx)
    store = FeatureStore(UserTable.from_impressions(train, articles), articles, args.device)

    fit, val = train.split_by_day(last_days=1)
    print(f"fit {fit.n_impressions:,} impressions / val {val.n_impressions:,}", flush=True)
    results = select(args.kinds, store, fit, val, args)
    kind = choose(results, args.auc_margin, args.ll_margin)
    epochs = results[kind]["best_epoch"]
    selection = {"chosen": kind, "epochs": epochs, "rule": {"auc_margin": args.auc_margin,
                 "ll_margin": args.ll_margin}, "models": results}
    (args.out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    print("val summary:", {k: {m: round(r["val"][m], 4) for m in ("auc", "log_loss", "group_auc", "ece")}
                           | {"epoch": r["best_epoch"]} for k, r in results.items()}, flush=True)
    print(f"chosen: {kind}, {epochs} epochs", flush=True)
    del fit, val
    gc.collect()

    print("=== refit on full train", flush=True)
    torch.manual_seed(args.seed)
    model = cm.build(kind, store)
    refit_history = cm.train(model, store, train, epochs=epochs, schedule_epochs=args.max_epochs,
                             batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    cm.save(model, encoder=MODEL_NAME, epochs=epochs, schedule_epochs=args.max_epochs,
            batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    del store, train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=== test", flush=True)
    test = load_processed("test")
    test_users = UserTable.from_impressions(test, articles)
    np.savez(user_table_path("test"), **test_users.__dict__)
    store = FeatureStore(test_users, articles, args.device)
    users, arts, _ = cm.candidate_tensors(test, store.device)
    probs = cm.predict(model, store, users, arts)
    np.save(args.out_dir / "test_probs.npy", probs.astype(np.float32))
    cos = torch.cat([store.cosine(users[s:s + 65_536], arts[s:s + 65_536])
                     for s in range(0, len(users), 65_536)]).cpu().numpy()

    unseen = ~np.isin(test.cand_idx, train_articles)
    labels = test.cand_label
    sim = ClickSimulator(model, store, np.flatnonzero(test_users.hist_len >= 1), np.unique(test.cand_idx))
    metrics = {
        "model": kind, "epochs": epochs, "encoder": MODEL_NAME,
        "test": score(probs, test, ranking=True),
        "test_calibration_bins": calibration(probs, labels)["bins"],
        "test_unseen_articles": {**global_metrics(probs[unseen], labels[unseen]),
                                 "share_of_candidates": float(unseen.mean())},
        "test_seen_articles": global_metrics(probs[~unseen], labels[~unseen]),
        "test_baseline_cosine_only": {"auc": global_metrics(1 / (1 + np.exp(-cos)), labels)["auc"],
                                      **impression_metrics(cos, labels, test.cand_ptr)},
        "simulator_spread": spread(sim),
        "refit_history": refit_history,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    t = metrics["test"]
    print("test:", {k: round(t[k], 4) for k in ("auc", "group_auc", "mrr", "ndcg@5", "ndcg@10",
                                                "log_loss", "ece", "mean_pred", "observed_ctr")})
    print("test unseen-article AUC", round(metrics["test_unseen_articles"]["auc"], 4),
          "share", round(metrics["test_unseen_articles"]["share_of_candidates"], 3))
    print("cosine-only group AUC", round(metrics["test_baseline_cosine_only"]["group_auc"], 4))
    print("spread:", json.dumps(metrics["simulator_spread"], indent=2))


if __name__ == "__main__":
    main()
