"""Refit every click-model family with disjoint selection and calibration days."""

import argparse
import gc
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from scripts.calibrate_click_model import fit_calibration, sigmoid
from scripts.final_validation.common import OUT, SEED, fingerprint, is_complete, save_npz, write_json
from scripts.train_click_model import choose, default_device, score, select
from src.data.mind import load_processed, load_processed_news
from src.simulator import click_model as cm
from src.simulator.encode import EMBED_PATH, MODEL_NAME, load_embeddings
from src.simulator.features import ArticleTable, FeatureStore, UserTable


def day_names(times):
    return [str(np.datetime64(int(d), "D")) for d in np.unique(times // 86400)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=default_device())
    p.add_argument("--max-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()
    dest = OUT / "temporal_simulator"
    provenance = fingerprint({"seed": args.seed, "epochs": args.max_epochs, "batch_size": args.batch_size,
                              "protocol": "fit Nov9-12, select Nov13, refit Nov9-13, calibrate Nov14, evaluate Nov15",
                              "primary_calibrator": "intercept"},
                             [Path(__file__), Path("src/simulator/click_model.py"),
                              Path("dataset/processed/train_impressions.npz"),
                              Path("dataset/processed/test_impressions.npz"), EMBED_PATH])
    if is_complete(dest / "summary.json", provenance):
        return
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    train = load_processed("train")
    days = train.imp_time // 86400
    unique_days = np.unique(days)
    if len(unique_days) != 6:
        raise ValueError(f"Expected six training dates, found {unique_days}")
    select_day, calib_day = unique_days[-2:]
    fit, validation = train.subset(days < select_day), train.subset(days == select_day)
    refit, calibration = train.subset(days < calib_day), train.subset(days == calib_day)
    assert fit.imp_time.max() < validation.imp_time.min()
    assert refit.imp_time.max() < calibration.imp_time.min()
    articles = ArticleTable.from_news(load_processed_news(), load_embeddings())
    users = UserTable.from_impressions(train, articles)
    store = FeatureStore(users, articles, args.device)
    selection_args = Namespace(seed=args.seed, max_epochs=args.max_epochs, batch_size=args.batch_size, lr=1e-3)
    selection_path = dest / "selection.json"
    if selection_path.exists():
        selection_record = json.loads(selection_path.read_text())
        if selection_record["provenance"] != provenance:
            raise RuntimeError("Selection checkpoint protocol differs from this run")
        selection = selection_record["models"]
    else:
        selection = select(cm.KINDS, store, fit, validation, selection_args)
        write_json(selection_path, {"provenance": provenance, "models": selection})
    primary = choose(selection, 0.005, 0.005)
    protocol = {"fit_dates": day_names(fit.imp_time), "selection_dates": day_names(validation.imp_time),
                "refit_dates": day_names(refit.imp_time), "calibration_dates": day_names(calibration.imp_time),
                "primary_family": primary, "primary_calibrator": "intercept",
                "evaluation_status": "Previously inspected development day; locked retrospective validation"}
    write_json(dest / "protocol.json", {"provenance": provenance, **protocol})
    # A strict stream excludes any article in an earlier candidate or history.
    earlier_ids = np.unique(np.r_[train.cand_idx, train.hist_idx])
    save_npz(dest / "earlier_article_rows.npz", article_rows=earlier_ids)
    del fit, validation, train
    gc.collect()
    calib_u, calib_a, _ = cm.candidate_tensors(calibration, store.device)
    for kind in cm.KINDS:
        family_dir = dest / kind
        family_dir.mkdir(exist_ok=True)
        raw_path = family_dir / "uncalibrated.pt"
        if not raw_path.exists():
            torch.manual_seed(args.seed)
            model = cm.build(kind, store)
            history = cm.train(model, store, refit, epochs=selection[kind]["best_epoch"],
                               schedule_epochs=args.max_epochs, batch_size=args.batch_size, seed=args.seed)
            cm.save(model, raw_path, encoder=MODEL_NAME, seed=args.seed, protocol=protocol,
                    epochs=selection[kind]["best_epoch"], refit_history=history)
        else:
            model = cm.load(raw_path, args.device)
        z = cm.predict(model, store, calib_u, calib_a, logits=True).astype(np.float64)
        _, intercept, _ = fit_calibration(z, calibration.cand_label, platt=False)
        scale, bias, _ = fit_calibration(z, calibration.cand_label, platt=True)
        if scale <= 0:
            raise ValueError("Fitted Platt scale would reverse rankings")
        record = {"identity": [1.0, 0.0], "intercept": [1.0, intercept], "platt": [scale, bias]}
        write_json(family_dir / "calibrators.json", record)
        for method, (a, b) in record.items():
            with torch.no_grad():
                model.calib_scale.fill_(a)
                model.calib_bias.fill_(b)
            cm.save(model, family_dir / f"{method}.pt", encoder=MODEL_NAME, seed=args.seed,
                    protocol=protocol, calibration={"method": method, "scale": a, "bias": b})
        del model, z
        gc.collect()
    del store, users, refit, calibration, calib_u, calib_a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test = load_processed("test")
    assert str(np.datetime64(int((test.imp_time // 86400).min()), "D")) == "2019-11-15"
    user_table = UserTable.from_impressions(test, articles)
    store = FeatureStore(user_table, articles, args.device)
    test_u, test_a, _ = cm.candidate_tensors(test, store.device)
    strict = ~np.isin(test.cand_idx, earlier_ids)
    result = {}
    for kind in cm.KINDS:
        family_dir = dest / kind
        model = cm.load(family_dir / "uncalibrated.pt", args.device)
        z = cm.predict(model, store, test_u, test_a, logits=True).astype(np.float64)
        methods = json.loads((family_dir / "calibrators.json").read_text())
        result[kind] = {"selected_epoch": selection[kind]["best_epoch"], "methods": {}}
        for method, (a, b) in methods.items():
            probs = sigmoid(a * z + b).astype(np.float32)
            m = score(probs, test, ranking=True)
            m["strict_article_holdout"] = {"pairs": int(strict.sum()), "articles": int(len(np.unique(test.cand_idx[strict]))),
                                           **score_subset(probs[strict], test.cand_label[strict])}
            result[kind]["methods"][method] = m
            if method == "intercept":
                np.save(family_dir / "test_probs.npy", probs)
            print(kind, method, {k: m[k] for k in ("auc", "group_auc", "log_loss", "ece")}, flush=True)
        del model, z
    write_json(dest / "summary.json", {"provenance": provenance, "protocol": protocol, "results": result})


def score_subset(probs, labels):
    from src.evaluation.metrics import global_metrics, calibration
    return {**global_metrics(probs, labels), "ece": calibration(probs, labels)["ece"]}


if __name__ == "__main__":
    main()
