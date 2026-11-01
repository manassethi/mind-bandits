"""Post-hoc calibration of the saved click model's logit z.

Two methods:

    anchor   (default) p = sigmoid(z + b), with b chosen so the mean predicted
             CTR over the test day's logged (user, candidate) pairs equals the
             average CTR over all train days. Uses no test labels.
    val-fit  intercept-only and Platt (sigmoid(a * z + b)) fit by log loss on the
             last train day (Nov 14). Rejected for this project: Nov 14 is a
             low-CTR day (3.86% vs 4.05% train average), so it pulls the model
             toward one unusual day.

Both are monotone, so rankings (AUC, group AUC, MRR, nDCG) do not change. The
chosen (a, b) is written into the checkpoint, which ClickSimulator then uses.

    python -m scripts.calibrate_click_model [--method anchor|val-fit]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.mind import load_processed, load_processed_news
from src.evaluation.metrics import calibration, global_metrics, impression_metrics
from src.simulator import click_model as cm
from src.simulator.encode import load_embeddings
from src.simulator.features import ArticleTable, FeatureStore, UserTable
from src.simulator.simulator import ClickSimulator, user_table_path
from scripts.train_click_model import default_device, spread


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def anchor_bias(z: np.ndarray, target: float, tol: float = 1e-10) -> float:
    """Bias b with mean(sigmoid(z + b)) == target (bisection; the mean is increasing in b)."""
    lo, hi = -5.0, 5.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sigmoid(z + mid).mean() < target else (lo, mid)
    return (lo + hi) / 2


def fit_calibration(z: np.ndarray, y: np.ndarray, platt: bool) -> tuple[float, float, float]:
    """(scale, bias, log loss) minimizing binary cross-entropy of sigmoid(scale * z + bias)."""
    zt, yt = torch.from_numpy(z).double(), torch.from_numpy(y).double()
    scale = torch.ones((), dtype=torch.float64, requires_grad=platt)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    params = [scale, bias] if platt else [bias]
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=200, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(scale * zt + bias, yt)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        loss = loss_fn(scale * zt + bias, yt).item()
    return float(scale.detach()), float(bias.detach()), loss


def val_fit(model, articles, device) -> tuple[str, float, float, dict]:
    train = load_processed("train")
    _, val = train.split_by_day(last_days=1)
    store = FeatureStore(UserTable.from_impressions(train, articles), articles, device)
    users, arts, _ = cm.candidate_tensors(val, store.device)
    z = cm.predict(model, store, users, arts, logits=True).astype(np.float64)
    y = val.cand_label.astype(np.float64)
    fits = {name: dict(zip(("scale", "bias", "val_log_loss"), fit_calibration(z, y, platt)))
            for name, platt in (("intercept", False), ("platt", True))}
    chosen = min(fits, key=lambda k: fits[k]["val_log_loss"])
    return chosen, fits[chosen]["scale"], fits[chosen]["bias"], {"fits": fits}


def summary(probs: np.ndarray, imps) -> dict:
    return {**global_metrics(probs, imps.cand_label), "ece": calibration(probs, imps.cand_label)["ece"],
            **impression_metrics(probs, imps.cand_label, imps.cand_ptr)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("anchor", "val-fit"), default="anchor")
    p.add_argument("--model", type=Path, default=cm.MODEL_PATH)
    p.add_argument("--device", default=default_device())
    p.add_argument("--out", type=Path, default=Path("results/click_model/calibration.json"))
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    if "calibration" in ckpt:
        raise SystemExit(f"{args.model} is already calibrated: {ckpt['calibration']}")
    model = cm.load(args.model, device=args.device)
    articles = ArticleTable.from_news(load_processed_news(), load_embeddings())

    test = load_processed("test")
    with np.load(user_table_path("test")) as f:
        test_users = UserTable(**{k: f[k] for k in f.files})
    store = FeatureStore(test_users, articles, args.device)
    users, arts, _ = cm.candidate_tensors(test, store.device)
    z_test = cm.predict(model, store, users, arts, logits=True).astype(np.float64)

    if args.method == "anchor":
        target = float(load_processed("train").cand_label.mean())
        a, b = 1.0, anchor_bias(z_test, target)
        method, extra = "anchor", {"target_ctr": target, "anchored_on": "test-day logged pairs, no labels"}
    else:
        method, a, b, extra = val_fit(model, articles, args.device)
    print(f"{method}: scale={a:.5f} bias={b:.5f}", extra, flush=True)

    before = summary(sigmoid(z_test), test)
    after_probs = sigmoid(a * z_test + b)
    after = summary(after_probs, test)

    with torch.no_grad():
        model.calib_scale.fill_(a)
        model.calib_bias.fill_(b)
    sim = ClickSimulator(model, store, np.flatnonzero(test_users.hist_len >= 1), np.unique(test.cand_idx))
    result = {"method": method, "scale": a, "bias": b, **extra,
              "test_before": before, "test_after": after,
              "test_calibration_bins_after": calibration(after_probs, test.cand_label)["bins"],
              "simulator_spread_after": spread(sim)}
    args.out.write_text(json.dumps(result, indent=2))
    np.save(args.out.parent / "test_probs.npy", after_probs.astype(np.float32))

    ckpt["state_dict"] = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt["calibration"] = {"method": method, "scale": a, "bias": b, **extra}
    torch.save(ckpt, args.model)

    keys = ("auc", "group_auc", "mrr", "ndcg@10", "log_loss", "ece", "mean_pred", "observed_ctr")
    print("test before:", {k: round(before[k], 4) for k in keys})
    print("test after: ", {k: round(after[k], 4) for k in keys})
    print("spread after:", json.dumps(result["simulator_spread_after"], indent=2))


if __name__ == "__main__":
    main()
