"""Click models P(click | user, article) over the pair features in features.py.

Three model families, from simplest to richest:

    logistic   sigmoid(w . x + b)
    bilinear   logistic + (P u) . (Q a), a rank-r user x article interaction
    mlp        one hidden ReLU layer over x

All standardize x with statistics fit on a training sample.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from src.data.mind import Impressions
from src.simulator.features import FeatureStore

MODEL_PATH = Path("models/click_model.pt")
KINDS = ("logistic", "bilinear", "mlp")


class ClickModel(nn.Module):
    def __init__(self, n_features: int, dim: int):
        super().__init__()
        self.n_features, self.dim = n_features, dim
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        # Post-hoc Platt calibration of the logit, fit on held-out data; identity by default.
        self.register_buffer("calib_scale", torch.ones(()))
        self.register_buffer("calib_bias", torch.zeros(()))

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def calibrated_logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.calib_scale * self(x) + self.calib_bias

    def set_base_rate(self, ctr: float) -> None:
        """Start the output bias at the base rate so early steps are spent on the weights."""
        with torch.no_grad():
            self.output_bias().fill_(math.log(ctr / (1 - ctr)))


class LogisticClickModel(ClickModel):
    def __init__(self, n_features: int, dim: int):
        super().__init__(n_features, dim)
        self.linear = nn.Linear(n_features, 1)

    def output_bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.standardize(x)).squeeze(-1)


class BilinearClickModel(LogisticClickModel):
    def __init__(self, n_features: int, dim: int, rank: int = 64):
        super().__init__(n_features, dim)
        self.rank = rank
        self.p = nn.Linear(dim, rank, bias=False)
        self.q = nn.Linear(dim, rank, bias=False)
        nn.init.normal_(self.p.weight, std=0.01)
        nn.init.normal_(self.q.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature layout is [u*a, a, u, scalars]; raw unit vectors feed the bilinear term.
        a, u = x[:, self.dim:2 * self.dim], x[:, 2 * self.dim:3 * self.dim]
        return super().forward(x) + (self.p(u) * self.q(a)).sum(-1) * math.sqrt(self.dim)


class MLPClickModel(ClickModel):
    def __init__(self, n_features: int, dim: int, hidden: int = 256):
        super().__init__(n_features, dim)
        self.hidden = hidden
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def output_bias(self):
        return self.net[-1].bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.standardize(x)).squeeze(-1)


def build(kind: str, store: FeatureStore) -> ClickModel:
    cls = {"logistic": LogisticClickModel, "bilinear": BilinearClickModel, "mlp": MLPClickModel}[kind]
    return cls(store.n_features, store.dim).to(store.device)


def candidate_tensors(imps: Impressions, device: torch.device):
    users = torch.from_numpy(imps.imp_user[imps.cand_imp()]).to(device)
    articles = torch.from_numpy(imps.cand_idx).to(device)
    labels = torch.from_numpy(imps.cand_label).to(device)
    return users, articles, labels


@torch.no_grad()
def fit_standardizer(model: ClickModel, store: FeatureStore, users, articles,
                     n_sample: int = 1_000_000, seed: int = 0) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randint(len(users), (n_sample,), generator=g).to(users.device)
    s1 = torch.zeros(store.n_features, device=users.device, dtype=torch.float64)
    s2 = torch.zeros_like(s1)
    for chunk in idx.split(65_536):
        x = store.pair_features(users[chunk], articles[chunk]).double()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    mean = s1 / n_sample
    std = (s2 / n_sample - mean**2).clamp(min=0).sqrt()
    model.mean.copy_(mean.float())
    model.std.copy_(torch.where(std > 1e-8, std, torch.ones_like(std)).float())


def train(model: ClickModel, store: FeatureStore, imps: Impressions, epochs: int = 2,
          schedule_epochs: int | None = None, batch_size: int = 8192, lr: float = 1e-3,
          seed: int = 0, log_every: int = 2000,
          on_epoch_end: Callable[[int], dict] | None = None) -> list[dict]:
    """Mini-batch Adam over every candidate in ``imps``.

    The cosine learning-rate schedule spans ``schedule_epochs`` (default ``epochs``),
    so a run stopped early at epoch k follows the same path as a longer run did up
    to epoch k. ``on_epoch_end`` (e.g. a validation pass) is merged into the log.
    """
    device = store.device
    users, articles, labels = candidate_tensors(imps, device)
    n = len(labels)
    fit_standardizer(model, store, users, articles, seed=seed)
    model.set_base_rate(float(imps.cand_label.mean()))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = math.ceil(n / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=(schedule_epochs or epochs) * steps_per_epoch)
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)
    history, t0 = [], time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        running = torch.zeros((), device=device)
        for step, b in enumerate(perm.split(batch_size)):
            loss = loss_fn(model(store.pair_features(users[b], articles[b])), labels[b].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.detach()
            if (step + 1) % log_every == 0:
                print({"epoch": epoch + 1, "step": step + 1, "of": steps_per_epoch,
                       "train_loss": round(running.item() / log_every, 5),
                       "elapsed_s": round(time.time() - t0)}, flush=True)
                running.zero_()
        rec = {"epoch": epoch + 1, "elapsed_s": round(time.time() - t0)}
        if on_epoch_end is not None:
            rec.update(on_epoch_end(epoch + 1))
        history.append(rec)
        print(rec, flush=True)
    return history


@torch.no_grad()
def predict(model: ClickModel, store: FeatureStore, users: torch.Tensor,
            articles: torch.Tensor, batch_size: int = 65_536, logits: bool = False) -> np.ndarray:
    """Calibrated click probabilities (or calibrated logits)."""
    was_training = model.training
    model.eval()
    out = [model.calibrated_logit(store.pair_features(users[s:s + batch_size], articles[s:s + batch_size]))
           for s in range(0, len(users), batch_size)]
    model.train(was_training)
    out = torch.cat(out)
    return (out if logits else torch.sigmoid(out)).cpu().numpy()


def config(model: ClickModel) -> dict:
    kind = {LogisticClickModel: "logistic", BilinearClickModel: "bilinear",
            MLPClickModel: "mlp"}[type(model)]
    cfg = {"kind": kind, "n_features": model.n_features, "dim": model.dim}
    if kind == "bilinear":
        cfg["rank"] = model.rank
    if kind == "mlp":
        cfg["hidden"] = model.hidden
    return cfg


def snapshot(model: ClickModel) -> dict:
    return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})


def save(model: ClickModel, path: Path = MODEL_PATH, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config(model), **meta}, path)


def load(path: Path = MODEL_PATH, device: str = "cpu") -> ClickModel:
    ckpt = torch.load(path, map_location=device)
    cfg = dict(ckpt["config"])
    cls = {"logistic": LogisticClickModel, "bilinear": BilinearClickModel,
           "mlp": MLPClickModel}[cfg.pop("kind")]
    model = cls(**cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    # Checkpoints saved before calibration existed lack the calib_* buffers (identity).
    if unexpected or set(missing) - {"calib_scale", "calib_bias"}:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.eval()
