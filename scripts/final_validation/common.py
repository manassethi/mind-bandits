"""Shared output conventions and paired uncertainty for follow-up experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(os.environ.get("FINAL_OUT", "results/final_validation_20261121"))
SEED = 20261121


def interval(values) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Intervals require a finite vector of independent pool-level observations")
    half = 0.0 if len(x) < 2 else stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return {"mean": float(x.mean()), "half_width_95": float(half), "n": len(x), "runs": x.tolist()}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez(temp, **arrays)
    temp.replace(path)


def fingerprint(config: dict, inputs: list[Path]) -> dict:
    return {"config": json.loads(json.dumps(config)), "inputs_sha256": {str(p): file_hash(p) for p in inputs}}


def is_complete(path: Path, provenance: dict) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if data.get("provenance") != provenance:
        raise RuntimeError(f"Existing output has different inputs/settings: {path}")
    return True
