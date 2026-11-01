"""Fetch Neuronpedia auto-interp labels for the SAE features a representation uses.

For each representation (e.g. g12b_sae_L41) reads its 128 selected feature ids from
dataset/processed/policy_features/meta.json and stores, per feature, every
explanation Neuronpedia has. Output: results/q3/labels/<rep>.json

    python -m scripts.fetch_neuronpedia_labels g12b_sae_L12 g12b_sae_L41 g1b_sae_L17
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FEATS_META = Path("dataset/processed/policy_features/meta.json")
OUT = Path("results/q3/labels")
MODEL = {"g1b": "gemma-3-1b", "g12b": "gemma-3-12b"}


def fetch(url: str, retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(2 ** attempt)
    return None


def labels_for(rep: str, feature_ids: list[int]) -> dict:
    short, _, layer = rep.partition("_sae_L")
    model, source = MODEL[short], f"{layer}-gemmascope-2-res-16k"

    def one(fid: int) -> dict:
        d = fetch(f"https://www.neuronpedia.org/api/feature/{model}/{source}/{fid}")
        if d is None:
            return {"id": fid, "found": False, "explanations": []}
        return {"id": fid, "found": True, "sae_id": (d.get("source") or {}).get("saelensSaeId"),
                "frac_nonzero": d.get("frac_nonzero"),
                "explanations": [e.get("description") for e in d.get("explanations") or []]}

    with ThreadPoolExecutor(8) as pool:
        feats = list(pool.map(one, feature_ids))
    return {"rep": rep, "model": model, "source": source, "features": feats}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("reps", nargs="+")
    args = p.parse_args()
    meta = json.loads(FEATS_META.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for rep in args.reps:
        res = labels_for(rep, meta[rep]["feature_ids"])
        (OUT / f"{rep}.json").write_text(json.dumps(res, indent=1))
        f = res["features"]
        labeled = [x for x in f if x["explanations"]]
        print(f"{rep}: {len(labeled)}/{len(f)} labeled, {sum(x['found'] for x in f)} found, "
              f"sae ids {sorted({x.get('sae_id') for x in f if x['found']})}", flush=True)


if __name__ == "__main__":
    main()
