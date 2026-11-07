"""Does the LLM know which headlines get clicked? A cleaner test on well-observed articles.

Groups of test-day articles that each had >= 100 impressions (so their real CTR is
reliable): 200 groups of 5 drawn across all categories, and 200 groups of 5 from a
single category (headline judgment beyond topic). The LLM ranks each group once
(temperature 0); the simulator's population CTR ranks the same groups.

    python -m scripts.prior_probe build                         # local: groups + simulator CTR
    python -m scripts.prior_probe run --model google/gemma-3-12b-it   # GPU: LLM rankings
    python -m scripts.prior_probe analyze                       # local: compare to real CTR
"""

import argparse
import json

import numpy as np
from scipy import stats

from scripts.build_q4_envs import OUT
from scripts.train_click_model import default_device

GROUPS, SIZE, MIN_IMPRESSIONS, SEED = 200, 5, 100, 7_000_000
PATH = OUT / "prior_groups.json"


def build(device: str | None = None) -> None:
    from src.data.mind import load_processed, load_processed_news
    from src.simulator.simulator import ClickSimulator

    test, news = load_processed("test"), load_processed_news()
    n = np.bincount(test.cand_idx, minlength=len(news)).astype(float)
    c = np.bincount(test.cand_idx, weights=test.cand_label, minlength=len(news))
    pool = np.flatnonzero(n >= MIN_IMPRESSIONS)
    cats = news["category"].to_numpy()
    rng = np.random.default_rng(SEED)
    groups = [{"type": "mixed", "rows": rng.choice(pool, SIZE, replace=False).tolist()} for _ in range(GROUPS)]
    by_cat = {k: pool[cats[pool] == k] for k in np.unique(cats[pool])}
    big = [k for k, v in by_cat.items() if len(v) >= 3 * SIZE]
    weights = np.array([len(by_cat[k]) for k in big], float)
    for k in rng.choice(big, GROUPS, p=weights / weights.sum()):
        groups.append({"type": "same_category", "category": str(k),
                       "rows": rng.choice(by_cat[k], SIZE, replace=False).tolist()})

    sim = ClickSimulator.load(split="test", device=device or default_device())
    users = sim.sample_users(5000, np.random.default_rng(0))
    rows = np.unique([r for g in groups for r in g["rows"]])
    pop = dict(zip(rows.tolist(), sim.prob_matrix(users, rows).mean(0).tolist()))
    for g in groups:
        g["titles"] = [news["title"].iloc[r] for r in g["rows"]]
        g["real_ctr"] = [float(c[r] / n[r]) for r in g["rows"]]
        g["impressions"] = [int(n[r]) for r in g["rows"]]
        g["sim_ctr"] = [pop[r] for r in g["rows"]]
    PATH.write_text(json.dumps(groups, indent=1))
    print(f"{len(groups)} groups from {len(pool):,} articles; categories used: {sorted(set(g.get('category', '') for g in groups))}")


def run(model: str) -> None:
    from scripts.run_q4_llm import vllm_engine
    from src.agents.llm_agent import parse_ranking, prior_prompt

    groups = json.loads(PATH.read_text())
    generate = vllm_engine(model, max_tokens=60, temperature=0.0, tp=1)
    outs = generate([[{"role": "user", "content": prior_prompt(g["titles"])}] for g in groups])
    for g, (text, _, _) in zip(groups, outs):
        g.setdefault("llm", {})[model] = {"raw": text, "ranking": parse_ranking(text, len(g["titles"]))}
    PATH.write_text(json.dumps(groups, indent=1))
    print(f"parsed {sum(g['llm'][model]['ranking'] is not None for g in groups)}/{len(groups)}")


def analyze() -> None:
    groups = json.loads(PATH.read_text())
    models = sorted({m for g in groups for m in g.get("llm", {})})
    out, md = {}, ["# LLM headline prior vs real CTR on well-observed articles\n",
                   f"Groups of {SIZE} test-day articles, each with ≥ {MIN_IMPRESSIONS} impressions. Spearman is "
                   f"per group, averaged (± 95% CI). Chance for top pick = {1 / SIZE:.0%}.\n",
                   "| Groups | Ranker | Spearman vs real CTR | Top pick = real best |", "|---|---|---|---|"]
    for gtype in ("mixed", "same_category"):
        gs = [g for g in groups if g["type"] == gtype]
        rankers = {"Simulator CTR": lambda g: np.array(g["sim_ctr"])}
        for m in models:
            rankers[f"LLM ({m.split('/')[-1]})"] = (
                lambda g, m=m: None if g["llm"][m]["ranking"] is None else -np.argsort(g["llm"][m]["ranking"]))
        for name, score in rankers.items():
            rho, top = [], []
            for g in gs:
                s = score(g)
                if s is None:
                    continue
                r = np.array(g["real_ctr"])
                rho.append(stats.spearmanr(s, r)[0])
                top.append(int(np.argmax(s)) == int(np.argmax(r)))
            rho = np.array(rho)
            m_, h = rho.mean(), stats.t.ppf(0.975, len(rho) - 1) * rho.std(ddof=1) / np.sqrt(len(rho))
            out[f"{gtype}|{name}"] = {"spearman": [float(m_), float(h)], "top1": float(np.mean(top)), "groups": len(rho)}
            md.append(f"| {gtype.replace('_', ' ')} | {name} | {m_:.2f} ± {h:.2f} | {np.mean(top):.0%} |")
    (OUT / "prior_probe.json").write_text(json.dumps(out, indent=1))
    (OUT / "prior_probe.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=("build", "run", "analyze"))
    p.add_argument("--model", default="google/gemma-3-12b-it")
    p.add_argument("--device", default=default_device(), help="simulator device for the build step")
    args = p.parse_args()
    {"build": lambda: build(args.device), "run": lambda: run(args.model), "analyze": analyze}[args.step]()


if __name__ == "__main__":
    main()
