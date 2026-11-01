"""Check the LLM's headline prior and the simulator's click rates against real logged CTR.

Real CTR = clicks / impressions of each article as a candidate in the MIND test
day (Nov 15), shrunk towards the day's overall CTR with a Beta prior of strength
20 impressions so rarely shown articles do not dominate. Logged CTR comes from
MSN's own recommender (which articles it showed to whom), so it is a reference,
not ground truth for a uniformly random reader.

Three comparisons:
1. Q4 runs: Spearman of (a) the simulator's population CTR and (b) the LLM's prior
   ranking against real CTR within each run's K articles; how often each picks the
   real best article.
2. All test-day articles with >= 100 impressions: simulator population CTR (random
   test users) vs real CTR.
3. Same articles: the click model's mean prediction over each article's actual
   logged impressions vs real CTR (same readers, so this isolates the model).

    python -m scripts.validate_q4_prior
Writes results/q4/prior_validation.json and .md.
"""

import json

import numpy as np
from scipy import stats

from scripts.build_q4_envs import ARMS, OUT
from src.data.mind import load_processed

PRIOR_STRENGTH, MIN_IMPRESSIONS = 20, 100


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main() -> None:
    test = load_processed("test")
    n = np.bincount(test.cand_idx, minlength=test.cand_idx.max() + 1).astype(float)
    c = np.bincount(test.cand_idx, weights=test.cand_label, minlength=len(n))
    base = c.sum() / n.sum()
    real = (c + PRIOR_STRENGTH * base) / (n + PRIOR_STRENGTH * base + PRIOR_STRENGTH * (1 - base))

    envs = np.load(OUT / "envs.npz")
    llm_meta = json.loads((OUT / "llm_gemma-3-12b-it.json").read_text())
    out, md = {"day_ctr": float(base)}, ["# Headline prior and simulator vs real logged CTR (MIND test day)\n",
                                         f"Real CTR = test-day clicks / impressions, shrunk to the day's CTR "
                                         f"({base:.2%}) with a Beta prior worth {PRIOR_STRENGTH} impressions.\n",
                                         "## 1. Within Q4 runs\n",
                                         "| K | Simulator vs real (Spearman) | LLM prior vs real (Spearman) | Simulator's best = real best | "
                                         "LLM's top pick = real best | Chance | Median impressions per article |",
                                         "|---|---|---|---|---|---|---|"]
    for K in ARMS:
        rows, mu = envs[f"K{K}_rows"], envs[f"K{K}_mu"]
        ranks = llm_meta["prior_rankings"][f"K{K}"]
        sim_rho, llm_rho, sim_top, llm_top = [], [], [], []
        for s in range(len(rows)):
            r = real[rows[s]]
            sim_rho.append(stats.spearmanr(mu[s], r)[0])
            sim_top.append(mu[s].argmax() == r.argmax())
            if ranks[s] is not None:
                llm_rho.append(stats.spearmanr(-np.argsort(ranks[s]), r)[0])  # rank position -> score
                llm_top.append(ranks[s][0] == r.argmax())
        res = {"sim_vs_real": ci(sim_rho), "llm_vs_real": ci(llm_rho), "sim_top_is_real_best": float(np.mean(sim_top)),
               "llm_top_is_real_best": float(np.mean(llm_top)), "chance": 1 / K,
               "median_impressions": float(np.median(n[rows]))}
        out[f"K{K}"] = res
        md.append(f"| {K} | {res['sim_vs_real'][0]:.2f} ± {res['sim_vs_real'][1]:.2f} | "
                  f"{res['llm_vs_real'][0]:.2f} ± {res['llm_vs_real'][1]:.2f} | {res['sim_top_is_real_best']:.0%} | "
                  f"{res['llm_top_is_real_best']:.0%} | {1 / K:.0%} | {res['median_impressions']:.0f} |")

    # 2-3. All well-observed test-day articles.
    from src.simulator.simulator import ClickSimulator

    sim = ClickSimulator.load(split="test", device="mps")
    arts = np.flatnonzero(n >= MIN_IMPRESSIONS)
    users = sim.sample_users(5000, np.random.default_rng(0))
    pop = np.concatenate([sim.prob_matrix(users, arts[i:i + 500]).mean(0) for i in range(0, len(arts), 500)])
    probs = np.load("results/click_model/test_probs.npy").astype(float)
    logged_pred = np.bincount(test.cand_idx, weights=probs, minlength=len(n))[arts] / n[arts]
    raw = c[arts] / n[arts]
    out["all_articles"] = {
        "n_articles": int(len(arts)), "min_impressions": MIN_IMPRESSIONS,
        "sim_population_vs_real_spearman": float(stats.spearmanr(pop, raw)[0]),
        "model_on_logged_impressions_vs_real_spearman": float(stats.spearmanr(logged_pred, raw)[0]),
        "model_on_logged_impressions_vs_real_pearson": float(stats.pearsonr(logged_pred, raw)[0]),
        "real_ctr_quantiles": np.percentile(raw, [10, 50, 90]).round(4).tolist(),
        "sim_population_ctr_quantiles": np.percentile(pop, [10, 50, 90]).round(4).tolist(),
    }
    a = out["all_articles"]
    md += ["\n## 2–3. All test-day articles with ≥ 100 impressions\n",
           f"{a['n_articles']:,} articles. Real CTR deciles (10/50/90%): {a['real_ctr_quantiles']}; "
           f"simulator population CTR: {a['sim_population_ctr_quantiles']}.\n",
           "| Comparison | Spearman |", "|---|---|",
           f"| Simulator population CTR (random test readers) vs real CTR | {a['sim_population_vs_real_spearman']:.2f} |",
           f"| Click model averaged over each article's actual logged readers vs real CTR | "
           f"{a['model_on_logged_impressions_vs_real_spearman']:.2f} (Pearson {a['model_on_logged_impressions_vs_real_pearson']:.2f}) |"]
    (OUT / "prior_validation.json").write_text(json.dumps(out, indent=1))
    (OUT / "prior_validation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
