"""Q4 analysis: LLM agents vs bandit baselines on identical environments and click draws.

Baselines (numpy, same uniforms as the agents): Thompson sampling, UCB1 (c = 1 and
the Q1-tuned c = 0.1), epsilon-greedy (0.1), greedy, uniform, and "LLM prior only"
(always show the headline the LLM ranked first, never learning).

Metrics, following Krishnamurthy et al. (2024):
    regret             sum_t mu* - mu[a_t]
    best-arm share     overall and in the last 100 rounds
    suffix failure     share of runs that never pick the best arm in the second half
    K * MinFrac        K x the smallest share of pulls any arm received (exploration;
                       1 = uniform, 0 = some arm never tried)
    stuck              share of runs where one non-best arm takes >= 90% of the last half

    python -m scripts.analyze_q4
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from scripts.build_q4_envs import ARMS, OUT
from src.bandits.policies import UCB1, EpsilonGreedy, Thompson, Uniform

INK, MUTED, GRID = "#1f1e1d", "#5f5e5a", "#ecebe6"
COLOR = {"LLM": "#2a78d6", "LLM + tool": "#eb6834", "Thompson sampling": "#1baf7a", "UCB1 (c=0.1)": "#eda100",
         "UCB1 (c=1)": "#e87ba4", "ε-greedy (0.1)": "#008300", "Greedy": "#6250d6", "Uniform": "#8a8984",
         "LLM prior only": "#e34948"}


def ci(x) -> tuple[float, float]:
    x = np.asarray(x, float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def run_baseline(policy, mu, unif) -> np.ndarray:
    S, T = unif.shape
    rows = np.arange(S)
    choices = np.empty((S, T), int)
    for t in range(T):
        a = policy.select(t)
        policy.update(a, (unif[:, t] < mu[rows, a]).astype(float))
        choices[:, t] = a
    return choices


def metrics(choices: np.ndarray, mu: np.ndarray) -> dict:
    S, T = choices.shape
    K = mu.shape[1]
    rows = np.arange(S)[:, None]
    best = mu.argmax(1)[:, None]
    inst = mu.max(1)[:, None] - mu[rows, choices]
    regret = inst.cumsum(1)
    half = choices[:, T // 2:]
    counts = np.stack([(choices == k).sum(1) for k in range(K)], 1) / T
    top_half = np.stack([(half == k).mean(1) for k in range(K)], 1)
    stuck = (top_half.max(1) >= 0.9) & (top_half.argmax(1) != best[:, 0])
    return {"regret": ci(regret[:, -1]), "regret_runs": regret[:, -1].tolist(), "curve": regret.mean(0).tolist(),
            "best_share": ci((choices == best).mean(1)), "best_share_last100": ci((choices[:, -100:] == best).mean(1)),
            "suffix_failure": float((half != best).all(1).mean()), "k_minfrac": ci(K * counts.min(1)),
            "stuck": float(stuck.mean())}


def main() -> None:
    envs = np.load(OUT / "envs.npz")
    llm_files = sorted(f for f in OUT.glob("llm_*.npz") if f.stem != "llm_mock")
    results = {}
    for K in ARMS:
        mu, unif = envs[f"K{K}_mu"], envs[f"K{K}_uniforms"]
        res = {}
        for f in llm_files:
            model = f.stem[4:]
            meta = json.loads(f.with_suffix(".json").read_text())
            z = np.load(f)
            decisions = sum(z[f"K{k}_{v}/choice"].size for k in ARMS for v in ("tool", "notool"))
            for variant, label in (("notool", "LLM"), ("tool", "LLM + tool")):
                tag = f"K{K}_{variant}"
                ch = z[f"{tag}/choice"]
                S, T = ch.shape
                m = metrics(ch, mu[:S])
                cfg = next(c for c in meta["configs"] if c["tag"] == tag)
                m.update(model=model, tool_call_rate=float(z[f"{tag}/tool_called"].mean()),
                         parse_fail_rate=float(z[f"{tag}/parse_failed"].mean()),
                         tokens_per_1k=cfg["tokens_per_1k_decisions"],
                         gpu_seconds_per_1k=meta["seconds_total"] / decisions * 1000)
                res[f"{label} [{model}]"] = m
            ranks = meta["prior_rankings"][f"K{K}"]
            ok = [i for i, r in enumerate(ranks) if r is not None]
            rho = [stats.spearmanr(np.argsort(ranks[i]), -mu[i])[0] for i in ok]  # rank position vs CTR
            top = np.array([ranks[i][0] for i in ok])
            res[f"LLM prior only [{model}]"] = {
                **metrics(np.repeat(top[:, None], T, 1), mu[ok][:, :]),
                "prior_spearman": ci(rho), "prior_top1_is_best": float((top == mu[ok].argmax(1)).mean()),
                "prior_parsed": len(ok)}
            S_llm, T_llm = S, T
        S, T = (S_llm, T_llm) if llm_files else unif.shape
        mu_s, unif_s = mu[:S], unif[:S, :T]
        rng = np.random.default_rng(K)
        for label, pol in (("Thompson sampling", Thompson(S, K, rng)), ("UCB1 (c=0.1)", UCB1(S, K, rng, c=0.1)),
                           ("UCB1 (c=1)", UCB1(S, K, rng, c=1.0)), ("ε-greedy (0.1)", EpsilonGreedy(S, K, rng, eps=0.1)),
                           ("Greedy", EpsilonGreedy(S, K, rng, eps=0.0)), ("Uniform", Uniform(S, K, rng))):
            res[label] = metrics(run_baseline(pol, mu_s, unif_s), mu_s)
        ref = np.array(res["Thompson sampling"]["regret_runs"])
        for r in res.values():
            if len(r["regret_runs"]) == len(ref):
                r["regret_minus_thompson"] = ci(np.array(r["regret_runs"]) - ref)
        results[f"K{K}"] = res

    (OUT / "summary.json").write_text(json.dumps(results, indent=1))
    write_md(results)
    plot(results)
    print((OUT / "summary.md").read_text())


def write_md(results: dict) -> None:
    md = ["# Q4 results: an LLM as the bandit policy\n",
          "Non-contextual bandits whose arms are real test-day headlines (true click rate = simulator "
          "average over 20k users). All policies see the same click draws. ± = 95% CI over runs; "
          "'− TS' is paired against Thompson sampling.\n"]
    for K, res in results.items():
        T = len(next(iter(res.values()))["curve"])
        md += [f"## {K[1:]} articles, {T} rounds\n",
               "| Policy | Regret | − TS | Best-arm share (last 100) | Suffix failure | K·MinFrac | Stuck | "
               "Tool calls | Tokens / 1k decisions |",
               "|---|---|---|---|---|---|---|---|---|"]
        for name, r in res.items():
            d = r.get("regret_minus_thompson")
            diff = "" if d is None else f"{d[0]:+.1f} ± {d[1]:.1f}"
            tool = f"{r['tool_call_rate']:.0%}" if "tool" in name else ""
            tokens = f"{r['tokens_per_1k']:,.0f}" if "tokens_per_1k" in r else ""
            md.append(f"| {name} | {r['regret'][0]:.1f} ± {r['regret'][1]:.1f} | {diff} | "
                      f"{r['best_share_last100'][0]:.0%} | {r['suffix_failure']:.0%} | {r['k_minfrac'][0]:.2f} | "
                      f"{r['stuck']:.0%} | {tool} | {tokens} |")
        for name, r in res.items():
            if "prior_spearman" in r:
                md.append(f"\n{name}: headline ranking vs true click rate Spearman {r['prior_spearman'][0]:.2f} ± "
                          f"{r['prior_spearman'][1]:.2f}; top-ranked headline is the best article in "
                          f"{r['prior_top1_is_best']:.0%} of runs (chance {1 / int(K[1:]):.0%}).")
        md.append("")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")


def plot(results: dict) -> None:
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.0), dpi=150, sharey=False)
    for ax, (K, res) in zip(np.atleast_1d(axes), results.items()):
        for name, r in res.items():
            base = name.split(" [")[0]
            ls = "--" if base in ("Uniform", "LLM prior only") else "-"
            ax.plot(np.arange(1, len(r["curve"]) + 1), r["curve"], color=COLOR.get(base, "#888"), lw=2, ls=ls,
                    label=name)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_xlabel("Round", color=MUTED)
        ax.set_ylabel("Cumulative regret (expected clicks lost)", color=MUTED)
        ax.set_title(f"{K[1:]} articles", loc="left", fontsize=11, color=INK)
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(fig_dir / "regret_curves.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
