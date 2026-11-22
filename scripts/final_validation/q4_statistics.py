"""Tool-only policies, rank-prior TS, and diagnostics on the original Q4 pools."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from scripts.analyze_q4 import metrics
from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json

BASE_CTR = 0.04051924803670631


def posterior_scores(shows, clicks, rng, draws=4000):
    a, b = 1 + clicks, 1 + shows - clicks
    samples = rng.beta(a[:, :, None], b[:, :, None], size=(*a.shape, draws))
    winners = samples.argmax(1)
    pbest = np.stack([(winners == k).mean(1) for k in range(a.shape[1])], 1)
    return {"mean": a / (a + b), "upper": stats.beta.ppf(0.975, a, b), "pbest": pbest}


def simulate(mu, uniforms, rule, seed, prior=None, strength=0, draws=4000):
    S, K = mu.shape
    T = uniforms.shape[1]
    rng = np.random.default_rng(seed)
    shows, clicks = np.zeros((S, K)), np.zeros((S, K))
    rows = np.arange(S)
    choices = np.empty((S, T), np.int16)
    for t in range(T):
        a, b = 1 + clicks, 1 + shows - clicks
        if rule == "ts":
            if prior is not None:
                a = a + strength * prior
                b = b + strength * (1 - prior)
            score = rng.beta(a, b)
        elif rule == "mean":
            score = a / (a + b)
        elif rule == "upper":
            score = stats.beta.ppf(0.975, a, b)
        else:
            score = posterior_scores(shows, clicks, rng, draws)["pbest"]
        arm = (score + 1e-12 * rng.random(score.shape)).argmax(1)
        reward = uniforms[:, t] < mu[rows, arm]
        shows[rows, arm] += 1
        clicks[rows, arm] += reward
        choices[:, t] = arm
    return choices


def make_prior(rankings, K):
    # Each arm has the same pseudo-count strength. Only its prior mean changes.
    # The rank-to-rate mapping is fixed before inspecting any outcomes.
    values = np.exp(np.linspace(1, -1, K))
    values = BASE_CTR * values / values.mean()
    prior = np.full((len(rankings), K), BASE_CTR)
    for i, ranking in enumerate(rankings):
        if ranking is not None:
            if sorted(ranking) != list(range(K)):
                raise ValueError("Malformed saved headline ranking")
            prior[i, ranking] = values
    return prior


def diagnostics(mu, uniforms, data, tag, draws):
    choice = data[f"{tag}/choice"]
    called = data[f"{tag}/tool_called"].astype(bool)
    S, T = choice.shape
    K = mu.shape[1]
    shows, clicks = np.zeros((S, K)), np.zeros((S, K))
    rows = np.arange(S)
    totals = {key: np.zeros(S) for key in ("mean", "upper", "pbest")}
    rng = np.random.default_rng(SEED + K)
    for t in range(T):
        active = np.flatnonzero(called[:, t])
        if len(active):
            scores = posterior_scores(shows[active], clicks[active], rng, draws)
            for key, score in scores.items():
                chosen_score = score[np.arange(len(active)), choice[active, t]]
                # Accept every maximizer, including ties from quantized Monte Carlo counts.
                totals[key][active] += np.isclose(chosen_score, score.max(1), rtol=0, atol=1e-12)
        arm = choice[:, t]
        reward = uniforms[:, t] < mu[rows, arm]
        assert np.array_equal(reward, data[f"{tag}/reward"][:, t].astype(bool))
        shows[rows, arm] += 1
        clicks[rows, arm] += reward
    count = called.sum(1)
    valid = count > 0
    return {
        "tool_rounds": int(count.sum()),
        "pools_with_tool_calls": int(valid.sum()),
        "agreement": {k: interval(v[valid] / count[valid]) for k, v in totals.items()} if valid.any() else {},
        "ever_tried_fraction": interval((shows > 0).mean(1)),
        "position_pull_fraction": (shows.sum(0) / shows.sum()).tolist(),
        "parse_failure_fraction": interval(data[f"{tag}/parse_failed"].mean(1)),
        "monte_carlo_note": "p(best) is recomputed with an independent 4000-draw sample; the original tool RNG also advanced on parsing failures.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=4000)
    args = p.parse_args()
    dest = OUT / "q4_statistics"
    inputs = [Path("results/q4/envs.npz"), Path("results/q4/llm_gemma-3-12b-it.npz"),
              Path("results/q4/llm_gemma-3-12b-it.json"), Path(__file__)]
    provenance = fingerprint({"seed": SEED, "draws": args.draws, "strengths": [0, 10, 100, 1000],
                              "prior_base_ctr": BASE_CTR}, inputs)
    if is_complete(dest / "summary.json", provenance):
        return
    env = np.load(inputs[0])
    saved = np.load(inputs[1])
    meta = json.loads(inputs[2].read_text())
    output, arrays = {}, {}
    for K in (5, 10):
        mu, uniforms = env[f"K{K}_mu"], env[f"K{K}_uniforms"]
        choices = {f"tool_{rule}": simulate(mu, uniforms, rule, SEED + 10 * K, draws=args.draws)
                   for rule in ("mean", "upper", "pbest")}
        choices["ts"] = simulate(mu, uniforms, "ts", SEED + 10 * K)
        rankings = meta["prior_rankings"][f"K{K}"]
        rng = np.random.default_rng(SEED + K)
        shuffled = [None if r is None else rng.permutation(r).tolist() for r in rankings]
        priors = {"llm": make_prior(rankings, K), "shuffled": make_prior(shuffled, K),
                  "simulator_rank": make_prior(np.argsort(-mu, axis=1).tolist(), K)}
        for source, prior in priors.items():
            for strength in (10, 100, 1000):
                label = f"prior_{source}_{strength}"
                choices[label] = simulate(mu, uniforms, "ts", SEED + 10 * K, prior, strength)
        for variant in ("tool", "notool"):
            choices[f"llm_{variant}"] = saved[f"K{K}_{variant}/choice"]
        results = {label: metrics(ch, mu) for label, ch in choices.items()}
        reference = np.array(results["ts"]["regret_runs"])
        for label, res in results.items():
            res["paired_minus_ts"] = interval(np.array(res["regret_runs"]) - reference)
            arrays[f"K{K}/{label}/choice"] = choices[label]
            print(K, label, res["regret"], flush=True)
        output[f"K{K}"] = {"policies": results, "diagnostics": {
            v: diagnostics(mu, uniforms, saved, f"K{K}_{v}", args.draws) for v in ("tool", "notool")}}
    save_npz(dest / "choices.npz", **arrays)
    write_json(dest / "summary.json", {"provenance": provenance, "results": output})


if __name__ == "__main__":
    main()
