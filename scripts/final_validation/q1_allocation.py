"""Reward-optimized fixed tests and batched-feedback bandits on matched Q1 pools."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.final_validation.common import OUT, SEED, fingerprint, interval, is_complete, save_npz, write_json
from scripts.run_q1 import load_env
from src.bandits.environment import BanditEnv
from src.bandits.policies import (DiscountedThompson, EpsilonGreedy, ExploreThenCommit,
                                 Policy, Thompson, UCB1, Uniform, ab_sample_size)

CONDITIONS = (("K2", 10000), ("K2", 100000), ("K10_T10k", 10000),
              ("K10", 100000), ("K10_abrupt", 100000), ("K10_gradual", 100000))


class KLUCB(Policy):
    def select(self, t):
        means = self.means()
        lo, hi = means.copy(), np.ones_like(means)
        budget = (np.log(max(t, 3)) + 3 * np.log(np.log(max(t, 3)))) / np.maximum(self.pulls, 1)
        p = np.clip(means, 1e-12, 1 - 1e-12)
        for _ in range(18):
            mid = (lo + hi) / 2
            q = np.clip(mid, 1e-12, 1 - 1e-12)
            divergence = p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
            good = divergence <= budget
            lo, hi = np.where(good, mid, lo), np.where(good, hi, mid)
        return self.argmax(self.untried_first(lo))


class SuccessiveElimination(Policy):
    def __init__(self, *args, horizon):
        super().__init__(*args)
        self.active = np.ones((self.S, self.K), bool)
        self.logterm = np.log(4 * self.K * horizon**2 / 0.05)

    def select(self, t):
        n = np.maximum(self.pulls, 1)
        mean = self.means()
        radius = np.sqrt(2 * mean * (1 - mean) * self.logterm / n) + 3 * self.logterm / n
        best_lower = np.where(self.active, mean - radius, -np.inf).max(1)
        self.active &= mean + radius >= best_lower[:, None]
        return self.argmax(np.where(self.active, -self.pulls, -1e30))


class PeriodicETC(ExploreThenCommit):
    def __init__(self, *args, period, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period

    def select(self, t):
        if t > 0 and t % self.period == 0:
            self.pulls.fill(0)
            self.wins.fill(0)
            self.choice.fill(-1)
            self.commit_time.fill(-1)
        return super().select(t % self.period)


def optimized_test_size(mu, horizon, seed):
    S, K = mu.shape
    candidates = np.unique(np.geomspace(1, max(2, horizon // (2 * K)), 32).astype(int))
    values = []
    for n in candidates:
        rng = np.random.default_rng(seed)
        counts = rng.binomial(n, mu[:, :, None], size=(S, K, 4096))
        winner = (counts + rng.random(counts.shape) * 1e-6).argmax(1)
        deploy = mu[np.arange(S)[:, None], winner].mean(1)
        values.append(float((n * mu.sum(1) + (horizon - K * n) * deploy).mean()))
    best = int(np.argmax(values))
    return int(candidates[best]), {"per_arm_grid": candidates.tolist(), "training_expected_clicks": values,
                                   "selected_per_arm": int(candidates[best]), "draws_per_training_pool": 4096}


def normal_test_size(mu, horizon):
    # Feit & Berman (2019), Eq. 10, with prior moments estimated from training pools.
    # Bernoulli rewards use the normal approximation. This formula is two-arm only.
    variance = max(float(mu.var()), 1e-10)
    noise = float((mu * (1 - mu)).mean())
    ratio = noise / variance
    root = np.sqrt(horizon * ratio / 4 + (3 * ratio / 4)**2)
    return max(1, int(np.ceil((horizon * ratio / 4) / (root + 3 * ratio / 4))))


def make_policy(name, env, batch, per_arm, coefficient=0.1, discount=0.9999):
    rng = np.random.default_rng(SEED)
    S, K, T = env.S, env.K, env.horizon
    if name.startswith("etc_"):
        n = int(np.ceil(per_arm * K / batch) * batch / K)
        return ExploreThenCommit(S, K, rng, n, name)
    if name == "periodic_etc":
        period = max(batch, T // 10)
        n = max(1, int(0.1 * period / K))
        n = int(np.ceil(n * K / batch) * batch / K)
        return PeriodicETC(S, K, rng, n, name, period=period)
    if name == "ucb_tuned":
        return UCB1(S, K, rng, c=coefficient)
    if name == "kl_ucb":
        return KLUCB(S, K, rng)
    if name == "ts":
        return Thompson(S, K, rng)
    if name == "discounted_ts":
        return DiscountedThompson(S, K, rng, discount)
    if name == "elimination":
        return SuccessiveElimination(S, K, rng, horizon=T)
    if name == "epsilon_0.1":
        return EpsilonGreedy(S, K, rng, eps=0.1)
    return Uniform(S, K, rng)


def simulate(policy, env, batch, diagnostics=True):
    S, K, T = env.S, env.K, env.horizon
    rows = np.arange(S)
    regret, expected, realized = np.zeros(S), np.zeros(S), np.zeros(S)
    choices = np.empty((S, T), np.int16)
    pending_arms, pending_rewards = [], []
    marks = np.unique(np.r_[np.geomspace(10, T, 40).astype(int), T])
    curves = np.empty((S, len(marks)))
    c = 0
    for t in range(T):
        # A balanced first batch avoids starving arms before any feedback is visible.
        arm = np.full(S, t % K) if t < max(K, batch) and not isinstance(policy, ExploreThenCommit) else policy.select(t)
        mu = env.mu(t)
        reward = env.uniforms[:, t] < env.click_prob(t, arm)
        pending_arms.append(arm)
        pending_rewards.append(reward)
        if (t + 1) % batch == 0 or t + 1 == T:
            for a, r in zip(pending_arms, pending_rewards):
                policy.update(a, r)
            pending_arms.clear()
            pending_rewards.clear()
        regret += mu.max(1) - mu[rows, arm]
        expected += mu[rows, arm]
        realized += reward
        choices[:, t] = arm
        if t + 1 == marks[c]:
            curves[:, c] = regret
            c += 1
    terminal_mu = env.mu(T - 1)
    recommendation = policy.argmax(policy.means())
    simple = terminal_mu.max(1) - terminal_mu[rows, recommendation]
    if not diagnostics:
        return {"regret": regret}
    current_best = np.stack([env.mu(t).argmax(1) for t in range(T // 2, T)], 1)
    share = (choices[:, T // 2:] == current_best).mean(1)
    recovery = np.full(S, T - T // 2, int)
    recovered = np.zeros(S, bool)
    for start in range(T // 2, T - 999, 1000):
        best = np.stack([env.mu(t).argmax(1) for t in range(start, start + 1000)], 1)
        good = (choices[:, start:start + 1000] == best).mean(1) >= .8
        newly = good & ~recovered
        recovery[newly] = start + 1000 - T // 2
        recovered |= good
    return {"regret": regret, "expected_ctr": expected / T, "realized_clicks": realized,
            "terminal_correct": np.isclose(simple, 0, rtol=0, atol=1e-10).astype(float),
            "terminal_simple_regret": simple, "second_half_best_share": share,
            "recovery_rounds": recovery, "recovery_observed": recovered.astype(float),
            "curve": curves, "marks": marks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, choices=range(len(CONDITIONS)), required=True)
    args = p.parse_args()
    condition, horizon = CONDITIONS[args.index]
    dest = OUT / "q1_allocation" / f"{condition}_T{horizon}"
    inputs = [Path(__file__), Path("src/bandits/policies.py"), Path("results/q1/tuned.json")]
    inputs += [Path(f"results/q1/envs/{s}_{condition}.npz") for s in ("train", "test")]
    provenance = fingerprint({"seed": SEED, "batches": [1, 100, 1000], "condition": condition, "horizon": horizon}, inputs)
    if is_complete(dest / "summary.json", provenance):
        return
    marker = dest / "provenance.json"
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial Q1 run was made with different inputs/settings")
    write_json(marker, provenance)
    train = load_env("train", condition)
    raw = load_env("test", condition)
    env = BanditEnv(raw.P, raw.articles, horizon, raw.drift, raw.perm, raw.seed)
    n, sizing = optimized_test_size(train.mu0, horizon, SEED)
    fractions = [.005, .01, .02, .05, .1, .2, .5]
    sizes = {f"etc_frac_{f}": max(1, int(f * horizon / env.K)) for f in fractions}
    sizes["etc_reward_optimized"] = n
    if env.K == 2:
        sizes["etc_test_roll_normal"] = normal_test_size(train.mu0, horizon)
    tuned = json.loads(inputs[2].read_text())[condition]
    # Power sizing uses the training CTR rather than the evaluation pool's unknown mean.
    mde = float(tuned["A/B test (power analysis)"].split("_")[-1])
    sizes["etc_power"] = int(ab_sample_size(np.array([train.mu0.mean()]), mde)[0])
    original_fraction = float(tuned["A/B test (tuned length)"].split("_")[-1])
    sizes["etc_original_train_tuned"] = max(1, int(original_fraction * horizon / env.K))
    coefficient = float(tuned["UCB1"].split("_")[-1])
    discount = float(tuned["Discounted Thompson"].split("_")[-1])
    # Fractional grid is evaluated on training only and chosen separately for each batch.
    results, arrays, selected = {}, {}, {}
    for batch in (1, 100, 1000):
        tr_env = BanditEnv(train.P, train.articles, horizon, train.drift, train.perm, train.seed)
        train_values = {}
        for label in [f"etc_frac_{f}" for f in fractions]:
            value = simulate(make_policy(label, tr_env, batch, sizes[label]), tr_env, batch, diagnostics=False)
            train_values[label] = float(value["regret"].mean())
        selected[str(batch)] = min(train_values, key=train_values.get)
        labels = [selected[str(batch)], "etc_reward_optimized", "etc_original_train_tuned", "etc_power",
                  "ucb_tuned", "kl_ucb", "ts", "discounted_ts", "elimination", "epsilon_0.1", "uniform"]
        if env.K == 2:
            labels.append("etc_test_roll_normal")
        if env.drift != "none":
            labels.append("periodic_etc")
        by_policy = {}
        for label in labels:
            key = f"B{batch}/{label}"
            path = dest / f"B{batch}_{label}.npz"
            if path.exists():
                with np.load(path) as f:
                    res = {k: f[k] for k in f.files}
            else:
                pol = make_policy(label, env, batch, sizes.get(label, n), coefficient, discount)
                res = simulate(pol, env, batch)
                save_npz(path, **res)
            arrays.update({f"{key}/{k}": v for k, v in res.items()})
            by_policy[label] = res
            print(condition, horizon, batch, label, float(res["regret"].mean()), flush=True)
        ref = by_policy["ts"]["regret"]
        results[str(batch)] = {label: {
            **{k: interval(v) for k, v in res.items() if v.ndim == 1 and k != "marks"},
            "paired_minus_ts": interval(res["regret"] - ref)} for label, res in by_policy.items()}
    save_npz(dest / "runs.npz", **arrays)
    write_json(dest / "summary.json", {"provenance": provenance, "test_sizing": sizing,
        "per_arm_sizes": sizes, "batch_train_selected_fraction": selected,
        "warmup": "balanced first max(K,B) decisions for adaptive policies, with feedback released at batch ends",
        "test_roll_citation": "https://arxiv.org/abs/1811.00457; normal approximation only for K=2",
        "results": results})


if __name__ == "__main__":
    main()
