"""Live simulation engines for the Streamlit demo (NumPy only, no torch, no GPU).

race()        Q1 setting: K headlines with fixed population click rates. Runs the repository's
              own policy classes (src/bandits/policies.py) on shared coin flips, so every policy
              sees the same readers and the same click draws, as in the paper.
churn_run()   Q2/Q3 setting: 20 live articles, one replaced every 100 rounds. A NumPy port of
              src/bandits/contextual.py (same schedule, features phi = [1, a, u * a], ridge
              LinUCB with Sherman-Morrison updates, per-article Beta TS). It additionally keeps
              snapshots of the policy state so the app can explain individual choices.
              The random streams differ from the torch original, so single runs are not
              identical to the paper's, but averages over pools should agree (checked in
              demo/check_engines.py).
posterior_stats()  the Q4 statistics tool: Beta posterior mean, 95% upper bound, P(best).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

from src.bandits.environment import drift_permutation
from src.bandits.policies import DiscountedThompson, EpsilonGreedy, ExploreThenCommit, Thompson, UCB1

POLICY_NAMES = ("A/B test", "ε-greedy", "UCB", "Thompson sampling", "Discounted Thompson")


# ----------------------------------------------------------------------------------------------
# Q1: non-contextual race on fixed click rates
# ----------------------------------------------------------------------------------------------

def contract(mu: np.ndarray, lam: float) -> np.ndarray:
    """Shrink each pool's click rates toward the pool mean by lam (milestone report, eq. 2)."""
    m = mu.mean(-1, keepdims=True)
    return m + lam * (mu - m)


def make_policies(S: int, K: int, horizon: int, seed: int, ab_frac: float, eps_c: float,
                  ucb_c: float, gamma: float, which=POLICY_NAMES) -> dict:
    """Each policy gets its own random stream (as in policy_suite)."""
    makers = {
        # A/B: round-robin over the K headlines for ab_frac of the horizon, then commit.
        "A/B test": lambda r: ExploreThenCommit(S, K, r, max(1, int(ab_frac * horizon / K)), "ab"),
        # epsilon_t = min(1, c K / t): explores a lot early, less later.
        "ε-greedy": lambda r: EpsilonGreedy(S, K, r, decay_c=eps_c),
        "UCB": lambda r: UCB1(S, K, r, c=ucb_c),
        "Thompson sampling": lambda r: Thompson(S, K, r),
        "Discounted Thompson": lambda r: DiscountedThompson(S, K, r, gamma=gamma),
    }
    seeds = np.random.SeedSequence(seed).spawn(len(POLICY_NAMES))
    return {n: makers[n](np.random.default_rng(s)) for n, s in zip(POLICY_NAMES, seeds) if n in which}


def race(mu: np.ndarray, horizon: int, seed: int = 0, ab_frac: float = 0.2, eps_c: float = 30.0,
         ucb_c: float = 0.1, gamma: float = 0.999, drift: bool = False, lam: float = 1.0,
         which=POLICY_NAMES) -> dict:
    """Run policies on S pools of fixed click rates ``mu`` (S, K) for ``horizon`` rounds.

    Rewards are clicks with probability mu_t[arm]; with ``drift`` the click rates are permuted
    halfway (the best headline changes), as in the Q1 abrupt-drift condition.

    Returns per policy: arms (S, T), clicks (S, T), regret (S, T) cumulative expected clicks lost
    to the best headline, plus the true click-rate paths for the oracle.
    """
    mu = contract(np.atleast_2d(np.asarray(mu, np.float64)), lam)
    S, K = mu.shape
    rng = np.random.default_rng(seed)
    uniforms = rng.random((S, horizon))                    # common random numbers across policies
    mu1 = np.take_along_axis(mu, drift_permutation(mu, rng), 1) if drift else mu
    switch = horizon // 2 if drift else horizon
    rows = np.arange(S)
    out = {}
    for name, pol in make_policies(S, K, horizon, seed + 1, ab_frac, eps_c, ucb_c, gamma, which).items():
        arms = np.empty((S, horizon), np.int16)
        clicks = np.empty((S, horizon), bool)
        lost = np.empty((S, horizon))
        for t in range(horizon):
            m = mu if t < switch else mu1
            a = pol.select(t)
            r = uniforms[:, t] < m[rows, a]
            pol.update(a, r)
            arms[:, t], clicks[:, t], lost[:, t] = a, r, m.max(1) - m[rows, a]
        out[name] = {"arms": arms, "clicks": clicks, "regret": lost.cumsum(1)}
    out["_mu"], out["_mu1"], out["_switch"] = mu, mu1, switch
    return out


def posterior_stats(shows: np.ndarray, clicks: np.ndarray, draws: int = 4000, seed: int = 0) -> dict:
    """The Q4 statistics tool for one pool: Beta(1 + clicks, 1 + misses) posteriors."""
    a, b = 1 + clicks, 1 + shows - clicks
    rng = np.random.default_rng(seed)
    samples = rng.beta(a, b, size=(draws, len(a)))
    return {"mean": a / (a + b), "upper95": beta_dist.ppf(0.975, a, b),
            "p_best": np.bincount(samples.argmax(1), minlength=len(a)) / draws}


# ----------------------------------------------------------------------------------------------
# Q2/Q3: contextual bandits on a churning article pool
# ----------------------------------------------------------------------------------------------

def churn_run(P: np.ndarray, art_x: np.ndarray | None, user_x: np.ndarray | None, alpha: float,
              n_live: int, lifetime: int, horizon: int, seed: int = 0, snap_every: int = 100,
              new_window: int | None = None) -> dict:
    """Play one pool. P (U, M) click probabilities; art_x (M, d) and user_x (U, d) or None.

    alpha is LinUCB's exploration weight; art_x None means per-article Beta Thompson sampling.
    Regret is against the per-user oracle (best live article for the arriving reader).
    """
    P = P.astype(np.float64)
    U, M = P.shape
    step = max(1, lifetime // n_live)
    new_window = new_window or max(1, lifetime // 10)
    rng = np.random.default_rng(seed)
    user_seq = rng.integers(U, size=horizon)
    unif = rng.random(horizon)
    linear = art_x is not None
    if linear:
        A_x, U_x = art_x.astype(np.float64), user_x.astype(np.float64)
        d = A_x.shape[1]
        D = 1 + 2 * d
        A_inv, b = np.eye(D), np.zeros(D)                 # ridge lambda = 1
    else:
        wins, pulls = np.zeros(M), np.zeros(M)

    live = np.arange(n_live)
    born = np.zeros(n_live, np.int64)
    next_article = n_live
    chosen = np.empty(horizon, np.int32)
    click = np.empty(horizon, bool)
    lost = np.empty(horizon)
    lost_new = np.empty(horizon)
    p_chosen = np.empty(horizon)
    snaps = []
    for t in range(horizon):
        if t > 0 and t % step == 0 and next_article < M:
            slot = (t // step - 1) % n_live
            live[slot] = next_article
            born[slot] = t
            next_article += 1
        u = user_seq[t]
        p_live = P[u, live]
        if linear:
            a = A_x[live]                                          # (K, d)
            phi = np.concatenate([np.ones((n_live, 1)), a, U_x[u] * a], 1)
            theta = A_inv @ b
            mean = phi @ theta
            width = np.sqrt(np.clip(np.einsum("kd,de,ke->k", phi, A_inv, phi), 0, None))
            k = int((mean + alpha * width).argmax())
        else:
            sample = rng.beta(1 + wins[live], 1 + pulls[live] - wins[live])
            k = int(sample.argmax())
        if t % snap_every == 0:
            snap = {"t": t, "user": int(u), "live": live.copy(), "born": born.copy(), "choice": k,
                    "p_live": p_live.copy()}
            if linear:
                snap.update(theta=theta.copy(), mean=mean.copy(), width=width.copy())
            else:
                snap.update(wins=wins[live].copy(), pulls=pulls[live].copy())
            snaps.append(snap)
        m = live[k]
        r = unif[t] < p_live[k]
        if linear:
            x = phi[k]
            Ax = A_inv @ x
            A_inv -= np.outer(Ax, Ax) / (1 + x @ Ax)
            b += r * x
        else:
            pulls[m] += 1
            wins[m] += r
        best = int(p_live.argmax())
        chosen[t], click[t], lost[t] = m, r, p_live[best] - p_live[k]
        p_chosen[t] = p_live[k]
        lost_new[t] = lost[t] if (t - born[best]) < new_window else 0.0
    out = {"chosen": chosen, "clicks": click, "regret": lost.cumsum(), "regret_new": lost_new.cumsum(),
           "exp_ctr": float(p_chosen.mean()), "users": user_seq, "snaps": snaps}
    if linear:
        out["theta"] = A_inv @ b
    return out


def explain_choice(snap: dict, art_x: np.ndarray, user_x: np.ndarray, labels: list[str],
                   top: int = 8) -> list[dict]:
    """Per-feature reasons for the chosen article at one snapshot of a linear policy.

    The predicted click score is theta . [1, a, u * a]. Feature j contributes
    theta_a[j] * a[j] + theta_u[j] * u[j] * a[j]. Only differences between live articles can change
    a choice, so each contribution is reported relative to the average over the live articles
    (the same centring Q3 uses for decision importance).
    """
    d = art_x.shape[1]
    theta = snap["theta"]
    ta, tu = theta[1:1 + d], theta[1 + d:]
    a = art_x[snap["live"]]                                    # (K, d)
    u = user_x[snap["user"]]
    contrib = ta * a + tu * u * a                              # (K, d)
    rel = contrib[snap["choice"]] - contrib.mean(0)
    order = np.argsort(-np.abs(rel))[:top]
    return [{"feature": labels[j], "coordinate": int(j), "push": float(rel[j]),
             "article_value": float(a[snap["choice"], j]), "reader_value": float(u[j])} for j in order]
