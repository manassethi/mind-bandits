"""Q2: contextual bandits on a churning article pool, vectorized over S runs on a torch device.

Environment. K articles are live at once. Every ``lifetime / K`` rounds the
oldest slot retires and a never-seen article takes its place, so each article
lives ``lifetime`` rounds and new articles arrive steadily. Each round one user
(from a fixed population of U) arrives; P[s, u, m] is the simulator's click
probability for user u and the run's m-th article.

Regret is measured against the per-user oracle (the best live article for this
user), the most a contextual policy could get. Rounds where that best article
is younger than ``new_window`` count towards cold-start regret.

Policies (all share one model across articles, so they can score new ones):
    beta_ts    per-article Beta-Bernoulli Thompson sampling, no features
    linucb     ridge regression + alpha * sqrt(phi' A^-1 phi)
    lints      theta ~ N(theta_hat, v^2 A^-1)
    greedy     ridge regression, no exploration
with phi(u, a) = [1, a, u * a]: article features a and the user's mean history
features u in the same space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ChurnSchedule:
    n_live: int
    lifetime: int
    horizon: int

    @property
    def step(self) -> int:
        return max(1, self.lifetime // self.n_live)

    @property
    def n_articles(self) -> int:
        return self.n_live + self.horizon // self.step + 1


def build_churn_env(sim, n_runs: int, n_users: int, sched: ChurnSchedule, seed: int) -> dict:
    """Draw per-run user populations and article streams; tabulate P with the simulator."""
    rng = np.random.default_rng(seed)
    M = sched.n_articles
    P = np.empty((n_runs, n_users, M), np.float32)
    users = np.empty((n_runs, n_users), np.int64)
    articles = np.empty((n_runs, M), np.int64)
    for s in range(n_runs):
        users[s] = sim.sample_users(n_users, rng)
        articles[s] = sim.sample_articles(M, rng)
        P[s] = sim.prob_matrix(users[s], articles[s])
    return {"P": P, "users": users, "articles": articles, "n_live": sched.n_live,
            "lifetime": sched.lifetime, "horizon": sched.horizon, "seed": seed}


def article_features(feats: np.ndarray, d: int) -> np.ndarray:
    """First d columns, scaled so the mean squared row norm is 1 (comparable across representations)."""
    X = feats[:, :d].astype(np.float32)
    return X / np.sqrt((X**2).sum(1).mean())


def user_features(X: np.ndarray, user_rows: np.ndarray, hist_ptr: np.ndarray, hist_idx: np.ndarray) -> np.ndarray:
    """Mean article features of each user's click history (rows of the split's user table)."""
    flat = user_rows.ravel()
    lens = hist_ptr[flat + 1] - hist_ptr[flat]
    starts = hist_ptr[flat]
    idx = np.concatenate([hist_idx[s:s + n] for s, n in zip(starts, lens)])
    owner = np.repeat(np.arange(len(flat)), lens)
    out = np.zeros((len(flat), X.shape[1]), np.float32)
    np.add.at(out, owner, X[idx])
    out /= np.maximum(lens, 1)[:, None]
    return out.reshape(*user_rows.shape, X.shape[1])


class Policy:
    def __init__(self, S: int, M: int, D: int, device, gen: torch.Generator):
        self.S, self.M, self.D, self.device, self.gen = S, M, D, device, gen
        self.rows = torch.arange(S, device=device)


class BetaTS(Policy):
    name = "beta_ts"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.wins = torch.zeros(self.S, self.M, device=self.device)
        self.pulls = torch.zeros_like(self.wins)

    def select(self, live, phi):
        w, n = self.wins.gather(1, live), self.pulls.gather(1, live)
        a = torch.distributions.Gamma(1 + w, torch.ones_like(w)).sample()
        b = torch.distributions.Gamma(1 + n - w, torch.ones_like(w)).sample()
        return (a / (a + b)).argmax(1)

    def update(self, arm_ids, phi_chosen, r):
        self.pulls[self.rows, arm_ids] += 1
        self.wins[self.rows, arm_ids] += r


class LinearPolicy(Policy):
    """Shared ridge model with Sherman-Morrison updates of A^-1 (float64)."""

    def __init__(self, S, M, D, device, gen, lam: float = 1.0):
        super().__init__(S, M, D, device, gen)
        eye = torch.eye(D, device=device, dtype=torch.float64)
        self.A_inv = (eye / lam).expand(S, D, D).clone()
        self.b = torch.zeros(S, D, device=device, dtype=torch.float64)

    def theta(self):
        return torch.einsum("sij,sj->si", self.A_inv, self.b)

    def update(self, arm_ids, phi_chosen, r):
        x = phi_chosen.double()
        Ax = torch.einsum("sij,sj->si", self.A_inv, x)
        self.A_inv -= torch.einsum("si,sj->sij", Ax, Ax) / (1 + (x * Ax).sum(1))[:, None, None]
        self.b += r.double()[:, None] * x


class LinUCB(LinearPolicy):
    def __init__(self, *a, alpha: float, **k):
        super().__init__(*a, **k)
        self.alpha, self.name = alpha, f"linucb_{alpha}" if alpha > 0 else "greedy"

    def select(self, live, phi):
        phi = phi.double()
        mean = torch.einsum("skd,sd->sk", phi, self.theta())
        if self.alpha == 0:
            return (mean + 1e-12 * torch.rand(mean.shape, generator=self.gen, device=self.device,
                                               dtype=mean.dtype)).argmax(1)
        width = torch.einsum("skd,sde,ske->sk", phi, self.A_inv, phi).clamp(min=0).sqrt()
        return (mean + self.alpha * width).argmax(1)


class LinTS(LinearPolicy):
    def __init__(self, *a, v: float, **k):
        super().__init__(*a, **k)
        self.v, self.name = v, f"lints_{v}"

    def select(self, live, phi):
        cov = 0.5 * (self.A_inv + self.A_inv.transpose(1, 2))
        L = torch.linalg.cholesky(cov)
        z = torch.randn(self.S, self.D, generator=self.gen, device=self.device, dtype=torch.float64)
        theta = self.theta() + self.v * torch.einsum("sij,sj->si", L, z)
        return torch.einsum("skd,sd->sk", phi.double(), theta).argmax(1)


def make_policy(spec: str, S: int, M: int, D: int, device, seed: int):
    gen = torch.Generator(device=device).manual_seed(seed)
    kind, _, val = spec.partition("_")
    if spec == "beta_ts":
        return BetaTS(S, M, D, device, gen)
    if spec == "greedy":
        return LinUCB(S, M, D, device, gen, alpha=0.0)
    if kind == "linucb":
        return LinUCB(S, M, D, device, gen, alpha=float(val))
    if kind == "lints":
        return LinTS(S, M, D, device, gen, v=float(val))
    raise ValueError(spec)


def run(env: dict, art_x: np.ndarray | None, user_x: np.ndarray | None, spec: str,
        device: str | torch.device, seed: int = 0, new_window: int | None = None,
        n_marks: int = 30, n_contexts: int = 0) -> dict:
    """Play the churning pool. ``art_x`` (S, M, d) and ``user_x`` (S, U, d) are ignored by beta_ts."""
    device = torch.device(device)
    P = torch.from_numpy(env["P"]).to(device)
    S, U, M = P.shape
    K, T, life = env["n_live"], env["horizon"], env["lifetime"]
    step = max(1, life // K)
    new_window = new_window or max(1, life // 10)
    g = torch.Generator(device="cpu").manual_seed(env["seed"] + 7)
    user_seq = torch.randint(U, (S, T), generator=g).to(device)
    unif = torch.rand(S, T, generator=g).to(device)

    if art_x is not None:
        A = torch.from_numpy(art_x).to(device)
        Ux = torch.from_numpy(user_x).to(device)
        d = A.shape[-1]
    else:
        d = 0
    D = 1 + 2 * d
    policy = make_policy(spec, S, M, D, device, seed)
    rows = torch.arange(S, device=device)
    live = torch.arange(K, device=device).expand(S, K).clone()   # article index per slot
    born = torch.zeros(K, dtype=torch.long, device=device)       # round each slot's article went live
    next_article = K

    marks = np.unique(np.r_[np.geomspace(50, T, n_marks).astype(int), T])
    out = {k: np.zeros((S, len(marks))) for k in ("regret", "regret_new", "clicks", "rounds_new")}
    acc = {k: torch.zeros(S, device=device, dtype=torch.float64) for k in out}
    # Optionally keep (user, live set) at evenly spaced rounds, to re-score the final policy offline.
    ctx_rounds = set(np.linspace(0, T - 1, n_contexts).astype(int).tolist()) if n_contexts else set()
    ctx_users, ctx_live = [], []
    c = 0
    for t in range(T):
        if t > 0 and t % step == 0 and next_article < M:
            slot = (t // step - 1) % K
            live[:, slot] = next_article
            born[slot] = t
            next_article += 1
        u = user_seq[:, t]
        if t in ctx_rounds:
            ctx_users.append(u.cpu().numpy())
            ctx_live.append(live.cpu().numpy())
        p_live = P[rows[:, None], u[:, None], live]                     # (S, K)
        if d:
            a = A[rows[:, None], live]                                  # (S, K, d)
            ux = Ux[rows, u][:, None, :].expand_as(a)
            phi = torch.cat([torch.ones(S, K, 1, device=device), a, ux * a], dim=2)
        else:
            phi = None
        k = policy.select(live, phi)
        chosen = live[rows, k]
        p = p_live[rows, k]
        r = (unif[:, t] < p).float()
        policy.update(chosen, phi[rows, k] if d else None, r)

        best = p_live.max(1)
        is_new = (t - born[p_live.argmax(1)]) < new_window
        reg = (best.values - p).double()
        acc["regret"] += reg
        acc["regret_new"] += reg * is_new
        acc["rounds_new"] += is_new.double()
        acc["clicks"] += p.double()
        if t + 1 == marks[c]:
            for key in out:
                out[key][:, c] = acc[key].cpu().numpy()
            c += 1
    out["marks"] = marks
    if isinstance(policy, LinearPolicy):
        out["theta"] = policy.theta().cpu().numpy()                     # (S, 1 + 2d): [bias, a, u*a]
    if ctx_users:
        out["ctx_users"] = np.stack(ctx_users, 1)                       # (S, n)
        out["ctx_live"] = np.stack(ctx_live, 1)                         # (S, n, K)
    return out
