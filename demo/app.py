"""Interactive demo: Bandits vs. A/B Testing for News Recommendation (CSE 592).

    streamlit run demo/app.py

Pages
    Overview              what the project asks and how to read the demo
    A/B test vs. bandits  Q1: race policies on real MIND headlines, watch traffic shift
    Play the bandit       visitors choose headlines themselves, then compare with Thompson sampling
    SAE bandit            Q2/Q3: contextual bandits on a churning pool; features behind each choice
    LLM agent             Q4: the agent's stated reasoning, with and without the statistics tool
    Results               headline numbers and figures from the final validation campaign

Every simulation here is live but uses the stored simulator probabilities (demo/data/), so it is
semi-synthetic in exactly the way the paper's results are. Tables and figures on the Results,
SAE and LLM pages are read directly from results/ and are not recomputed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo.engines import POLICY_NAMES, churn_run, explain_choice, posterior_stats, race  # noqa: E402

DATA = ROOT / "demo" / "data"
FINAL = ROOT / "results" / "final_validation_20261121"
COLORS = {
    "A/B test": "#d62728", "ε-greedy": "#9467bd", "UCB": "#ff7f0e",
    "Thompson sampling": "#2ca02c", "Discounted Thompson": "#17becf", "You": "#1f77b4",
    "Per-article TS (no features)": "#7f7f7f", "LinUCB + categories": "#bcbd22",
    "LinUCB + dense Gemma": "#1f77b4", "LinUCB + SAE features": "#e377c2",
}
CHURN_POLICIES = {   # name -> representation key in churn.npz (None = no features)
    "Per-article TS (no features)": None,
    "LinUCB + categories": "category",
    "LinUCB + dense Gemma": "dense",
    "LinUCB + SAE features": "sae",
}
SEMI_SYNTHETIC = ("All results are **semi-synthetic**: real MIND headlines and click logs train a click "
                  "simulator, and the simulator's probabilities generate the clicks you see here.")

st.set_page_config(page_title="Bandits vs. A/B Testing", page_icon="📰", layout="wide")


# ----------------------------------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------------------------------

@st.cache_data
def load_headlines():
    with np.load(DATA / "headlines.npz") as f:
        mu = {5: f["K5_mu"], 10: f["K10_mu"]}
    meta = json.loads((DATA / "headlines.json").read_text())
    return mu, {5: meta["K5"], 10: meta["K10"]}


@st.cache_resource
def load_churn():
    with np.load(DATA / "churn.npz") as f:
        arrays = {k: f[k] for k in f.files}
    return arrays, json.loads((DATA / "churn.json").read_text(encoding="utf-8"))


@st.cache_data
def load_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


@st.cache_data
def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Simulating...")
def cached_race(K, pool, horizon, seed, ab_frac, eps_c, ucb_c, gamma, drift, lam, all_pools):
    mu, _ = load_headlines()
    m = mu[K] if all_pools else mu[K][pool:pool + 1]
    return race(m, horizon, seed, ab_frac, eps_c, ucb_c, gamma, drift, lam)


@st.cache_data(show_spinner="Running the contextual bandit...")
def cached_churn(pool, policy, horizon, seed):
    arrays, meta = load_churn()
    rep = CHURN_POLICIES[policy]
    P = arrays["P"][pool].astype(np.float64)
    if rep is None:
        art, usr, alpha = None, None, 0.0
    else:
        art, usr = arrays[f"{rep}_art"][pool], arrays[f"{rep}_user"][pool]
        alpha = float(meta["linucb"][rep].split("_")[1])     # tuned LinUCB weight (linucb_0.3)
    return churn_run(P, art, usr, alpha, meta["n_live"], meta["lifetime"], horizon, seed=seed)


def thin(T: int, n: int = 400) -> np.ndarray:
    return np.unique(np.linspace(0, T - 1, n).astype(int))


def mean_ci(x: np.ndarray):
    """Mean and 95% Student-t half-width over axis 0 (pools)."""
    if len(x) < 2:
        return x.mean(0), np.zeros(x.shape[1:])
    return x.mean(0), stats.t.ppf(0.975, len(x) - 1) * x.std(0, ddof=1) / np.sqrt(len(x))


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


# ----------------------------------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------------------------------

def page_overview():
    st.title("📰 Bandits vs. A/B Testing for News Recommendation")
    st.caption("Mansi Sethi · Samee Arif · Andrew Koss · University of Michigan · CSE 592")
    st.markdown(
        "A news site must decide which article to show each reader, and it only learns what works "
        "from the clicks it gets. **An A/B test** splits traffic evenly for a while, then ships the "
        "winner. **A bandit** shifts traffic toward better articles as evidence arrives. We ask:")
    cols = st.columns(4)
    qs = [("Q1 · Allocation", "How many clicks does an A/B test give up versus a bandit, and when is it competitive?"),
          ("Q2 · New articles", "Do text features help a bandit recommend articles it has never shown?"),
          ("Q3 · Interpretability", "Can sparse autoencoder (SAE) features tell us what the policy rewards?"),
          ("Q4 · LLM agents", "Can an LLM explore well from headlines, and does a statistics tool help?")]
    for c, (h, q) in zip(cols, qs):
        with c.container(border=True):
            st.markdown(f"**{h}**")
            st.write(q)
    st.subheader("How the simulator works")
    st.markdown(
        "1. A click model is fit on MIND-large logs (readers' click histories x article text).\n"
        "2. Its predicted click probabilities become the **ground truth** for every policy.\n"
        "3. A policy picks an article; a click is drawn with that probability; the policy sees only "
        "click or no click.\n"
        "4. **Regret** = expected clicks lost compared with always showing the best article. Lower is better.")
    st.info(SEMI_SYNTHETIC)
    st.subheader("Try it")
    st.markdown(
        "- **A/B test vs. bandits**: pick a pool of real headlines and watch each policy's traffic and regret evolve.\n"
        "- **Play the bandit**: choose headlines yourself, then see how you did against Thompson sampling.\n"
        "- **SAE bandit**: see which interpretable features pushed a contextual bandit toward each article.\n"
        "- **LLM agent**: read Gemma-3-12B's stated reasoning, with and without the statistics tool.\n"
        "- **Results**: the main numbers from the final validation campaign.")


# ----------------------------------------------------------------------------------------------
# Q1: A/B test vs. bandits
# ----------------------------------------------------------------------------------------------

def headline_table(titles, cats, mu=None, reveal=False):
    df = pd.DataFrame({"#": np.arange(1, len(titles) + 1), "Headline": titles, "Category": cats})
    if reveal and mu is not None:
        df["True click rate"] = [pct(x) for x in mu]
        df["Best"] = np.where(np.arange(len(mu)) == mu.argmax(), "★", "")
    return df


def page_race():
    st.title("A/B test vs. bandits")
    st.markdown("Each round one reader arrives and the policy shows one headline. All policies face "
                "**the same readers and the same coin flips**, so differences come from their choices alone.")
    mu_all, meta_all = load_headlines()
    with st.sidebar:
        st.header("Setup")
        K = st.segmented_control("Headlines in the pool", [5, 10], default=10)
        pool = st.number_input("Headline pool (1 to 30)", 1, 30, 1) - 1
        horizon = st.select_slider("Readers (rounds)", [1_000, 2_000, 5_000, 10_000, 20_000], value=10_000)
        all_pools = st.toggle("Average over all 30 pools", help="Mean and 95% interval across pools, as in the paper.")
        st.header("Stress tests")
        drift = st.toggle("Reader tastes change halfway", help="Click rates are permuted at the midpoint, so the best headline changes (Q1 abrupt drift).")
        lam = st.slider("Gap between headlines", 0.0, 1.0, 1.0, 0.05,
                        help="1 = simulator click rates. Smaller values shrink every headline toward the pool mean, making them harder to tell apart (milestone gap-contraction test).")
        with st.expander("Policy settings"):
            ab_frac = st.slider("A/B test length (% of readers)", 1, 50, 20) / 100
            ucb_c = st.select_slider("UCB exploration weight c", [0.01, 0.03, 0.1, 0.3, 1.0], value=0.1)
            eps_c = st.select_slider("ε-greedy decay c (ε_t = cK/t)", [1.0, 10.0, 30.0, 100.0], value=30.0)
            gamma = st.select_slider("Discounted TS memory γ", [0.999, 0.9995, 0.9999, 0.99999], value=0.9999,
                                     help="Old evidence is multiplied by γ every round, so the policy remembers roughly the last 1/(1-γ) readers.")
            seed = st.number_input("Random seed", 0, 10_000, 0)
            st.caption("Defaults follow the training-tuned settings for K=10, T=10,000 in results/q1/tuned.json, "
                       "except γ (tuned value 0.99999), which is shortened so drift adaptation is visible at demo horizons.")
    K = K or 10
    titles = [a["title"] for a in meta_all[K][pool]]
    cats = [a["category"] for a in meta_all[K][pool]]
    res = cached_race(K, pool, horizon, seed, ab_frac, eps_c, ucb_c, gamma, drift, lam, all_pools)
    row = pool if all_pools else 0
    mu = res["_mu"][row]

    reveal = st.toggle("Reveal true click rates (the policies never see these)")
    st.dataframe(headline_table(titles, cats, mu, reveal), hide_index=True, width="stretch")

    shown = st.multiselect("Policies", POLICY_NAMES, default=list(POLICY_NAMES))
    idx = thin(horizon)
    fig = go.Figure()
    summary = []
    for name in shown:
        reg = res[name]["regret"]
        m, h = mean_ci(reg[:, idx])
        if all_pools:
            fig.add_trace(go.Scatter(x=np.r_[idx, idx[::-1]] + 1, y=np.r_[m + h, (m - h)[::-1]], fill="toself",
                                     fillcolor=COLORS[name], opacity=0.15, line=dict(width=0),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=idx + 1, y=m, name=name, line=dict(color=COLORS[name], width=2.5)))
        fm, fh = mean_ci(reg[:, -1:])
        clicks = res[name]["clicks"].sum(1)
        summary.append({"Policy": name, "Regret (expected clicks lost)": f"{fm[0]:.1f}" + (f" ± {fh[0]:.1f}" if all_pools else ""),
                        "Clicks": f"{clicks.mean():.1f}", "CTR": pct(clicks.mean() / horizon)})
    if drift:
        fig.add_vline(x=res["_switch"], line_dash="dot", annotation_text="tastes change")
    fig.update_layout(title="Cumulative regret" + (" (mean and 95% interval over 30 pools)" if all_pools else ""),
                      xaxis_title="Readers so far", yaxis_title="Expected clicks lost", height=420,
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=50, b=10))
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig, width="stretch")
    c2.markdown("**After all readers**")
    c2.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    best_rate = res["_mu"][:, :].max(1).mean()
    c2.caption(f"Always showing the best headline would earn about {best_rate * horizon:.0f} expected clicks. "
               "Clicks are realized draws; regret uses expected values, so it is less noisy.")

    st.divider()
    st.subheader("Watch one policy learn")
    pol = st.segmented_control("Policy", shown or list(POLICY_NAMES), default=(shown or ["Thompson sampling"])[-2 if len(shown) > 1 else 0])
    pol = pol or "Thompson sampling"
    arms, clicks = res[pol]["arms"][row].astype(int), res[pol]["clicks"][row]
    play = st.button("▶ Play from the start")
    t_slot = st.empty()
    view = st.empty()

    def render(t: int):
        with view.container():
            onehot = np.eye(K)[arms[:t]]
            share_idx = thin(t, 150) if t > 1 else np.array([0])
            cum = onehot.cumsum(0)[share_idx]
            share = cum / (share_idx[:, None] + 1)
            left, right = st.columns([1, 1])
            f2 = go.Figure()
            for k in range(K):
                f2.add_trace(go.Scatter(x=share_idx + 1, y=share[:, k], stackgroup="one", name=f"#{k + 1}",
                                        hovertext=titles[k], line=dict(width=0.5)))
            f2.update_layout(title=f"Share of readers shown each headline (first {t:,} readers)", height=360,
                             yaxis=dict(range=[0, 1], tickformat=".0%"), xaxis_title="Readers so far",
                             margin=dict(t=50, b=10))
            left.plotly_chart(f2, width="stretch", key=f"share_{t}")
            shows = np.bincount(arms[:t], minlength=K)
            got = np.bincount(arms[:t], weights=clicks[:t], minlength=K)
            ps = posterior_stats(shows, got)
            nxt = arms[t] if t < len(arms) else None
            tab = pd.DataFrame({"#": np.arange(1, K + 1), "Shown": shows, "Clicks": got.astype(int),
                                "Observed CTR": [pct(c / n) if n else "n/a" for c, n in zip(got, shows)],
                                "Posterior mean": [pct(x) for x in ps["mean"]],
                                "95% upper bound": [pct(x) for x in ps["upper95"]],
                                "P(best)": [f"{x:.0%}" for x in ps["p_best"]],
                                "Next pick": ["👉" if k == nxt else "" for k in range(K)]})
            if reveal:
                tab["True rate"] = [pct(x) for x in mu]
            right.markdown(f"**What {pol} knows after {t:,} readers** (the Q4 statistics tool, Beta(1,1) prior)")
            right.dataframe(tab, hide_index=True, width="stretch", height=38 * (K + 1))
            recent = [f"{'✅' if clicks[i] else '·'} #{arms[i] + 1}" for i in range(max(0, t - 20), t)]
            st.caption("Last 20 recommendations (✅ = click): " + "  ".join(recent))

    if play:
        for t in np.unique(np.geomspace(10, horizon, 30).astype(int)):
            t_slot.markdown(f"Readers so far: **{t:,}**")
            render(int(t))
            time.sleep(0.25)
    else:
        t = t_slot.slider("Readers so far", 1, horizon, min(500, horizon), key="watch_t")
        render(t)
    if pol == "A/B test":
        st.caption(f"The A/B test shows headlines in turn for the first {max(1, int(ab_frac * horizon / K)) * K:,} readers, then commits to the best observed one.")


# ----------------------------------------------------------------------------------------------
# Play the bandit
# ----------------------------------------------------------------------------------------------

def new_game(rounds: int, boost: float):
    mu_all, meta_all = load_headlines()
    rng = np.random.default_rng()
    pool = int(rng.integers(30))
    seed = int(rng.integers(1_000_000))
    mu = np.minimum(mu_all[5][pool] * boost, 1.0)
    st.session_state.game = {
        "pool": pool, "seed": seed, "mu": mu, "rounds": rounds, "boost": boost,
        "titles": [a["title"] for a in meta_all[5][pool]],
        "u": np.random.default_rng(seed).random((1, rounds))[0],   # same coin flips race() will use
        "arms": [], "clicks": [], "order": rng.permutation(5),
    }


def page_play():
    st.title("Play the bandit")
    st.markdown("You run the top-story slot. Each button press is one reader. Pick a headline; you only "
                "learn whether that reader clicked. Try to collect as many clicks as you can.")
    with st.sidebar:
        rounds = st.select_slider("Readers per game", [20, 40, 80], value=40)
        boost = st.select_slider("Click-rate multiplier", [1, 3, 5], value=5,
                                 help="Real click rates are a few percent, so at 1x most games end with one or two clicks. "
                                      "The multiplier scales every headline equally to make short games playable.")
        if st.button("New game", type="primary") or "game" not in st.session_state:
            new_game(rounds, boost)
    g = st.session_state.game
    t = len(g["arms"])
    if g["boost"] > 1:
        st.caption(f"Click rates multiplied by {g['boost']}x for playability.")
    done = t >= g["rounds"]
    st.progress(t / g["rounds"], text=f"Reader {min(t + 1, g['rounds'])} of {g['rounds']} · clicks so far: {int(sum(g['clicks']))}")

    def choose(k):
        gg = st.session_state.game
        i = len(gg["arms"])
        if i < gg["rounds"]:
            gg["arms"].append(k)
            gg["clicks"].append(bool(gg["u"][i] < gg["mu"][k]))

    arms = np.array(g["arms"], int)
    clicks = np.array(g["clicks"], bool)
    if t and not done:
        st.toast("Click! ✅" if clicks[-1] else "No click")
    for k in g["order"]:
        n = int((arms == k).sum())
        c = int(clicks[arms == k].sum())
        col1, col2 = st.columns([5, 1])
        col1.button(g["titles"][k], key=f"h{k}", on_click=choose, args=(int(k),), disabled=done, width="stretch")
        col2.markdown(f"shown **{n}** · clicks **{c}**")

    if done:
        st.divider()
        st.subheader("How did you do?")
        mu = g["mu"]
        res = race(mu[None], g["rounds"], g["seed"], which=("Thompson sampling", "UCB"))
        best = mu.max() * g["rounds"]
        yours = mu[arms].sum()
        rows = [{"Player": "You", "Clicks": str(int(clicks.sum())), "Expected clicks": f"{yours:.2f}",
                 "Regret": f"{best - yours:.2f}"}]
        for name in ("Thompson sampling", "UCB"):
            a = res[name]["arms"][0].astype(int)
            rows.append({"Player": name, "Clicks": str(int(res[name]["clicks"][0].sum())),
                         "Expected clicks": f"{mu[a].sum():.2f}", "Regret": f"{best - mu[a].sum():.2f}"})
        rows.append({"Player": "Always the best headline", "Clicks": "", "Expected clicks": f"{best:.2f}", "Regret": "0"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Everyone faced the same readers and coin flips. Expected clicks remove luck: they add up the true click rate of each headline shown.")
        df = pd.DataFrame({"Headline": g["titles"], "True click rate": [pct(x) for x in mu],
                           "You showed it": np.bincount(arms, minlength=5),
                           "Thompson showed it": np.bincount(res["Thompson sampling"]["arms"][0], minlength=5)})
        st.dataframe(df.iloc[np.argsort(-mu)], hide_index=True, width="stretch")
        st.markdown("With few clicks, a headline that got lucky early can look best. Bandits keep some doubt "
                    "about headlines they have rarely shown; that is the exploration trade-off.")


# ----------------------------------------------------------------------------------------------
# Q2/Q3: SAE bandit on a churning pool
# ----------------------------------------------------------------------------------------------

def page_sae():
    st.title("New articles and SAE features")
    arrays, meta = load_churn()
    K, life = meta["n_live"], meta["lifetime"]
    st.markdown(f"**{K} articles are live at once**; every {life // K} readers the oldest retires and a brand-new "
                f"article arrives (each lives {life:,} readers). A policy without features must learn every new "
                "article from scratch. A **contextual bandit (LinUCB)** scores articles from their text features and "
                "the reader's history, so it can judge an article before showing it. Regret is measured against the "
                "best live article **for each reader**.")
    with st.sidebar:
        st.header("Setup")
        pool = st.number_input(f"Article stream (1 to {meta['n_pools']})", 1, meta["n_pools"], 1) - 1
        horizon = st.select_slider("Readers", [2_000, 5_000, 10_000, 20_000], value=10_000)
        seed = st.number_input("Random seed", 0, 10_000, 0, key="sae_seed")
        chosen = st.multiselect("Policies", list(CHURN_POLICIES), default=list(CHURN_POLICIES))
        st.caption("Features have 32 dimensions (categories: 18). LinUCB uses the tuned weight α = 0.3 "
                   "(results/q2/tuned.json). The dense and SAE features both come from Gemma-3-12B.")
    runs = {p: cached_churn(pool, p, horizon, seed) for p in chosen}

    idx = thin(horizon)
    fig, fig2, rows = go.Figure(), go.Figure(), []
    for p, r in runs.items():
        fig.add_trace(go.Scatter(x=idx + 1, y=r["regret"][idx], name=p, line=dict(color=COLORS[p], width=2.5)))
        fig2.add_trace(go.Scatter(x=idx + 1, y=r["regret_new"][idx], name=p, line=dict(color=COLORS[p], width=2.5)))
        rows.append({"Policy": p, "Regret": f"{r['regret'][-1]:.0f}", "of which on new articles": f"{r['regret_new'][-1]:.0f}",
                     "Expected CTR": pct(r["exp_ctr"])})
    for f, title in ((fig, "Cumulative regret vs. each reader's best live article"),
                     (fig2, f"Regret when the reader's best article is under {life // 10} readers old")):
        f.update_layout(title=title, height=360, xaxis_title="Readers so far", yaxis_title="Expected clicks lost",
                        legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=10))
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig, width="stretch")
    c2.plotly_chart(fig2, width="stretch")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("One article stream and one seed. The paper averages 30 streams: per-article TS 1,338 ± 28 vs. "
               "Gemma-3-12B dense LinUCB 754 ± 44 at 20,000 readers (results/q2/summary.md).")

    linear = [p for p in runs if CHURN_POLICIES[p]]
    if not linear:
        return
    st.divider()
    st.subheader("Why did the policy pick this article?")
    pol = st.segmented_control("Explain", linear, default="LinUCB + SAE features" if "LinUCB + SAE features" in linear else linear[0])
    pol = pol or linear[0]
    rep = CHURN_POLICIES[pol]
    r = runs[pol]
    art, usr = arrays[f"{rep}_art"][pool], arrays[f"{rep}_user"][pool]
    labels = meta["feature_labels"][rep]
    snaps = r["snaps"]
    i = st.slider("Reader number", 0, len(snaps) - 1, min(len(snaps) - 1, 40),
                  format="%d", help="Decisions are recorded every 100 readers.")
    s = snaps[i]
    st.markdown(f"**Reader {s['t'] + 1:,}** recently clicked:")
    for h in meta["reader_history"][pool][s["user"]] or ["(no history)"]:
        st.markdown(f"- {h}")
    alpha = float(meta["linucb"][rep].split("_")[1])
    titles, cats = meta["titles"][pool], meta["categories"][pool]
    best = int(s["p_live"].argmax())
    df = pd.DataFrame({
        "": ["👉 chosen" if k == s["choice"] else ("★ best" if k == best else "") for k in range(len(s["live"]))],
        "Headline": [titles[m] for m in s["live"]],
        "Total": (s["mean"] + alpha * s["width"]).round(4),
        "Predicted score": s["mean"].round(4), "Uncertainty bonus": (alpha * s["width"]).round(4),
        "True click prob. for this reader": [pct(x) for x in s["p_live"]],
        "Age (readers)": s["t"] - s["born"], "Category": [cats[m] for m in s["live"]],
    }).sort_values("Total", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("LinUCB shows the article with the highest predicted score plus uncertainty bonus. "
               "Newer articles get bigger bonuses because the model has seen fewer articles like them.")

    reasons = explain_choice(s, art, usr, labels, top=10)
    fr = go.Figure(go.Bar(x=[x["push"] for x in reasons][::-1], y=[x["feature"] for x in reasons][::-1], orientation="h",
                          marker_color=["#2ca02c" if x["push"] > 0 else "#d62728" for x in reasons][::-1]))
    fr.update_layout(title=f"Features that pushed '{titles[s['live'][s['choice']]][:70]}' above the other live articles",
                     xaxis_title="Contribution to predicted score, relative to the live-article average", height=380,
                     margin=dict(t=50, b=10))
    st.plotly_chart(fr, width="stretch")
    if rep == "sae":
        st.caption("Labels are Neuronpedia descriptions of Gemma Scope 2 SAE features (layer 41). They are automated "
                   "explanations, not checked by people, and a feature can matter without matching its label.")

    with st.expander("What this run learned to reward overall"):
        d = art.shape[1]
        th = r["theta"]
        tdf = pd.DataFrame({"Feature": labels, "Weight on article": th[1:1 + d], "Weight x reader history": th[1 + d:]})
        tdf["size"] = tdf[["Weight on article", "Weight x reader history"]].abs().max(axis=1)
        st.dataframe(tdf.sort_values("size", ascending=False).drop(columns="size").head(15).round(3), hide_index=True)
        st.caption("Positive 'x reader history' weights mean the policy shows articles with that feature to readers "
                   "who clicked similar articles before.")

    if rep == "sae":
        st.divider()
        st.subheader("Are these features really driving choices? (Q3 results)")
        w = load_csv("results/q3/g12b_sae_L41/weights.csv").head(10)
        st.dataframe(w.rename(columns={"rank_train": "Rank", "feature_id": "SAE feature", "label": "Label",
                                       "importance_train": "Importance (train)", "importance_test": "Importance (test)"})
                     [["Rank", "SAE feature", "Label", "Importance (train)", "Importance (test)"]].round(4), hide_index=True)
        c = load_csv("results/final_validation_20261121/tables/q3_necessity_contrasts.csv")
        c = c[c.group == "g12b41"].copy()
        c["Top minus random removal (regret)"] = c.apply(lambda x: f"{x.top_minus_random:+.0f} ± {x.ci95:.0f}", axis=1)
        st.markdown("Removing the policy's top-ranked features versus the same number of random features "
                    "(positive = top features mattered more), 30 pools, Holm-adjusted p-values:")
        st.dataframe(c.rename(columns={"teacher": "Response model", "removed_coordinates": "Features removed", "p_holm": "Holm p"})
                     [["Response model", "Features removed", "Top minus random removal (regret)", "Holm p"]].round(4), hide_index=True)
        st.caption("Source: results/final_validation_20261121/tables/q3_necessity_contrasts.csv. These are algorithmic "
                   "interventions inside the simulator, not claims about what makes real readers click.")


# ----------------------------------------------------------------------------------------------
# Q4: LLM agent
# ----------------------------------------------------------------------------------------------

def page_llm():
    st.title("LLM agent")
    st.markdown("Gemma-3-12B-IT runs the top-story slot for 500 readers. Each round it sees the headlines and how "
                "often each was shown and clicked. With the **statistics tool** it may ask for each headline's "
                "posterior mean, 95% upper bound and probability of being best before choosing.")
    tr = load_json("results/q4/llm_gemma-3-12b-it_transcripts.json")
    mu_all, _ = load_headlines()
    with st.sidebar:
        K = st.segmented_control("Headlines", [5, 10], default=5, key="llm_k") or 5
        tool = st.toggle("Statistics tool available", value=True)
        run = st.radio("Recorded run", [1, 2], horizontal=True) - 1
    log = tr[f"K{K}_{'tool' if tool else 'notool'}"][run]
    mu = mu_all[K][run]
    st.caption("Transcripts were saved every 50 rounds of the original run (results/q4/llm_gemma-3-12b-it_transcripts.json). "
               f"Best headline in this pool: #{mu.argmax() + 1} ({pct(mu.max())} true click rate).")
    rounds = [x["round"] for x in log]
    rd = st.select_slider("Round", rounds, value=rounds[1])
    x = log[rounds.index(rd)]
    with st.expander("System prompt"):
        st.write(x["messages"][0]["content"])
    for m in x["messages"][1:]:
        role = "assistant" if m["role"] == "assistant" else "user"
        with st.chat_message(role, avatar="🤖" if role == "assistant" else ("🧮" if m["content"].startswith("Statistics tool") else "📰")):
            st.text(m["content"])
    ch = x["choice"]
    st.markdown(f"**Choice:** headline #{ch} (true click rate {pct(mu[ch - 1])}"
                f"{', the best one' if ch - 1 == mu.argmax() else ''}) · reader {'clicked ✅' if x['click'] else 'did not click'}")

    st.divider()
    st.subheader("Results over 30 pools, 500 rounds")
    q = load_csv("results/final_validation_20261121/tables/q4_matched_conditions.csv")
    q = q[q.condition == "original"].copy()
    q["Regret"] = q.apply(lambda r: f"{r.regret:.2f} ± {r.regret_ci95:.2f}", axis=1)
    q["Tool"] = q.configuration.str.endswith("_tool").map({True: "yes", False: "no"})
    q["K"] = q.configuration.str.extract(r"K(\d+)")[0].astype(int)
    q["tokens_per_1000"] = q.tokens_per_1000.round().astype(int)
    st.dataframe(q[["model", "K", "Tool", "Regret", "tokens_per_1000"]].rename(
        columns={"model": "Model", "tokens_per_1000": "Tokens per 1,000 decisions"}), hide_index=True)
    c = load_csv("results/final_validation_20261121/tables/q4_matched_contrasts.csv")
    c = c[(c.family == "optional_tool") & c.left.str.contains("/original/")].copy()
    c["Model"] = c.left.str.split("/").str[0]
    c["K"] = c.left.str.extract(r"K(\d+)_")[0].astype(int)
    c["Tool minus no tool (regret)"] = c.apply(lambda r: f"{r.difference:+.2f} ± {r.ci95:.2f}", axis=1)
    c["Holm p"] = c.p_holm_within_family.map(lambda p: f"{p:.4f}")
    st.markdown("**Does the tool help?** Paired differences over the same 30 pools (negative = tool helps):")
    st.dataframe(c[["Model", "K", "Tool minus no tool (regret)", "Holm p"]], hide_index=True)
    st.caption("In the paper's full comparison, no agent configuration has a resolved advantage over simply "
               "choosing the headline with the highest posterior mean (README, Agent comparisons).")
    fig = FINAL / "figures" / "matched_agent_baselines.png"
    if fig.exists():
        st.image(str(fig), caption=load_json("results/final_validation_20261121/figures/manifest.json")["figures"]
                 .get("matched_agent_baselines", {}).get("caption", ""))


# ----------------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------------

def page_results():
    st.title("Results from the final validation campaign")
    st.info(SEMI_SYNTHETIC + " ± is a 95% interval over 30 paired article/user pools.")
    sim = load_csv("results/final_validation_20261121/tables/temporal_simulator.csv")
    st.subheader("Is the click simulator realistic?")
    st.markdown("Fit on Nov 9-12, selected on Nov 13, calibrated on Nov 14, evaluated on the Nov 15 day of MIND-large.")
    st.dataframe(sim.round(4), hide_index=True)

    st.subheader("Q1 · Allocation (K=10, T=10,000, immediate feedback)")
    q1 = load_csv("results/final_validation_20261121/tables/q1_allocation.csv")
    q1 = q1[(q1.condition == "K10_T10k_T10000") & (q1.feedback_batch == 1)].copy()
    q1["Regret"] = q1.apply(lambda r: f"{r.regret:.0f} ± {r.regret_ci95:.0f}", axis=1)
    q1["Picked best at end"] = q1.terminal_correct.map(lambda x: f"{x:.0%}")
    st.dataframe(q1[["policy", "Regret", "Picked best at end"]].sort_values("policy"), hide_index=True)

    manifest = load_json("results/final_validation_20261121/figures/manifest.json")["figures"]
    for name, title in [("temporal_model_validation", "Simulator validation"),
                        ("feedback_retuning", "Q1 · Delayed feedback and retuning"),
                        ("matched_feature_controls", "Q2 · SAE vs. dense and random features"),
                        ("feature_interventions", "Q3 · Removing top features"),
                        ("matched_agent_baselines", "Q4 · Agents vs. direct baselines")]:
        path = FINAL / "figures" / f"{name}.png"
        if path.exists():
            st.subheader(title)
            st.image(str(path), caption=manifest.get(name, {}).get("caption", ""))


pages = [st.Page(page_overview, title="Overview", icon="🏠", default=True),
         st.Page(page_race, title="A/B test vs. bandits", icon="🏁", url_path="race"),
         st.Page(page_play, title="Play the bandit", icon="🎮", url_path="play"),
         st.Page(page_sae, title="SAE bandit", icon="🔍", url_path="sae"),
         st.Page(page_llm, title="LLM agent", icon="🤖", url_path="llm"),
         st.Page(page_results, title="Results", icon="📊", url_path="results")]
st.navigation(pages).run()
