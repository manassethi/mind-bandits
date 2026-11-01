"""Q4: an LLM as the bandit policy.

Each round the agent sees the K candidate headlines and a running summary of
how often each was shown and clicked (the "summarized history" setting of
Krishnamurthy et al., 2024), then picks one. The tool variant may first call a
statistics tool that returns, per article, the posterior mean click rate, a 95%
upper credible bound and the probability that the article is the best one.

The engine is any callable ``generate(conversations) -> [(text, prompt_tokens,
completion_tokens)]`` so the logic can be tested without a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

TOOL_CALL = "CALL_TOOL"
CHOICE_RE = re.compile(r"choice\s*[:=]\s*\[?\s*(\d+)", re.IGNORECASE)
RANK_RE = re.compile(r"ranking\s*:\s*([\d,\s]+)", re.IGNORECASE)


def system_prompt(K: int, T: int) -> str:
    return (f"You run the top story slot of a news website. Each round one reader visits and you choose "
            f"which of {K} articles to show them. The reader either clicks or does not. Your goal is to "
            f"get as many clicks as possible over all {T} rounds. Click rates are low (usually a few "
            f"percent), differ between articles, and you only learn them from the clicks you observe.")


def round_prompt(t: int, T: int, headlines: list[str], shows: np.ndarray, clicks: np.ndarray,
                 tool: bool) -> str:
    lines = [f"Round {t + 1} of {T}.", "", "Articles so far:"]
    for i, h in enumerate(headlines):
        rate = f"{clicks[i] / shows[i]:.1%}" if shows[i] else "n/a"
        lines.append(f"[{i + 1}] {h}  (shown {int(shows[i])} times, {int(clicks[i])} clicks, click rate {rate})")
    lines.append("")
    if tool:
        lines.append(f"You may call a statistics tool that returns, for each article, the posterior mean click "
                     f"rate, a 95% upper confidence bound, and the probability that it is the best article. "
                     f"To call it, reply with exactly {TOOL_CALL}. Otherwise, think briefly (at most two "
                     f"sentences) and end with a line 'Choice: <article number>'.")
    else:
        lines.append("Think briefly (at most two sentences) and end with a line 'Choice: <article number>'.")
    return "\n".join(lines)


def tool_output(shows: np.ndarray, clicks: np.ndarray, rng: np.random.Generator, draws: int = 4000) -> str:
    a, b = 1 + clicks, 1 + shows - clicks
    mean = a / (a + b)
    upper = stats.beta.ppf(0.975, a, b)
    samples = rng.beta(a[:, None], b[:, None], size=(len(a), draws))
    p_best = np.bincount(samples.argmax(0), minlength=len(a)) / draws
    rows = ["Statistics tool output (Beta(1,1) prior on each article's click rate):"]
    for i in range(len(a)):
        rows.append(f"[{i + 1}] posterior mean {mean[i]:.2%}, 95% upper bound {upper[i]:.2%}, "
                    f"probability best {p_best[i]:.0%}")
    rows.append("")
    rows.append("Now think briefly (at most two sentences) and end with a line 'Choice: <article number>'.")
    return "\n".join(rows)


def parse_choice(text: str, K: int) -> int | None:
    m = CHOICE_RE.findall(text)
    if m:
        k = int(m[-1]) - 1
        if 0 <= k < K:
            return k
    return None


def prior_prompt(headlines: list[str]) -> str:
    lines = ["Here are headlines that could be shown on a general news website's homepage:"]
    lines += [f"[{i + 1}] {h}" for i, h in enumerate(headlines)]
    lines += ["", "Rank them from most to least likely to be clicked by a typical reader. Reply with one line "
              "'Ranking: <numbers separated by commas>' and nothing else."]
    return "\n".join(lines)


def parse_ranking(text: str, K: int) -> list[int] | None:
    m = RANK_RE.search(text)
    if not m:
        return None
    order = [int(x) - 1 for x in re.findall(r"\d+", m.group(1))]
    seen = [k for i, k in enumerate(order) if 0 <= k < K and k not in order[:i]]
    return seen if len(seen) == K else None


@dataclass
class AgentRun:
    """One agent playing one bandit run."""

    headlines: list[str]
    mu: np.ndarray
    uniforms: np.ndarray
    tool: bool
    seed: int
    T: int = field(init=False)
    shows: np.ndarray = field(init=False)
    clicks: np.ndarray = field(init=False)

    def __post_init__(self):
        K = len(self.headlines)
        self.T = len(self.uniforms)
        self.shows, self.clicks = np.zeros(K), np.zeros(K)
        self.rng = np.random.default_rng(self.seed)
        self.log = {k: [] for k in ("choice", "reward", "tool_called", "parse_failed",
                                    "prompt_tokens", "completion_tokens")}
        self.transcript: list[dict] = []

    def base_messages(self, t: int) -> list[dict]:
        return [{"role": "system", "content": system_prompt(len(self.headlines), self.T)},
                {"role": "user", "content": round_prompt(t, self.T, self.headlines, self.shows, self.clicks, self.tool)}]

    def finish_round(self, t: int, text: str, tool_called: bool, ptok: int, ctok: int, keep: bool,
                     convo: list[dict]) -> None:
        K = len(self.headlines)
        k = parse_choice(text, K)
        failed = k is None
        if failed:  # unparseable answer: show a uniformly random article and record it
            k = int(self.rng.integers(K))
        r = float(self.uniforms[t] < self.mu[k])
        self.shows[k] += 1
        self.clicks[k] += r
        for key, v in (("choice", k), ("reward", r), ("tool_called", tool_called), ("parse_failed", failed),
                       ("prompt_tokens", ptok), ("completion_tokens", ctok)):
            self.log[key].append(v)
        if keep:
            self.transcript.append({"round": t + 1, "messages": convo + [{"role": "assistant", "content": text}],
                                    "choice": k + 1, "click": r})


def play_round(runs: list[AgentRun], t: int, generate, keep_transcript: bool = False) -> None:
    """Advance every run by one round, batching all LLM calls."""
    convos = [r.base_messages(t) for r in runs]
    first = generate(convos)
    second_idx, second_convos = [], []
    for i, (r, (text, pt, ct)) in enumerate(zip(runs, first)):
        if r.tool and TOOL_CALL in text and parse_choice(text, len(r.headlines)) is None:
            second_idx.append(i)
            second_convos.append(convos[i] + [{"role": "assistant", "content": TOOL_CALL},
                                              {"role": "user", "content": tool_output(r.shows, r.clicks, r.rng)}])
    second = dict(zip(second_idx, generate(second_convos))) if second_convos else {}
    for i, r in enumerate(runs):
        text, pt, ct = first[i]
        convo = convos[i]
        called = i in second
        if called:
            text2, pt2, ct2 = second[i]
            text, pt, ct = text2, pt + pt2, ct + ct2
            convo = second_convos[second_idx.index(i)]
        r.finish_round(t, text, called, pt, ct, keep_transcript, convo)
