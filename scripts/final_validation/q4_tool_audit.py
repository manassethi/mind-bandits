"""Reconstruct the tool's displayed statistics with each saved agent RNG stream."""

import json
import re
from pathlib import Path

import numpy as np

from scripts.final_validation.common import OUT, fingerprint, interval, is_complete, write_json
from src.agents.llm_agent import tool_output

PATTERN = re.compile(r"\[(\d+)\] posterior mean ([\d.]+)%, 95% upper bound ([\d.]+)%, probability best ([\d.]+)%")


def main():
    files = [Path("results/q4/llm_gemma-3-12b-it.npz"), Path("results/q4/envs.npz"),
             Path("results/q4/llm_gemma-3-12b-it_transcripts.json"), Path(__file__), Path("src/agents/llm_agent.py")]
    provenance = fingerprint({"draws": 4000, "score_precision": "exact displayed percentages", "restore_agent_rng": True}, files)
    path = OUT / "q4_tool_audit" / "summary.json"
    if is_complete(path, provenance):
        return
    saved, env = np.load(files[0]), np.load(files[1])
    transcripts = json.loads(files[2].read_text())
    output = {}
    for K in (5, 10):
        tag = f"K{K}_tool"
        choice = saved[f"{tag}/choice"]
        S, T = choice.shape
        counts = np.zeros(S)
        matches = np.zeros((S, 3))
        transcript_checks = 0
        for s in range(S):
            rng = np.random.default_rng(1000 * K + 10 * s + 1)
            shows, clicks = np.zeros(K), np.zeros(K)
            examples = {r["round"] - 1: r for r in transcripts[tag][s]} if s < len(transcripts[tag]) else {}
            for t in range(T):
                arm = int(choice[s, t])
                if saved[f"{tag}/tool_called"][s, t]:
                    text = tool_output(shows, clicks, rng)
                    values = np.array([[float(v) for v in match[1:]] for match in PATTERN.findall(text)])
                    assert values.shape == (K, 3)
                    matches[s] += np.isclose(values[arm], values.max(0), rtol=0, atol=1e-12)
                    counts[s] += 1
                    if t in examples:
                        stored = [m["content"] for m in examples[t]["messages"] if m["role"] == "user" and
                                  m["content"].startswith("Statistics tool output")]
                        if len(stored) != 1 or stored[0] != text:
                            raise RuntimeError(f"Tool RNG reconstruction disagrees with recorded transcript at {tag}/{s}/{t}")
                        transcript_checks += 1
                if saved[f"{tag}/parse_failed"][s, t]:
                    assert int(rng.integers(K)) == arm
                reward = env[f"K{K}_uniforms"][s, t] < env[f"K{K}_mu"][s, arm]
                assert reward == bool(saved[f"{tag}/reward"][s, t])
                shows[arm] += 1
                clicks[arm] += reward
        valid = counts > 0
        output[f"K{K}"] = {"displayed_statistic_argmax_agreement": {
            name: interval(matches[valid, j] / counts[valid]) for j, name in enumerate(("mean", "upper", "pbest"))},
            "sampled_tool_transcripts_verified_exactly": transcript_checks,
            "all_rewards_and_fallback_draws_verified": True, "tool_rounds": int(counts.sum())}
        print(K, "verified transcripts", transcript_checks,
              {k: round(v["mean"], 3) for k, v in output[f"K{K}"]["displayed_statistic_argmax_agreement"].items()}, flush=True)
    write_json(path, {"provenance": provenance, "results": output})


if __name__ == "__main__":
    main()
