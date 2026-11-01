"""Q4: run LLM agents (with and without the statistics tool) on the Q4 environments.

All runs of all configurations advance together, one batched vLLM call per
round (two when tool calls happen). Also asks the model, once per run, to rank
the headlines by expected clicks (its prior, before any feedback).

    python -m scripts.run_q4_llm --model google/gemma-3-12b-it
    python -m scripts.run_q4_llm --mock          # engine-free smoke test
Writes results/q4/llm_<name>.npz, llm_<name>.json (config, prior rankings, cost) and
llm_<name>_transcripts.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scripts.build_q4_envs import ARMS, OUT
from src.agents.llm_agent import AgentRun, parse_ranking, play_round, prior_prompt


def mock_engine(seed: int = 0):
    rng = np.random.default_rng(seed)

    def generate(convos):
        out = []
        for c in convos:
            text = c[-1]["content"]
            K = text.count("\n[")
            if "Ranking" in text:
                out.append(("Ranking: " + ", ".join(map(str, rng.permutation(K) + 1)), 100, 10))
            elif "CALL_TOOL" in text and rng.random() < 0.5:
                out.append(("CALL_TOOL", 300, 3))
            else:
                out.append((f"Seems good. Choice: {rng.integers(1, K + 1)}", 300, 20))
        return out
    return generate


def vllm_engine(model: str, max_tokens: int, temperature: float, tp: int):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.88, tensor_parallel_size=tp, seed=0)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def generate(convos):
        outs = llm.chat(convos, params, use_tqdm=False)
        return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outs]
    return generate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/gemma-3-12b-it")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--horizon", type=int, default=None, help="override (smoke tests)")
    p.add_argument("--runs", type=int, default=None, help="override (smoke tests)")
    args = p.parse_args()
    name = "mock" if args.mock else args.model.split("/")[-1]
    generate = mock_engine() if args.mock else vllm_engine(args.model, args.max_tokens, args.temperature, args.tp)

    envs = np.load(OUT / "envs.npz")
    heads = json.loads((OUT / "envs.json").read_text())
    configs, runs = [], []
    for K in ARMS:
        mu, unif = envs[f"K{K}_mu"], envs[f"K{K}_uniforms"]
        n_runs = args.runs or len(mu)
        T = args.horizon or unif.shape[1]
        for tool in (False, True):
            cfg = {"K": K, "tool": tool, "runs": []}
            for s in range(n_runs):
                h = [a["title"] for a in heads[f"K{K}"][s]]
                runs.append(AgentRun(h, mu[s], unif[s, :T], tool, seed=1000 * K + 10 * s + tool))
                cfg["runs"].append(len(runs) - 1)
            configs.append(cfg)

    # Prior probe: one ranking per (K, run), before any feedback.
    prior = {}
    for K in ARMS:
        n_runs = args.runs or len(envs[f"K{K}_mu"])
        convos = [[{"role": "user", "content": prior_prompt([a["title"] for a in heads[f"K{K}"][s]])}]
                  for s in range(n_runs)]
        prior[f"K{K}"] = [parse_ranking(text, K) for text, _, _ in generate(convos)]

    T = runs[0].T
    t0 = time.time()
    for t in range(T):
        play_round(runs, t, generate, keep_transcript=(t % 50 == 0 or t == T - 1))
        if t % 50 == 0:
            print(f"round {t + 1}/{T}: {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0

    arrays, meta = {}, {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens,
                        "seconds_total": elapsed, "prior_rankings": prior, "configs": []}
    transcripts = {}
    for cfg in configs:
        tag = f"K{cfg['K']}_{'tool' if cfg['tool'] else 'notool'}"
        rs = [runs[i] for i in cfg["runs"]]
        for key in rs[0].log:
            arrays[f"{tag}/{key}"] = np.array([r.log[key] for r in rs])
        tok = arrays[f"{tag}/prompt_tokens"].sum() + arrays[f"{tag}/completion_tokens"].sum()
        meta["configs"].append({"tag": tag, "K": cfg["K"], "tool": cfg["tool"], "runs": len(rs),
                                "tokens_per_1k_decisions": float(tok / arrays[f"{tag}/choice"].size * 1000)})
        transcripts[tag] = [r.transcript for r in rs[:2]]
    np.savez(OUT / f"llm_{name}.npz", **arrays)
    (OUT / f"llm_{name}.json").write_text(json.dumps(meta, indent=1))
    (OUT / f"llm_{name}_transcripts.json").write_text(json.dumps(transcripts, indent=1))
    print(json.dumps({c["tag"]: c["tokens_per_1k_decisions"] for c in meta["configs"]}), f"{elapsed:.0f}s")


if __name__ == "__main__":
    main()
