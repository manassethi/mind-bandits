"""One-factor LLM exploration checks on the fixed, paired headline environments."""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from scripts.final_validation.common import OUT, SEED, fingerprint, is_complete, save_npz, write_json
from scripts.run_q4_llm import vllm_engine
from src.agents.llm_agent import AgentRun, parse_ranking, play_round, prior_prompt, tool_output

CONDITIONS = ("temperature0", "shuffle", "anonymous", "warmup", "always_stats", "original")


def statistics_messages(agent, round_index):
    """Supply statistics in the existing user turn, preserving role alternation."""
    conversation = agent.base_messages(round_index)
    conversation[-1]["content"] += "\n\n" + tool_output(agent.shows, agent.clicks, agent.rng)
    return conversation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, choices=range(len(CONDITIONS)), required=True)
    p.add_argument("--model", default="google/gemma-3-12b-it")
    p.add_argument("--revision", help="Immutable Hugging Face model and tokenizer commit")
    p.add_argument("--runs", type=int, choices=range(2, 31), default=30)
    p.add_argument("--horizon", type=int, choices=range(1, 501), default=500)
    args = p.parse_args()
    # Serving uses small CPU operations alongside the GPU; avoid spinning
    # multiple Torch threads against the allocation's CPU quota.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    condition = CONDITIONS[args.index]
    dest = OUT / "q4_ablation" / args.model.split("/")[-1] / condition
    marker = dest / "provenance.json"
    if marker.exists():
        recorded_revision = json.loads(marker.read_text())["config"].get("revision")
        if args.revision and args.revision != recorded_revision:
            raise ValueError("Requested revision differs from the saved run")
        revision = recorded_revision
    elif args.revision:
        revision = args.revision
    elif args.model == "google/gemma-3-12b-it":
        revision = "96b6f1eccf38110c56df3a15bffe176da04bfd80"
    elif args.model == "Qwen/Qwen3-8B":
        revision = "b968826d9c46dd6066d109eabc6255188de91218"
    else:
        from huggingface_hub import model_info
        revision = model_info(args.model).sha
    chat_kwargs = {"enable_thinking": False} if "qwen3" in args.model.lower() else None
    provenance = fingerprint({"model": args.model, "condition": condition, "runs": args.runs,
                              "horizon": args.horizon, "seed": SEED, "max_tokens": 160,
                              "chat_template_kwargs": chat_kwargs, "revision": revision, "serving_cpu_threads": 1},
                             [Path(__file__), Path("src/agents/llm_agent.py"),
                              Path("scripts/run_q4_llm.py"), Path("src/evaluation/bandit_metrics.py"),
                              Path("results/q4/envs.npz"), Path("results/q4/envs.json")])
    if is_complete(dest / "summary.json", provenance):
        return
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise RuntimeError("Partial LLM run has different settings")
    write_json(marker, provenance)
    env = np.load("results/q4/envs.npz")
    heads = json.loads(Path("results/q4/envs.json").read_text())
    agents, specifications = [], []
    for K in (5, 10):
        for tool in ([True] if condition == "always_stats" else [False, True]):
            for s in range(args.runs):
                rng = np.random.default_rng(SEED + K * 100 + s)
                permutation = rng.permutation(K) if condition == "shuffle" else np.arange(K)
                titles = [heads[f"K{K}"][s][i]["title"] for i in permutation]
                if condition == "anonymous":
                    titles = [f"Article {i + 1}" for i in range(K)]
                mu = env[f"K{K}_mu"][s, permutation]
                uniforms = env[f"K{K}_uniforms"][s, :args.horizon]
                if len(uniforms) != args.horizon:
                    raise ValueError("Horizon exceeds stored paired reward draws")
                agent = AgentRun(titles, mu, uniforms, tool and condition != "always_stats",
                                 seed=1000 * K + 10 * s + tool)
                agents.append(agent)
                specifications.append({"K": K, "tool": tool, "pool": s, "permutation": permutation.tolist()})
    generate = vllm_engine(args.model, max_tokens=160, temperature=0.0 if condition == "temperature0" else 1.0,
                            tp=1, chat_template_kwargs=chat_kwargs, revision=revision)
    import torch
    device = torch.cuda.get_device_properties(0)
    hardware = {"hostname": platform.node(), "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": int(device.total_memory), "torch_version": torch.__version__,
                "torch_cpu_threads": int(torch.get_num_threads())}
    elapsed_before = 0.0
    start_round = 0
    # Resume completed-round state without replaying earlier LLM decisions.
    state_path = dest / "checkpoint.npz"
    if state_path.exists():
        with np.load(state_path) as state:
            start_round = int(state["round"])
            elapsed_before = float(state["elapsed"])
            for i, agent in enumerate(agents):
                agent.shows[:] = state[f"{i}/shows"]
                agent.clicks[:] = state[f"{i}/clicks"]
                agent.rng.bit_generator.state = json.loads(str(state[f"{i}/rng"]))
                for key in agent.log:
                    agent.log[key] = state[f"{i}/{key}"].tolist()
    t0 = time.monotonic()
    for t in range(start_round, args.horizon):
        active = []
        for i, agent in enumerate(agents):
            K = specifications[i]["K"]
            if condition == "warmup" and t < K:
                agent.finish_round(t, f"Choice: {t + 1}", False, 0, 0, True, [])
            else:
                active.append(agent)
        if condition == "always_stats":
            conversations = [statistics_messages(agent, t) for agent in active]
            outputs = generate(conversations)
            for agent, convo, (response, prompt_tokens, completion_tokens) in zip(active, conversations, outputs):
                agent.finish_round(t, response, True, prompt_tokens, completion_tokens, t % 50 == 0, convo)
        elif active:
            play_round(active, t, generate, keep_transcript=t % 50 == 0 or t == args.horizon - 1)
        if (t + 1) % 25 == 0 or t + 1 == args.horizon:
            state = {"round": t + 1, "elapsed": elapsed_before + time.monotonic() - t0}
            for i, agent in enumerate(agents):
                state[f"{i}/shows"] = agent.shows
                state[f"{i}/clicks"] = agent.clicks
                state[f"{i}/rng"] = np.array(json.dumps(agent.rng.bit_generator.state))
                state.update({f"{i}/{key}": np.array(value) for key, value in agent.log.items()})
            save_npz(state_path, **state)
            print(condition, "round", t + 1, "seconds", round(float(state["elapsed"])), flush=True)
    decision_seconds = elapsed_before + time.monotonic() - t0
    prior_rankings, prior_raw, prior_tokens = {}, {}, 0
    prior_started = time.monotonic()
    if condition == "original":
        # Fresh, stateless prompts contain no feedback. Run after the action loop
        # to keep the action-generation RNG stream consistent across ablations.
        for K in (5, 10):
            conversations = [[{"role": "user", "content": prior_prompt([a["title"] for a in heads[f"K{K}"][s]])}]
                             for s in range(args.runs)]
            responses = generate(conversations)
            prior_rankings[f"K{K}"] = [parse_ranking(text, K) for text, _, _ in responses]
            prior_raw[f"K{K}"] = [text for text, _, _ in responses]
            prior_tokens += sum(pt + ct for _, pt, ct in responses)
    prior_seconds = time.monotonic() - prior_started
    arrays, configs = {}, []
    from src.evaluation.bandit_metrics import metrics
    for K in (5, 10):
        for tool in ([True] if condition == "always_stats" else [False, True]):
            indexes = [i for i, spec in enumerate(specifications) if spec["K"] == K and spec["tool"] == tool]
            tag = f"K{K}_{'tool' if tool else 'notool'}"
            for key in agents[0].log:
                values = np.array([agents[i].log[key] for i in indexes])
                if key == "choice":
                    values = np.stack([np.array(specifications[i]["permutation"])[values[j]] for j, i in enumerate(indexes)])
                arrays[f"{tag}/{key}"] = values
            result = metrics(arrays[f"{tag}/choice"], env[f"K{K}_mu"][:args.runs])
            tokens = arrays[f"{tag}/prompt_tokens"].sum() + arrays[f"{tag}/completion_tokens"].sum()
            configs.append({"tag": tag, **result, "tokens_per_1k": float(tokens / (args.runs * args.horizon) * 1000),
                            "parse_failure_rate": float(arrays[f"{tag}/parse_failed"].mean()),
                            "tool_call_rate": float(arrays[f"{tag}/tool_called"].mean())})
    save_npz(dest / "decisions.npz", **arrays)
    write_json(dest / "transcripts.json", {str(i): a.transcript for i, a in enumerate(agents) if specifications[i]["pool"] < 2})
    write_json(dest / "summary.json", {"provenance": provenance, "specifications": specifications,
        "seconds_total": elapsed_before + time.monotonic() - t0,
        "seconds_decisions": decision_seconds, "seconds_prior_probe": prior_seconds,
        "prior_rankings": prior_rankings, "prior_raw": prior_raw, "prior_probe_tokens": prior_tokens,
        "prior_probe_note": "Fresh headline-only prompts, generated after the decision loop with frozen weights and no feedback in the input.",
        "hardware": hardware, "model_revision": revision, "resumed_at_round": start_round,
        "resume_note": "Agent counts, decisions, and tool RNG resume exactly. vLLM generation RNG restarts after a resumed process.",
        "results": configs})


if __name__ == "__main__":
    main()
