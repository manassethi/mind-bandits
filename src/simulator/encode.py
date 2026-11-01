"""Simulator-side article encoder (Qwen3-Embedding-0.6B).

These embeddings define the simulator's ground truth and must never be given
to a policy, otherwise that policy would share the simulator's features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_PATH = Path("dataset/processed/sim_qwen3_embeddings.npy")
# Longest MIND article (title + abstract) is 674 Qwen3 tokens; the model allows 32k.
MAX_SEQ_LENGTH = 1024


def article_text(news: pd.DataFrame) -> list[str]:
    return [f"{t}. {a}" if a else t for t, a in zip(news["title"], news["abstract"])]


def encode_articles(news: pd.DataFrame, batch_size: int = 64, max_seq_length: int = MAX_SEQ_LENGTH,
                    device: str | None = None) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = SentenceTransformer(MODEL_NAME, device=device, model_kwargs={"torch_dtype": dtype})
    model.max_seq_length = max_seq_length
    texts = article_text(news)

    # Documents are encoded without an instruction prompt (Qwen3 uses prompts for queries only).
    lengths = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
    if max(lengths) > max_seq_length:
        raise ValueError(f"{sum(l > max_seq_length for l in lengths)} articles exceed "
                         f"max_seq_length={max_seq_length} (longest {max(lengths)} tokens)")
    print(f"token lengths: mean {np.mean(lengths):.1f}, max {max(lengths)}; no truncation")

    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=True)
    return emb.astype(np.float16)


def load_embeddings(path: Path = EMBED_PATH) -> np.ndarray:
    return np.load(path).astype(np.float32)
