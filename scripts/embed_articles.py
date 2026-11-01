"""Encode every article with the simulator's Qwen3-Embedding encoder.

    python -m scripts.embed_articles
"""

import argparse
from pathlib import Path

import numpy as np

from src.data.mind import load_processed_news
from src.simulator.encode import EMBED_PATH, encode_articles


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=EMBED_PATH)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    news = load_processed_news()
    emb = encode_articles(news, batch_size=args.batch_size, device=args.device)
    np.save(args.out, emb)
    print(f"saved {emb.shape} to {args.out}")


if __name__ == "__main__":
    main()
