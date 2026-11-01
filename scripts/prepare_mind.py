"""Parse raw MIND TSVs into dataset/processed/{news.parquet, <split>_impressions.npz}.

    python -m scripts.prepare_mind
"""

import argparse
from pathlib import Path

from src.data.mind import (PROCESSED_DIR, RAW_DIR, SPLITS, impressions_path,
                           load_news, news_path, parse_behaviors)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    news = load_news(args.raw_dir)
    news.to_parquet(news_path(args.out_dir))
    index = dict(zip(news["news_id"], range(len(news))))
    print(f"news: {len(news):,} articles")

    for split in SPLITS:
        imps = parse_behaviors(args.raw_dir / split / "behaviors.tsv", index)
        imps.save(impressions_path(split, args.out_dir))
        n = len(imps.cand_label)
        print(f"{split}: {imps.n_impressions:,} impressions, {imps.n_users:,} user rows, "
              f"{n:,} candidates, CTR {imps.cand_label.mean():.4f}")


if __name__ == "__main__":
    main()
