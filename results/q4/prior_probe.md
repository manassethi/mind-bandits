# LLM headline prior vs real CTR on well-observed articles

Groups of 5 test-day articles, each with ≥ 100 impressions. Spearman is per group, averaged (± 95% CI). Chance for top pick = 20%.

| Groups | Ranker | Spearman vs real CTR | Top pick = real best |
|---|---|---|---|
| mixed | Simulator CTR | 0.44 ± 0.06 | 44% |
| mixed | LLM (gemma-3-12b-it) | 0.08 ± 0.07 | 28% |
| same category | Simulator CTR | 0.44 ± 0.06 | 42% |
| same category | LLM (gemma-3-12b-it) | 0.11 ± 0.07 | 29% |
