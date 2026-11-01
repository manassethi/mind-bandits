# Headline prior and simulator vs real logged CTR (MIND test day)

Real CTR = test-day clicks / impressions, shrunk to the day's CTR (4.08%) with a Beta prior worth 20 impressions.

## 1. Within Q4 runs

| K | Simulator vs real (Spearman) | LLM prior vs real (Spearman) | Simulator's best = real best | LLM's top pick = real best | Chance | Median impressions per article |
|---|---|---|---|---|---|---|
| 5 | 0.18 ± 0.20 | -0.12 ± 0.18 | 37% | 3% | 20% | 22 |
| 10 | 0.20 ± 0.15 | -0.01 ± 0.09 | 27% | 13% | 10% | 23 |

## 2–3. All test-day articles with ≥ 100 impressions

2,125 articles. Real CTR deciles (10/50/90%): [0.0059, 0.0223, 0.074]; simulator population CTR: [0.0091, 0.0235, 0.0567].

| Comparison | Spearman |
|---|---|
| Simulator population CTR (random test readers) vs real CTR | 0.56 |
| Click model averaged over each article's actual logged readers vs real CTR | 0.63 (Pearson 0.55) |
