# Q4 results: an LLM as the bandit policy

Non-contextual bandits whose arms are real test-day headlines (true click rate = simulator average over 20k users). All policies see the same click draws. ± = 95% CI over runs; '− TS' is paired against Thompson sampling.

## 5 articles, 500 rounds

| Policy | Regret | − TS | Best-arm share (last 100) | Suffix failure | K·MinFrac | Stuck | Tool calls | Tokens / 1k decisions |
|---|---|---|---|---|---|---|---|---|
| LLM [gemma-3-12b-it] | 13.7 ± 4.8 | +4.2 ± 4.3 | 30% | 43% | 0.04 | 67% |  | 360,562 |
| LLM + tool [gemma-3-12b-it] | 6.0 ± 1.8 | -3.5 ± 2.2 | 73% | 3% | 0.24 | 3% | 51% | 728,257 |
| LLM prior only [gemma-3-12b-it] | 15.9 ± 5.2 | +6.4 ± 4.5 | 20% | 80% | 0.00 | 80% |  |  |
| Thompson sampling | 9.5 ± 1.8 | +0.0 ± 0.0 | 55% | 0% | 0.49 | 0% |  |  |
| UCB1 (c=0.1) | 8.1 ± 1.6 | -1.4 ± 1.8 | 58% | 7% | 0.33 | 10% |  |  |
| UCB1 (c=1) | 13.4 ± 2.6 | +3.9 ± 1.7 | 26% | 0% | 0.86 | 0% |  |  |
| ε-greedy (0.1) | 7.2 ± 2.6 | -2.4 ± 2.6 | 57% | 0% | 0.12 | 20% |  |  |
| Greedy | 9.9 ± 4.8 | +0.4 ± 4.8 | 43% | 53% | 0.08 | 53% |  |  |
| Uniform | 15.0 ± 3.1 | +5.4 ± 2.2 | 20% | 0% | 0.88 | 0% |  |  |

LLM prior only [gemma-3-12b-it]: headline ranking vs true click rate Spearman -0.03 ± 0.18; top-ranked headline is the best article in 20% of runs (chance 20%).

## 10 articles, 500 rounds

| Policy | Regret | − TS | Best-arm share (last 100) | Suffix failure | K·MinFrac | Stuck | Tool calls | Tokens / 1k decisions |
|---|---|---|---|---|---|---|---|---|
| LLM [gemma-3-12b-it] | 15.2 ± 4.2 | +0.5 ± 3.6 | 20% | 67% | 0.04 | 70% |  | 551,389 |
| LLM + tool [gemma-3-12b-it] | 9.3 ± 2.6 | -5.4 ± 2.8 | 49% | 0% | 0.30 | 3% | 33% | 932,264 |
| LLM prior only [gemma-3-12b-it] | 16.4 ± 4.7 | +1.7 ± 4.0 | 13% | 87% | 0.00 | 87% |  |  |
| Thompson sampling | 14.7 ± 2.6 | +0.0 ± 0.0 | 25% | 0% | 0.61 | 0% |  |  |
| UCB1 (c=0.1) | 9.9 ± 1.5 | -4.8 ± 2.1 | 37% | 3% | 0.40 | 3% |  |  |
| UCB1 (c=1) | 16.8 ± 3.1 | +2.1 ± 1.0 | 12% | 0% | 0.90 | 0% |  |  |
| ε-greedy (0.1) | 10.2 ± 3.3 | -4.5 ± 2.9 | 33% | 13% | 0.09 | 20% |  |  |
| Greedy | 13.4 ± 4.4 | -1.3 ± 3.8 | 30% | 70% | 0.05 | 70% |  |  |
| Uniform | 17.3 ± 3.1 | +2.6 ± 1.0 | 10% | 0% | 0.78 | 0% |  |  |

LLM prior only [gemma-3-12b-it]: headline ranking vs true click rate Spearman 0.05 ± 0.14; top-ranked headline is the best article in 13% of runs (chance 10%).

