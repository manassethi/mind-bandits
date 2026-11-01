# Q3 results: reading and testing the SAE-feature policy

LinUCB on SAE features (d = 32) in the Q2 main environment. Feature importance is ranked on train-day runs only; everything else is measured on the 30 test-day runs. Importance = how much a feature's score contribution differs across the live articles of a decision (a contribution shared by all articles cannot change the choice).

## Overview

| Layer | Test regret | Top-10 labeled | Importance rank: train vs test (Spearman) | Split-half stability (test) | Post-hoc: top-4 removal changes | Random-4 changes | Retrain: top-4 − random-4 regret |
|---|---|---|---|---|---|---|---|
| Gemma-3-12B, layer 41 (main) | 841 ± 39 | 8/10 | 0.87 | 0.89 | 46% of decisions | 42% | +6 ± 29 clicks |
| Gemma-3-12B, layer 31 | 966 ± 85 | 10/10 | 0.85 | 0.74 | 53% of decisions | 47% | -36 ± 30 clicks |
| Gemma-3-1B, layer 17 | 920 ± 55 | 6/10 | 0.78 | 0.58 | 42% of decisions | 45% | +17 ± 80 clicks |

## Gemma-3-12B, layer 41 (main)

### Top 10 features (train-ranked)

w_article: weight on the feature for everyone; w_personal: extra weight when the user's history has the feature. ± = 95% CI over test runs.

| Rank | Feature | Neuronpedia label | w_article | w_personal |
|---|---|---|---|---|
| 1 | 813 | sports | -0.165 ± 0.134 | +0.287 ± 0.084 |
| 2 | 922 | sports teams | +0.022 ± 0.072 | +0.192 ± 0.100 |
| 3 | 377 | code and URLs | +0.003 ± 0.049 | -0.074 ± 0.034 |
| 4 | 414 | foreign language | -0.052 ± 0.049 | -0.089 ± 0.035 |
| 5 | 780 | (unlabeled f780) | +0.071 ± 0.025 | +0.467 ± 0.053 |
| 6 | 54 | outperform others | -0.134 ± 0.069 | +0.047 ± 0.064 |
| 7 | 544 | asking questions | +0.006 ± 0.050 | -0.119 ± 0.040 |
| 8 | 4776 | football | +0.186 ± 0.086 | +0.575 ± 0.084 |
| 9 | 1876 | body and description | +0.007 ± 0.058 | +0.446 ± 0.112 |
| 10 | 486 | (unlabeled f486) | -0.004 ± 0.024 | -0.049 ± 0.039 |

### Post-hoc ablation (final test policies, recorded decisions)

| k removed | Top-k: decisions changed | Random-k: decisions changed | Top-k: clicks lost / 1k | Random-k: clicks lost / 1k | Top-k beats random draws |
|---|---|---|---|---|---|
| 1 | 37% | 18% ± 3% | 11.5 | 3.5 ± 1.2 | 94% |
| 2 | 39% | 28% ± 3% | 11.6 | 5.9 ± 1.4 | 82% |
| 4 | 46% | 42% ± 3% | 13.0 | 10.6 ± 1.9 | 66% |
| 8 | 71% | 57% ± 3% | 26.5 | 16.5 ± 2.5 | 92% |
| 16 | 92% | 71% ± 2% | 43.2 | 25.5 ± 2.9 | 100% |

### Retrain ablation (features removed before learning; paired regret increase)

| k removed | Top-k | Random-k (mean of 5 draws) | Top − random |
|---|---|---|---|
| 1 | +28 ± 32 | +21 ± 19 | +7 ± 23 |
| 4 | +40 ± 36 | +33 ± 26 | +6 ± 29 |
| 8 | +251 ± 51 | +139 ± 36 | +111 ± 51 |

### What the policy favors, by reader group

Everyone: highest effective weights. Groups (users' most-read category, ≥5% of users): concepts weighted more for that group than for everyone.

| Group | Share | Top concepts |
|---|---|---|
| everyone | 100% | football; sports or games; (unlabeled f780); professional sports; model |
| finance | 5% | Here; (unlabeled f469); (unlabeled f486); security indicators; outperform others |
| lifestyle | 11% | security indicators; emotive; (unlabeled f756); model; weather forecast |
| news | 46% | (unlabeled f780); Here; legislative; methods and techniques; elections and voting |
| sports | 16% | football; body and description; sports; sports teams; athletic actions |

Hand-check sheet: `results/q3/g12b_sae_L41/handcheck.csv`. Full weights: `results/q3/g12b_sae_L41/weights.csv`.

## Gemma-3-12B, layer 31

### Top 10 features (train-ranked)

w_article: weight on the feature for everyone; w_personal: extra weight when the user's history has the feature. ± = 95% CI over test runs.

| Rank | Feature | Neuronpedia label | w_article | w_personal |
|---|---|---|---|---|
| 1 | 221 | programming code examples | -0.020 ± 0.102 | -0.067 ± 0.029 |
| 2 | 300 | punctuation and labels | -0.110 ± 0.076 | -0.093 ± 0.027 |
| 3 | 107 | urls and code commands | +0.020 ± 0.094 | -0.082 ± 0.026 |
| 4 | 604 | team sports and players | -0.087 ± 0.086 | +0.530 ± 0.098 |
| 5 | 10119 | title clinched or cycle arrest | +0.175 ± 0.096 | +0.246 ± 0.083 |
| 6 | 1083 | play tic tac toe with | +0.049 ± 0.080 | +0.008 ± 0.036 |
| 7 | 342 | The neuron strongly activates on numeric tokens (e.g. years, percentages, counts, large numbers), so it’s detecting numbers/statistical values in the text. | -0.065 ± 0.025 | +0.172 ± 0.045 |
| 8 | 203 | win tournament | -0.126 ± 0.057 | +0.171 ± 0.074 |
| 9 | 4038 | football and bowl events | +0.050 ± 0.084 | +0.668 ± 0.120 |
| 10 | 249 | The neuron is simply a “sentence-position” detector – it has its strongest activations on the first few real tokens of a sentence (e.g. the initial determiner or auxiliary) and then decays as you move further into the sentence. | +0.026 ± 0.049 | -0.081 ± 0.046 |

### Post-hoc ablation (final test policies, recorded decisions)

| k removed | Top-k: decisions changed | Random-k: decisions changed | Top-k: clicks lost / 1k | Random-k: clicks lost / 1k | Top-k beats random draws |
|---|---|---|---|---|---|
| 1 | 36% | 17% ± 4% | 9.6 | 2.9 ± 1.0 | 94% |
| 2 | 37% | 30% ± 3% | 8.3 | 6.6 ± 1.4 | 72% |
| 4 | 53% | 47% ± 3% | 13.3 | 12.3 ± 1.7 | 72% |
| 8 | 67% | 63% ± 2% | 21.5 | 21.0 ± 1.7 | 66% |
| 16 | 80% | 76% ± 2% | 38.3 | 28.2 ± 1.8 | 78% |

### Retrain ablation (features removed before learning; paired regret increase)

| k removed | Top-k | Random-k (mean of 5 draws) | Top − random |
|---|---|---|---|
| 1 | -22 ± 63 | +6 ± 27 | -28 ± 43 |
| 4 | -10 ± 64 | +26 ± 46 | -36 ± 30 |
| 8 | +42 ± 70 | +56 ± 44 | -14 ± 42 |

### What the policy favors, by reader group

Everyone: highest effective weights. Groups (users' most-read category, ≥5% of users): concepts weighted more for that group than for everyone.

| Group | Share | Top concepts |
|---|---|---|
| everyone | 100% | title clinched or cycle arrest; combat sports; universities; football and bowl events; attacking opponent actions |
| finance | 5% | The neuron strongly activates on numeric tokens (e.g. years, percentages, counts, large numbers), so it’s detecting numbers/statistical values in the text.; The neuron flags unusually informative or low‐frequency “content” words (i.e. non‐function words that carry specific meaning).; news websites and sources; The neuron is triggered by numeric expressions and measurements (digits, quantities, and related numeric tokens).; programming code examples |
| lifestyle | 11% | news websites and sources; play tic tac toe with; exploit, abuse, endanger children; skating and hockey; fire and flammable risks |
| news | 46% | harm, unusual, suspect; The neuron strongly activates on numeric tokens (e.g. years, percentages, counts, large numbers), so it’s detecting numbers/statistical values in the text.; threatened, Power, Used, Simple, wave; political campaign campaigning; upcoming announcements and events |
| sports | 16% | team sports and players; football and bowl events; title clinched or cycle arrest; win tournament; attacking opponent actions |

Hand-check sheet: `results/q3/g12b_sae_L31/handcheck.csv`. Full weights: `results/q3/g12b_sae_L31/weights.csv`.

## Gemma-3-1B, layer 17

### Top 10 features (train-ranked)

w_article: weight on the feature for everyone; w_personal: extra weight when the user's history has the feature. ± = 95% CI over test runs.

| Rank | Feature | Neuronpedia label | w_article | w_personal |
|---|---|---|---|---|
| 1 | 561 | sports players | +0.085 ± 0.122 | +0.229 ± 0.080 |
| 2 | 4656 | HTTP requests and headers | +0.006 ± 0.085 | +0.289 ± 0.074 |
| 3 | 1009 | (unlabeled f1009) | -0.117 ± 0.061 | +0.171 ± 0.104 |
| 4 | 5173 | I can express capability | +0.091 ± 0.075 | +0.483 ± 0.092 |
| 5 | 2414 | coach of your team | +0.057 ± 0.065 | +0.487 ± 0.121 |
| 6 | 497 | (unlabeled f497) | -0.040 ± 0.070 | +0.317 ± 0.087 |
| 7 | 1284 | local city or county | +0.005 ± 0.029 | +0.225 ± 0.055 |
| 8 | 1509 | (unlabeled f1509) | -0.102 ± 0.088 | +0.049 ± 0.060 |
| 9 | 220 | tic tac toe game | -0.034 ± 0.031 | +0.033 ± 0.054 |
| 10 | 1014 | (unlabeled f1014) | +0.053 ± 0.058 | +0.230 ± 0.082 |

### Post-hoc ablation (final test policies, recorded decisions)

| k removed | Top-k: decisions changed | Random-k: decisions changed | Top-k: clicks lost / 1k | Random-k: clicks lost / 1k | Top-k beats random draws |
|---|---|---|---|---|---|
| 1 | 30% | 19% ± 2% | 6.8 | 4.1 ± 0.9 | 84% |
| 2 | 30% | 28% ± 2% | 4.3 | 6.6 ± 1.2 | 66% |
| 4 | 42% | 45% ± 2% | 9.4 | 13.0 ± 1.6 | 32% |
| 8 | 63% | 59% ± 2% | 21.1 | 17.5 ± 1.9 | 74% |
| 16 | 85% | 72% ± 2% | 37.4 | 26.1 ± 1.8 | 98% |

### Retrain ablation (features removed before learning; paired regret increase)

| k removed | Top-k | Random-k (mean of 5 draws) | Top − random |
|---|---|---|---|
| 1 | +94 ± 112 | +41 ± 57 | +52 ± 79 |
| 4 | +68 ± 80 | +51 ± 73 | +17 ± 80 |
| 8 | +121 ± 107 | +79 ± 75 | +42 ± 64 |

### What the policy favors, by reader group

Everyone: highest effective weights. Groups (users' most-read category, ≥5% of users): concepts weighted more for that group than for everyone.

| Group | Share | Top concepts |
|---|---|---|
| everyone | 100% | I can express capability; sports players; coach of your team; (unlabeled f1014); (unlabeled f69) |
| finance | 5% | code definitions; news articles and journalism; (unlabeled f325); housing and residences; (unlabeled f340) |
| lifestyle | 11% | government official communication; (unlabeled f194); news articles and journalism; housing and residences; emergency and aid |
| news | 46% | Error handling and successful operations; (unlabeled f692); code definitions; (unlabeled f69); government official communication |
| sports | 16% | coach of your team; I can express capability; HTTP requests and headers; (unlabeled f1014); sports players |

Hand-check sheet: `results/q3/g1b_sae_L17/handcheck.csv`. Full weights: `results/q3/g1b_sae_L17/weights.csv`.
