# Q3 results: reading and testing the SAE-feature policy

LinUCB on SAE features (d = 32) in the Q2 main environment. Feature importance is ranked on train-day runs only; everything else is measured on the 30 test-day runs. Importance = how much a feature's score contribution differs across the live articles of a decision (a contribution shared by all articles cannot change the choice).

Post-hoc random-control intervals summarize 50 feature-mask draws. Retraining and weight intervals summarize 30 evaluation pools. Post-hoc top-feature values are pooled mean-score diagnostics, not full pre-action UCB interventions.

## Overview

| Layer | Test regret | Top-10 labeled | Importance rank: train vs test (Spearman) | Split-half stability (test) | Post-hoc: top-4 removal changes | Random-4 changes | Retrain: top-4 − random-4 regret |
|---|---|---|---|---|---|---|---|
| Gemma-3-12B, layer 41 (main) | 841 ± 39 | 8/10 | 0.89 | 0.94 | 55% of decisions | 38% | +58 ± 30 clicks |
| Gemma-3-12B, layer 31 | 966 ± 85 | 10/10 | 0.91 | 0.87 | 36% of decisions | 44% | +5 ± 58 clicks |
| Gemma-3-1B, layer 17 | 920 ± 55 | 6/10 | 0.78 | 0.74 | 45% of decisions | 38% | -32 ± 53 clicks |

## Gemma-3-12B, layer 41 (main)

### Top 10 features (train-ranked)

w_article: weight on the feature for everyone; w_personal: extra weight when the user's history has the feature. ± = 95% CI over test runs.

| Rank | Feature | Neuronpedia label | w_article | w_personal |
|---|---|---|---|---|
| 1 | 813 | sports | -0.165 ± 0.134 | +0.287 ± 0.084 |
| 2 | 377 | code and URLs | +0.003 ± 0.049 | -0.074 ± 0.034 |
| 3 | 414 | foreign language | -0.052 ± 0.049 | -0.089 ± 0.035 |
| 4 | 780 | (unlabeled f780) | +0.071 ± 0.025 | +0.467 ± 0.053 |
| 5 | 922 | sports teams | +0.022 ± 0.072 | +0.192 ± 0.100 |
| 6 | 54 | outperform others | -0.134 ± 0.069 | +0.047 ± 0.064 |
| 7 | 544 | asking questions | +0.006 ± 0.050 | -0.119 ± 0.040 |
| 8 | 1275 | Here | -0.010 ± 0.030 | +0.229 ± 0.035 |
| 9 | 486 | (unlabeled f486) | -0.004 ± 0.024 | -0.049 ± 0.039 |
| 10 | 1876 | body and description | +0.007 ± 0.058 | +0.446 ± 0.112 |

### Post-hoc ablation (final test policies, recorded decisions)

| k removed | Top-k: decisions changed | Random-k: decisions changed | Top-k: clicks lost / 1k | Random-k: clicks lost / 1k | Top-k beats random draws |
|---|---|---|---|---|---|
| 1 | 34% | 16% ± 3% | 8.3 | 3.0 ± 1.0 | 96% |
| 2 | 44% | 25% ± 3% | 10.3 | 5.5 ± 1.2 | 100% |
| 4 | 55% | 38% ± 3% | 16.6 | 10.0 ± 1.7 | 98% |
| 8 | 65% | 54% ± 2% | 22.5 | 18.0 ± 2.3 | 92% |
| 16 | 84% | 71% ± 2% | 42.4 | 29.3 ± 2.5 | 100% |

### Retrain ablation (features removed before learning; paired regret increase)

| k removed | Top-k | Random-k (mean of 5 draws) | Top − random |
|---|---|---|---|
| 1 | +28 ± 32 | +21 ± 19 | +7 ± 23 |
| 4 | +91 ± 27 | +33 ± 26 | +58 ± 30 |
| 8 | +174 ± 48 | +139 ± 36 | +35 ± 45 |

### What the policy favors, by reader group

Everyone: highest effective weights. Groups (users' most-read category, ≥5% of users): concepts weighted more for that group than for everyone.

| Group | Share | Top concepts |
|---|---|---|
| everyone | 100% | football; sports or games; (unlabeled f780); professional sports; model |
| finance | 5% | Here; (unlabeled f469); (unlabeled f486); security indicators; outperform others |
| lifestyle | 11% | security indicators; emotive; (unlabeled f756); model; weather forecast |
| news | 46% | (unlabeled f780); Here; legislative; methods and techniques; elections and voting |
| sports | 16% | football; body and description; sports; sports teams; athletic actions |

Feature example sheet (not human annotations): `results/q3/g12b_sae_L41/handcheck.csv`. Full weights: `results/q3/g12b_sae_L41/weights.csv`.

## Gemma-3-12B, layer 31

### Top 10 features (train-ranked)

w_article: weight on the feature for everyone; w_personal: extra weight when the user's history has the feature. ± = 95% CI over test runs.

| Rank | Feature | Neuronpedia label | w_article | w_personal |
|---|---|---|---|---|
| 1 | 221 | programming code examples | -0.020 ± 0.102 | -0.067 ± 0.029 |
| 2 | 300 | punctuation and labels | -0.110 ± 0.076 | -0.093 ± 0.027 |
| 3 | 107 | urls and code commands | +0.020 ± 0.094 | -0.082 ± 0.026 |
| 4 | 1083 | play tic tac toe with | +0.049 ± 0.080 | +0.008 ± 0.036 |
| 5 | 604 | team sports and players | -0.087 ± 0.086 | +0.530 ± 0.098 |
| 6 | 203 | win tournament | -0.126 ± 0.057 | +0.171 ± 0.074 |
| 7 | 342 | The neuron strongly activates on numeric tokens (e.g. years, percentages, counts, large numbers), so it’s detecting numbers/statistical values in the text. | -0.065 ± 0.025 | +0.172 ± 0.045 |
| 8 | 249 | The neuron is simply a “sentence-position” detector – it has its strongest activations on the first few real tokens of a sentence (e.g. the initial determiner or auxiliary) and then decays as you move further into the sentence. | +0.026 ± 0.049 | -0.081 ± 0.046 |
| 9 | 25 | upcoming announcements and events | -0.092 ± 0.035 | +0.197 ± 0.048 |
| 10 | 10119 | title clinched or cycle arrest | +0.175 ± 0.096 | +0.246 ± 0.083 |

### Post-hoc ablation (final test policies, recorded decisions)

| k removed | Top-k: decisions changed | Random-k: decisions changed | Top-k: clicks lost / 1k | Random-k: clicks lost / 1k | Top-k beats random draws |
|---|---|---|---|---|---|
| 1 | 32% | 14% ± 3% | 4.6 | 2.4 ± 0.7 | 96% |
| 2 | 36% | 26% ± 3% | 6.7 | 5.5 ± 1.1 | 84% |
| 4 | 36% | 44% ± 3% | 6.0 | 10.5 ± 1.4 | 14% |
| 8 | 56% | 60% ± 2% | 12.8 | 19.1 ± 1.6 | 20% |
| 16 | 76% | 74% ± 1% | 29.2 | 27.0 ± 1.4 | 70% |

### Retrain ablation (features removed before learning; paired regret increase)

| k removed | Top-k | Random-k (mean of 5 draws) | Top − random |
|---|---|---|---|
| 1 | -22 ± 63 | +6 ± 27 | -28 ± 43 |
| 4 | +31 ± 32 | +26 ± 46 | +5 ± 58 |
| 8 | +15 ± 70 | +56 ± 44 | -41 ± 42 |

### What the policy favors, by reader group

Everyone: highest effective weights. Groups (users' most-read category, ≥5% of users): concepts weighted more for that group than for everyone.

| Group | Share | Top concepts |
|---|---|---|
| everyone | 100% | title clinched or cycle arrest; combat sports; universities; football and bowl events; attacking opponent actions |
| finance | 5% | The neuron strongly activates on numeric tokens (e.g. years, percentages, counts, large numbers), so it’s detecting numbers/statistical values in the text.; The neuron flags unusually informative or low‐frequency “content” words (i.e. non‐function words that carry specific meaning).; news websites and sources; The neuron is triggered by numeric expressions and measurements (digits, quantities, and related numeric tokens).; programming code examples |
| lifestyle | 11% | news websites and sources; play tic tac toe with; exploit, abuse, endanger children; skating and hockey; fire and flammable risks |
| news | 46% | harm, unusual, suspect; The neuron strongly activates on numeric tokens (e.g. years, percentages, counts, large numbers), so it’s detecting numbers/statistical values in the text.; threatened, Power, Used, Simple, wave; political campaign campaigning; upcoming announcements and events |
| sports | 16% | team sports and players; football and bowl events; title clinched or cycle arrest; win tournament; attacking opponent actions |

Feature example sheet (not human annotations): `results/q3/g12b_sae_L31/handcheck.csv`. Full weights: `results/q3/g12b_sae_L31/weights.csv`.

## Gemma-3-1B, layer 17

### Top 10 features (train-ranked)

w_article: weight on the feature for everyone; w_personal: extra weight when the user's history has the feature. ± = 95% CI over test runs.

| Rank | Feature | Neuronpedia label | w_article | w_personal |
|---|---|---|---|---|
| 1 | 561 | sports players | +0.085 ± 0.122 | +0.229 ± 0.080 |
| 2 | 4656 | HTTP requests and headers | +0.006 ± 0.085 | +0.289 ± 0.074 |
| 3 | 1284 | local city or county | +0.005 ± 0.029 | +0.225 ± 0.055 |
| 4 | 497 | (unlabeled f497) | -0.040 ± 0.070 | +0.317 ± 0.087 |
| 5 | 1509 | (unlabeled f1509) | -0.102 ± 0.088 | +0.049 ± 0.060 |
| 6 | 1009 | (unlabeled f1009) | -0.117 ± 0.061 | +0.171 ± 0.104 |
| 7 | 2414 | coach of your team | +0.057 ± 0.065 | +0.487 ± 0.121 |
| 8 | 220 | tic tac toe game | -0.034 ± 0.031 | +0.033 ± 0.054 |
| 9 | 1014 | (unlabeled f1014) | +0.053 ± 0.058 | +0.230 ± 0.082 |
| 10 | 5173 | I can express capability | +0.091 ± 0.075 | +0.483 ± 0.092 |

### Post-hoc ablation (final test policies, recorded decisions)

| k removed | Top-k: decisions changed | Random-k: decisions changed | Top-k: clicks lost / 1k | Random-k: clicks lost / 1k | Top-k beats random draws |
|---|---|---|---|---|---|
| 1 | 23% | 15% ± 2% | 6.4 | 3.1 ± 0.7 | 80% |
| 2 | 30% | 23% ± 2% | 9.9 | 4.2 ± 0.8 | 86% |
| 4 | 45% | 38% ± 2% | 12.4 | 8.9 ± 1.2 | 88% |
| 8 | 57% | 53% ± 2% | 18.5 | 14.5 ± 1.4 | 88% |
| 16 | 78% | 69% ± 1% | 35.0 | 24.5 ± 1.2 | 100% |

### Retrain ablation (features removed before learning; paired regret increase)

| k removed | Top-k | Random-k (mean of 5 draws) | Top − random |
|---|---|---|---|
| 1 | +94 ± 112 | +41 ± 57 | +52 ± 79 |
| 4 | +19 ± 66 | +51 ± 73 | -32 ± 53 |
| 8 | +113 ± 101 | +79 ± 75 | +34 ± 54 |

### What the policy favors, by reader group

Everyone: highest effective weights. Groups (users' most-read category, ≥5% of users): concepts weighted more for that group than for everyone.

| Group | Share | Top concepts |
|---|---|---|
| everyone | 100% | I can express capability; sports players; coach of your team; (unlabeled f1014); (unlabeled f69) |
| finance | 5% | code definitions; news articles and journalism; (unlabeled f325); housing and residences; (unlabeled f340) |
| lifestyle | 11% | government official communication; (unlabeled f194); news articles and journalism; housing and residences; emergency and aid |
| news | 46% | Error handling and successful operations; (unlabeled f692); code definitions; (unlabeled f69); government official communication |
| sports | 16% | coach of your team; I can express capability; HTTP requests and headers; (unlabeled f1014); sports players |

Feature example sheet (not human annotations): `results/q3/g1b_sae_L17/handcheck.csv`. Full weights: `results/q3/g1b_sae_L17/weights.csv`.
