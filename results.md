# Experiment Results

All results below are averages over **10,000 recommendation rounds per run**. Higher clicks and lower regret are better. Regret measures expected clicks lost relative to the simulator's best available choice.

Compare algorithms within each table: the simulator and evaluation setup differ between experiments. These are simulated recommendation results, not measured performance on live users. Bold values indicate the best averages, not necessarily statistically significant differences.

## Experiment 1: Fixed Article Click Probabilities

Each article has a fixed click probability averaged over a user cohort. Results use 30 random seeds.

| Algorithm | Mean clicks ↑ | Mean regret ↓ |
|---|---:|---:|
| Explore-then-commit | 464.43 | 99.10 |
| Epsilon-greedy | 508.70 | **53.27** |
| UCB1 | 394.40 | 164.34 |
| Thompson sampling | **509.33** | 54.84 |

## Experiment 2: Personalized Recommendations and New Articles

Click probabilities depend on the user and article. Candidate sets change, and new articles are introduced halfway through each run. These are the initial results using one article pool and 20 random seeds.

| Algorithm | Mean clicks ↑ | Mean regret ↓ |
|---|---:|---:|
| LinUCB — category | 882.60 | 143.38 |
| LinUCB — reading history | 879.15 | 146.29 |
| Linear Thompson — reading history | 885.80 | 132.10 |
| Epsilon-greedy | **897.25** | **127.97** |
| UCB1 | 650.50 | 367.62 |
| Thompson sampling | 886.70 | 133.90 |

## Text-Aware Simulator: Representation Comparisons

Results use five article pools and 20 random seeds. LinUCB rows use **α = 0.3**. Original and normalized GPT-2/SAE results come from separate sweeps with matching evaluation settings and frozen rewards.

| Algorithm / representation | Mean clicks ↑ | Mean regret ↓ |
|---|---:|---:|
| LinUCB — TF-IDF + SVD | **875.60** | **72.49** |
| LinUCB — normalized GPT-2 + PCA | 873.05 | 73.06 |
| LinUCB — GPT-2 + PCA | 871.51 | 74.96 |
| LinUCB — normalized SAE | 865.59 | 81.79 |
| LinUCB — SAE | 864.54 | 84.58 |
| LinUCB — MiniLM + PCA | 861.40 | 85.35 |
| Thompson sampling | 830.10 | 114.70 |
| Epsilon-greedy, ε = 0.1 | 801.18 | 145.85 |
| LinUCB — reading history | 729.66 | 216.75 |
| LinUCB — category | 705.65 | 241.97 |
| UCB1 | 699.34 | 247.37 |

TF-IDF + SVD LinUCB has the best average performance in this representation comparison. This finding applies to the fixed text-aware simulator and does not establish superiority on real users. The displayed hyperparameters were evaluated in sweeps; independent confirmation results are not included in these tables.

## Source Outputs

The tables reproduce the results previously reviewed in this project from:

- `results/q1/summary.csv`
- `results/q2/summary.csv`
- `results/q2_sae_sweep/summary.csv`
- `results/q2_sae_normalized_sweep/summary.csv`

Paired comparisons and confidence intervals are stored in the corresponding experiment output folders. They should be used when assessing differences between policies, rather than ranking averages alone.
