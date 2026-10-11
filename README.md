# News recommendation experiments

## Experiment 1: A/B testing versus Bernoulli bandits

From the project directory:

```sh
source env/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/prepare_simulator.py
python scripts/run_experiment.py
```

The preparation command fits a logistic click model on a random sample of 20,000 MIND-small training impressions (all their candidate labels are retained). It evaluates on 5,000 validation impressions and saves AUC, log loss, Brier score and observed/predicted CTR. No validation labels enter training. Pre-impression history supplies category affinity, subcategory affinity, history length and whether the candidate has already been read; candidate category/subcategory are also used. This is a lightweight starting model, not a validated high-fidelity recommender. It omits article text and may give similar scores to different articles.

The simulator takes the five most exposed articles in the sampled training data and averages predicted click probabilities over the first recorded histories of 1,000 sampled, unique validation users. This cohort is fixed. Each article becomes a stationary Bernoulli arm. The bandits receive only selected-arm rewards; they do not see model probabilities or user histories. Using validation histories defines the simulator's user population, not a separate policy generalization test. AUC evaluates the click model on logged candidates; it does not validate counterfactual predictions or simulator realism.

The runner compares explore-then-commit (the A/B baseline), epsilon greedy, UCB1 and Beta(1,1) Thompson sampling for 10,000 rounds across 30 seeds. Explore-then-commit explores equally for 1,000 rounds then permanently selects its estimated winner. This is a fixed-budget baseline, not a statistical significance test. Epsilon is fixed at 0.1. All methods share round/arm potential click outcomes per seed. Policy randomness uses separate reproducible streams.

Regret is cumulative expected regret: sum of the best fixed-arm probability minus the chosen-arm probability. The oracle curve is expected clicks, not realized clicks. Shading and summary intervals use 1.96 times the standard error across seeds (approximate 95% intervals); they do not capture click-model uncertainty. These are simulator results, not evidence of real deployment lift. No algorithm is guaranteed to win this particular horizon and environment.

Outputs:
- `models/click_model.joblib`: fitted model and feature vocabulary.
- `dataset/processed/q1_environment.json`: article pool and click probabilities.
- `results/metrics/click_model.json`: held-out model metrics.
- `results/q1/runs.csv`: per-seed clicks, CTR, regret and arm selections.
- `results/q1/summary.csv`: aggregate means, standard deviations and intervals.
- `results/q1/comparison.png`: clicks and regret curves.
- `results/q1/config.json`: complete settings and simulator snapshot.

Use more data for the main run:

```sh
python scripts/prepare_simulator.py --train-impressions 156965 --dev-impressions 73152 --cohort-size 5000
python scripts/run_experiment.py --rounds 50000 --seeds 50 --exploration-rounds 5000 --output-dir results/q1_long
```

Full preparation constructs sparse candidate features in batches but still uses more memory and time. Start with the default run. Change exploration budgets and epsilon for sensitivity analyses; keep the article pool fixed when comparing settings.

Run correctness checks:

```sh
python -m unittest discover -s tests -v
```

Dataset archives, models, environment and outputs are excluded from version control. The original PDF supplied in the conversation describes a different game-agent project; this implementation follows the pasted news-recommendation plan and subsequent requests.

## Experiment 1 robustness sweep

Prepare a bank of 50 popular training articles, then sample fixed pools of five:

```sh
python scripts/prepare_simulator.py --arms 50
python scripts/run_sweep.py
```

For all MIND-small training and validation impressions:

```sh
python scripts/prepare_simulator.py --full-data --arms 50 --cohort-size 5000
python scripts/run_sweep.py
```

Preparation now constructs sparse features in batches to reduce peak memory. Full data still takes more time and memory than the default sample. Preparation replaces the saved model/environment; existing experiment results retain their environment snapshot.

The default sweep uses five pools, 30 seeds, horizons of 1,000/10,000/50,000 rounds, A/B exploration fractions of 5%/10%/20%, and epsilon values of 0.01/0.05/0.1, plus UCB1 and Thompson sampling. All variants use identical pools and potential outcomes within a seed. Outcome prefixes and policy random streams are shared across horizons. Pools are sampled without replacement internally; pools can overlap. They come from the top 50 exposed articles, not the entire news catalog.

This is 73.2 million policy decisions and can take several minutes. Each pool/horizon completion is printed, and completed blocks are written to disk. This does not implement automatic resume.

For a faster check:

```sh
python scripts/run_sweep.py --pools 2 --rounds 1000 5000 --seeds 5 --output-dir results/q1_sweep_quick
```

Outputs in `results/q1_sweep/`:
- `runs.csv`: raw policy results for every pool, horizon and seed.
- `per_pool_summary.csv`: means and 95% Student-t confidence intervals for individual pools.
- `summary.csv`: means across pools and intervals across seed-level pool averages.
- `paired_comparisons.csv`: adaptive-policy click gains and regret reductions against each A/B setting, with paired 95% intervals. Positive values favor the adaptive policy; intervals containing zero do not establish a difference.
- `comparison.png`: mean click rate and regret per round at each horizon.
- `config.json`: arguments, complete article pools, model metrics and environment snapshot.

Aggregate intervals describe simulation randomness conditional on the selected pools and fitted model. They do not estimate uncertainty over all possible article pools or models. Inspect the per-pool table for variation between pools. The intervals are pointwise and do not correct for multiple comparisons. Report all settings; choosing the best setting from these results requires independent evaluation before claiming its superiority.

## Experiment 2: contextual bandits and policy cold start

Reuse the fitted Q1 click model. To prepare per-user probabilities and category/history features:

```sh
python scripts/prepare_contextual.py --articles 50 --users 1000
```

Quick check:

```sh
python scripts/run_contextual.py --rounds 2000 --seeds 3 --cold-window 200 --output-dir results/q2_quick
```

Main run:

```sh
python scripts/run_contextual.py --rounds 10000 --seeds 20
```

Policies: category-only LinUCB, category/history LinUCB, category/history linear Thompson sampling, noncontextual epsilon greedy, UCB1 and Bernoulli Thompson sampling. Shared linear models transfer evidence between articles; they use a bias, article-category one-hot features and category-specific history affinity interactions. All features have norm at most one. Linear Thompson sampling uses a Gaussian linear approximation to binary rewards, not a Bernoulli posterior. LinUCB alpha and linear Thompson scale default to 0.1; they are starting settings, not tuned optima. Change them with `--alpha`, `--ts-scale`; change the basic baseline with `--epsilon`.

A fixed pool of 20 articles is sampled from 50 popular training articles. Ten are eligible initially; ten additional articles enter halfway through. Each round samples a user uniformly from the fixed cohort and five candidate articles uniformly from the eligible pool. All policies receive identical candidate sets and potential click outcomes per seed. Each learns only from its selected reward, starting with no interaction training. History features remain fixed rather than updating from simulated clicks. Policies never receive simulator probabilities.

Regret compares each choice against the highest click probability for that user among that round's offered candidates. This differs from Q1's best fixed-arm comparator, so regrets from Q1 and Q2 are not directly comparable. The before/after candidate distribution also changes; before/after raw CTR alone does not measure cold-start improvement.

The cold-start definition is articles unseen by the policies, not unseen by the simulator's fitted click model. New articles are sampled without reference to oracle probabilities and are not forced into every post-introduction candidate slate. `runs.csv` includes new-article choices and opportunities. `cold_window` reports all decisions during the first 1,000 rounds after introduction (configurable), not just decisions involving new articles.

Outputs under `results/q2/`:
- `runs.csv`: per-seed results for all rounds, before introduction, after introduction and the early cold-start window.
- `summary.csv`: per-policy and per-phase means and 95% Student-t intervals.
- `paired_comparisons.csv`: paired click gains/regret reductions relative to noncontextual Thompson sampling; positive values favor the named policy.
- `comparison.png`: mean cumulative clicks and user-specific regret, with an introduction marker.
- `config.json`: settings, article identities, feature schema and fitted-model checksum.

Intervals describe simulation randomness conditional on this fixed article pool/cohort/model and are pointwise, without correction for multiple comparisons. The same category/history inputs also influence the simulator's probabilities, which can favor feature-based policies. These experiments test learning within this simulator, not whether contextual methods necessarily improve real recommendation traffic. Text embeddings, genuinely held-out article evaluation, more article pools and a richer independent reward model remain future work.

## Experiment 2 robustness sweep

Reuse the Q2 environment (no retraining needed):

```sh
python scripts/run_contextual_sweep.py
```

Defaults: five fixed 20-article pools, 10,000 rounds, 20 seeds, LinUCB alpha values 0.03/0.1/0.3 for category and history variants, linear Thompson scales 0.03/0.1/0.3, epsilon baselines 0.05/0.1, and UCB1/Bernoulli Thompson. This is 13 million policy decisions. Progress is printed per pool/seed and raw results are saved after each seed; automatic resume is not supported.

Quick check:

```sh
python scripts/run_contextual_sweep.py --pools 2 --rounds 2000 --seeds 3 --cold-window 200 --output-dir results/q2_sweep_quick
```

Outputs in `results/q2_sweep/`: `runs.csv`, `per_pool_summary.csv`, `summary.csv`, `paired_comparisons.csv`, `comparison.png` and `config.json`.

In addition to all/before/after/cold_window phases, `new_offered` evaluates all rounds containing at least one new article and `cold_new_offered` restricts those opportunities to the early cold-start window. These masks depend on offered candidates, not a policy's selection, so every method is evaluated on the same rounds. They measure performance on slates containing new articles; they do not isolate reward from choosing new articles. The selected-new-article count is recorded separately. Empty phases are omitted.

Every policy setting is directly compared against each epsilon baseline and basic Thompson sampling. Positive click/CTR gains and positive regret reductions favor the named policy. Intervals average pools within each seed first, then use a Student-t interval across seeds, conditional on these pools/cohort/model. Pools may overlap. Per-pool results show pool variation; the aggregate intervals do not quantify uncertainty over all possible pools. Intervals are pointwise without multiple-comparison correction. Treat settings discovered here as exploratory and evaluate selected settings on new pools/seeds before claiming superiority.

## Text-feature comparison under a text-aware reward model

This first text baseline uses **TF-IDF + truncated SVD**, not pretrained neural embeddings. It runs locally with existing dependencies and no external model downloads.

```sh
python scripts/prepare_text_simulator.py
python scripts/run_contextual_sweep.py --environment dataset/processed/q2_text_environment.npz --include-text --output-dir results/q2_text_sweep
```

Preparation fits an SGD logistic click model on 20,000 sampled training impressions over three passes and evaluates 5,000 sampled dev impressions. It includes hashed title/abstract terms, category/history features and cosine similarity between candidate lexical features and prior clicked-article lexical features. Binary labels are not class-balanced. Numerical and text features are built in batches. Training labels never enter history construction or latent policy representation fitting. AUC, log loss, Brier score and observed/predicted CTR are printed and saved in `results/metrics/text_click_model.json`. This richer feature set is not guaranteed to outperform the earlier click model; inspect held-out metrics before interpreting simulator experiments.

TF-IDF vocabulary and SVD are fitted on training-news titles/abstracts without click labels. Each article has a normalized 16-dimensional latent vector. The user's text profile averages vectors of prior clicked articles and is normalized. Text-policy features append the candidate vector and elementwise candidate/user interaction to the existing category/history block; those baseline features retain their original scale. Candidate and interaction features are divided by sqrt(2), giving each text block bounded norm. The text policy therefore has more dimensions and a larger total feature norm; sweep alpha rather than assuming equal alpha means identical exploration.

The reward model uses lexical hashes rather than the policy's SVD representation. This reduces direct representation matching but the simulator remains a learned, semi-synthetic reward model, not independent evidence of real-world text benefit. Cold articles remain unseen by policies, not necessarily by the reward model.

`--include-text` adds text LinUCB alongside category/history LinUCB at each alpha. Linear Thompson continues to use category/history features. All variants are evaluated on identical text-aware rewards, schedules and cold-start masks. The paired table also compares against category/history settings directly. To isolate added text, compare text and history variants at the same alpha; to choose an overall best method, tune each feature variant and confirm selected settings using fresh pools/seeds. Results from this reward model cannot be compared directly with absolute clicks/regret in the earlier simulator.

For all MIND-small labels (takes substantially longer):

```sh
python scripts/prepare_text_simulator.py --full-data
```

Quick end-to-end check:

```sh
python scripts/prepare_text_simulator.py --train-impressions 1000 --dev-impressions 500 --epochs 2 --users 100 --output dataset/processed/q2_text_quick_environment.npz
python scripts/run_contextual_sweep.py --environment dataset/processed/q2_text_quick_environment.npz --include-text --pools 2 --rounds 2000 --seeds 3 --cold-window 200 --output-dir results/q2_text_quick
```

The quick model is only a pipeline check, not the model for reported research results. Preparation saves `models/text_click_model.joblib`, `models/text_representation.joblib`, and the requested environment NPZ. These are separate from the existing category-only model and Q2 environment. The environment stores per-user probabilities, full policy features, feature boundaries, training settings, held-out metrics and reward-model checksum. Any text preparation run replaces the text model/representation/metric files, but already saved NPZs retain their frozen probabilities and provenance.

## Pretrained embeddings versus TF-IDF/SVD

Install the additional local inference dependencies, then extend the existing text environment without changing rewards:

```sh
python -m pip install -r requirements-embeddings-lock.txt
python scripts/prepare_embeddings.py
python scripts/run_contextual_sweep.py --environment dataset/processed/q2_embeddings_environment.npz --include-embeddings --output-dir results/q2_embeddings_sweep
```

Preparation uses frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings of article title/abstract text. It downloads the public pretrained model on first use; inference runs locally. SentenceTransformer chooses available hardware by default; `--device cpu` or `--device mps` overrides this. No model fine-tuning occurs. The resolved model commit is recorded and can be pinned with `--revision`.

A 16-dimensional PCA projection is fitted on 5,000 sampled training-news embeddings, without click labels. This matches the default TF-IDF/SVD dimensionality and keeps online matrix updates practical. Candidates and normalized mean-history profiles use the projected vectors. The added policy block consists of article embeddings and elementwise user/article interactions, scaled like the TF-IDF block. This tests compressed pretrained embeddings, not full 384-dimensional vectors. Vary `--dimensions` for a sensitivity study.

The exact source cohort is reconstructed by replaying source preparation sampling and checking its category/history features. All saved source probabilities and TF-IDF features are preserved unchanged. Required articles (candidates, user history and projection sample) are encoded and cached; no need to encode unused articles. Cache identity incorporates exact texts, article IDs, model ID and resolved commit.

The extended environment contains both representations. `--include-embeddings` automatically includes text variants. Text LinUCB sees history+TF-IDF/SVD only; embeddings LinUCB sees history+pretrained/PCA only, not both. Category/history and noncontextual baselines are retained. All policies share schedules and rewards. `paired_comparisons.csv` includes embeddings versus text at each exploration strength; inspect same-alpha comparisons, then validate selected settings on fresh pools/seeds.

Quick comparison after preparation:

```sh
python scripts/run_contextual_sweep.py --environment dataset/processed/q2_embeddings_environment.npz --include-embeddings --pools 2 --rounds 2000 --seeds 3 --cold-window 200 --output-dir results/q2_embeddings_quick
```

Confirmation after selecting settings:

```sh
python scripts/run_contextual_sweep.py --environment dataset/processed/q2_embeddings_environment.npz --include-embeddings --alphas 0.3 --ts-scales 0.1 --pool-seed 2027 --seed 1000 --output-dir results/q2_embeddings_validation
```

Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 . The fixed reward model uses lexical features, so it can favor lexical TF-IDF representations; better semantic embeddings are not guaranteed to win this simulator. Existing statistical and cold-start limitations still apply. SAE features and interpretability are not implemented in this step.

## SAE features and interpretability

Uses the existing embedding dependencies. Train a local top-k SAE on GPT-2 article activations and preserve the frozen simulator:

```sh
python scripts/prepare_sae.py
python scripts/run_contextual_sweep.py --environment dataset/processed/q2_sae_environment.npz --include-sae --output-dir results/q2_sae_sweep
python scripts/analyze_sae.py
```

Defaults: mean-pooled GPT-2 layer-8 hidden states for titles/abstracts truncated to 128 tokens; 5,000 sampled training-news articles, split 90/10 for SAE training/held-out reconstruction; 1,024 dictionary features with at most 16 active per article; 20 training epochs. The pretrained GPT-2 model is frozen and downloaded once. This is a locally trained SAE on pooled article states, not a published pretrained token-level SAE. No click labels train the SAE or select its policy features.

The policy uses the 32 dictionary features with greatest mean activation on SAE training articles; it appends article codes and user-history interactions to the existing category/history features. Thus the policy uses a selected subset of a larger sparse dictionary, not all 1,024 units. `selected_training_activation_mass_fraction` reports how much activation mass that selection retains; `training_active_dictionary_fraction` reports dead-feature coverage. Sparse codes are scaled by their training 95th-percentile norm and clipped to unit norm; no extra activations are introduced. A matched-width dense GPT-2/PCA control is fitted on the same training article states. It separates source-model effects from the sparse representation. TF-IDF and MiniLM retain their existing 16-dimensional projections while SAE and GPT-2 dense use 32 by default, so comparisons against those smaller representations also change feature capacity.

`--include-sae` automatically includes the existing representation variants. Each policy sees only its own representation block. All source probabilities and prior features remain unchanged. The paired table includes SAE comparisons against dense GPT-2, MiniLM, TF-IDF, category/history and basic bandits. SAE reconstruction quality is not itself proof of policy quality or interpretability.

Outputs:
- `models/gpt2_sae.pt`: SAE state and feature selection/scaling.
- `dataset/processed/q2_sae_environment.npz`: extended features, unchanged probabilities and metadata.
- `results/sae/reconstruction.json`: train/held-out reconstruction curve, activity and retained-mass diagnostics.
- `results/sae/feature_examples.json`: highly activating article examples for each selected unit, with numeric IDs rather than automatically assigned concept labels.
- `results/q2_sae_sweep/`: policy comparison outputs.
- `results/sae_interpretability/feature_importance.csv`: feature importance for one fitted bandit.
- `results/sae_interpretability/choice_explanations.json`: selected articles, linear predicted-reward contributions and separately reported uncertainty bonuses.
- `results/sae_interpretability/ablations.csv`: top-feature removal versus equal-size random-feature controls.

The interpretability runner fits one SAE LinUCB policy on a 10,000-round cold-start schedule, ranks features by mean absolute reward contribution on a separate diagnostic sample, freezes the policy, and evaluates on 2,000 fresh user/candidate contexts. Both content and user-interaction coordinates for a removed feature are zeroed; weights, covariance and reward probabilities are held fixed. It reports changed-choice frequency, expected CTR and expected regret. This is an input perturbation sensitivity analysis, not a causal intervention on article meaning, and does not retrain after removal. Reports from one fitted policy do not quantify seed uncertainty. Top-feature ranking uses predicted reward contributions; UCB uncertainty is not attributed to individual concepts. Review article examples before giving units semantic labels.

A lighter pipeline check (not a research-quality SAE):

```sh
python scripts/prepare_sae.py --train-articles 1000 --epochs 5 --cohort-limit 100 --output dataset/processed/q2_sae_quick_environment.npz
python scripts/run_contextual_sweep.py --environment dataset/processed/q2_sae_quick_environment.npz --include-sae --alphas 0.3 --ts-scales 0.1 --pools 2 --rounds 2000 --seeds 3 --cold-window 200 --output-dir results/q2_sae_quick
python scripts/analyze_sae.py --environment dataset/processed/q2_sae_quick_environment.npz --rounds 2000 --evaluation-contexts 500 --output-dir results/sae_interpretability_quick
```

Reconstruction/examples/model files are replaced by each preparation run; NPZ environments retain their own frozen features and metrics. Examine held-out reconstruction and dead-feature diagnostics before trusting feature interpretations. Model documentation: https://huggingface.co/docs/transformers/model_doc/gpt2 .

## Repeated SAE policy ablations

```sh
python scripts/run_sae_interpretability_sweep.py
```

Defaults: three fixed article pools, five fitted-policy seeds per pool, 10,000 training rounds, 2,000 fresh evaluation contexts per policy, top-three feature removal and ten random three-feature controls. Policy-training, policy-selection, importance-ranking, evaluation and random-control streams use distinct SeedSequence keys. Rankings are computed on diagnostic contexts before evaluation. Raw per-policy artifacts are retained under pool/seed subdirectories.

Quick check:

```sh
python scripts/run_sae_interpretability_sweep.py --pools 2 --seeds 3 --rounds 2000 --evaluation-contexts 500 --random-controls 3 --output-dir results/sae_interpretability_sweep_quick
```

Outputs in `results/sae_interpretability_sweep/`: `report.md`, `feature_review.md`, `runs.csv`, `summary.csv`, `per_pool_summary.csv`, `feature_stability.csv` and configuration. Positive `excess_ctr_drop` means top-feature removal reduces expected CTR more than random removal. CTR differences are absolute percentage points when displayed as percentages, not relative lift. Aggregate intervals average pools within each seed before a 95% Student-t interval across seeds. They do not cover SAE retraining uncertainty, all possible article pools, or real traffic.

Feature stability is descriptive frequency of ranking among the top removal features across fitted policies. The review lists highly activating article examples with numeric feature IDs and does not invent concept labels. Examples are omitted when the current SAE checkpoint does not match the environment. Frozen-policy perturbations and stable importance do not establish semantic disentanglement or causal concept interventions. No policies retrain after ablation.

## SAE redundancy and length diagnostics

```sh
python scripts/diagnose_sae.py
```

No SAE or bandit retraining. It uses the matching saved checkpoint, re-encodes article text, and checks that policy-pool SAE features reproduce the frozen environment before proceeding. The primary sample contains 1,000 articles found in the dev news catalog but absent from the train catalog; these were not used to train the SAE. The 50 policy articles form a separately reported stratum. GPT-2 states are cached for repeated diagnostics.

Outputs in `results/sae_diagnostics/`:
- `report.md`: highly correlated feature pairs and associations for frequently important units.
- `feature_pairs.csv`: raw/scaled correlations, Spearman correlation, active-set Jaccard overlap, top-article overlap and correlation after linear adjustment for effective token length and state norm.
- `feature_length_correlations.csv`: each unit's activation associations with full token length, truncated length and hidden-state norm.
- `heldout_examples.json`: activating examples on the unseen article sample.
- `article_measurements.csv`: lengths, norms and overall SAE activation for each article.
- `summary.json`: sample sizes, truncation rates, overall code associations and source-feature agreement.

These are descriptive diagnostics, not causal explanations or multiple-comparison-adjusted significance tests. Residual correlation removes linear associations only. Activation correlations and top examples can reflect article style, content, truncation or common factors. Constant/inactive features have undefined correlations. Top-article overlap excludes zero activations, uses the smaller active top-list size as denominator and is undefined when either list is empty. Feature examples support inspection, not verified concept labels.

Quick check:

```sh
python scripts/diagnose_sae.py --sample-articles 100 --output-dir results/sae_diagnostics_quick
```

If the saved SAE checkpoint differs from the environment, use its original checkpoint with `--checkpoint`; the script fails rather than combining incompatible features. `--max-tokens` must match extraction (128 by default); frozen feature reproduction checks catch mismatches. Current input texts are also checked indirectly by reproduced policy features.


### Normalized-state SAE control

Remove each pooled GPT-2 state's magnitude by dividing by its L2 norm before fitting the SAE and matched dense PCA. Training-only centering and RMS scaling still follow this step. Cached raw states, sampled articles, rewards, cohort and existing representations remain the same. This tests magnitude sensitivity; it does not guarantee semantic features. Newly trained feature IDs are not directly comparable to the original dictionary. Reconstruction losses across the two input spaces are not directly comparable either.

```sh
python scripts/prepare_sae.py --state-normalization unit
python scripts/diagnose_sae.py --environment dataset/processed/q2_sae_normalized_environment.npz --checkpoint models/gpt2_sae_normalized.pt --output-dir results/sae_normalized_diagnostics
python scripts/run_contextual_sweep.py --include-sae --environment dataset/processed/q2_sae_normalized_environment.npz --output-dir results/q2_sae_normalized_sweep
```

The control uses separate environment, checkpoint and reconstruction/example directories by default; original SAE artifacts remain available. Explicit `--checkpoint`, `--report-dir` and `--output` customize destinations. Diagnostics reproduce the checkpoint preprocessing and retain raw state norm for association checks. Compare normalized results with the original sweep using identical seeds, pools, horizons and hyperparameters. Inspect redundancy, length/norm associations and held-out examples before treating features as concepts.
