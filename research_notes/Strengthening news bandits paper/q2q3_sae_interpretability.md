# SAE features for bandit decision-making (Q2) and faithfulness of feature-based explanations (Q3)

Scope note: these notes come from 17 search/fetch calls (Oct 2026). Items marked **[UNVERIFIED]** come from background knowledge and were not confirmed in this session. Check them before citing.

## 1. Do SAE features beat dense/linear baselines downstream? What baselines do reviewers expect?

### Takeaway
The 2025-2026 literature mostly agrees that SAE features lose to dense baselines (linear probes on raw activations) on supervised downstream tasks, and that SAEs often barely beat *random* SAE-like baselines. Our finding (SAE costs 22-35% more regret than same-dim PCA but beats no-features) fits this consensus, so a reviewer will treat it as expected, not surprising. What makes the paper credible is (a) a full baseline ladder, especially random-SAE baselines, and (b) a claim focused on what SAEs buy (interpretability or auditability) at that cost.

### Cited Findings
- Kantamneni, Engels, Rajamanoharan, Tegmark, Nanda, "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing" (ICML 2025, arXiv:2502.16681):
  - Setup: they test SAE probes against baselines in four regimes: data scarcity, class imbalance, label noise, and covariate shift.
  - Result: SAEs occasionally win on individual datasets, but ensembles that add SAEs to baselines do not consistently beat baseline-only ensembles.
  - Apparent SAE wins on spurious-correlation detection, dataset quality, and multi-token probes were matched by simple non-SAE baselines.
  - The authors call for rigorous downstream evaluation against strong baselines.
  - [arXiv](https://arxiv.org/abs/2502.16681); [PMLR](https://proceedings.mlr.press/v267/kantamneni25a.html)
- Smith, Rajamanoharan, Conmy, McDougall, Lieberum, Kramár, Shah, Nanda, "Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research" (GDM Mech Interp Progress Update #2, Alignment Forum, 26 Mar 2025):
  - On OOD harmful-intent probing on Gemma-2-9B-IT layer 20, dense linear probes reached about 0.999 AUROC. k-sparse SAE probes (k about 20) transferred worse OOD.
  - Single-latent SAE probes often could not fit the training set.
  - Finetuning the SAE on chat data closed about half the gap.
  - Probes on SAE *reconstructions* did worse than probes on the raw residual, which points to information loss.
  - SAEs were still useful for dataset debugging, e.g. finding a question-mark latent that separated the classes spuriously.
  - The authors recommend reporting frequency-weighted autointerp scores alongside uniform averages.
  - [Alignment Forum](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)
- Korznikov et al., "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?" (arXiv:2602.14111, Feb 2026; listed at ICML 2026, and a NeurIPS 2026 page also exists):
  - On synthetic data with ground-truth features, SAEs recover only 9% of the true features despite 71% explained variance.
  - They propose three random baselines that constrain decoder directions, encoder directions, or activation patterns to random values.
  - These baselines roughly match fully trained SAEs on interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal editing (0.73 vs 0.72).
  - The authors offer the baselines as a reusable protocol.
  - [arXiv](https://arxiv.org/html/2602.14111v1); [ICML 2026](https://icml.cc/virtual/2026/79403)
- Heap, Lawson, Farnik, Aitchison, "Sparse autoencoders can interpret randomly initialized transformers" (arXiv:2501.17727, Jan 2025, revised Jan 2026; no peer-reviewed venue listed):
  - SAEs trained on randomly initialized Pythia models often score similarly to SAEs on trained models, on both auto-interp and reconstruction metrics.
  - The authors recommend routine randomized baselines and measures of feature "abstractness".
  - [arXiv](https://arxiv.org/abs/2501.17727)
- Karvonen, Rager, et al., "SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability" (ICML 2025, arXiv:2503.09532):
  - Eight metrics covering interpretability, disentanglement, and practical uses such as unlearning; over 200 SAEs across 7 architectures.
  - Gains on proxy metrics do not reliably carry over to practical performance. For example, Matryoshka SAEs underperform on proxy metrics but win on disentanglement.
  - Results are browsable at neuronpedia.org/sae-bench.
  - [arXiv](https://arxiv.org/html/2503.09532v3); [PMLR](https://proceedings.mlr.press/v267/karvonen25a.html)
- Gemma Scope 2 (Google DeepMind, announced 19 Dec 2025):
  - Open SAEs and transcoders for the Gemma 3 family (270M-27B, pretrained and IT), with SAEs at three sites for every layer, plus crosscoders and CLTs for the small models.
  - The technical report is a PDF linked from the Hugging Face README. I found no arXiv version.
  - [HF](https://huggingface.co/google/gemma-scope-2); [DeepMind blog](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior); [InfoQ](https://infoq.com/news/2026/01/google-gemma-scope-2/)
- Gemma Scope 1 (Lieberum et al., arXiv:2408.05147): JumpReLU SAEs on Gemma 2. [arXiv](https://arxiv.org/pdf/2408.05147)
- Counterpoint: the position paper "Use Sparse Autoencoders to Discover Unknowns" (arXiv:2506.23845) argues that SAEs are weaker for acting on *known* concepts but especially powerful for *discovering unknown* ones. [alphaXiv](https://alphaxiv.org/abs/2506.23845)
- Counterpoint: one SAE-for-classification paper reports macro F1 > 0.8 with SAE features, beating hidden-state and bag-of-words baselines, with cross-model transfer from Gemma 2 2B to 9B-IT. I saw only the arXiv id in the search snippet (arXiv:2502.11367) and did not verify the title or details. [arXiv](https://arxiv.org/abs/2502.11367v1)

### Inferences
- Frame our Q2 result as an independent confirmation of Kantamneni et al. and the GDM negative results in a *sequential decision-making / bandit* setting. As far as I found, this is a new setting: all the cited negative results are for static probing.
- Baseline ladder reviewers will expect at the same d (16-128), with cost estimates:
  1. **Random-direction "SAE" baseline**, following Korznikov et al. Use JumpReLU with random encoder directions, thresholds calibrated to the same L0 (about 60), and the same mean-pool and top-variance selection. Cheap: no training, just a matmul over cached residuals.
  2. **SAE on a random-init Gemma-3-1B**, following Heap et al. Moderate cost, since it needs SAE training. Optional.
  3. **Random Gaussian projection** of the dense residual to d dims (Johnson-Lindenstrauss baseline). Trivial.
  4. **Top-variance raw residual neurons** (same selection rule as for SAE features). Trivial. This isolates "sparse basis vs selection rule".
  5. **ICA (FastICA) to d components** on pooled residuals. A sparse, linear, non-learned-dictionary alternative. Trivial.
  6. **SAE reconstruction projected by PCA**, i.e. dense info kept vs lost. Answers the GDM finding that reconstructions lose information.
  7. **PCA on the full 16k SAE code** (dense compression of sparse features). Separates "SAE information" from "variance top-k selection is a bad compressor".
  8. **Width and L0 sweep** using Gemma Scope 2 variants available on HF (e.g. 16k vs 65k/262k width, l0_small/medium/big). Moderate cost: needs re-encoding cached activations.
- Our current comparison may confound "SAE basis is worse" with "top-variance selection of 128 out of 16k sparse features is a poor dimensionality reducer". Baseline 7 plus supervised selection (section 5) addresses this directly and is probably the most important addition.

### Gaps
- I did not retrieve the Gemma Scope 2 technical report PDF, so the exact available widths/L0s per layer for 1B and 12B are unverified.
- Exact author list of Korznikov et al. beyond the first author is not recorded here.
- I found no SAE-vs-dense comparison in bandit/RL settings specifically.

## 2. SAEs and concept-based interpretable representations for recommender systems and user modeling

### Takeaway
Since late 2024 there is an active line of work applying SAEs to *recommender-model embeddings* (CF and sequential recommenders) for interpretability and steering. There is also older LLM-concept / natural-language-profile work. Nobody I found uses *LLM-SAE features of item text as bandit contexts*, so this is our novelty claim. These papers must be cited as related work.

### Cited Findings
- RecSAE, "Understanding Internal Representations of Recommendation Models with Sparse Autoencoders" (arXiv:2411.06112):
  - Applies SAEs to general, graph-based, and sequential recommenders and links latents to semantic concepts verified by human experts.
  - Leaves the original model's performance intact.
  - [arXiv](https://arxiv.org/html/2411.06112v3)
- "Extracting Interaction-Aware Monosemantic Concepts in Recommender Systems" (arXiv:2511.18024):
  - Trains an SAE on user/item embeddings with an objective routed through the frozen recommender, so latents still predict user-item affinities.
  - Latents map to genre, popularity, and temporal trends, and support filtering or promotion without retraining.
  - [arXiv](https://arxiv.org/abs/2511.18024)
- Steerable collaborative filtering (arXiv:2601.11182): an SAE disentangles the user embedding of a CF autoencoder, and the dimensions are labeled and exposed as user/editor controls. Title not verified. [arXiv](https://arxiv.org/pdf/2601.11182v1)
- Sber AI Lab (arXiv:2507.12202): SAEs on transformer sequential recommenders give directions that are more monosemantic than hidden units and can steer recommendations. [arXiv](https://arxiv.org/pdf/2507.12202v2)
- Movva, Peng, Garg, Kleinberg, Pierson, "Sparse Autoencoders for Hypothesis Generation" (HypotheSAEs, ICML 2025, arXiv:2502.04382). This is highly relevant because it uses **headline click data**.
  - Data and model: a TopK SAE trained on OpenAI text-embedding-3-small embeddings.
  - Feature selection: L1 regression with a binary search on λ to get exactly H features.
  - Labeling: GPT-4o labels each feature from 10 high- and 10 weak-activating texts, generating 3 candidates.
  - Fidelity scoring: F1 of GPT-4o-mini annotations vs the neuron's activations on 100 positive and 100 negative samples. The best candidate is kept.
  - On Upworthy A/B headlines (pairwise click-rate prediction) it reaches AUC 0.69 with 15 significant hypotheses, vs BERTopic 0.62, NLParam 0.66, HypoGeniC 0.65.
  - Example concept: "surprise or shock".
  - [arXiv](https://arxiv.org/html/2502.04382v1)
- LACE, a concept-value bottleneck for text recommendation with editable user concept profiles (SIGIR 2023, arXiv:2304.04250). [ar5iv](https://ar5iv.labs.arxiv.org/html/2304.04250)
- Natural-language user profiles for scrutable recommendation (arXiv:2402.05810); LangPTune, end-to-end-trained LLM language profiles (arXiv:2410.18870). [arXiv 2402.05810](https://arxiv.org/pdf/2402.05810); [HF 2410.18870](https://huggingface.co/papers/2410.18870)

### Inferences
- HypotheSAEs gives a direct, citable protocol for both supervised feature selection and in-domain relabeling with fidelity scoring, on click data. Mirror it in Q3.
- The recommender-SAE papers interpret the *recommender's* embeddings. Ours interprets the *content* representation that a linear policy consumes. Make this distinction explicitly, since our approach keeps the policy itself transparent (linear weights over concept features).

### Gaps
- I found no 2025 paper combining SAE features with contextual bandits or with MIND specifically. This is worth one more targeted search before submission.

## 3. Faithfulness of feature-based explanations: protocols and statistics

### Takeaway
Our post-hoc ablation (top-k vs random-k) plus retrain ablation is essentially a deletion test plus ROAR. Reviewers familiar with ROAR critiques will ask for:
- deletion *curves* (not single k) with random and bottom-k controls
- an "insertion"/keep-only-top-k complement
- paired statistics across runs
- awareness that retraining lets correlated features compensate

### Cited Findings
- Hooker, Erhan, Kindermans, Kim, ROAR (arXiv:1806.10758; NeurIPS 2019 **[venue UNVERIFIED in session]**):
  - Method: remove top-ranked features from train and test, retrain, and measure degradation. This avoids confounding with distribution shift from ablation.
  - Finding: many attribution methods were no better than a random assignment of importance.
  - [arXiv](https://arxiv.org/pdf/1806.10758v2)
- Song, Cha, Seo, "On Pitfalls of RemOve-And-Retrain" (arXiv:2304.13836): attributions with *less* information about the decision function can score better under ROAR (a "blurriness bias"). [arXiv](https://arxiv.org/html/2304.13836v4)
- GOAR (Park et al., arXiv:2407.12401): ROAR and ROAD can fail to remove important information because of redundancy, giving little degradation. [arXiv](https://www.arxiv.org/pdf/2407.12401)
- Korznikov et al. report "causal editing" scores in which random baselines match trained SAEs (0.73 vs 0.72). Causal-intervention faithfulness should therefore be compared against random-feature controls, not just reported in absolute terms. [arXiv](https://arxiv.org/html/2602.14111v1)
- **[UNVERIFIED]** Deletion/insertion curves (Petsiuk et al., RISE, BMVC 2018, arXiv:1806.07421) are the standard "area under deletion curve" metric.
- **[UNVERIFIED]** Causal scrubbing (Chan et al., Redwood Research, LessWrong/AF 2022) is a hypothesis-testing framework for mechanistic claims. It may be overkill for a linear policy.

### Inferences
- Since LinUCB is linear, the post-hoc ablation is *exactly* faithful by construction for the learned weights. The interesting faithfulness questions are therefore:
  1. Does importance rank predict *decision* change better than random? We already test this.
  2. Is the concept the policy relies on irreplaceable, or do correlated features compensate? This is the ROAR/retrain question, and GOAR's redundancy critique is exactly why at other layers our retrain effects are weak.
- Recommended reporting:
  - Full deletion curves over k ∈ {1, 2, 4, 8, 16, 32}, comparing top-k, random-k (≥20 draws), and bottom-k.
  - Area-between-curves (top minus random) with paired bootstrap CIs over the 30 runs.
  - A Wilcoxon signed-rank test per k, with Holm correction across k and layers.
  - A "keep-only-top-k" (insertion) arm for retraining: train with only the top-8 features and compare to random-8 and the full model.
  - A redundancy-aware ablation: remove top-k *plus* features with |corr| > 0.5 with them, or ablate in a whitened/PCA basis, to answer the GOAR critique.

### Gaps
- I did not find a bandit-specific faithfulness protocol in the 2023-2026 literature. Our "expected clicks lost" metric appears novel; present it as such.

## 4. Auto-interp label quality, domain shift, and in-domain relabeling

### Takeaway
There is a standard, cheap, open protocol for producing and *scoring* labels:
- Delphi (EleutherAI) detection and fuzzing scores
- HypotheSAEs fidelity F1

Neuronpedia's Gemma Scope 2 labels are LLM-generated on web text and were still being rolled out in early 2026. Relabeling our top features on MIND headlines and scoring both labels in-domain directly fixes our observed mismatch (formatting features with misleading web-text labels).

### Cited Findings
- Paulo, Mallen, Juang, Belrose, "Automatically Interpreting Millions of Features in Large Language Models" (ICML 2025, arXiv:2410.13928):
  - Open pipeline (Delphi) that both generates and scores labels.
  - **Detection** score: a scorer LLM decides which whole contexts activate a feature, given the label.
  - **Fuzzing** score: token-level marking of activating tokens, with 5 shuffled examples per prompt.
  - Also defines intervention, surprisal, and embedding scoring.
  - Cost: about $1300 (Llama 3.1) to $8500 (Claude 3.5 Sonnet) for 1.5M GPT-2 features.
  - Code: github.com/EleutherAI/delphi.
  - [arXiv](https://arxiv.org/pdf/2410.13928); [PMLR](https://proceedings.mlr.press/v267/paulo25a.html); [EleutherAI blog](https://blog.eleuther.ai/autointerp/)
- Neuronpedia's Gemma Scope 2 labels (e.g. Gemma 3 27B-IT pages) are generated by models such as gemini-2.5-flash-lite, with a "New Auto-Interp" option using claude-4-5-haiku. A rolling-release notice said label generation would finish by about 26 Jan 2026. [Neuronpedia Gemma Scope 2](https://www.neuronpedia.org/gemma-scope-2); [example feature](https://www.neuronpedia.org/gemma-3-27b-it/41-gemmascope-2-res-262k/78)
- HypotheSAEs in-domain relabeling protocol: GPT-4o proposes 3 labels from 10 top and 10 weak examples, then GPT-4o-mini annotates 100 positive and 100 negative samples; label fidelity is the F1 against activations. [arXiv](https://arxiv.org/html/2502.04382v1)
- Jiang, Sun, Dunlap, Smith, Nanda, "Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit" (arXiv:2512.10092, ICML 2026 poster):
  - Reuses existing labels but *relabels* latents when needed: the top 200 latents for unlabeled dataset diffing.
  - Relabeling with in-distribution data (Pile, LMSYS) improved retrieval on similar distributions. This is direct evidence that domain-matched labels matter.
  - [arXiv](https://arxiv.org/html/2512.10092v1)
- GDM recommends frequency-weighted autointerp scores alongside uniform averages. [AF](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)
- Heap et al. show that high aggregate autointerp scores also occur for SAEs on random transformers, so autointerp scores need a random baseline. [arXiv](https://arxiv.org/abs/2501.17727)
- Other label-scoring work: Semantic Regexes, structured auto-interp descriptions (arXiv:2510.06378) [arXiv](https://arxiv.org/pdf/2510.06378); CE-Bench, contrastive evaluation of SAE interpretability (arXiv:2509.00691) [arXiv](https://arxiv.org/pdf/2509.00691)

### Inferences
- Concrete protocol for our paper (fits a 1-month budget):
  1. For the top-32 decision-important features per layer, collect the top-20 activating MIND articles at the document level (mean-pooled, as used by the policy) plus 20 random or low-activation ones.
  2. An LLM proposes an in-domain label (3 candidates).
  3. Score both the Neuronpedia label and the in-domain label using Delphi-style detection and HypotheSAEs-style F1 on held-out MIND articles.
  4. Report the fraction of top features whose Neuronpedia label scores at chance in-domain. This quantifies the "formatting features mislabeled" observation.
  5. Include random-feature and bottom-importance features as controls.
- Feasibility: Delphi can run with a local open model on the L40S, or via an API. Cost is negligible for about 100-200 features.

### Gaps
- Gemma Scope 2's own label-quality evaluation (if any) was not retrieved.
- I found no published study that quantifies auto-interp label degradation under domain shift. The Jiang et al. relabeling result is indirect evidence.

## 5. Pooling choices and feature selection for document-level SAE features

### Takeaway
Max-pooling is the choice of the most relevant recent "SAE features as document embeddings" work (Jiang et al.), and CE-Bench reports max pooling as best for its benchmark. Our mean-pooling plus unsupervised top-variance selection is the weakest-justified design choice and a cheap ablation target. Supervised selection (L1/Lasso, as in HypotheSAEs) is the established alternative.

### Cited Findings
- Jiang et al. 2025 build document embeddings by taking the **max** activation over tokens per latent (Goodfire SAE, Llama 3.3 70B layer 50, 65k width). [arXiv](https://arxiv.org/html/2512.10092v1)
- CE-Bench (arXiv:2509.00691) concludes max pooling is the best aggregation strategy for its contrastive interpretability benchmark (from the search snippet; not read in full). [arXiv](https://arxiv.org/pdf/2509.00691)
- A LessWrong thread on the predecessor post ("Towards data-centric interpretability with sparse autoencoders") asks why max-pooling was used. The answer was not retrieved. [LessWrong](https://www.lesswrong.com/posts/a4EDinzAYtRwpNmx9/towards-data-centric-interpretability-with-sparse)
- HypotheSAEs selects features with L1-regularized regression on the target, tuned to give exactly H features. [arXiv](https://arxiv.org/html/2502.04382v1)
- Kantamneni et al. likewise select SAE latents for sparse probes with supervised criteria **[details UNVERIFIED this session]**. [arXiv](https://arxiv.org/abs/2502.16681)

### Inferences
- Top-variance selection on mean-pooled codes is biased toward dense, high-frequency features (BOS, position, formatting), which matches our observation that many top features are formatting features. Mean-pooling also dilutes rare content features.
- Ablation grid (cached activations make this cheap):
  - Pooling: {mean, max, sum, mean-excluding-BOS, last-token}
  - Selection: {top-variance, top-frequency, mutual information with click on train logs, L1-logistic on train clicks, random}
  - d ∈ {16, 32, 64, 128}
  - Report regret relative to PCA.
- Also try **excluding BOS and position tokens** before pooling, and **dropping features with activation density > 50%**. Both are standard hygiene that reviewers may ask about.

### Gaps
- I found no paper directly comparing mean vs max vs last-token pooling of SAE codes for *downstream prediction* in a rigorous way.

## 6. Steering, causal interventions, and policy-relevant concepts (e.g. clickbait)

### Takeaway
I found no published work on SAE-based *clickbait/sensationalism* detection. HypotheSAEs on Upworthy headlines is the nearest neighbor: it surfaces "surprise/shock" concepts that predict clicks. Steering recommendations via SAE latents is shown for recommender-embedding SAEs. Steering a *bandit policy* via LLM-SAE concepts appears open.

### Cited Findings
- No published SAE clickbait/sensationalism detector was found. Adjacent work:
  - SAE features for artificial-text detection style analysis (ACL Findings 2025). [ACL Anthology](https://preview.aclanthology.org/watermark/2025.findings-acl.1321)
  - SAE detoxification with Neuronpedia-screened features, keeping 3-8 per layer (arXiv:2505.14536). [arXiv](https://arxiv.org/pdf/2505.14536)
- HypotheSAEs on Upworthy finds headline concepts such as surprise or shock that predict higher click rates (pairwise AUC 0.69). [arXiv](https://arxiv.org/html/2502.04382v1)
- Recommendation steering via SAE latents is demonstrated in arXiv:2507.12202 (sequential recommenders) and arXiv:2511.18024 (filtering and promotion without retraining). [arXiv 2507.12202](https://arxiv.org/pdf/2507.12202v2); [arXiv 2511.18024](https://arxiv.org/abs/2511.18024)
- Korznikov et al. find that random baselines match SAEs on causal editing, so any steering claim needs a random-feature control. [arXiv](https://arxiv.org/html/2602.14111v1)

### Inferences
- "Policy steering" experiment (cheap, because the policy is linear in features):
  1. Identify a concept set, e.g. sensational/clickbait, sports, or politics, via in-domain labels or by correlating features with an LLM-rated clickbait score on MIND headlines.
  2. Clamp or zero those weights (or add a penalty term on them) at decision time.
  3. Measure:
     - change in the fraction of sensational articles recommended
     - expected clicks lost (the "price of de-sensationalizing")
     - a matched random-feature control
  This turns Q3 from descriptive into an actionable auditing/control result, which is the type of downstream use the "SAEs for unknowns" position paper argues for.
- Validation for the concept set: compare an SAE-concept "clickbait score" against an LLM-judge clickbait label and against a linear probe on dense features. Expect the dense probe to win on accuracy (per Kantamneni/GDM); the SAE's advantage is interpretability and no labels.
- Activation-level steering of Gemma itself (adding SAE decoder directions to the residual stream) is out of scope and not needed. The policy-level intervention is cleaner.

### Gaps
- I found no MIND-specific clickbait labels. A published clickbait dataset (e.g. Webis-Clickbait-17 **[UNVERIFIED]**) or LLM-judge labels would be needed.

---

## Prioritized proposed experiments (impact vs effort, ~1 month, L40S)

These items go here rather than as a new top-level section, to keep within the three-section format per question. Everything assumes cached residual activations for 1B and 12B layers.

| # | Experiment | Addresses reviewer concern | Impact | Effort |
|---|---|---|---|---|
| 1 | **Random-SAE baselines (Korznikov-style)**: random encoder directions plus JumpReLU thresholds calibrated to L0 ≈ 60, same pooling and selection, in LinUCB at d=16-128 | "Do SAE features beat random sparse codes?" Without this, the Q2 SAE result is hard to interpret | High | Low (1-2 days, no training) |
| 2 | **Selection/pooling ablation**: {mean, max, mean-no-BOS} × {variance, frequency, MI/L1-on-train-clicks, PCA-of-full-16k-code}; drop features with density > 50% | Separates "SAE basis is worse" from "our selection is bad"; may close part of the 22-35% gap | High | Low-Med (3-4 days) |
| 3 | **In-domain relabeling plus label scoring** (Delphi detection/fuzzing, HypotheSAEs F1) of the top-32 features per layer, with random-feature controls; report the fraction of Neuronpedia labels at chance on MIND | Quantifies the formatting-feature/label-mismatch finding; makes Q3 concept profiles trustworthy | High | Med (4-5 days; local LLM or cheap API) |
| 4 | **Full deletion/insertion curves with stats**: top-k vs random-k (≥20 draws) vs bottom-k for k=1..32; keep-only-top-k retrain; paired bootstrap CIs over 30 runs; Wilcoxon with Holm correction | Turns "beats random at most k" into a clean, defensible faithfulness claim | High | Low (2-3 days; reruns of existing code) |
| 5 | **Redundancy-aware ROAR**: remove top-k plus correlated features (|r|>0.5), or ablate in a decorrelated basis | Answers the GOAR redundancy critique; may explain weak faithfulness at other layers | Med | Low (2 days) |
| 6 | **Cheap dense/linear baselines at the same d**: Gaussian random projection, ICA, top-variance raw neurons, PCA of SAE reconstruction | Completes the baseline ladder; tests information loss through the SAE | Med | Low (1-2 days) |
| 7 | **Policy steering on a concept set** (clickbait/sensational, topic): zero or penalize those weights, measure the composition shift and the price in clicks, with a random-feature control | Shows a downstream *use* of SAE interpretability, not just description | Med-High | Med (4-5 days, incl. LLM-judge clickbait labels) |
| 8 | **Width/L0 sweep** on Gemma Scope 2 (16k vs 65k/262k; l0 small/medium/big) at the main layer | "Is 16k/l0_medium just a bad choice?" | Med | Med (re-encode activations; 3-4 days) |
| 9 | **SAE on a random-init Gemma-3-1B** (Heap-style) | Strongest sanity check, but requires SAE training | Low-Med | High (≥1 week) — skip unless time allows |

Suggested 1-month plan:
- Week 1: experiments 1, 4, 6.
- Week 2: experiment 2.
- Week 3: experiments 3 and 5.
- Week 4: experiment 7, and experiment 8 if time allows.

Recommended framing: "consistent with Kantamneni et al. and the GDM negative results, SAE features underperform dense baselines for decision-making too, but they beat random-SAE baselines (if experiment 1 confirms this) and allow auditing and steering of the policy that dense PCA features do not."
