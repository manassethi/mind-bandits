# Q2: Contextual bandits for news recommendation with text/LLM representations and cold-start

Scope: literature (emphasis 2020-2026, plus classics), baselines and experiments reviewers will expect, and a prioritized experiment list for the Q2 workshop-paper results (MIND-large, semi-synthetic click simulator, churning pool of 20 articles, shared linear model on [1, a, u*a]).

Verification legend:
- (V) = checked in this session via search/fetch of the primary page or a direct snippet.
- (K) = from background knowledge, citation believed correct but NOT re-verified in this session. Check arXiv id/venue before citing.

---

## 1. Key papers: contextual bandits for news, cold-start, content features (classic + neural + Bayesian-deep + LLM-era)

### Takeaway
The Q2 setup sits at the intersection of three lines: (a) linear contextual bandits for news (LinUCB / hybrid LinUCB, linear TS) where item features handle new articles; (b) "deep representation, shallow exploration" methods (Neural-Linear, Neural-LinUCB, epinet/ENR) which are the closest modern analogue to "frozen LLM features + linear bandit head"; and (c) a small, very recent (2024-2026) literature on LLMs in bandits that almost entirely uses LLMs as *priors / pseudo-reward generators*, not as frozen feature extractors evaluated layer-by-layer. That last gap is the paper's clearest novelty claim.

### Cited Findings

Classic linear / news bandits
- (K) Li, Chu, Langford, Schapire, "A Contextual-Bandit Approach to Personalized News Article Recommendation," WWW 2010, arXiv:1003.0146. Introduces LinUCB (disjoint) and **hybrid LinUCB** (shared parameters over user-article interaction features + per-article parameters) on Yahoo! Today Module. Hybrid LinUCB is the canonical baseline for "shared model + per-article model" and is the most direct ancestor of the paper's [1, a, u*a] shared model. Cited as the news-bandit reference by the 2026 calibration-gated paper — [Pershin et al. 2026](https://arxiv.org/html/2604.14961).
- (V) Li, Chu, Langford, Wang, "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms," WSDM 2011 — replay evaluation on uniformly-random logged data — [arXiv:1003.5956](https://arxiv.org/pdf/1003.5956).
- (K) Chu, Li, Reyzin, Schapire, "Contextual Bandits with Linear Payoff Functions," AISTATS 2011 (LinUCB regret analysis).
- (K) Agrawal & Goyal, "Thompson Sampling for Contextual Bandits with Linear Payoffs," ICML 2013, arXiv:1209.3352 (LinTS).
- (K) Chapelle & Li, "An Empirical Evaluation of Thompson Sampling," NeurIPS 2011 — logistic-regression TS with a Laplace (diagonal) posterior on Yahoo! news and display ads; the standard reference for "Bernoulli clicks -> logistic TS" and for delayed/batched updates.
- (K) Abbasi-Yadkori, Pál, Szepesvári, "Improved Algorithms for Linear Stochastic Bandits," NeurIPS 2011 (OFUL; confidence sets used in modern LinUCB) — also cited by [Pershin et al. 2026](https://arxiv.org/html/2604.14961).
- (K) Chakrabarti, Kumar, Radlinski, Upfal, "Mortal Multi-Armed Bandits," NeurIPS 2008 — arms with finite lifetimes; the classic framing for the paper's article-churn / lifetime manipulation (500/2000/10000 rounds). Reviewers in the bandit community may expect this cited.

Generalized linear (Bernoulli) bandits
- (K) Filippi, Cappé, Garivier, Szepesvári, "Parametric Bandits: The Generalized Linear Case," NeurIPS 2010 (GLM-UCB).
- (K) Li, Lu, Zhou, "Provably Optimal Algorithms for Generalized Linear Contextual Bandits," ICML 2017, arXiv:1703.00048 (UCB-GLM, SupCB-GLM).
- (K) Faury, Abeille, Calauzènes, Fercoq, "Improved Optimistic Algorithms for Logistic Bandits," ICML 2020, arXiv:2002.07530 — removes the bad dependence on the logistic curvature constant kappa; relevant because CTRs in news are small (~a few %), exactly the regime where kappa is large and linear-Gaussian models are most misspecified.

Neural bandits (full-network exploration)
- (K) Zhou, Li, Gu, "Neural Contextual Bandits with UCB-based Exploration" (NeuralUCB), ICML 2020, arXiv:1911.04462.
- (K) Zhang, Zhou, Li, Gu, "Neural Thompson Sampling" (NeuralTS), ICLR 2021, arXiv:2010.00827.
- (K) Ban, Yan, Banerjee, He, "EE-Net: Exploitation-Exploration Neural Networks in Contextual Bandits," ICLR 2022, arXiv:2110.03177.
- (V) Xu, Wen, Zhao, Gu, "Neural Contextual Bandits with Deep Representation and Shallow Exploration" (Neural-LinUCB), arXiv Dec 2020, ICLR 2022. Uses the last hidden layer of a ReLU net as features and does UCB only in the final linear layer; O~(sqrt(T)) regret; argues NeuralUCB/NeuralTS regret has effective dimension that can scale with O(TK) because they explore the whole parameter space, and that last-layer exploration is far cheaper — [arXiv:2012.01780](https://arxiv.org/pdf/2012.01780); venue [ICLR 2022](https://iclr.cc/virtual/2022/poster/6858) (one aggregator lists NeurIPS 2021; ICLR listing is the primary source).

Bayesian deep bandits / approximate TS
- (V) Riquelme, Tucker, Snoek, "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling," ICLR 2018, arXiv:1802.09127. Main finding: many approximate-posterior methods that work in supervised learning underperform in sequential decisions, partly because slowly converging uncertainty estimates do not adapt online — [Google Research](https://research.google/pubs/deep-bayesian-bandits-showdown/), [alphaxiv](https://alphaxiv.org/abs/1802.09127).
- (K) In the same paper, **NeuralLinear** (Bayesian linear regression on the last layer of a NN, re-fit periodically) was among the most robust methods across tasks — this is the most important single citation for justifying "frozen representation + exact Bayesian linear head" as a strong design, i.e. exactly what the paper does with Gemma features. (Not re-verified this session; the snippets did not name methods.)
- (K) Zahavy & Mannor, "Deep Neural Linear Bandits: Overcoming Catastrophic Forgetting through Likelihood Matching," arXiv:1901.08612.
- (K) Lu & Van Roy, "Ensemble Sampling," NeurIPS 2017, arXiv:1705.07347.
- (V) Osband, Wen, Asghari, Dwaracherla, Ibrahimi, Lu, Van Roy, "Approximate Thompson Sampling via Epistemic Neural Networks," UAI 2023 (PMLR 216:1586-1595), arXiv:2302.09205. The epinet is a small additive network that matches large ensembles at orders of magnitude lower compute; approximate posterior samples help decisions only if they approximate **joint** (not just marginal) predictive distributions — [PMLR](https://proceedings.mlr.press/v216/osband23a.html), [arXiv](https://arxiv.org/pdf/2302.09205).
- (K) Osband et al., "Epistemic Neural Networks," arXiv:2107.08924 (NeurIPS 2023).
- (V) Zhu & Van Roy, "Scalable Neural Contextual Bandit for Recommender Systems," CIKM 2023, arXiv:2306.14834. Epistemic Neural Recommendation (ENR): TS at scale; reports >=9% CTR and >=6% rating gains over SOTA neural contextual bandits, matches the best baseline with >=29% fewer interactions, orders of magnitude less compute — [arXiv](https://arxiv.org/abs/2306.14834). A search snippet says it uses MIND's logged impressions as ground-truth feedback for evaluation — [search result](https://arxiv.org/pdf/2306.14834) (dataset not confirmed from the abstract page; check the paper's experiments section — likely the closest prior work using MIND as a bandit benchmark).

Neural bandits on MIND
- (V) "Two-Stage Neural Contextual Bandits for Personalised News Recommendation," arXiv:2206.14648 — hierarchical topic-then-news UCB on MIND with NRMS as base predictor; motivates exploration to avoid feedback loops — [arXiv](https://arxiv.org/pdf/2206.14648). (Authors not captured in snippet; I believe Mengyan Zhang et al., unverified.)
- (V) Off-policy evaluation of contextual bandit algorithms for news on MIND (IPW/SNIPS), EAI 2026 — [EUDL](https://eudl.eu/doi/10.4108/eai.22-5-2026.2365082) (low-profile venue; cite only if needed).
- (V) "Simulated Contextual Bandits for Personalization Tasks from Recommendation Datasets," arXiv:2210.10631 — general recipe for building semi-synthetic bandit environments from rec datasets (MovieLens, Last.fm); relevant methodological precedent for the simulator — [arXiv](https://arxiv.org/pdf/2210.10631).

LLMs + bandits / LLM cold-start (2023-2026)
- (V) Alamdari, Cao, Wilson, "Jump Starting Bandits with LLM-Generated Prior Knowledge" (CBLI), EMNLP 2024, arXiv:2406.19317. LLM generates synthetic personas/interactions to pre-train a contextual bandit; user contexts are OpenAI text embeddings; reduces early regret — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.1107/). Page numbers differ across sources; use the Anthology entry.
- (V) Robustness follow-up: "Robustness of LLM-Initialized Bandits for Recommendation Under Noisy Priors" (Bayley et al., GenAIRecP 2025 workshop) — diminishing / reversing benefit as LLM-prior noise grows — [PDF](https://genai-personalization.github.io/assets/papers/GenAIRecP2025/7_Bayley.pdf), [RBC Borealis](https://rbcborealis.com/publications/robustness-of-llm-initialized-bandits-for-recommendation-under-noisy-priors/).
- (V) Pershin, Golovanov, Baltabaev, Trankova, "Calibration-Gated LLM Pseudo-Observations for Online Contextual Bandits," arXiv:2604.14961 (Apr 2026). Disjoint LinUCB + GPT-4o-mini pseudo-rewards for unplayed arms; on MIND-small with K=5, feature-hashed contexts, Bernoulli rewards (~38% of rounds have a clickable candidate); single seed, T=100: regret NoLLM 32 vs task-prompted LLM 26 (~19% lower), calibration-gated 33; LLM pseudo-observations hurt on Mushroom — [arXiv HTML](https://arxiv.org/html/2604.14961). Very weak experimental bar (1 seed, T=100) -> the paper's much larger, multi-seed MIND-large study compares favourably.
- (V) Related LLM-bandit works cited there: LOLA (Ye et al. 2024, arXiv:2406.02611); LLM-enhanced MAB (Sun et al. 2025, arXiv:2502.01118); LLM-augmented contextual bandit (Baheri & Alm 2023, arXiv:2311.02268); EVOLvE (Nie et al., ICML 2025) on in-context exploration; Chen et al. EMNLP 2024 "Efficient sequential decision making with LLMs" — [Pershin et al. reference list](https://arxiv.org/html/2604.14961).
- (V) Lee, Choi, Kang, Jang, "LLM-Derived Priors for Thompson Sampling in Cold-Start Comment Recommendation," arXiv:2608.03382 (Aug 2026). LLM turns comment text into Beta priors for per-segment TS; online A/B/C test; priors help most in sparse-feedback regimes — [arXiv](https://arxiv.org/abs/2608.03382). Note: this is exactly "per-arm Beta TS + LLM text prior", i.e. a natural competitor to the paper's "none" baseline.
- (V) Zhao et al., "Dynamic Prior Thompson Sampling for Cold-Start Exploration in Recommender Systems," arXiv:2602.00943 (Feb 2026) — cold-start prior design for new items in TS (not text-based) — [arXiv](https://arxiv.org/abs/2602.00943).
- (V) LLM cold-start (non-bandit): LLM interaction simulator / ColdLLM, arXiv:2402.09176; LLM2Rec (KDD 2025) fine-tunes LLMs into item embedding models for sequential recommendation, arXiv:2506.21579 — [LLM2Rec](https://arxiv.org/pdf/2506.21579); survey of LLM-enhanced recsys, arXiv:2412.13432 — [survey](https://arxiv.org/pdf/2412.13432).
- (V) Survey: "A Survey on LLM-based News Recommender Systems," arXiv:2502.09797 — [arXiv](https://arxiv.org/pdf/2502.09797) (PDF could not be parsed this session; no numbers extracted).

### Inferences
- Existing LLM-bandit work uses LLMs as (i) synthetic-data / prior generators (CBLI, LLM-derived priors) or (ii) reward oracles (calibration-gated pseudo-observations, LOLA). None found that systematically compares **frozen LLM residual-stream layers, SAE features, and sentence-embedding models as bandit features under item churn**. That framing ("which representation, which layer, how many dims, under how fast churn") is the paper's distinct contribution and should be stated explicitly in related work.
- The paper's design (frozen encoder + Bayesian linear head) is a NeuralLinear / Neural-LinUCB instance with a pretrained, rather than online-trained, body. Citing Riquelme et al. 2018 and Xu et al. 2022 lets the authors frame it as "deep representation, shallow exploration with a foundation-model body" rather than as a naive baseline.
- The 2024-2026 LLM-prior papers suggest one cheap extra arm: an LLM-prior variant of the per-article Beta TS baseline (see experiments).

### Gaps
- Could not confirm the datasets in Zhu & Van Roy 2023 from the abstract page alone (snippet suggests MIND).
- Found no paper that runs neural/linear bandits with frozen *decoder-LLM layer-wise* features on MIND; absence is from ~15 searches, not an exhaustive arXiv sweep.
- Hybrid LinUCB, GLM-UCB, NeuralUCB/TS, EE-Net, ensemble sampling, ENN citations are (K) — verify ids before camera-ready.

---

## 2. Baselines reviewers will expect beyond LinUCB / LinTS / greedy

### Takeaway
The minimum set a bandit-savvy reviewer will look for: (1) hybrid/disjoint LinUCB (Li et al. 2010) as the historical news baseline; (2) a Bernoulli-aware model (logistic TS with Laplace/diagonal posterior, or GLM-UCB) because rewards are clicks with low CTR; (3) one "learned representation" method (NeuralLinear or Neural-LinUCB, possibly an epinet/ensemble TS head) to show whether fine-tuning the representation online beats frozen LLM features; (4) epsilon-greedy and a random policy as sanity anchors. Diagonal/approximate posteriors are worth one ablation because at d=128 with [1,a,u*a] features (dim ~257) the full covariance is cheap, but production systems use diagonal approximations.

### Cited Findings
- Hybrid LinUCB is the original news-bandit design combining shared and per-article parameters (K) — Li et al. 2010, arXiv:1003.0146.
- Logistic TS with Laplace approximation and diagonal covariance is the standard Bernoulli-click baseline (K) — Chapelle & Li 2011.
- GLM bandit theory: GLM-UCB (Filippi et al. 2010), UCB-GLM (Li et al. 2017, arXiv:1703.00048), logistic-bandit improvements in small-CTR regime (Faury et al. 2020, arXiv:2002.07530) (all K).
- Last-layer exploration with a learned body (Neural-LinUCB) is computationally efficient and has better regret dependence than full-network exploration — [Xu et al., arXiv:2012.01780](https://arxiv.org/pdf/2012.01780).
- Approximate posteriors that work in supervised learning often fail in bandits; benchmark across approaches before claiming superiority — [Riquelme et al. 2018](https://research.google/pubs/deep-bayesian-bandits-showdown/).
- Epinet matches large ensembles at much lower compute for approximate TS — [Osband et al., UAI 2023](https://proceedings.mlr.press/v216/osband23a.html); scaled to recommendation as ENR — [Zhu & Van Roy, CIKM 2023](https://arxiv.org/abs/2306.14834).
- Representation selection in linear bandits: (K) Papini, Tirinzoni, Restelli, Lazaric, Pirotta, "Leveraging Good Representations in Linear Contextual Bandits," ICML 2021, arXiv:2104.03781 — algorithms that adaptively pick among candidate feature maps; relevant because the paper compares many representations (layers, dims, encoders) and a reviewer may ask "can the bandit pick the layer online?"

### Inferences
- Recommended concrete additions (cheap, all CPU/GPU-light because d<=257):
  1. **Hybrid LinUCB** (shared [1,a,u*a] + per-article disjoint ridge). Directly tests whether per-article residuals fix the category-one-hot failure and whether they help or hurt under fast churn.
  2. **Logistic TS (Laplace, full and diagonal)** and/or **GLM-UCB** on the same features. Clicks are Bernoulli with low CTR; if linear LinUCB is already near logistic, that is a useful negative result; if logistic wins, rankings may change.
  3. **NeuralLinear / Neural-LinUCB**: small MLP on top of frozen Gemma/bge features (or on raw category) retrained every N rounds, Bayesian linear last layer. Answers "is the frozen LLM representation already good enough?"
  4. **Ensemble sampling or epinet head** (e.g., 10-member ensemble of ridge heads with bootstrapped/perturbed data) — optional, one row.
  5. **epsilon-greedy** + **uniform random** as anchors; greedy is already present.
- NeuralUCB/NeuralTS/EE-Net are expensive (full-gradient covariance) and known to be brittle; one representative (NeuralTS with diagonal approx) is enough, or justify omission by citing Xu et al.'s argument for shallow exploration.

### Gaps
- No direct source found comparing logistic vs linear TS on MIND specifically.

---

## 3. MIND news-recommendation accuracy levels (to position the simulator: AUC 0.733, gAUC 0.686, MRR 0.328, nDCG@10 0.427)

### Takeaway
Supervised SOTA on MIND-large is roughly AUC 0.67-0.73, MRR 0.33-0.37, nDCG@10 0.42-0.47 (dev in papers; hidden test on leaderboard). The simulator's ranking metrics (MRR 0.328, nDCG@10 0.427) are at about NRMS level; its reported AUC 0.733 is at or above leaderboard top numbers, which is an internal inconsistency worth explaining (probably global AUC over pooled impressions vs the standard per-impression-averaged AUC; the 0.686 "group AUC" is the number comparable to the literature and lands right at NRMS/UNBERT level).

### Cited Findings
- MIND official leaderboard (hidden test): UNBERT (Feb 2021) AUC 0.7207, MRR 0.3677, nDCG@5 0.4041, nDCG@10 0.4602; MINER (Sep 2021) AUC 0.7275, MRR 0.3724, nDCG@5 0.4102, nDCG@10 0.4661 — [msnews.github.io](https://msnews.github.io/).
- UNBERT paper (IJCAI 2021), MIND-large: NRMS AUC 0.6766, MRR 0.3325, nDCG@5 0.3628, nDCG@10 0.4198; UNBERT 0.7068 / 0.3568 / 0.3913 / 0.4478; ensemble AUC 0.7183, nDCG@10 0.4581 — [IJCAI PDF](https://www.ijcai.org/proceedings/2021/0462.pdf).
- MINER (Findings of ACL 2022), MIND-large: AUC 71.51, MRR 36.18, nDCG@5 39.72, nDCG@10 45.34 — [ACL Anthology](https://aclanthology.org/2022.findings-acl.29.pdf).
- Paper vs leaderboard numbers differ (e.g. UNBERT 0.7068 vs 0.7207) — different splits/evaluation; do not mix them — [UNBERT](https://www.ijcai.org/proceedings/2021/0462.pdf) vs [leaderboard](https://msnews.github.io/).
- LLM-era: ONCE/GENRE (Liu, Chen, Sakai, Wu; WSDM 2024) reports up to 19.32% relative improvement over BERT-based baselines on MIND (metric ambiguous, possibly nDCG@5) — [summary](https://papers.lunadong.com/paper/10475), [repo](https://github.com/Jyonn/GENRE-requests). LLM-generated category descriptions give up to 5.8% AUC gain on MIND for NAML/NRMS/NPA (Yada & Yamana) — [arXiv:2405.13007](https://arxiv.org/html/2405.13007v1). Other relevant: PUNR (arXiv:2304.12633), SPAR (arXiv:2402.10555) — [PUNR](https://arxiv.org/pdf/2304.12633), [SPAR](https://arxiv.org/pdf/2402.10555).
- (K) Classic MIND baselines: NRMS (Wu et al., EMNLP 2019), NAML (Wu et al., IJCAI 2019), LSTUR (An et al., ACL 2019), PLM-NR (Wu et al., SIGIR 2021, arXiv:2104.07413); MIND dataset paper: Wu et al., ACL 2020. The MIND paper reports AUC/MRR/nDCG@5/@10 averaged over impressions.

### Inferences
- In the paper, report the simulator next to NRMS (dev) and UNBERT/MINER (dev) in one small table, and state which AUC convention is used. Since the user renamed MIND dev to "test", the comparable literature numbers are the paper-reported (dev) ones, not leaderboard (hidden test).
- Framing: "our simulator ranks about as well as NRMS, a standard neural news recommender, so it encodes a realistic but imperfect user model" — this preempts the reviewer worry that the simulator is either trivial or unrealistically sharp.
- Because gAUC 0.686 < leaderboard SOTA, there is headroom; reward is semi-synthetic, so bandit conclusions are about learning a *realistic-strength* preference function, which is fine for a workshop paper.

### Gaps
- Could not extract ONCE/GENRE, LKPNR, Prompt4NR, PLM-NR exact MIND-large numbers this session (survey PDF unparsable). Fetch arXiv:2502.09797 tables or the individual papers before citing numbers.

---

## 4. Which LLM layer to take representations from (relevance to "deeper layers better")

### Takeaway
The dominant 2024-2026 finding is that **intermediate layers often beat the final layer** for embeddings (Skean et al., ICML 2025 oral) and for recommendation specifically (MARC, SIGIR 2026 "Mid-layer Representation Advantage"; IMFuse 2026). The paper's "deeper is better" result therefore needs precise wording: if performance rises monotonically to the last layer, that is a mild *contrast* worth discussing (mean-pooling + correlation PCA + linear bandit may favor the more linearly-separable late layers for topical similarity); if it peaks before the final layer, it *agrees* with this literature. Either way, plot regret vs relative depth for 1B and 12B on the same axis.

### Cited Findings
- Skean, Arefin, Zhao, Patel, Naghiyev, LeCun, Shwartz-Ziv, "Layer by Layer: Uncovering Hidden Representations in Language Models," ICML 2025 (oral), arXiv:2502.02013. Intermediate layers often outperform the final layer across 32 text-embedding tasks; proposes information-theoretic/geometric/robustness metrics (e.g., prompt entropy, curvature) for layer quality across transformers and SSMs; code github.com/OFSkean/information_flow — [PMLR](https://proceedings.mlr.press/v267/skean25a.html), [arXiv](https://arxiv.org/abs/2502.02013v1).
- Xi, Zhu, Lin, Chen, Tang, Yu, Zhang, "Modular Representation Compression: Adapting LLMs for Efficient and Effective Recommendations" (MARC), SIGIR 2026, arXiv:2604.18146: "representations from middle layers of LLMs outperform those from final layers in recommendation tasks"; attributed to the final layer specializing in the proxy (next-token) task; compressing at the final layer is suboptimal; 2.82% eCPM lift online — [arXiv](https://arxiv.org/abs/2604.18146).
- IMFuse (arXiv:2607.27002): multi-layer fusion for LLM-enhanced sequential rec; different layers capture different granularity; final-layer representations can suffer spectral collapse — [arXiv](https://arxiv.org/pdf/2607.27002). FLEXRec (arXiv:2608.17316): fusion of layer-wise exits in compact LLMs for rec — [arXiv](https://arxiv.org/pdf/2608.17316) (results not verified).
- (K) Earlier: Skean et al., "Does Representation Matter? Exploring Intermediate Layers in Large Language Models," arXiv:2412.09563 (workshop precursor); Liu et al., "Linguistic Knowledge and Transferability of Contextual Representations," NAACL 2019; Tenney et al., "BERT Rediscovers the Classical NLP Pipeline," ACL 2019.

### Inferences
- Add a layer-quality diagnostic (e.g., Skean et al.'s prompt entropy, or simply linear-probe AUC for click prediction offline) per layer and correlate it with bandit regret. If offline probe quality predicts online regret, that is a cheap, publishable "selection rule" result.
- Since the task is topical similarity between user history and candidate (largely semantic/topic content), later layers being good is plausible; MARC's mechanism (final layer specialized to next-token prediction) predicts a dip at the very last 1-2 layers — check for it explicitly.
- Multi-layer concatenation/fusion (IMFuse-style) is an obvious reviewer question: one experiment concatenating e.g. middle + late layers before PCA.

### Gaps
- No bandit-specific (online) layer-selection paper found; all layer evidence is supervised/offline.

---

## 5. Dimension/compression, position bias, slate/cascading bandits

### Takeaway
Dimension sweeps are expected and already present (d in {16,32,64,128}); reviewers may add "random projection vs PCA" and "SAE sparsity." Slate/cascading bandits are a well-established extension (CascadeLinUCB/TS, position-based model) but are not required for a workshop paper on representations; a single top-K cascade variant would preempt "single-article is unrealistic" criticism. Position bias matters only if a slate is shown.

### Cited Findings
- Zong, Ni, Sung, Ke, Wen, Kveton, "Cascading Bandits for Large-Scale Recommendation Problems," UAI 2016, arXiv:1603.05359: recommend K of L items; attraction probability linear in item features, so regret independent of L; CascadeLinUCB (regret O(dK sqrt(T) ln T)) and CascadeLinTS — [arXiv](https://arxiv.org/abs/1603.05359), [UAI PDF](https://www.auai.org/uai2016/proceedings/papers/96.pdf).
- Cheung, Tan, Zhong, "Thompson Sampling Algorithms for Cascading Bandits," arXiv:1810.01187 — linear TS for cascades with regret depending on dimension, not number of items — [arXiv](https://arxiv.org/pdf/1810.01187v3).
- Online clustering of contextual cascading bandits, arXiv:1711.08594 — [arXiv](https://arxiv.org/pdf/1711.08594).
- (K) Kveton, Szepesvári, Wen, Ashkan, "Cascading Bandits: Learning to Rank in the Cascade Model," ICML 2015, arXiv:1502.02763; Lagrée, Vernade, Cappé, "Multiple-Play Bandits in the Position-Based Model," NeurIPS 2016, arXiv:1608.03023; Ie et al., "SlateQ" (IJCAI 2019) and RecSim (arXiv:1909.04847) for slate simulation.
- MARC argues high-dimensional LLM representations require compression and that where you compress (which layer) matters — [arXiv:2604.18146](https://arxiv.org/abs/2604.18146).

### Inferences
- A **cascade/top-K extension** is straightforward with the existing simulator: show K=5 articles, user scans in order, clicks the first with simulated p>U (or position-discounted PBM). Swap LinUCB for CascadeLinUCB (same ridge stats, update on examined items). Effort: ~3-5 days. Reports whether the feature-ranking conclusions transfer.
- Dimension: add a **random Gaussian projection** baseline at the same d (separates "PCA picks good directions" from "LLM space is good"), and plot regret vs d with the theoretical d·sqrt(T) dependence in mind (larger d costs exploration; best d should shrink with faster churn/shorter horizon).

### Gaps
- No source found specifically on position bias in MIND impression logs.

---

## 6. Why category one-hot can be worse than no features (misspecification in shared-parameter bandits)

### Takeaway
Theory says a misspecified linear model pays a linear-in-T term ~ eps*sqrt(d)*T, which per-article (tabular) Beta TS does not pay; with only ~18 coarse categories, a shared [1, a, u*a] model cannot distinguish articles within a category, so its optimistic choices are systematically biased and never corrected, whereas per-article TS eventually learns each article's CTR. Under churn the per-article learner loses its advantage, which is why rich text features (low eps) win but coarse features (high eps) lose.

### Cited Findings
- (V, via secondary summary) Lattimore, Szepesvári, Weisz (ICML 2020) lower bound: under uniform eps-misspecification, regret must be Omega(d sqrt(T) + eps sqrt(d) T); eps sqrt(d) T is "the price of misspecification" — summarized in [Foster et al. NeurIPS 2020 / arXiv:2107.05745](https://arxiv.org/pdf/2107.05745). (K) Original: "Learning with Good Feature Representations in Bandits and in RL with a Generative Model," arXiv:1911.07676.
- Foster, Gentile, Mohri, Zimmert, "Adapting to Misspecification in Contextual Bandits," NeurIPS 2020: oracle-efficient algorithm achieving O~(d sqrt(T) + eps sqrt(d) T) without knowing eps — [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/84c230a5b1bc3495046ef916957c7238-Abstract.html).
- Gu et al.-line ICML 2023, "On the Interplay Between Misspecification and Sub-optimality Gap in Linear Contextual Bandits," arXiv:2303.09390: earlier algorithms incur linear regret under misspecification; learnable efficiently when eps is small relative to gap/sqrt(d) — [arXiv](https://arxiv.org/pdf/2303.09390).
- Hybrid LinUCB (Li et al. 2010; K) is the classical practical fix: add per-article parameters so misspecification of the shared part is absorbed.
- (K) Papini et al., ICML 2021 (arXiv:2104.03781): representation quality (spanning/"HLS" condition) determines whether linear bandits get constant vs sqrt(T) regret — a theoretical vocabulary for "good vs bad features."

### Inferences
- The paper can explain the category result with a one-paragraph misspecification argument plus one empirical diagnostic: estimate eps for each feature set as the RMSE (or max error) of the best linear fit of the simulator's true click probabilities on [1,a,u*a] (computable offline because the simulator is known). Plot regret vs estimated eps across all feature sets: expect monotone. This turns an anecdote into a mechanism and is cheap.
- Hybrid LinUCB and "category + per-article" are the two direct tests that should rescue categories; if hybrid fixes it at slow churn but not fast churn, that cleanly shows the churn-dependent value of shared content features.

### Gaps
- No paper found that empirically shows "coarse categorical features worse than no features" in news bandits specifically; the theory supports it but the empirical claim is the paper's own.

---

## 7. Prioritized proposed experiments for Q2 (~1 month, single L40S)

All bandit runs are cheap (d<=257 ridge updates); GPU is only needed for new feature extraction. Effort estimates assume existing pipeline, cached embeddings (do not retrain the simulator — per project memory).

| # | Experiment | Why reviewers want it | Effort | Impact |
|---|---|---|---|---|
| 1 | **Misspecification diagnostic**: for each feature set/d, fit best linear (and logistic) model of simulator click-prob on [1,a,u*a]; report eps (RMSE/max error, and R^2); scatter regret vs eps | Explains category < none and Gemma ≈ Qwen ceiling mechanistically; ties to Lattimore 2020 / Foster 2020 | 1-2 days, CPU | **High** |
| 2 | **Hybrid LinUCB** (shared + per-article disjoint) for category, bge, Gemma | Canonical news baseline (Li 2010); tests whether per-article residuals fix coarse features; interacts with churn | 2 days | **High** |
| 3 | **Logistic TS (Laplace, full + diagonal) and/or GLM-UCB** on same features | Rewards are Bernoulli with low CTR; Chapelle & Li 2011, Faury 2020 | 2-3 days | **High** |
| 4 | **Seeds/CIs + horizon sensitivity**: ≥10 seeds, 95% CIs, paired differences; report regret at several T | Standard rigor; contrasts with weak LLM-bandit papers (1 seed, T=100) | 1-2 days compute | **High** |
| 5 | **NeuralLinear / Neural-LinUCB on frozen features** (small MLP retrained every N rounds, Bayesian last layer) and on category/raw IDs | "Is frozen LLM representation enough, or does online representation learning help?" (Riquelme 2018, Xu 2022) | 4-5 days | Med-High |
| 6 | **Layer analysis upgrade**: regret vs relative depth for 1B & 12B on one plot; check final-layer dip; add one multi-layer concat (mid+late) ; correlate per-layer offline probe AUC (or Skean prompt-entropy) with regret | Directly engages Skean 2025 / MARC 2026 "mid-layer advantage" | 3-4 days (features mostly cached) | Med-High |
| 7 | **Random-projection and raw (no-PCA) controls at same d**; PCA vs correlation-PCA | Separates encoder quality from dimensionality reduction | 1-2 days | Medium |
| 8 | **LLM-prior per-article TS baseline**: Beta priors from a cosine-to-user-history or LLM-scored relevance (CBLI / Lee 2026 style) | Represents the 2024-26 "LLM as prior" line; fair competitor for "none" | 3 days | Medium |
| 9 | **Top-K cascade / PBM slate variant** (K=5, CascadeLinUCB/TS) for best 3 feature sets | Preempts "single-article unrealistic"; Zong 2016 | 4-5 days | Medium |
| 10 | **Ensemble/epinet TS head** (e.g., 10 bootstrapped ridge heads) | Modern approximate-TS line (Osband 2023, Zhu & Van Roy 2023) | 3 days | Low-Med |
| 11 | **Representation selection online** (run bandit over layers/d; e.g., Papini 2021-style or simple meta-bandit over representations) | Answers "can you pick the layer without the oracle sweep?" | 4-5 days | Low-Med (nice-to-have) |
| 12 | Full NeuralUCB/NeuralTS/EE-Net | Completeness; costly and brittle — justify omission instead | 1+ week | Low |

Suggested 1-month plan: week 1 = #1, #2, #4; week 2 = #3, #7, #6; week 3 = #5, #8; week 4 = #9 (or #10) + writing. Also add the simulator-vs-literature table (Section 3) and related-work paragraphs on Mortal bandits, NeuralLinear/Neural-LinUCB, LLM-prior bandits, and layer-wise representation papers.

### Gaps
- Effort estimates are judgment calls, not sourced.
