# Simulator Validity and Off-Policy Evaluation for a Semi-Synthetic MIND News-Bandit Study

Scope: (a) recommendation/bandit simulators, semi-synthetic evaluation, and their critiques; (b) OPE methods and datasets with known propensities that could provide real-data evidence; (c) concrete credibility experiments for a paper whose ground truth is a learned MLP click simulator on MIND-large (Qwen3-Embedding-0.6B article embeddings, mean-history user plus category affinities, intercept anchored to 4.05% CTR).

Verification legend: items marked **[verified]** were confirmed against a primary or near-primary page in this session. Items marked **[prior knowledge, not re-verified]** come from the researcher's background knowledge. Check their arXiv ids, venues, and numbers before citing.

---

## Q1. Which simulators and semi-synthetic protocols exist, how do they validate themselves, and what are the known critiques?

### Takeaway
Most recommender simulators validate only weakly. They fit observed logs, report predictive metrics such as AUC or likelihood, or show qualitative "stylized facts". Few show that policy *rankings* in the simulator match rankings online, and Accordion (Netflix) and Li et al.'s replay paper are rare exceptions. The critique literature agrees that simulator conclusions are conditional on the simulator's assumptions. Its main recommendations are to report those assumptions, show sensitivity to them, and tie simulation to real data wherever possible. The MIND simulator's validation (held-out AUC 0.733, group AUC, Spearman against article-level CTR) is already stronger than many published simulators. What it lacks is evidence that *policy-level conclusions* survive under changed assumptions.

### Cited Findings

**Simulator platforms**
- **RecSim** (Ie et al., Google, 2019). A configurable platform for sequential recommendation with user latent state, a document model, and a user choice model. It is a framework for *stylized* environments, not one fitted to data. — [Google Research page](https://research.google/pubs/recsim-a-configurable-simulation-platform-for-recommender-systems/); arXiv:1909.04847 [verified exists via search]
- **RecSim NG** (Mladenov et al., 2021). A differentiable, probabilistic successor built on TensorFlow/Edward2 that models multi-agent ecosystems and uncertainty. — [Google Research page](https://research.google/pubs/recsim-ng-toward-principled-uncertainty-modeling-for-recommender-ecosystems/); arXiv:2103.08057 [verified]
- **Accordion** (McInerney, Elahi, Basilico, Raimond, Jebara; RecSys 2021, pp. 102–113). A trainable simulator based on Poisson processes that models visits over time, trained on large-scale data and evaluated on two real interactive datasets. Its validation criterion is notable: the authors claim it is more *sensitive to hyperparameter tuning and offline A/B testing* than comparison methods. That is, it is judged on whether it reproduces *decision-relevant differences*, not only on fit. — [OpenAlex record, DOI 10.1145/3460231.3474259](https://api.openalex.org/works/doi:10.1145%2F3460231.3474259) [verified]
- **SARDINE** (Deffayet, Thonet, Hwang, Lehoux, Renders, de Rijke, 2024). Gymnasium environments for slate recommendation (SlateTopK Bored/BoredInf/Uncertain, SlateRerank Static/Bored). MIT licence, `pip install sardine-rec`. These are synthetic environments, not fitted to real logs. — [PyPI sardine-rec](https://pypi.org/project/sardine-rec/) [verified; I found no arXiv id]
- **KuaiSim** (NeurIPS 2023 Datasets & Benchmarks). A user simulator trained on Kuaishou data with multi-behaviour feedback (immediate feedback, leave, return). It supports request-level, session-level, and cross-session tasks, and its code is public. — [NeurIPS 2023 poster page](https://neurips.cc/virtual/2023/poster/73528); arXiv:2309.12645 [verified exists]
- **Virtual Taobao** (Shi et al., AAAI 2019, arXiv:1805.10000). A GAN-based (GAN-SD) customer simulator with multi-agent adversarial imitation learning. It reportedly validated by comparing simulated vs real distributions of customer features and behaviour, and by showing that a policy trained in simulation improved online. — [prior knowledge, not re-verified]
- **RecoGym** (Rohde et al., 2018, arXiv:1808.00720). Criteo's environment mixing organic and bandit (ad-click) behaviour. It is synthetic, built to study offline vs online gaps. — [prior knowledge, not re-verified]. Jeunen, Rohde & Vasile (2019) used RecoGym to show that traditional offline evaluation can mislead. — [arXiv:1907.12384](https://arxiv.org/pdf/1907.12384) [verified]
- **RL4RS** (Wang et al., NetEase, arXiv:2110.11073). A real-world RL recsys dataset with a learned simulator. It explicitly includes "environment simulation evaluation" and offline policy evaluation components. — [prior knowledge, not re-verified; venue reportedly NeurIPS 2023 D&B, unverified]
- **Open Bandit Pipeline synthetic** (`obp.dataset.SyntheticBanditDataset`) and **multiclass-to-bandit conversion** (`MultiClassToBanditReduction`). The conversion follows Dudík, Langford & Li (ICML 2011, "Doubly Robust Policy Evaluation and Learning", arXiv:1103.4601): a supervised label defines the reward and a known logging policy is applied, so true policy values are known exactly. — OBP docs [prior knowledge for the class names]; [OBP paper arXiv:2008.07146](https://arxiv.org/pdf/2008.07146) [verified]

**LLM-based user simulators (2023–2026)**
- **RecAgent** (Wang et al., arXiv:2306.02552, later "User Behavior Simulation with LLM-based Agents"). A sandbox where LLM agents browse, chat, and post. — [prior knowledge, not re-verified]
- **Agent4Rec / "On Generative Agents in Recommendation"** (Zhang et al., SIGIR 2024, arXiv:2310.10108). It runs 1,000 LLM agents initialised from MovieLens, Steam, and Amazon-Book. The authors report "both the alignment and deviation" between agents and real user preferences, and they reproduce filter-bubble effects. — [arXiv:2310.10108](https://arxiv.org/pdf/2310.10108) [verified]. The 1,000-agent figure is from prior knowledge.
- **SimUSER** (Bougie & Watanabe, Woven by Toyota; arXiv:2504.12722; ACL 2025 Industry). Persona, memory (episodic plus knowledge graph), perception (visual), and "brain" modules. It claims closer alignment with humans at the micro and macro level and uses the simulator to tune a real recommender. — [arXiv:2504.12722](https://arxiv.org/abs/2504.12722v1); [ACL Anthology](https://aclanthology.org/2025.acl-industry.5) [verified]
- **LLM-Powered User Simulator for Recommender System** (Zhang et al., AAAI 2025, arXiv:2412.16984). It criticises existing simulators for "opacity of user preference modeling and the incapability of evaluating simulation accuracy" and proposes an ensemble of logical and statistical models. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33456) [verified]
- **"How Reliable is Your Simulator?"** (arXiv:2403.16416, 2024). It analyses LLM user simulators for conversational recsys (iEvaLM on ReDial/OpenDialKG) and finds three problems: data leakage that inflates scores, over-reliance on conversational history, and poor controllability through a single prompt. — [arXiv:2403.16416](https://arxiv.org/abs/2403.16416) [verified]
- **DGDPO** (arXiv:2508.12645) argues that static, single-step prompt-based profiles in LLM simulators are inaccurate and incomplete. — [arXiv:2508.12645](https://arxiv.org/html/2508.12645v3) [verified]

**Critiques and position papers**
- **Li, Chu, Langford, Wang** (WSDM 2011, arXiv:1003.5956) give the classic critique: "creating simulator itself is often difficult and modeling bias is usually unavoidably introduced". Their motivation for replay is to avoid simulator bias. They validated replay against *online bucket* results on Yahoo! Front Page. — [arXiv:1003.5956](https://arxiv.org/pdf/1003.5956) [verified]
- **Aouali, Benhalloum, Bompaire, Heymann, Jeunen, Rohde, Sakhi, Vasile** (Criteo, 2022), "Offline Evaluation of Reward-Optimizing Recommender Systems: The Case of Simulation". This position paper argues *for* simulation. Its reasoning is that proxy offline metrics correlate poorly with online metrics and that counterfactual estimators rely on assumptions that are rarely met. It also concedes that simulation differs from reality. — [arXiv:2209.08642](https://arxiv.org/pdf/2209.08642) [verified]
- **Sim2Rec** (arXiv:2305.04832) states that building "a user simulator with no reality-gap... is unrealistic". It proposes training over an *ensemble of simulators* with varied user behaviours. **S2R-Rec** (NeurIPS 2021) treats the sim-to-real gap as off-dynamics RL. — [Sim2Rec](https://export.arxiv.org/pdf/2305.04832); [S2R-Rec](https://neurips.cc/virtual/2021/48857) [verified]
- **Bernardi, Batra, Bruscantini** (Booking.com, arXiv:2109.06723) give an industry view of simulation and propose design principles for simulation platforms. — [arXiv:2109.06723](https://arxiv.org/pdf/2109.06723) [verified exists]
- **Winecoff et al.**, "Simulation as Experiment: An Empirical Critique of Simulation Research on Recommender Systems" (arXiv:2107.14333). They survey recsys simulation papers and criticise weak reporting, assumptions left unexamined, and the lack of validation or sensitivity analysis. — [arXiv:2107.14333](https://arxiv.org/pdf/2107.14333) [verified exists; content summary from prior knowledge]
- Surveys: "Synthetic Data-Based Simulators for Recommender Systems: A Survey" ([arXiv:2206.11338](https://arxiv.org/pdf/2206.11338)) and "Synthetic Data and Simulators for Recommendation Systems: Current State and Future Directions" ([arXiv:2112.11022](https://arxiv.org/pdf/2112.11022)) [verified exist]
- **Chaney, Stewart, Engelhardt** (RecSys 2018, arXiv:1710.11214), "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity". This is a simulation study of feedback loops, often cited on simulation assumptions. — [prior knowledge, not re-verified]
- **Huang, Oosterhuis, de Rijke, van Hoof** (RecSys 2020), "Keeping Dataset Biases out of the Simulation: A Debiased Simulator for RL-based Recommender Systems". It shows that simulators fitted to biased logs inherit exposure and popularity bias, which matters for the MIND simulator fitted on logged impressions. — [prior knowledge, not re-verified]
- **"Are we evaluating rigorously?"** (Sun et al., RecSys 2020, "Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison") and **Ferrari Dacrema, Cremonesi, Jannach** (RecSys 2019, "Are We Really Making Much Progress?", arXiv:1907.06902). These concern rigour in *offline* evaluation in general (baselines, tuning, splits), not simulators specifically. — [prior knowledge, not re-verified]

### Inferences
- **What the literature counts as validation.** Validation runs in rough tiers, weakest first: (1) predictive fit on held-out logs (AUC, likelihood); (2) matching marginal statistics or stylized facts (CTR distribution, popularity curves); (3) correlation with real per-item or per-user outcomes; (4) *decision-level* agreement, meaning policy or hyperparameter rankings in simulation match online or replay rankings (Accordion; Li et al. 2011). The MIND simulator has tiers 1–3. Tier 4 is the gap reviewers will target, and it can only be approached through real logged data with known propensities (see Q3).
- The Huang et al. debiasing critique applies directly. MIND impressions come from a production recommender, so the simulator's click model is fitted on a non-random exposure distribution. This should be acknowledged.
- The two LLM-simulator critiques (leakage, controllability) support the paper's design choice of using LLM features from a different model family than the simulator encoder. The "leakage ceiling" result is a good defence and should be framed explicitly as a leakage test.

### Gaps
- I did not retrieve the full Winecoff et al. checklist. Its specific reporting recommendations are not quoted here.
- I could not confirm a "challenges of simulators for RL-based recsys" paper by that exact title. The closest items are Sim2Rec, S2R-Rec, Aouali et al., and Huang et al. Treat the requested title as unverified.
- KuaiSim's and Virtual Taobao's exact validation metrics were not re-read.

---

## Q2. Robustness of conclusions to simulator misspecification: is there precedent for checking rank stability across simulators, and how should the paper test sensitivity?

### Takeaway
I found no recsys paper that systematically reports "policy rank stability across alternative simulators" as a standard metric. That makes such an analysis a real contribution, not just a box-ticking exercise. Close precedents are ensemble-of-simulators training (Sim2Rec), robustness evaluation of OPE estimators across configurations (IEOE), and robotics sim-to-real rank-consistency metrics (SIMPLER's MMRV). The paper can borrow these: re-run the main comparisons under a family of perturbed simulators and report Kendall tau, pairwise win-rate preservation, and MMRV-style rank violations.

### Cited Findings
- **SIMPLER** (robot manipulation) proposes the Mean Maximum Rank Violation (MMRV) metric to measure how well simulated policy rankings match real ones. Each mis-ranked pair is weighted by the real performance gap. The metric is directly reusable for comparing sim to sim (base vs perturbed simulators). — [SIMPLER project page](https://simpler-env.github.io/) [verified]
- **Sim2Rec** trains across a set of simulators with varied user behaviour because any single simulator has a reality gap. This shows the field accepts *simulator families* as a way to address misspecification. — [arXiv:2305.04832](https://export.arxiv.org/pdf/2305.04832) [verified]
- **IEOE, "Evaluating the Robustness of Off-Policy Evaluation"** (Saito, Udagawa, Kiyohara, Mogi, Narita, Tateno; RecSys 2021; arXiv:2108.13703). This is a protocol for reporting the *distribution* of evaluation outcomes over hyperparameters and evaluation policies rather than one point. It is applied to the Open Bandit Dataset. — [arXiv:2108.13703](https://arxiv.org/pdf/2108.13703) [verified]
- **Accordion** is judged on its sensitivity to A/B-test and hyperparameter differences, which is a decision-level criterion. — [OpenAlex](https://api.openalex.org/works/doi:10.1145%2F3460231.3474259) [verified]
- **RecUserSim** (arXiv:2507.22897) reports consistency across LLM backbones (Pearson/Spearman > 0.8 on action ratings). This is a sim-to-sim agreement check of the same kind. — [arXiv:2507.22897](https://arxiv.org/pdf/2507.22897) [verified via search snippet]
- **SharpeRatio@k** (Kiyohara et al., ICLR 2024, arXiv:2311.18207) evaluates OPE by the risk–return of the top-k policies it selects. It is relevant if the paper's claim is "policy X is chosen". — [prior knowledge, not re-verified]

### Inferences: a concrete sensitivity design for Q1/Q2 of the paper
Define a **simulator family** S = {S0 (current MLP), S1 ... Sk}. Re-run every headline comparison (policy A vs B, feature set X vs Y) under each Sk with the same seeds. Report per-Sk regret and CTR with CIs, plus Kendall tau of policy rankings vs S0, the fraction of pairwise conclusions preserved, and MMRV. Candidate perturbations, roughly cheapest first:
1. **Functional form:** logistic and bilinear click models (already trained during model selection, so this is nearly free).
2. **CTR spread / temperature:** scale the logits by tau in {0.5, 0.75, 1.5, 2} and re-anchor the intercept to 4.05%. This changes the reward gap between good and bad arms, which governs bandit regret orderings.
3. **Base-rate shift:** anchor the mean CTR to 2% and 8%.
4. **Encoder swap:** refit the simulator on bge-m3 (or another family, such as e5 or gte) embeddings. This tests whether "LLM features from family F beat F′" depends on which family the simulator uses. It also gives a second leakage ceiling.
5. **Additive heterogeneity / noise:** add per-(user, item) Gaussian logit noise (unexplainable variance), plus user-level random intercepts (per-user click propensity).
6. **Popularity / recency term:** add beta·log(popularity) and a freshness decay, since MIND news CTR is strongly time-dependent.
7. **Position bias / slates:** see Q4.
8. **Bootstrap simulators:** refit the MLP on bootstrap resamples of the training data. This captures epistemic uncertainty of the fitted simulator, not just stochastic clicks.

A conclusion that holds across perturbations 1–6 (and 8) is much harder to dismiss as an artefact of the simulator.

### Gaps
- I found no published recsys benchmark that reports rank stability across simulators as a standard table. This claim of absence rests on searches that are not exhaustive.
- The SIMPLER paper's arXiv id was not retrieved (I believe it is arXiv:2405.05941; unverified).

---

## Q3. Off-policy evaluation: methods, applicability to MIND, and datasets with known propensities (Yahoo! R6, OBD, KuaiRand, Coat, Yahoo! R3)

### Takeaway
Standard OPE (IPS, SNIPS, DR, and their variants) **cannot be applied to MIND without propensities**. MIND logs come from an unknown production policy, and the impression order is shuffled. Estimating propensities post hoc would rest on untestable assumptions. The natural real-data check is **Li et al. replay on Yahoo! R6A/R6B**: uniformly random news logs where replay is provably unbiased. It is the same domain (news, front-page module) and the standard benchmark for LinUCB-style contextual bandits. The **Open Bandit Dataset** (uniform-random and Bernoulli-TS logs with true propensities) is a second, easier-access option, though it is fashion e-commerce rather than news. **KuaiRand** offers randomly exposed items for unbiased evaluation in short video.

### Cited Findings

**Estimators**
- **Replay** (Li, Chu, Langford, Wang, WSDM 2011, arXiv:1003.5956). With logs from a uniform random policy, an event is kept only when the evaluated policy's choice matches the logged arm. This is provably unbiased, and the paper validated it against online buckets on Yahoo! Front Page data. — [arXiv:1003.5956](https://arxiv.org/pdf/1003.5956) [verified]. Data efficiency is about 1/K of events (K = pool size), which is the practical limitation. [prior knowledge]
- **IPS / DM / DR**: Dudík, Langford & Li (ICML 2011, arXiv:1103.4601). — [prior knowledge, not re-verified]
- **SNIPS**: Swaminathan & Joachims, "The Self-Normalized Estimator for Counterfactual Learning" (NeurIPS 2015). — [prior knowledge]
- **Switch-DR**: Wang, Agarwal, Dudík (ICML 2017, arXiv:1612.01205). — [prior knowledge]
- **DR with optimistic shrinkage (DRos)**: Su, Dimakopoulou, Krishnamurthy, Dudík (ICML 2020, arXiv:1907.09623). — [prior knowledge]
- **MIPS**: Saito & Joachims, "Off-Policy Evaluation for Large Action Spaces via Embeddings" (ICML 2022, arXiv:2202.06317). It applies importance weights over *action embeddings* rather than actions and is unbiased under a "no direct effect" assumption. — [arXiv id prior knowledge; existence confirmed via OffCEM paper](https://arxiv.org/pdf/2305.08062)
- **OffCEM**: Saito, Ren, Joachims (ICML 2023, PMLR 202, arXiv:2305.08062). It splits reward into a cluster effect (handled by importance weights over clusters) and a residual effect (handled by regression). It is unbiased under "local correctness" and strictly generalises MIPS. — [PMLR](https://proceedings.mlr.press/v202/saito23b.html); [arXiv:2305.08062](https://arxiv.org/pdf/2305.08062) [verified]
- **Slate OPE / pseudo-inverse (PI) estimator**: Swaminathan, Krishnamurthy, Agarwal, Dudík, Langford, Jose, Zitouni (NeurIPS 2017, arXiv:1605.04812). It assumes additive slot-level rewards. — [prior knowledge]
- **Offline A/B testing for recommender systems**: Gilotte et al. (WSDM 2018, arXiv:1801.07030). Capped/normalised IPS for slates at Criteo. — [prior knowledge]
- **Open Bandit Pipeline** implements IPS, SNIPS, DM, DR, Switch-DR, DRos, MIPS, slate estimators, and synthetic or multiclass-reduction data generators. — [OBP paper](https://arxiv.org/pdf/2008.07146); [OBP docs](https://zr-obp.readthedocs.io/en/stable/about.html) [verified; estimator list partly from prior knowledge]

**Datasets with known propensities**
- **Open Bandit Dataset** (Saito, Aihara, Matsutani, Narita; NeurIPS 2021 Datasets & Benchmarks; arXiv:2008.07146). ZOZOTOWN fashion e-commerce. Logs come from an A/B test of a **uniform Random** policy vs **Bernoulli Thompson Sampling**: 7 days in late November 2019, three campaigns (all, men's, women's), top-3 slate positions, true propensities included. Per-campaign Bernoulli TS counts include 4,077,727 and 7,765,497 impressions. — [arXiv:2008.07146](https://arxiv.org/pdf/2008.07146v5); [zr-obp docs](https://zr-obp.readthedocs.io/en/latest/quickstart.html) [verified]. The total size (about 26M rows) and item counts (about 80 items in "all") are from prior knowledge and **unverified**. Because it contains *two* logging policies with known on-policy values, you can evaluate Random→BTS ("OPE of BTS using Random logs, compared with BTS's actual on-policy CTR"). This is a built-in ground truth for OPE validity. The licence is reportedly CC BY 4.0 [unverified].
- **Yahoo! R6A / R6B** (Webscope, "Yahoo! Front Page Today Module User Click Log"). Uniformly random article display in the Featured slot, used for the LinUCB paper (Li et al., WWW 2010, arXiv:1003.0146) and replay. Secondary sources confirm R6A/R6B are Front Page Today Module click logs, and that Webscope is limited to **non-commercial, academic users** (requests from .edu/university email; faculty, researchers, students). — [News recsys review arXiv:2009.04964](https://arxiv.org/pdf/2009.04964); [Campus Technology on Webscope eligibility](https://campustechnology.com/articles/2016/01/20/yahoo-releases-machine-learning-dataset-for-academic-researchers.aspx) [verified for access terms at a general Webscope level]. Details from prior knowledge, **unverified**: R6A covers May 1–10, 2009, about 45.8M events, with 6-dim user and article features. R6B covers October 2011, about 28M events, with 136-dim binary user features and raw IDs only. Approval can take days to weeks, so request access immediately.
- **KuaiRand** (Gao et al., CIKM 2022, arXiv:2208.08696). Kuaishou short video. Randomly chosen videos were inserted into the normal feed for two weeks without users knowing, with 12 feedback signals (click, like, view time, and others). Hosted at kuairand.com and on Zenodo (KuaiRand total about 11.1 GB; a 2026 record of about 6.9 GB adds video captions). — [arXiv:2208.08696](https://arxiv.org/pdf/2208.08696); [Zenodo](https://zenodo.org/records/10439422); [Zenodo 2026](https://zenodo.org/records/18159199) [verified]. **Licence not found in search** (believed CC BY-SA 4.0, unverified). The 2026 caption file matters because text features would allow an LLM-feature pipeline on KuaiRand.
- **KuaiRec** (Gao et al., CIKM 2022, arXiv:2202.10842). A nearly fully observed user–item matrix, so every user–item pair has ground truth. — [prior knowledge]
- **Coat** (Schnabel et al., ICML 2016, "Recommendations as Treatments", arXiv:1602.05352) and **Yahoo! R3** (Marlin & Zemel 2009). These are MNAR ratings with a random-exposure test set. They are rating data, not clicks or bandit logs, so they are weak fits for news bandits. — [prior knowledge]

**MIND specifically**
- The official MIND documentation states that "The orders of news in a impressions have been shuffled". The impression column lists displayed news and click labels, and the logging propensities are not released. — [MIND introduction doc](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md) [verified]

### Inferences
- **OPE on MIND is not credible.** Possible estimates (impression-level frequencies of exposure, or a learned propensity model of the production recommender) would be unidentified because of the shuffled order and the unknown policy. Reviewers who know OPE would distrust it. Say so explicitly in the paper. Use the semi-synthetic simulator as ground truth and bring in real data elsewhere.
- **Yahoo! R6 replay is the most natural real-data check.** It is the same domain (news), has uniform-random logging (replay is unbiased, so no propensity modelling is needed), and is the historical benchmark for the exact algorithm class (LinUCB, Thompson sampling). The main limitation is that **R6A has no article text**: only 6-dim features and article IDs. LLM text features (the paper's key manipulation) therefore cannot be reproduced on R6A unless article text is recovered, which is generally not possible. Q1/Q2 conclusions about *algorithms* (exploration strategy, linear vs neural, and so on) can be checked on R6. Conclusions about *LLM feature representations* cannot. **Verify whether R6B contains article text or only IDs before committing.** (I believe it contains only IDs and binary user features; unverified.)
- **OBD and KuaiRand** can test "does ranking of policies/feature sets in a learned simulator match an unbiased real-data estimate?" on a *different* domain:
  - **OBD:** fit the same kind of MLP simulator on OBD Random logs. Compare (a) policy rankings in that simulator with (b) replay or IPS rankings computed on the held-out Random logs. Also compare the simulator's predicted CTR for Bernoulli TS with the *actual* logged on-policy CTR of Bernoulli TS. This directly validates the *methodology* "learned simulator → policy ranking" on real data with known truth.
  - **KuaiRand:** using the random-exposure logs plus the 2026 caption file, the same check could be done *with text features*. This is the only listed dataset where LLM features and unbiased real evaluation can coexist. The domain is short video, not news.
- **Multiclass-to-bandit (Dudík 2011)** is weaker evidence than a learned simulator for the reviewers' concern, because it is fully synthetic. It is useful only as a sanity check of the bandit implementations (true values known).

### Gaps
- I could not retrieve the official Yahoo! Webscope R6 page or licence text. R6 eligibility and terms are inferred from general Webscope descriptions (non-commercial, academic). Whether R6 is still distributed in 2026 is **unverified**.
- KuaiRand's licence was not confirmed.
- I did not confirm the OBD total row count or licence.
- No source checked whether MIND's raw logs (pre-shuffle) are obtainable. They are presumed unavailable.

---

## Q4. Position bias and impression-level effects in MIND: should the simulator model position or slates?

### Takeaway
MIND cannot identify position bias, because the impression order was shuffled before release. Any position-bias parameter in the simulator would be *assumed*, not *fitted*. The defensible approach is to (i) state that the simulator is a single-slot (or exchangeable-slate) model fitted on position-agnostic data, and (ii) add a position-based model (PBM) or cascade layer with literature-plausible examination curves *as a sensitivity perturbation*, checking whether conclusions change.

### Cited Findings
- MIND impressions are shuffled, so the released data contain no display position. — [MIND doc](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md) [verified]
- Craswell, Zoeter, Taylor, Ramsey (WSDM 2008), "An experimental comparison of click position-bias models". It found the **cascade model** the best explanation for early-rank position bias in web search, based on rank-perturbation experiments. — [Microsoft Research page](https://www.microsoft.com/en-us/research/?p=156688) [verified exists; finding from search summary]
- Gilotte et al. note that uniformly shuffling a portion of recommendations is the classical way to overcome position bias, at a cost to user experience. This supports the reading that MIND's shuffle removes positional information rather than reflecting a randomised display. — [Ranking metrics on non-shuffled traffic](https://paperswithcode.com/paper/ranking-metrics-on-non-shuffled-traffic) [verified via search snippet]
- OBD logs slates of 3 positions with position-level propensities, so position effects *are* measurable there. — [arXiv:2008.07146](https://arxiv.org/pdf/2008.07146) [verified that it is a top-3 layout]
- Chuklin, Markov, de Rijke, *Click Models for Web Search* (Morgan & Claypool, 2015) is the standard reference for PBM, cascade, DBN, and UBM. — [prior knowledge]

### Inferences
- For a single-arm contextual bandit (one article per round), position bias does not matter. If the paper's policies select **one** article per round, state this and move on.
- If policies select slates (top-k), add a PBM perturbation P(click | item, pos k) = theta_k · sigma(f(u, i)), with theta_k drawn from a decaying curve (for example theta_k = 1/k^eta, eta in {0.5, 1}). Also try a cascade variant. Re-run and report rank stability. The examination curve could be *calibrated* from OBD's 3-position data as a real-data anchor in another domain.
- Impression-level "competition" effects (CTR depends on the other items in the impression) are also not modelled by an independent per-item MLP. One option is an MNL/softmax choice model over the candidate set with a no-click option, fitted on MIND impressions (identifiable without position). It would make a useful alternative simulator in the Q2 family.

### Gaps
- I found no published estimate of MSN News position-bias curves.

---

## Q5. Best-practice reporting for semi-synthetic bandit studies

### Takeaway
Reviewers expect multiple seeds that cover *both* simulator stochasticity and simulator-fitting uncertainty, confidence intervals rather than mean ± std of a few runs, effect sizes for paired policy comparisons, explicit statements of the simulator's assumptions, and a sensitivity analysis. The OPE and RL communities offer concrete templates (IEOE distributions, stratified bootstrap CIs, IQM).

### Cited Findings
- IEOE recommends reporting evaluation outcomes as distributions over hyperparameter and evaluation-policy choices rather than single configurations. — [arXiv:2108.13703](https://arxiv.org/pdf/2108.13703) [verified]
- Aouali et al. argue that simulation should be used with its assumptions laid out and is useful as a comparative tool, not as a forecast of absolute online numbers. — [arXiv:2209.08642](https://arxiv.org/pdf/2209.08642) [verified at abstract level]
- Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep RL at the Edge of the Statistical Precipice" (NeurIPS 2021, arXiv:2108.13264). It recommends stratified bootstrap CIs, IQM, and performance profiles for comparisons with few seeds. — [prior knowledge, not re-verified]
- Henderson et al., "Deep Reinforcement Learning that Matters" (AAAI 2018), shows seed variance can flip conclusions. — [prior knowledge]
- Winecoff et al. call for assumptions to be transparent and for simulations to be validated. — [arXiv:2107.14333](https://arxiv.org/pdf/2107.14333) [exists; summary from prior knowledge]

### Inferences: a reporting checklist for the paper
1. At least 10 seeds for headline results (bandit randomness × simulator Bernoulli draws), with **paired** seeds across policies (common random numbers) so differences have tight CIs.
2. 95% bootstrap CIs on cumulative regret/CTR and on *differences* between policies. Report the fraction of seeds where A beats B.
3. A **simulator-uncertainty** layer: 3–5 bootstrap-refit simulators, with results reported across them.
4. A sensitivity table (the Q2 family) with Kendall tau / pairwise-preservation / MMRV.
5. Report absolute numbers as *simulator-relative*. Phrase claims as orderings ("A > B robustly across simulators"), not real-traffic lifts.
6. Validation summary: tiers 1–3 achieved, tier 4 attempted on R6/OBD/KuaiRand (Q3).
7. Leakage controls: different encoder family, plus the leakage-ceiling result.

### Gaps
- No recsys-specific consensus checklist for semi-synthetic bandit studies was found. The items above are synthesised from adjacent communities.

---

## Prioritized credibility experiments (impact vs effort; ~1 month, single L40S)

Ranked by (reviewer impact) / (effort). GPU needs are small for everything except re-embedding.

| # | Experiment | Impact | Effort | Notes |
|---|---|---|---|---|
| 1 | **Simulator-family sensitivity**: re-run headline Q1/Q2 comparisons under the logistic and bilinear simulators (already fitted), logit temperature tau ∈ {0.5, 1.5, 2} re-anchored to 4.05%, base CTR ∈ {2%, 8%}, and added logit noise / user random intercepts. Report Kendall tau, pairwise preservation, and MMRV against S0. | Very high: directly answers "is it a simulator artefact?" | Low (2–4 days, CPU/GPU-light) | Most headline credibility for least work. |
| 2 | **Bootstrap-refit simulators** (3–5 MLPs on resampled training data) combined with paired seeds and bootstrap CIs on policy *differences*. | High | Low (2–3 days) | Separates epistemic simulator uncertainty from noise. |
| 3 | **Alternative-encoder simulator** (bge-m3 or gte/e5 family): refit the MLP and redo the representation comparisons, including a second leakage ceiling. | High for the LLM-feature claims specifically | Medium (embedding MIND-large articles is about 100k+ texts; a few GPU-hours on L40S; about 4–5 days total) | Shows representation conclusions don't hinge on Qwen3 geometry. |
| 4 | **Yahoo! R6A/R6B replay** of the bandit *algorithms* (LinUCB, TS, epsilon-greedy, neural variants), checking that algorithm orderings match the MIND-simulator orderings. | High: real, unbiased, same domain | Medium; **access risk** (Webscope approval, academic email). Replay itself is cheap. | Request access on day 1. Cannot test LLM text features (no article text in R6A; R6B unverified). |
| 5 | **Learned-simulator vs real-ground-truth check on OBD**: fit an MLP simulator on OBD Random logs. Compare (a) simulator-predicted vs actual on-policy CTR of Bernoulli TS, and (b) policy rankings in the simulator vs IPS/replay rankings on held-out Random logs. | High: validates the *method* "learned simulator → ranking" against truth | Medium (about 1 week; OBP tooling and public data) | Different domain (fashion), but the cleanest decision-level validation available. |
| 6 | **Slate/position sensitivity**: PBM (theta_k = 1/k^eta) and cascade layers, plus an MNL-over-impression choice simulator. Re-run any slate experiments. | Medium (high if policies select slates) | Low–Medium (3–4 days) | Explicitly note that MIND order is shuffled, so position cannot be fitted. |
| 7 | **KuaiRand with captions**: replay/IPS evaluation of LLM-feature vs ID-feature bandits on randomly exposed items using text features. | Medium–High (only dataset combining text features with unbiased exposure) | High (about 1.5–2 weeks; data wrangling, caption quality, licence check) | Stretch goal; confirm licence first. |
| 8 | **Popularity/recency-augmented simulator** (beta·log pop + freshness decay) as another family member. | Medium | Low (1–2 days) | Addresses the "news CTR is driven by recency" critique. |
| 9 | **Multiclass-to-bandit sanity check** (Dudík 2011) of the bandit implementations. | Low for the transfer concern | Low | Only if a reviewer questions implementation correctness. |

**Suggested 4-week plan:**
- Week 1: experiments #1 and #2, plus the Webscope request.
- Week 2: #3, plus #5 started.
- Week 3: #5 finished, plus #6/#8; #4 if access has arrived.
- Week 4: #4 or #7 (whichever is feasible), then the write-up with the sensitivity table and a "validation tiers" paragraph.

**Framing for the paper:** state that OPE on MIND is not possible (no propensities, shuffled order). Present the simulator as validated at tiers 1–3 (fit, marginal statistics, item-CTR correlation). Show decision-level robustness across a simulator family (tier "4-sim"). Add real-data decision-level checks on R6 (same domain, algorithms) and OBD (methodology). Claim *orderings*, not absolute lifts.
