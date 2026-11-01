# Q1: Cost of A/B Testing vs Adaptive Experimentation / Bandits, and Non-Stationary Bandits for News

Verification legend. **[V]** means I checked the paper this session through search or the arXiv abstract page. **[M]** means the citation comes from my own knowledge of the field and was not re-checked this session. [M] items are standard, widely cited works, but check volume, page and arXiv numbers before camera-ready. Anything marked **FLAG** has a specific uncertainty, which is noted next to it.

---

## KQ1. Which papers quantify the regret or opportunity cost of A/B testing compared with bandits?

### Takeaway
The closest prior work to our Q1 is Feit & Berman (2019), "Test & Roll". It shows that **A/B tests sized by power analysis are much larger than the profit-maximizing size, and that a properly sized test & roll gets close to Thompson-sampling regret.** A reviewer will very likely ask why our power-sized A/B test is not a strawman, so we must cite this paper and compare against it. Theory papers (Garivier, Lattimore & Kaufmann 2016 on explore-then-commit; Bubeck et al. 2009 on simple vs cumulative regret) explain *why* fixed-length ETC loses. Recent industry papers from Amazon, eBay, Adobe and Spotify show how practitioners actually trade regret against decision quality.

### Cited Findings
**A/B test cost and sizing (closest to our claim)**
- [V] Feit, E. M. & Berman, R. (2019). "Test & Roll: Profit-Maximizing A/B Tests." *Marketing Science* 38(6). arXiv:1811.00457. They derive a closed-form profit-maximizing test size that is "substantially smaller than typical hypothesis-test recommendations". The resulting design achieves "nearly the same expected regret as the multi-armed bandit" across a wide range of conditions. Tested on website, display-ad and catalog data. — [arXiv](https://arxiv.org/abs/1811.00457); [Drexel summary](https://lebow.drexel.edu/news/spending-too-much-time-ab-testing-can-hinder-profits)
- [M] Garivier, A., Lattimore, T. & Kaufmann, E. (2016). "On Explore-Then-Commit Strategies." NeurIPS 2016. arXiv:1605.08988. Shows that ETC, even with optimal tuning and a known gap, is asymptotically suboptimal by a factor of 2 compared with fully sequential policies such as UCB/TS (for two Gaussian arms). This is the theoretical reason "A/B (= ETC)" loses even when well sized.
- [M] Bubeck, S., Munos, R. & Stoltz, G. (2009). "Pure Exploration in Multi-armed Bandits Problems." ALT 2009 (journal version TCS 2011). arXiv:0802.2655. Shows a fundamental trade-off: policies with low cumulative regret have poor simple regret (best-arm identification), and the reverse also holds. This is the formal basis for our "BAI accuracy vs clicks" framing.
- [M] Azevedo, E. M., Deng, A., Montiel Olea, J. L., Rao, J. & Weyl, E. G. (2020). "A/B Testing with Fat Tails." *Journal of Political Economy* 128(12). With fat-tailed effects it can be better to run many small, lean experiments than a few large, well-powered ones. This also argues against power-sized tests.
- [M] Scott, S. L. (2015). "Multi-armed bandit experiments in the online service economy." *Applied Stochastic Models in Business and Industry* 31(1):37–45. Also Scott (2010), "A modern Bayesian look at the multi-armed bandit," ASMBI 26(6). Google Analytics content experiments used TS-based allocation. These are the early industry argument that bandits cut the cost of A/B tests.

**Best-arm identification (BAI) and adaptive experimentation for decision making**
- [M] Russo, D. (2020). "Simple Bayesian Algorithms for Best-Arm Identification." *Operations Research* 68(6):1625–1647 (COLT 2016 version). arXiv:1602.08448. Introduces Top-Two Thompson Sampling (TTTS), which fixes TS's tendency to under-sample the runner-up arm when the goal is identification.
- [M] Jourdan, M., Degenne, R., Baudry, D., de Heide, R. & Kaufmann, E. (2022). "Top Two Algorithms Revisited." NeurIPS 2022. arXiv:2206.05979.
- [M] Qin, C., Klabjan, D. & Russo, D. (2017). "Improving the Expected Improvement Algorithm." NeurIPS 2017 (TTEI).
- [M] Kaufmann, E., Cappé, O. & Garivier, A. (2016). "On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models." *JMLR* 17(1):1–42. Also Garivier, A. & Kaufmann, E. (2016), "Optimal Best Arm Identification with Fixed Confidence" (Track-and-Stop), COLT 2016, arXiv:1602.04589.
- [M] Audibert, J.-Y., Bubeck, S. & Munos, R. (2010). "Best Arm Identification in Multi-Armed Bandits." COLT 2010 (Successive Rejects, fixed budget). Karnin, Z., Koren, T. & Somekh, O. (2013). "Almost Optimal Exploration in Multi-Armed Bandits." ICML 2013 (Sequential Halving). Even-Dar, E., Mannor, S. & Mansour, Y. (2006). "Action Elimination and Stopping Conditions for the Multi-Armed Bandit and Reinforcement Learning Problems." *JMLR* 7 (Successive Elimination).
- [V] Kasy, M. & Sautmann, A. (2021). "Adaptive Treatment Assignment in Experiments for Policy Choice." *Econometrica* 89(1):113–132. Proposes "exploration sampling", where assignment probabilities are a concave function of the posterior probability that each arm is optimal. — [CESifo WP](https://www.cesifo.org/DocDL/cesifo1_wp7778.pdf)
- [V] Ariu, K., Kato, M., Komiyama, J., McAlinn, K. & Qin, C. (2021). "Policy Choice and Best Arm Identification: Comments on 'Adaptive Treatment Assignment in Experiments for Policy Choice'." arXiv:2109.08229. Shows that policy choice is equivalent to BAI, that parts of Kasy & Sautmann's Theorem 1 are incorrect, and gives a counterexample to Thm 1(3). **If we cite Kasy & Sautmann, cite this comment as well.** — [arXiv](https://arxiv.org/abs/2109.08229)
- [V] Horn, S. & Sloman, S. J. (2022). "A Comparison of Methods for Adaptive Experimentation." ReALML workshop @ ICML 2022. arXiv:2207.00683. Compares TS, tempered TS and exploration sampling on welfare and estimation accuracy across numbers of waves. TS's relative performance depends on the number of waves, and exploration sampling performs about as well as random assignment in most cases. — [arXiv](https://arxiv.org/abs/2207.00683)
- [V] Che, E. & Namkoong, H. (2023). "Adaptive Experimentation at Scale: A Computational Framework for Flexible Batches." arXiv:2303.11582. The title changed between versions. **FLAG:** venue unconfirmed; a NeurIPS 2022 workshop listing exists. Argues that standard bandits assume continual reallocation, which is infeasible with delays and batches. Their Residual Horizon Optimization "significantly improves statistical power" over Bayesian bandits such as TS when there are few reallocation epochs. A companion benchmark, AExGym, is at arXiv:2408.04531. — [arXiv](https://arxiv.org/abs/2303.11582); [AExGym](https://arxiv.org/html/2408.04531v1)

**Industry papers on adaptive experimentation**
- [V] Fiez, T., Nassif, H., Chen, Y.-C., Gamez, S. & Jain, L. (2024). "Best of Three Worlds: Adaptive Experimentation for Digital Marketing in Practice." WWW 2024. doi:10.1145/3589334.3645504. arXiv:2402.10870. Amazon reports that AED guarantees are "poorly understood" outside stationary conditions. They build a framework that targets the counterfactual best arm, limits opportunity cost and is robust to drift. **This is the most relevant recent paper for our drift results.** (A third-party summary says their algorithm identified the best arm faster than TS with low regret; not verified against the paper.) — [arXiv](https://arxiv.org/abs/2402.10870); [Amazon Science](https://amazon.science/publications/best-of-three-worlds-adaptive-experimentation-for-digital-marketing-in-practice)
- [V] Zhang, Z. & Yuan, T. (2023). "Practical Batch Bayesian Sampling Algorithms for Online Adaptive Traffic Experimentation." arXiv:2305.14704 (eBay; preprint, no venue). Batched TS and Top-Two TS variants that use only summary batch statistics. Naive TS "can falsely identify a single best arm" when several arms are equally good. They recommend WB-TTTS as an alternative to fixed-horizon A/B. — [arXiv](https://arxiv.org/abs/2305.14704)
- [M] Hill, D. N., Nassif, H., Liu, Y., Iyer, A. & Vishwanathan, S. V. N. (2017). "An Efficient Bandit Algorithm for Realtime Multivariate Optimization." KDD 2017. arXiv:1810.09558 (Amazon, TS over layouts).
- [M] Agarwal, A. et al. (2016). "Making Contextual Decisions with Low Technical Debt." arXiv:1606.03966 (Microsoft Decision Service).
- [M] McInerney, J., Lacker, B., Hansen, S., Higley, K., Bouchard, H., Gruson, A. & Mehrotra, R. (2018). "Explore, Exploit, and Explain: Personalizing Explainable Recommendations with Bandits." RecSys 2018 (Spotify).
- [M] Bendada, W., Salha, G. & Bontempelli, T. (2020). "Carousel Personalization in Music Streaming Apps with Contextual Bandits." RecSys 2020. arXiv:2009.06546 (Deezer). Includes a semi-synthetic simulation setup that is a good methodological analogue to ours.
- [M] Kohavi, R., Tang, D. & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing.* Cambridge University Press. The standard reference for how A/B tests are run in practice (Microsoft, LinkedIn, etc.). Cite it to justify our power-analysis sizing as "standard practice".
- **FLAG:** I found no Netflix or LinkedIn paper that directly measures A/B-vs-bandit opportunity cost. Their bandit papers are about personalization (e.g., the Netflix artwork blog, 2017), not experiment cost.
- [V] There is a non-peer-reviewed 2026 arXiv paper (arXiv:2607.23696, "Offline-to-Online Creative Optimization with Generative Models and Adaptive Testing"). It reportedly finds that TS cut in-experiment regret by 66% at K=30, T=1e5. That setup is very close to ours, but I did not check the paper's date, authors or claims. — [arXiv](https://arxiv.org/pdf/2607.23696)

### Inferences
- Our headline "power-sized A/B loses ~9x more clicks than TS" is consistent with the literature, but a reviewer will say it overstates the case. Feit & Berman already show that most of the gap closes when the test is sized for profit rather than power. We need a Test & Roll–sized ETC and an "oracle-optimal-length ETC" (the ETC-opt we already have). We should then report what share of the gap comes from **sizing** and what share from **non-adaptivity**. Garivier et al. 2016 predicts that roughly a 2x factor survives optimal sizing.
- The literature now frames this as a **multi-objective** problem: in-experiment regret, decision quality (PCS / simple regret) and inference. A paper that reports only cumulative clicks will look dated against Simchi-Levi & Wang (2023/2025) and Fiez et al. (2024).

### Gaps
- No peer-reviewed 2024–2026 paper I found gives a direct "X-fold more lost clicks than TS" number to put next to our ~9x. Feit & Berman report near-equal regret for profit-sized tests, not a ratio against power-sized tests (the ratio would need to be read from their tables).
- I could not confirm a LinkedIn or Netflix paper on bandit-vs-A/B experiment cost.

---

## KQ2. Which stronger A/B baselines would reviewers expect?

### Takeaway
At minimum: (a) **sequential tests with early stopping**, either group-sequential (O'Brien–Fleming / alpha spending) or always-valid (mSPRT / confidence sequences); (b) a **multi-arm elimination** design (Successive Elimination or Sequential Halving), because a Bonferroni-sized K=50 A/B test "never finishing" is exactly the failure that elimination designs fix; (c) **Test & Roll** profit-optimal sizing; (d) **Top-Two TS** as the adaptive BAI method. Spotify's public comparison shows group-sequential tests have higher power than always-valid methods when data arrives in batches, so a GST is the more realistic "best practice A/B" baseline.

### Cited Findings
- [V] Johari, R., Koomen, P., Pekelis, L. & Walsh, D. "Always Valid Inference: Continuous Monitoring of A/B Tests." *Operations Research* (DOI 10.1287/opre.2021.2135). Earlier preprint arXiv:1512.04922 ("Bringing Sequential Analysis to A/B Testing", by Johari, Pekelis & Walsh only). The mSPRT-based always-valid p-values are deployed on Optimizely. **FLAG:** sources disagree on the publication year (2021 online vs 2022 print, vol. 70(3)) and on whether Koomen is an author; check on INFORMS. — [arXiv](https://arxiv.org/abs/1512.04922)
- [M] Johari, R., Koomen, P., Pekelis, L. & Walsh, D. (2017). "Peeking at A/B Tests: Why It Matters, and What to Do About It." KDD 2017.
- [V] Maharaj, A. V., …, Ramdas, A. (2023). "Anytime-Valid Confidence Sequences in an Enterprise A/B Testing Platform." WWW 2023 Companion/Proceedings. arXiv:2302.10108 (Adobe). Deployed asymptotic confidence sequences that allow continuous monitoring and data-dependent stopping, evaluated on thousands of real experiments. — [arXiv](https://arxiv.org/abs/2302.10108)
- [M] Howard, S. R., Ramdas, A., McAuliffe, J. & Sekhon, J. (2021). "Time-uniform, nonparametric, nonasymptotic confidence sequences." *Annals of Statistics* 49(2). arXiv:1810.08240. Ramdas, A., Grünwald, P., Vovk, V. & Shafer, G. (2023). "Game-Theoretic Statistics and Safe Anytime-Valid Inference." *Statistical Science* 38(4). arXiv:2210.01948.
- [V] Spotify Engineering (2023). "Choosing a Sequential Testing Framework — Comparisons and Discussions" (blog, March 2023; authors from memory: M. Schultzberg & S. Ankargren). Compares GST, mSPRT, generalized AVI (Howard et al.), corrected-alpha and Bonferroni. Spotify chose **GST** because its data arrives in batches and it can estimate the maximum sample size. Its documentation states that GSTs with a fixed number of interim analyses had higher power than AVI. — [Spotify Engineering](https://engineering.atspotify.com/2023/3/choosing-sequential-testing-framework-comparisons-and-discussions); [Confidence docs](https://confidence.spotify.com/docs/experiments/stats/sequential-tests)
- [M] Classic group-sequential designs: Pocock, S. J. (1977), *Biometrika* 64(2); O'Brien, P. C. & Fleming, T. R. (1979), *Biometrics* 35(3); Lan, K. K. G. & DeMets, D. L. (1983), "Discrete sequential boundaries for clinical trials," *Biometrika* 70(3) (alpha spending). mSPRT origin: Robbins, H. (1970), "Statistical methods related to the law of the iterated logarithm," *Ann. Math. Statist.* 41(5).
- [M] Lindon, M., Sanden, C. & Shirikian, V. (2022). "Rapid Regression Detection in Software Deployments through Sequential Testing." KDD 2022. arXiv:2205.14762 (Netflix; anytime-valid tests). **FLAG:** author list from memory.
- [M] Bayesian A/B: Deng, A. (2015). "Objective Bayesian Two Sample Hypothesis Testing for Online Controlled Experiments." WWW 2015 Companion (Microsoft). Stucchio, C. (2015) VWO "Bayesian A/B Testing at VWO" whitepaper (**FLAG:** grey literature).
- [V] Feit & Berman (2019), Test & Roll (above). This is the profit-optimal sizing baseline and includes an online calculator. — [arXiv](https://arxiv.org/abs/1811.00457)
- [V] Zhang & Yuan (2023, eBay), arXiv:2305.14704. Batched Top-Two TS proposed as a direct replacement for fixed-horizon A/B. — [arXiv](https://arxiv.org/abs/2305.14704)

### Inferences
- Recommended baseline ladder, from weakest to strongest "A/B family":
  1. power-sized fixed A/B (have);
  2. ETC with oracle-optimal length (have);
  3. **Test & Roll profit-optimal length** (needs only the prior on CTRs, which we can take from MIND);
  4. **group-sequential A/B (O'Brien–Fleming, 5–10 looks)** with early stopping, then commit;
  5. **mSPRT / confidence-sequence A/B** that stops when significant;
  6. for K>2: **Successive Elimination / Sequential Halving** (an "A/B tournament" that retires losers), plus a Bonferroni or Dunnett multi-arm test as the naive version.
  On the bandit/BAI side: **Top-Two TS**. Items 4–6 answer "A/B never finishes at 50 arms": with elimination, losing arms drop out early.
- Expected result: early stopping will cut A/B regret substantially at large gaps (easy problems) but not at small gaps or low CTR. Elimination at K=50 should close much of the gap to TS. Either outcome strengthens the paper, because the claim becomes "even best-practice A/B loses Y%".

### Gaps
- I found no paper that runs this full ladder (power-sized, Test & Roll, GST, mSPRT, SE, TTTS, TS) on one news benchmark. This is a contribution we could claim, but phrase it cautiously.

---

## KQ3. What does the literature say about the inference-vs-regret trade-off, and how should we report both?

### Takeaway
Adaptive allocation biases naive sample means (in general negatively for the selected arms) and breaks standard CIs. Valid inference needs IPW/AIPW with adaptive weights, batched-normal methods, or a designed mixture with uniform randomization. Simchi-Levi & Wang (2023/2025) formalize a Pareto frontier between regret and ATE estimation error. Liang & Bojinov's Mixture Adaptive Design gets anytime-valid inference for any bandit at small regret cost. We should report **cumulative regret, PCS / simple regret of the final arm, and bias + CI coverage of the effect estimate** for every method.

### Cited Findings
- [M] Nie, X., Tian, X., Taylor, J. & Zou, J. (2018). "Why Adaptively Collected Data Have Negative Bias and How to Correct for It." AISTATS 2018. arXiv:1708.01977.
- [M] Shin, J., Ramdas, A. & Rinaldo, A. (2019). "Are Sample Means in Multi-Armed Bandits Positively or Negatively Biased?" NeurIPS 2019. arXiv:1905.11397.
- [M] Deshpande, Y., Mackey, L., Syrgkanis, V. & Taddy, M. (2018). "Accurate Inference for Adaptive Linear Models." ICML 2018.
- [M] Zhang, K. W., Janson, L. & Murphy, S. A. (2020). "Inference for Batched Bandits." NeurIPS 2020. arXiv:2002.03217. Shows that OLS is not asymptotically normal under bandit allocation when the arm gap is zero, and proposes the Batched OLS estimator.
- [M] Hadad, V., Hirshberg, D. A., Zhan, R., Wager, S. & Athey, S. (2021). "Confidence Intervals for Policy Evaluation in Adaptive Experiments." *PNAS* 118(15):e2014602118. arXiv:1911.02768. Adaptively weighted AIPW gives asymptotically normal estimates.
- [V] Simchi-Levi, D. & Wang, C. (2023). "Multi-armed Bandit Experimental Design: Online Decision-Making and Adaptive Inference." AISTATS 2023 (PMLR 206). Extended version in *Management Science* 71(6):4828–4846 (2025), doi:10.1287/mnsc.2023.00492. Gives a necessary and sufficient condition for Pareto optimality between regret and ATE estimation error, and a tunable Pareto-optimal design. — [PMLR](https://proceedings.mlr.press/v206/simchi-levi23a.html); [RePEc](https://ideas.repec.org/a/inm/ormnsc/v71y2025i6p4828-4846.html)
- [V] Liang, B. & Bojinov, I. (2023+). "An Experimental Design for Anytime-Valid Causal Inference on Multi-Armed Bandits." arXiv:2311.05794. The Mixture Adaptive Design (MAD) mixes any bandit with Bernoulli randomization whose probability decays deterministically. It gives anytime-valid ATE inference, and its regret approaches that of the underlying bandit. **FLAG:** venue unconfirmed; the paper has several revisions. — [arXiv](https://arxiv.org/abs/2311.05794)
- [M] Erraqabi, A., Lazaric, A., Valko, M., Brunskill, E. & Liu, Y.-E. (2017). "Trading off Rewards and Errors in Multi-Armed Bandits." AISTATS 2017.
- [M] Offer-Westort, M., Coppock, A. & Green, D. P. (2021). "Adaptive Experimental Design: Prospects and Applications in Political Science." *American Journal of Political Science* 65(4):826–844. Shows that TS gives poor estimates for non-best arms, and proposes "control-augmented" TS that keeps a floor of allocation to control.
- [V] Horn & Sloman (2022). They measure both welfare and estimation accuracy, and propose hybrid loss measures. — [arXiv](https://arxiv.org/abs/2207.00683)
- [V] Zhang & Yuan (2023). They point to false best-arm claims from naive TS when arms are tied, which is a decision-quality failure. — [arXiv](https://arxiv.org/abs/2305.14704)

### Inferences
- Concrete reporting proposal:
  1. **In-experiment regret** (have).
  2. **Total regret over a population N**, Test & Roll style: in-experiment loss plus the rollout loss from committing to the chosen arm for the remaining N−T users. This is what a news product actually cares about, and it penalizes pure-regret bandits that never "decide".
  3. **PCS and simple regret** of the arm each method would deploy at T. For bandits, use the most-pulled arm or the posterior argmax.
  4. **Bias and 95% CI coverage** of the estimated lift (best vs control): naive means vs Hadad AW-AIPW (or plain IPW with known propensities, which is easy for TS via Monte Carlo propensities).
  5. Optional **Pareto plot** of regret vs estimation error / PCS, with tempered TS, TTTS, MAD or a uniform-floor TS tracing the frontier.
- Our finding that "A/B BAI accuracy is only 60–80%" needs the matching TS/TTTS PCS numbers. Otherwise a reviewer will ask whether TS is any better at deciding. The literature suggests TTTS > A/B ≥ TS on PCS at fixed budget, but that is our hypothesis, not a cited result.

### Gaps
- I did not verify whether Hadad et al. provide a public package. I believe Athey's lab released R code for adaptive weighting; check before relying on it.

---

## KQ4. Which non-stationary bandit methods will reviewers expect for drift, and what news-specific work exists?

### Takeaway
Discounted TS on its own is not enough. Reviewers will expect at least **SW-UCB and D-UCB** (Garivier & Moulines), a **change-detection restart method** (CUSUM-UCB, M-UCB or GLR-klUCB), and possibly **SW-TS**. A fair A/B counterpart under drift is **periodic re-testing**: re-run the A/B test every W steps. Yahoo! R6 has been a non-stationary bandit benchmark since about 2015, and M-UCB was evaluated on it. MIND has already been replayed in timestamp order for non-stationary bandit evaluation (Meta/Stanford, arXiv:2310.07786).

### Cited Findings
- [M] Kocsis, L. & Szepesvári, C. (2006). "Discounted UCB." 2nd PASCAL Challenges Workshop. Garivier, A. & Moulines, E. (2011). "On Upper-Confidence Bound Policies for Switching Bandit Problems." ALT 2011. arXiv:0805.3415 (D-UCB, SW-UCB, with lower bounds for abrupt changes).
- [M] Besbes, O., Gur, Y. & Zeevi, A. (2014). "Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards." NeurIPS 2014 (variation-budget model, the right framing for *gradual* drift).
- [M] Raj, V. & Kalyani, S. (2017). "Taming Non-stationary Bandits: A Bayesian Approach." arXiv:1707.09727 (discounted TS; the source for our dTS).
- [M] Trovò, F., Paladino, S., Restelli, M. & Gatti, N. (2020). "Sliding-Window Thompson Sampling for Non-Stationary Settings." *JAIR* 68.
- [M] Liu, F., Lee, J. & Shroff, N. (2018). "A Change-Detection Based Framework for Piecewise-Stationary Multi-Armed Bandit Problem." AAAI 2018. arXiv:1711.03539 (CUSUM-UCB, PHT-UCB).
- [V] Cao, Y., Wen, Z., Kveton, B. & Xie, Y. (2019). "Nearly Optimal Adaptive Procedure with Change Detection for Piecewise-Stationary Bandit." AISTATS 2019. arXiv:1802.03692 (M-UCB). The detector is a sliding-window two-sample mean test, with experiments on the Yahoo! dataset. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1802.03692)
- [M] Besson, L., Kaufmann, E., Maillard, O.-A. & Seznec, J. (2022). "Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits." *JMLR* 23. arXiv:1902.01575 (GLR-klUCB; implemented in SMPyBandits).
- [M] Auer, P., Gajane, P. & Ortner, R. (2019). "Adaptively Tracking the Best Bandit Arm with an Unknown Number of Distribution Changes." COLT 2019 (AdSwitch). Wei, C.-Y. & Luo, H. (2021). "Non-stationary Reinforcement Learning without Prior Knowledge: An Optimal Black-box Approach." COLT 2021 (MASTER).
- [V] Komiyama, J., Fouché, E. & Honda, J. (2024). "Finite-time Analysis of Globally Nonstationary Multi-Armed Bandits." *JMLR* 25. arXiv:2107.11419. ADR-bandit (adaptive windowing) covers both abrupt and gradual "global" changes, and the paper includes real-world experiments. — [JMLR](https://jmlr.org/papers/v25/21-0916.html)
- [V] Wei, L. & Srivastava, V. "On Abruptly-Changing and Slowly-Varying Multiarmed Bandit Problems." arXiv:1802.08380 (ACC 2018) (SW-UCB#, LM-DSEE; regret for both abrupt and slowly varying cases). — [arXiv](https://arxiv.org/abs/1802.08380)
- [V] A 2026 arXiv paper (arXiv:2604.21432, on restless/rested rotting bandits) states that Yahoo! R6A "has been used as a benchmark for non-stationary bandits since 2015". — [arXiv](https://arxiv.org/pdf/2604.21432)
- [V] SMPyBandits documentation distinguishes piecewise-stationary (abrupt) from slowly-varying bandits, and implements many of the methods above. It could be used as an off-the-shelf reference implementation. — [SMPyBandits docs](https://smpybandits.readthedocs.io/en/stable/NonStationaryBandits.html)
- Contextual / news-specific:
  - [M] Wu, Q., Iyer, N. & Wang, H. (2018). "Learning Contextual Bandits in a Non-stationary Environment." SIGIR 2018 (dLinUCB, evaluated on Yahoo! news). **FLAG:** I believe this is arXiv:1805.09365, but did not confirm.
  - [M] Russac, Y., Vernade, C. & Cappé, O. (2019). "Weighted Linear Bandits for Non-Stationary Environments." NeurIPS 2019. arXiv:1909.09146 (D-LinUCB).
  - [M] Cheung, W. C., Simchi-Levi, D. & Zhu, R. (2019). "Learning to Optimize under Non-Stationarity." AISTATS 2019 (SW-LinUCB, BOB).
  - [M] Liu, Y., Van Roy, B. & Xu, K. (2023). "Nonstationary Bandit Learning via Predictive Sampling." AISTATS 2023. arXiv:2205.01970.
  - [V] "Non-Stationary Contextual Bandit Learning via Neural Predictive Ensemble Sampling." arXiv:2310.07786 (**FLAG:** authors not confirmed; I believe Zhu, Liu, Xu et al., Meta/Stanford). It replays **MIND** in timestamp order to model non-stationary preferences (10,000 sampled users) and reports CTR gains over baselines. — [arXiv](https://arxiv.org/abs/2310.07786)
  - [V] Fiez et al. (2024, WWW), above. Robustness of adaptive experiments to drift in a real marketing system.
  - [M] Levine, N., Crammer, K. & Mannor, S. (2017). "Rotting Bandits." NeurIPS 2017. News CTR decays with article age, so "rotting" is a natural drift model to mention.

### Inferences
- Minimum drift baseline set: SW-UCB, D-UCB, SW-TS (or our dTS), M-UCB or GLR-klUCB (change detection plus restart), and "A/B with periodic re-test". Tune window and discount on a separate seed or condition, not on the test condition.
- For gradual drift, frame results with the Besbes et al. variation budget V_T so that drift intensity is a controlled axis.
- Our "A/B and eps-greedy lock in" result is expected by construction. Reviewers will only be convinced if a periodically refreshed A/B also loses. That is the realistic practice comparison.

### Gaps
- I found no paper that compares A/B (with or without re-testing) against non-stationary bandits on a news benchmark. The closest is Fiez et al. (marketing, not news).

---

## KQ5. Is it standard to test delayed or batched feedback?

### Takeaway
Yes. It is increasingly expected for any "practical experimentation" claim. Chapelle & Li (2011) already tested delayed/batched updates on Yahoo! news and found TS more robust than UCB. Batched bandits have a solid theory (Perchet et al.; Gao et al.; Kalkanli & Özgür for TS). Che & Namkoong and Spotify's GST choice both come from the fact that production data arrives in batches. A small batch-size sweep is cheap and directly tests whether our TS-vs-UCB ranking (UCB1 c=0.1 winning at low CTR) survives realistic update schedules. **This is a likely reviewer question, because deterministic UCB degrades more than TS under batching.**

### Cited Findings
- [V] Chapelle, O. & Li, L. (2011). "An Empirical Evaluation of Thompson Sampling." NIPS 2011. Uses Yahoo! front-page news data. Secondary sources report that their experiments found TS more robust to delayed/batched feedback than the alternatives. (I did not re-read the delay section directly; check the PDF.) — [NeurIPS proceedings](https://papers.nips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html)
- [V] Kalkanli, C. & Özgür, A. (2021). "Batched Thompson Sampling." NeurIPS 2021. Achieves O(log T) problem-dependent and O(√(T log T)) minimax regret with O(log T) batches, and O(log log T) batches in expectation, without knowing T. — [NeurIPS](https://neurips.cc/virtual/2021/poster/27500)
- [M] Perchet, V., Rigollet, P., Chassang, S. & Snowberg, E. (2016). "Batched Bandit Problems." *Annals of Statistics* 44(2). Gao, Z., Han, Y., Ren, Z. & Zhou, Z. (2019). "Batched Multi-armed Bandits Problem." NeurIPS 2019.
- [M] Joulani, P., György, A. & Szepesvári, C. (2013). "Online Learning under Delayed Feedback." ICML 2013. Vernade, C., Cappé, O. & Perchet, V. (2017). "Stochastic Bandit Models for Delayed Conversions." UAI 2017.
- [V] Wu, H. & Wager, S. (2022). "Thompson Sampling with Unrestricted Delays." arXiv:2202.12431 (title from memory, **FLAG**). Gives regret bounds for TS under arbitrary delay distributions; TS outperforms methods designed specifically for delays. — [arXiv](https://arxiv.org/abs/2202.12431)
- [M] Karbasi, A., Mirrokni, V. & Shadravan, M. (2021). "Parallelizing Thompson Sampling." NeurIPS 2021.
- [V] Che & Namkoong (2023), arXiv:2303.11582 (batched adaptive experiments, few reallocation epochs). Spotify (2023) chose GST because "data arrived in batches". — [arXiv](https://arxiv.org/abs/2303.11582); [Spotify](https://engineering.atspotify.com/2023/3/choosing-sequential-testing-framework-comparisons-and-discussions)
- [V] A 2025 news-headline bandit paper (arXiv:2506.06334, "Preference-based learning for news headline recommendation") simulates a 7-day feedback delay following Chapelle & Li. This is evidence that delay is now modeled in news bandit papers. — [arXiv](https://arxiv.org/pdf/2506.06334)

### Inferences
- Add a batch-size axis B ∈ {1, 100, 1,000, 10,000} (all methods update only at batch ends). Optionally add a geometric click-delay model. The A/B test is unaffected by batching, so batching *narrows* the bandit advantage. Showing how much survives is a strong robustness result.
- Our tuned UCB1 (c=0.1) win at low CTR may disappear under batching. Deterministic UCB pulls the same arm for a whole batch, while TS randomizes. This is worth checking before the claim goes in the paper.

### Gaps
- I did not find a news-specific paper that quantifies how A/B vs bandit regret ratios change with batch size.

---

## KQ6. Are there A/B-vs-bandit comparisons on MIND, Yahoo! R6, Open Bandit Dataset or other news data?

### Takeaway
Bandit evaluations on Yahoo! R6 are classic (Li et al. 2010/2011; Chapelle & Li 2011; M-UCB). MIND has been used in several contextual-bandit papers (2022–2026). **I found no paper that explicitly benchmarks a power-sized A/B test against bandits on MIND.** That supports a novelty claim, but phrase it as "to our knowledge".

### Cited Findings
- [M] Li, L., Chu, W., Langford, J. & Schapire, R. E. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation." WWW 2010. arXiv:1003.0146 (LinUCB, Yahoo! Today module).
- [M] Li, L., Chu, W., Langford, J. & Wang, X. (2011). "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms." WSDM 2011. arXiv:1003.5956 (replay method; the R6 evaluation protocol).
- [M] Agarwal, D., Chen, B.-C. & Elango, P. (2009). "Explore/Exploit Schemes for Web Content Optimization." ICDM 2009 (Yahoo! front page, an early news explore/exploit paper with time-varying CTR).
- [M] Wu, F. et al. (2020). "MIND: A Large-scale Dataset for News Recommendation." ACL 2020.
- [M] Saito, Y., Aihara, S., Matsutani, M. & Narita, Y. (2021). "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation." NeurIPS 2021 Datasets & Benchmarks. arXiv:2008.07146 (ZOZOTOWN fashion, not news; logged under uniform random and Bernoulli TS policies, so the logged random-vs-TS CTR is a real-data A/B-vs-bandit data point).
- [V] Zhang, M. et al. (2022). "Two-Stage Neural Contextual Bandits for Personalised News Recommendation." arXiv:2206.14648. MIND simulation (5 articles per user, 2,000 iterations, 5 trials); argues exploration reduces filter bubbles. **FLAG:** full author list and venue not confirmed. — [arXiv](https://arxiv.org/abs/2206.14648)
- [V] "Scalable Neural Contextual Bandit for Recommender Systems." arXiv:2306.14834 (Meta; ENR/"Epistemic Neural Recommendation"). Notes that MIND impressions give full ground truth over the shown slate, so the outcome of any chosen action can be evaluated. **FLAG:** authors and venue not confirmed (possibly CIKM 2023). — [arXiv](https://arxiv.org/abs/2306.14834)
- [V] arXiv:2310.07786 (non-stationary, MIND timestamp replay), above. — [arXiv](https://arxiv.org/abs/2310.07786)
- [V] "Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations." arXiv:2110.09905 (MIND, interactive exploration). — [arXiv](https://arxiv.org/abs/2110.09905)
- [V] "Calibration-Gated LLM Pseudo-Observations for Online Contextual Bandits." arXiv:2604.14961 (2026; MIND-small as an online bandit environment). — [arXiv](https://arxiv.org/abs/2604.14961)
- [M] Semi-synthetic simulators reviewers may expect us to position against: Rohde, D. et al. (2018), "RecoGym: A Reinforcement Learning Environment for the Problem of Product Recommendation in Online Advertising," arXiv:1808.00720; Ie, E. et al. (2019), "RecSim: A Configurable Simulation Platform for Recommender Systems," arXiv:1909.04847.
- [V] The search summary noted there is no single definitive MIND bandit simulator, and that protocols differ across papers. — (synthesis of the search results above)

### Inferences
- Position the paper as "to our knowledge, the first systematic A/B-vs-bandit cost comparison on MIND-large across K, T and drift". Cite 2206.14648, 2306.14834 and 2310.07786 as prior MIND bandit simulations, and Li et al. 2011 replay as the classical alternative.
- A cheap external-validity check is to rerun the non-contextual K-arm comparison on **Yahoo! R6A replay** (Li et al. 2011 replay estimator). Note that R6 access requires a Yahoo Webscope agreement, which the user must request themselves. The Open Bandit Dataset is freely downloadable.

### Gaps
- R6 access terms and turnaround time were not checked.
- I could not confirm whether any 2025–2026 RecSys/KDD paper compares A/B and bandits on MIND specifically. Searches found none.

---

## Additional notes on the "tuned UCB1 beats TS at low CTR" finding
- [M] Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47:235–256 (UCB1, UCB1-Tuned). Garivier, A. & Cappé, O. (2011). "The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond." COLT 2011. arXiv:1102.2490. Kaufmann, E., Cappé, O. & Garivier, A. (2012). "On Bayesian Upper Confidence Bounds for Bandit Problems." AISTATS 2012 (Bayes-UCB).
- [M] Kuleshov, V. & Precup, D. (2014). "Algorithms for Multi-Armed Bandit Problems." arXiv:1402.6028. An empirical study showing that simple tuned heuristics often beat theoretically motivated UCB variants. This is a precedent for tuned-constant results, and for the caveat that tuning on the test condition flatters a method.
- [V] Chapelle & Li (2011) recommend TS as a standard baseline. — [NeurIPS](https://papers.nips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html)
- Inference: reviewers will ask (i) whether c=0.1 was tuned on held-out conditions, (ii) whether **KL-UCB** (parameter-free, made for Bernoulli and strong at low rates) matches tuned UCB1, and (iii) whether **TS with an informative Beta prior** at the MIND base CTR (instead of Beta(1,1)) closes the gap. Beta(1,1) over-explores when CTR ≈ 3%. These are three cheap additions that make the claim robust.

---

## Prioritized proposed experiments for Q1 (impact vs effort, ~1 month)

Effort assumes the existing simulator and harness; "days" means person-days. Ordered by impact/effort ratio.

| # | Experiment | Why it matters (reviewer concern) | Effort | Feasible in 1 mo? |
|---|---|---|---|---|
| 1 | **Report decision quality and total-population regret for all methods**: PCS / simple regret of the deployed arm at T, plus Test & Roll "total regret" = in-test + rollout over N users | Answers "bandits don't decide" and makes the A/B comparison fair (Feit & Berman; Bubeck et al. 2009; Simchi-Levi & Wang). Our 60–80% A/B BAI number needs TS/UCB counterparts | 1–2 days (logging only) | Yes |
| 2 | **Stronger A/B family**: Test & Roll sizing (prior from MIND CTRs); group-sequential O'Brien–Fleming with 5–10 looks; mSPRT / confidence-sequence stopping; for K>2, Successive Elimination and Sequential Halving | The main "strawman" criticism. Shows how much of the ~9x gap comes from sizing vs non-adaptivity. Fixes "never finishes at K=50" | 4–6 days (SE/SH/T&R trivial; GST via alpha-spending formulas; mSPRT closed form for Bernoulli/normal) | Yes |
| 3 | **Top-Two TS** (plus optionally TS with a uniform/control floor) | The standard adaptive BAI method (Russo 2020; Jourdan et al. 2022). Lets us claim a position on the regret–identification frontier | 1 day | Yes |
| 4 | **Robustness of the UCB-vs-TS ranking**: KL-UCB, Bayes-UCB, TS with an informative Beta prior at the base CTR; tune c on held-out seeds/conditions | Protects the "tuned UCB1 beats TS at 3% CTR" claim from a tuning-leakage objection | 1–2 days | Yes |
| 5 | **Batched/delayed feedback sweep** (B ∈ {1, 1e2, 1e3, 1e4}; optional geometric delay) | Production realism (Chapelle & Li 2011; Kalkanli & Özgür 2021; Che & Namkoong; Spotify). Tests whether the bandit advantage and the UCB win survive batching | 2–3 days | Yes |
| 6 | **Non-stationary baselines**: SW-UCB, D-UCB, SW-TS, M-UCB or GLR-klUCB; plus **A/B with periodic re-test** | Discounted TS alone is thin. Periodic re-test is the realistic A/B under drift. Expected by anyone who knows Garivier & Moulines / Besbes et al. | 4–5 days (SMPyBandits or own implementations; window tuning on held-out conditions) | Yes |
| 7 | **Inference quality**: bias and 95% CI coverage of the best-vs-control lift. Compare naive means vs IPW/AW-AIPW (Hadad et al. 2021) for TS/TTTS, and vs a standard z-interval for A/B | Covers the inference-after-adaptivity literature (Nie et al.; Shin et al.; Zhang et al.; Hadad et al.). Turns "bandits win" into a nuanced, defensible trade-off | 4–6 days (propensities for TS by Monte Carlo) | Yes, if scoped to K=2 and K=10 |
| 8 | **Regret–estimation Pareto plot** (vary tempered-TS temperature, uniform-floor share or MAD mixing rate) | Connects to Simchi-Levi & Wang (2025) and Liang & Bojinov. High visual impact for a workshop paper | 2–3 days once #1 and #7 exist | Yes (stretch) |
| 9 | **External validity on a second dataset**: Open Bandit Dataset (random vs Bernoulli-TS logs) and/or Yahoo! R6A replay | Rules out "artifact of our simulator". OBD is freely available. R6 requires a Webscope agreement the user must request | 5–8 days | Partially (OBD yes; R6 depends on access) |
| 10 | **Drift-intensity axis via the variation budget V_T**, plus rotting-style CTR decay with article age | Grounds drift in theory (Besbes et al. 2014) and news reality (Levine et al. 2017; Agarwal et al. 2009) | 2–3 days | Yes (stretch) |

**Recommended core package (≈2.5–3 weeks):** #1, #2, #3, #4, #5, #6. These directly pre-empt the most likely reviewer objections: a strawman A/B, missing BAI/decision metrics, tuning leakage, unrealistic per-step updates, and thin drift baselines. **Stretch:** #7 and #8 (inference and Pareto framing) if time allows; #9 only for OBD.

**Must-cite minimum set** (if space is tight): Thompson 1933; Lai & Robbins 1985; Auer et al. 2002; Chapelle & Li 2011; Li et al. 2010/2011; Wu et al. 2020 (MIND); Feit & Berman 2019; Garivier, Lattimore & Kaufmann 2016; Russo 2020 (TTTS); Kaufmann, Cappé & Garivier 2016; Johari et al. (OR, always-valid); Kohavi, Tang & Xu 2020; Hadad et al. 2021; Simchi-Levi & Wang 2023/2025; Garivier & Moulines 2011; Besbes, Gur & Zeevi 2014; Cao et al. 2019 (M-UCB); Fiez et al. 2024; Kalkanli & Özgür 2021; Lattimore & Szepesvári 2020 (*Bandit Algorithms*, Cambridge UP) [M].
