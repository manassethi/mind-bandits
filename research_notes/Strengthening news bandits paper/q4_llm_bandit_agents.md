# Q4: LLMs as Bandit Agents and In-Context Exploration (literature for strengthening the news-bandits workshop paper)

Scope: Gemma-3-12B-it as a non-contextual K=5/10 bandit policy over MIND headlines, T=500, summarized history, short CoT, temp 1.0, optional stats tool (posterior mean, 95% UCB, P(best)). Notes compiled 2026-10-21. Items marked **[UNVERIFIED]** come from search snippets, secondary summaries, or my prior knowledge and were not checked against the primary paper text.

## 1. Krishnamurthy et al. 2024: exact setup, metrics, what worked

### Takeaway
Only one of many configurations explored robustly: GPT-4 with buttons scenario, suggestive framing, **summarized history**, reinforced CoT, temperature 0 (BSSC̃0). Everything else showed suffix failures or uniform-like failures. Our setup copies their summarized-history design, so our "alone gets stuck 67-70%" result is a negative finding for a modern open 12B model under their *best* history format. That is a meaningful and citable replication/extension.

### Cited Findings
- Paper: Krishnamurthy, Harris, Foster, Zhang, Slivkins, "Can large language models explore in-context?", NeurIPS 2024, arXiv:2403.15371. — [arXiv](https://arxiv.org/abs/2403.15371)
- Models: GPT-3.5-Turbo-0613, GPT-4-0613, Llama2-13B-chat (4-bit). — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3)
- Instances: Bernoulli. Hard instance K=5, Δ=0.2 (best arm 0.5+Δ/2, others 0.5−Δ/2). Easy instance K=4, Δ=0.5. — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3)
- Horizon T=100 in the main experiments. GPT-3.5 used N=20 replicates × 48 configs (~200K queries). GPT-4 used N=10 × 10 configs, with robustness runs at T=200 (N=20/40) and one basic config at T=500 (N=20). Llama2 used N=10 on 32 configs. Baselines got 1,000 replicates. — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3)
- Prompt design code L1–L5: Buttons/Advertisements scenario; Neutral/Suggestive framing ("you are an explorer"-style hint); Raw/Summarized history; no CoT / CoT / reinforced CoT; temperature 0/1 or "return a distribution" (D). — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3)
- Metric definitions — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3):
  - SuffFailFreq(t) is the fraction of replicates in which the best arm is never chosen in rounds [t, T]. They report SuffFailFreq(T/2).
  - MinFrac(t) is the minimum over arms of the fraction of rounds in [1, t] that arm was chosen, averaged over replicates. They plot K·MinFrac, which is in [0, 1]; values near 1 mean near-uniform play.
  - GreedyFrac is the fraction of rounds in which the chosen arm had the highest empirical mean so far.
  - MedianReward is the median over replicates of time-averaged reward.
- Baselines: UCB with bonus √(C/n_a) and C=1, Thompson sampling with Beta(1,1), Greedy (one initial pull per arm), and ε-greedy over a range of ε. — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3)
- BSSC̃0 had zero suffix failures at T=100, K·MinFrac slightly above TS, and reward comparable to TS. It held up at T=200, N=40. Its raw-history ablation (BSRC̃0) had a non-trivial suffix-failure rate. — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3)
- SuffFailFreq(T/2) ≥ 15% for all but five configurations. Basic GPT-4 (BNRN0) had suffix failures in >60% of replicates at T=500. Uniform-like failures appeared mainly with distributional output (D). Ranking: GPT-4 > GPT-3.5 > Llama2. High sensitivity to small prompt changes. — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3)
- Conclusion: external summarization matters but may not scale to complex settings. "Non-trivial algorithmic interventions" (fine-tuning, dataset curation, few-shot, tool use) may be needed. — [arXiv abstract](https://arxiv.org/abs/2403.15371)

### Inferences
- Our tool variant is essentially the "tool use" intervention they speculated about. We can frame Q4 as directly testing their open question with a real-data news environment rather than synthetic Bernoulli arms.
- Our T=500 matches their longest robustness run, where basic GPT-4 failed in >60% of replicates. Our 67-70% stuck rate is in the same range, which is a useful consistency anchor.
- Reviewers will expect reporting in their metric vocabulary. We already report SuffFail and K·MinFrac. We should add **GreedyFrac** and their temperature-0 and suggestive-framing ablations, because those were part of the only successful configuration.
- Our real headline arms differ from their unlabeled buttons. They found the semantic "advertisement" framing was not better than buttons. EVOLvE also varies semantic vs meaningless arm names. A "headlines replaced by Article A–J labels" ablation would separate prior-driven stickiness from exploration failure.

### Gaps
- I did not extract per-configuration GreedyFrac values or exact TS/UCB numeric baselines from the paper.

## 2. Follow-up work (2024–2026) and which interventions improved exploration

### Takeaway
The consistent picture is as follows. Raw-history LLMs under-explore and act greedily. Summarized statistics help larger models. **Putting algorithmic quantities such as UCB values into the prompt** helps most at inference time. Fine-tuning on UCB trajectories or RL fine-tuning closes much of the gap. A forced initial "try every arm" phase is a strong cheap fix. Even so, LLMs generally remain behind tuned classical algorithms. Our stats tool is closest to EVOLvE's "algorithm-guided support". Reviewers will compare against it.

### Cited Findings
- **EVOLvE / BanditBench** (Nie, Su, Chang, Lee, Chi, Le, Chen), ICML 2025, PMLR 267:46346-46376, arXiv:2410.06238. — [PMLR](https://proceedings.mlr.press/v267/nie25b.html); [arXiv](https://arxiv.org/abs/2410.06238)
  - Models: Gemma-2B, Gemma-9B, Gemini 1.5 Flash, Gemini 1.5 Pro. — [arXiv HTML v2](https://arxiv.org/html/2410.06238v2)
  - MAB settings vary K (5 or 20), arm description ("Videos" = meaningless labels vs "Clothes" = semantic names), Bernoulli vs Gaussian rewards, and gap Δmin (0.5 easy, 0.2 hard). The contextual bandit is a semi-synthetic MovieLens setup (K=10/30). The paper states horizons inconsistently (T=200/1000). — [arXiv HTML v2](https://arxiv.org/html/2410.06238v2)
  - Interventions:
    - Raw vs Summarized History (per-arm mean, count, time).
    - **Algorithm-Guided support (AG)**: the prompt includes each arm's empirical mean plus UCB exploration bonus, so the LLM only has to add and take an argmax.
    - Few-shot demos of 5 UCB trajectories.
    - **Oracle Behavior Fine-Tuning (OFT)** on UCB actions.

    — [arXiv HTML v2](https://arxiv.org/html/2410.06238v2)
  - Win-rates: UCB 90.6% (MAB). For Flash on MAB, few-shot with raw history reached 27.5% and few-shot with AG 50.2%. OFT Flash reached 65.6%. Pro with summarized history reached 60.0% vs 45.5% with raw history. Gemma-2B/9B "mostly stay in the linear-regret regime". Pro with AG/SH is nearly sublinear. They fit regret as f(T)=λ·log(T)^α/Δmin + βT + λ2. — [arXiv HTML v2](https://arxiv.org/html/2410.06238v2)
  - Few-shot demos from easy tasks beat those from hard tasks. OFT benefits more from hard-task data. — [arXiv HTML v2](https://arxiv.org/html/2410.06238v2)
- **LLMs are Greedy Agents** (Schmied, Bornschein, Grau-Moya, Wulfmeier, Pascanu), ICLR 2026, arXiv:2504.16078. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cf2fcda40de8520027af2eadcafdecd6-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2504.16078)
  - Setup: Gemma2 2B/9B/27B on MABs with 5/10/20 arms over 50 steps, a MovieLens contextual bandit, and Tic-tac-toe. — [arXiv HTML](https://arxiv.org/html/2504.16078v1)
  - **Greediness**: at 10 arms, action coverage is ~40% (2B) and ~65% (9B/27B). Without CoT, all models reach only ~25%. — [arXiv HTML](https://arxiv.org/html/2504.16078v1)
  - **Frequency bias**: the 2B model picks the most frequent in-context action 96% of the time, vs 14% for 27B. — [arXiv HTML](https://arxiv.org/html/2504.16078v1)
  - **Knowing-doing gap**: 27B computes correct UCB rationales 87% of the time, yet given a correct rationale it still picks the greedy action 58% of the time and the optimal action 21%. — [arXiv HTML](https://arxiv.org/html/2504.16078v1)
  - Interventions:
    - **"Try all actions first" gave the largest gain** and brought 27B nearly to UCB level.
    - An exploration bonus during RLFT raised coverage from ~50% to ~70%.
    - ε-greedy, self-consistency, and self-correction had mixed effects.
    - SFT on UCB expert data matched UCB regret.
    - A larger thinking budget (512 tokens) lifted 2B to roughly 9B+RLFT level.

    — [arXiv HTML](https://arxiv.org/html/2504.16078v1)
- **LLMs Are In-Context Bandit Reinforcement Learners** (Monea, Bosselut, Brantley, Artzi), COLM 2025, arXiv:2410.05362. Contextual-bandit ICRL on classification tasks, with models from 500M to 70B (Llama 3.1, Qwen2.5, Gemini 1.5 Flash). Key ingredients are relatively high sampling stochasticity and keeping only positive-reward episodes in context. They report instability. — [arXiv](https://arxiv.org/abs/2410.05362)
- **Should You Use Your LLM to Explore or Exploit?** (Harris & Slivkins), UAI 2026 / PMLR v337, arXiv:2502.00225. — [PMLR](https://proceedings.mlr.press/v337/harris26a.html); [arXiv HTML v3](https://arxiv.org/html/2502.00225v3)
  - Exploitation was tested on GPT-5.2, GPT-5-mini/nano, GPT-4, GPT-4o, GPT-3.5, Qwen-2.5, Gemma-3, Mistral-7B, and DeepSeek-R1. GPT-5-mini was perfect on MAB exploitation. GPT-3.5/4 degrade as history grows and gaps shrink. — [arXiv HTML v3](https://arxiv.org/html/2502.00225v3)
  - Mitigations were a code interpreter (tool use) and k-NN/k-means summarization. Even so, LLMs trail linear regression. — [arXiv HTML v3](https://arxiv.org/html/2502.00225v3)
  - Reasoning models are the most promising exploiters but too slow or costly. GPT-4 with code interpreter cost ~$70 and >6 h, vs a few dollars and <5 min without it. — [arXiv HTML v3](https://arxiv.org/html/2502.00225v3)
  - LLMs are useful as **exploration oracles**: the LLM proposes candidate arms in large semantic spaces, then UCB1 runs over them. — [arXiv HTML v3](https://arxiv.org/html/2502.00225v3)
- **Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents** (Xia, Liu, Yue, Li), Findings of ACL 2025, arXiv:2407.01887. — [arXiv](https://arxiv.org/abs/2407.01887); [ACL](https://preview.aclanthology.org/setup/2025.findings-acl.519)
  - Top LLMs reach low weak regret quickly but struggle to converge and exploit consistently. Results are prompt-sensitive. Models include GPT-3.5/4/4-Turbo, with LLaMA 3.1 and o1-preview added in later versions.
  - **LEAD** wraps a classic dueling-bandit algorithm around the LLM and inherits its regret guarantees. This is a hybrid analogous to our tool variant.
- **Large Language Models Think Too Fast To Explore Effectively**, arXiv:2501.18009. DeepSeek-R1 explored less effectively than humans and o1 in an open-ended combination task (Little Alchemy-style, not a bandit). — [arXiv PDF](https://arxiv.org/pdf/2501.18009) **[authors/details UNVERIFIED]**
- **TextBandit** (Oct 2025, arXiv:2510.13878): bandits with language-only feedback. Qwen3-4B reportedly had 89.2% best-arm selection, with larger models doing worse in that setup. — [arXiv HTML](https://arxiv.org/html/2510.13878v1) **[UNVERIFIED beyond snippet]**
- **Posterior Sampling via Autoregressive Generation** (Zhang, Cai, Namkoong, Russo), NeurIPS 2024, arXiv:2405.19466. Pretrains an autoregressive (LM-based) model on historical **news** click sequences with headline features. Generating imputed future outcomes implements Thompson sampling with a learned, headline-informed prior. — [arXiv HTML](https://arxiv.org/html/2405.19466v2); [NeurIPS](https://neurips.cc/virtual/2024/98881)

### Inferences
- **Intervention map against our design.**

  | Intervention | Status |
  |---|---|
  | Summarized history | Already used |
  | Short CoT | Already used |
  | Temperature 1.0 | Used. K et al.'s winner used temp 0 |
  | Suggestive "explore" framing | Not ablated |
  | Forced initial round-robin (try-all) | Not ablated. Cheap and strong per Schmied |
  | Algorithm-guided values in prompt | Our tool is close to AG, but optional (model decides when to call) |
  | Few-shot UCB/TS demos | Not ablated |
  | Fine-tuning on UCB/TS trajectories | Not done |
  | Reasoning model / longer thinking budget | Not done |

- Our "optional tool, called 33-51% of rounds" is a distinctive twist relative to EVOLvE's always-present AG. Reviewers will ask: what if the stats are **always shown in the prompt** (no tool call)? And does the LLM simply argmax the UCB bound or P(best)? Measuring **agreement between the LLM's choice and argmax-UCB / argmax-P(best)** on tool rounds would show whether the LLM adds anything beyond executing the algorithm. This mirrors the knowing-doing-gap analysis in Schmied et al.
- Schmied's frequency-bias and coverage metrics are cheap to compute from existing logs (coverage, P(choose most-frequent arm)).
- Gemma-3 was evaluated in Harris & Slivkins for exploitation, which is a direct citation hook for our model choice.

### Gaps
- I could not find a primary-source bandit evaluation of o1/R1-style reasoning models besides Harris & Slivkins (R1 for exploitation) and Xia et al. (o1-preview, dueling). I found no explicit scaling study of in-context exploration across Qwen3 or gpt-oss sizes (searched, no hits).
- No primary details checked for "LLM-enhanced Thompson sampling" under that exact name. The closest hits are listed in Section 3.

## 3. Hybrid designs: LLM prior + bandit algorithm (prior-informed TS) and cold-start evidence

### Takeaway
Combining an LLM headline prior with a principled bandit is a well-trodden, reviewer-friendly experiment. The closest precedent is **LOLA**: an LLM CTR prediction is used as a pseudo-count prior inside UCB on Upworthy headline A/B tests, and it beats pure UCB, pure LLM, and explore-then-commit, especially at small horizons. A 2026 paper shows LLM warm starts **hurt** when LLM preferences are misaligned with real users. That speaks directly to our weak prior (Spearman 0.08-0.11), so prior-informed TS on MIND would be an informative test of that theory.

### Cited Findings
- **LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments** (Ye, Yoganarasimhan, Zheng), arXiv:2406.02611. Appears in Marketing Science 44(5):995-1016 (2025). — [arXiv HTML v3](https://arxiv.org/html/2406.02611v3); [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/mksc.2024.0990); [RePEc](https://ideas.repec.org/a/inm/ormksc/v44y2025i5p995-1016.html)
  - Data: Upworthy, 17,681 tests and 77,245 headlines, 277M impressions, 3.7M clicks. — [arXiv HTML v3](https://arxiv.org/html/2406.02611v3)
  - Accuracy at picking the winning headline, against a random baseline of 33.02%. — [arXiv HTML v3](https://arxiv.org/html/2406.02611v3)

    | Method | Accuracy |
    |---|---|
    | GPT-3.5 zero-shot | 29.76% |
    | GPT-4 zero-shot | 37.85% |
    | GPT-4 2-shot | 39.96% |
    | OpenAI embeddings + linear regression | 46.28% |
    | LoRA fine-tuned Llama-3-8B (best) | 46.86% |

  - Hybrid (LLM-2UCBs): the LLM CTR prediction is treated as n_aux pseudo-impressions, giving a second upper bound U². The policy plays argmax min(U¹, U²). Tuned values are n_aux=1000 and α=0.08. Setting n_aux=0 gives UCB, and n_aux→∞ gives greedy-LLM. — [arXiv HTML v3](https://arxiv.org/html/2406.02611v3)
  - Results: at τ=50, LOLA beats E&C by ~8.7% and UCB by ~7.0%. At τ=1000 it beats UCB by ~2.2% and pure LLM by ~5.0%. Pure LLM is competitive only at short horizons. TS is not benchmarked in the main results. — [arXiv HTML v3](https://arxiv.org/html/2406.02611v3)
- **Jump Starting Bandits with LLM-Generated Prior Knowledge** (Alamdari, Cao, Wilson), EMNLP 2024, arXiv:2406.19317. The LLM simulates user preferences to build synthetic pretraining data for a contextual bandit (CBLI), which reduces early regret. Tested with an LLM-as-oracle setup and on real conjoint-survey data. — [arXiv](https://arxiv.org/abs/2406.19317); [ACL Anthology](https://preview.aclanthology.org/dois-2013-emnlp/2024.emnlp-main.1107)
- **Jump Start or False Start? A Theoretical and Empirical Evaluation of LLM-initialized Bandits** (Bayley, Zhu, Aoki, Cao, Wilson), arXiv:2604.02527, Apr 2026.
  - Warm start stays effective up to 30% label corruption, loses its advantage around 40%, and degrades beyond 50%.
  - **Systematic misalignment alone can make LLM priors give higher regret than cold start.**
  - Estimated LLM-user alignment predicts whether warm-starting helps.

  — [arXiv](https://arxiv.org/abs/2604.02527)
- **LLM-Derived Priors for Thompson Sampling in Cold-Start Comment Recommendation** (Lee, Choi, Kang, Jang; NAVER WEBTOON), arXiv:2608.03382, Aug 2026. LLM semantic signals are converted to Bayesian priors that warm-start TS, with per-segment posteriors. Online A/B/C test vs a uniform prior: LLM priors help most in sparse-feedback regimes, and effects vary by segment. The abstract does not give the prior-strength mapping. — [arXiv](https://arxiv.org/abs/2608.03382)
- **Generate to Explore, Select to Exploit** (arXiv:2609.15094): LLM-generated headline candidates plus Confidence-Aware Thompson Sampling for cold start. — [arXiv PDF](https://arxiv.org/pdf/2609.15094) **[UNVERIFIED beyond snippet]**
- **On the Importance of Uncertainty in Decision-Making with LLMs**, arXiv:2404.02649. Contextual bandit with LLM representations plus Bayesian/uncertainty heads, compared against greedy LLM policies. — [arXiv PDF](https://arxiv.org/pdf/2404.02649) **[content UNVERIFIED; from search hit + prior knowledge]**
- Harris & Slivkins also argue that LLMs are better used for proposing or shaping exploration than for exploitation. — [arXiv HTML v3](https://arxiv.org/html/2502.00225v3)

### Inferences
- **Proposed experiment, "LLM-prior TS".** Map the LLM's zero-shot ranking or CTR score to Beta(α0, β0) priors, e.g. mean = base CTR × rank-based multiplier, with strength n_aux ∈ {0, 10, 100, 1000}. Run TS and UCB (LOLA-style min-of-two-bounds) on the simulator. Sweep prior strength and report regret vs n_aux.
- Prediction from Bayley et al.: our ρ≈0.1 prior is near "misaligned". Strong priors should hurt and weak priors should be neutral. Two contrasts would make the weak-prior finding causal:
  - Simulator-derived priors (ρ=0.44) vs Gemma priors.
  - A shuffled-prior control.
- This experiment is cheap. The LLM prior is computed once per instance, and the bandit runs on CPU.
- A second hybrid is to make the LLM choose with the tool's numbers always in context, i.e. EVOLvE-AG. Our current tool design sits in between.

### Gaps
- I did not verify how the NAVER paper maps LLM signals to Beta pseudo-counts. Read the full text if we cite the mapping.

## 4. Models to compare and scale effects

### Takeaway
Reviewers will expect at least a second model family and a scale contrast. The literature shows larger and stronger models explore and exploit better (GPT-4 > GPT-3.5 > Llama2; Gemini Pro > Flash > Gemma; Gemma2 27B > 9B > 2B in coverage). Larger models trade frequency bias for greediness. Reasoning models help exploitation but are expensive. Nothing in the literature tests Gemma-3, Qwen3, or gpt-oss on in-context exploration at our horizon, which is a gap we can fill.

### Cited Findings
- GPT-4 > GPT-3.5 > Llama2-13B. — [Krishnamurthy et al.](https://arxiv.org/html/2403.15371v3)
- Gemma-2B/9B stay linear-regret, while Gemini Pro with AG/SH is near-sublinear. A fine-tuned Flash beats Pro. — [EVOLvE](https://arxiv.org/html/2410.06238v2)
- Gemma2 27B avoids frequency bias (14% vs 96% for 2B) but becomes greedy. Coverage rises 2B→9B, then plateaus at 27B. — [Schmied et al.](https://arxiv.org/html/2504.16078v1)
- Reasoning models (GPT-5-mini, DeepSeek-R1) are the most promising exploiters but slow and costly. Gemma-3 and Qwen-2.5 were included. — [Harris & Slivkins](https://arxiv.org/html/2502.00225v3)
- ICRL emerges across 500M–70B (Llama 3.1, Qwen2.5). — [Monea et al.](https://arxiv.org/abs/2410.05362)
- Qwen3-4B was strong on text-feedback bandits, and larger models were not uniformly better. — [TextBandit](https://arxiv.org/html/2510.13878v1) **[UNVERIFIED]**

### Inferences
- Recommended models, all feasible on L40S (48GB) with vLLM:

  | Model | Purpose | Fit on one L40S |
  |---|---|---|
  | Gemma-3-4B / 12B / 27B | Within-family scale | 27B in bf16 needs ~54GB, so use FP8/AWQ on one L40S or TP=2 |
  | Qwen3-8B or 14B | Second family. Toggles thinking vs non-thinking mode, giving a clean reasoning ablation with fixed weights | Fits |
  | gpt-oss-20b | Reasoning model with adjustable reasoning effort | Fits in ~16GB MXFP4 |
  | Llama-3.1-8B-Instruct | Widely used reference | Fits |

- Optional API anchor: one strong frontier model (e.g. GPT-4o/GPT-5-mini or Gemini Flash) on a single config with reduced N (10 runs). Cost estimate: 1000 decisions × ~700 tokens ≈ 0.7M tokens per run.
- The Qwen3 thinking toggle is the highest-value reasoning ablation: same model, same prompt, thinking on vs off, with token cost reported.

### Gaps
- I did not find published bandit results for Gemma-3, Qwen3 (beyond the TextBandit snippet), Llama-4, or gpt-oss.

## 5. LLM-as-recommender on news and headline popularity prediction (context for our weak-prior finding)

### Takeaway
Evidence consistently says LLMs are poor zero-shot predictors of which headline people actually click. On Upworthy, GPT-4 zero-shot is only ~5 points above random at picking A/B winners, and even fine-tuned models reach ~47% in a 3-way setting. LLM-simulated CTRs recover only a fraction of real effects. Our Spearman 0.08-0.11 is therefore expected and literature-consistent. Framing it alongside LOLA strengthens the paper.

### Cited Findings
- Upworthy winner prediction: GPT-4 zero-shot 37.85% vs 33.02% random. The best fine-tuned Llama-3-8B reached 46.86%. "None reliably predicts the best headline". — [LOLA](https://arxiv.org/html/2406.02611v3)
- Spotify Engineering (Aug 2026): gpt-4o-mini-predicted CTRs for headline variants recovered only 39% of the observed human treatment effect, with effects attenuated toward zero. — [Spotify Engineering blog](https://engineering.atspotify.com/2026/8/when-can-llms-replace-humans-in-a-b-tests/) **[secondary snippet; UNVERIFIED]**
- Fine-tuning LLMs on A/B outcomes can exploit superficial clickbait cues. A hypothesis-generation approach was proposed (Wang, Sudhir, Zhou, Yale SOM). — [news summary](https://letsdatascience.com/news/researchers-teach-llms-hypotheses-to-avoid-clickbait-5a8df083) **[secondary; primary paper not located]**
- ChatGPT zero-shot on MIND trails trained neural recommenders (NAML, NRMS). — [Li et al., arXiv:2306.10702](https://arxiv.org/pdf/2306.10702)
- Zero-shot ranking prompts are sometimes competitive, while rating prompts are weaker. Fine-tuning helps most. — [Exploring fine-tuning ChatGPT for news rec, arXiv:2311.05850](https://arxiv.org/abs/2311.05850v1)
- RecPrompt (GPT-4 + CoT prompt-tuning) reportedly beats neural baselines on MIND. — [summary](https://papers.lunadong.com/paper/10482) **[UNVERIFIED; not pure zero-shot]**
- Pre-LLM headline popularity: a multi-task GRU performed about as well as logistic regression on character n-grams. — [Lamprinidis, Hardt, Hovy, EMNLP 2018](https://preview.aclanthology.org/setup/D18-1068) (pre-2023, background only)

### Inferences
- Our finding that the simulator's ranking is ρ=0.44 vs the LLM's 0.08-0.11 parallels LOLA's result that trained embedding models beat prompting.
- A cheap add-on is to compute **pairwise winner accuracy** on MIND article pairs, giving a direct comparison to LOLA's numbers. Also elicit **numeric CTR estimates** instead of a ranking and check calibration.

### Gaps
- No primary-source zero-shot LLM headline-CTR correlation study on MIND was found.

## 6. Cost reporting conventions

### Takeaway
The standard expectation, set by Kapoor et al. "AI Agents That Matter", is to report cost alongside performance as a Pareto frontier, preferably in dollars, not just parameter counts. Bandit-LLM papers report query counts and wall-clock/$ for API runs.

### Cited Findings
- Kapoor, Stroebl, Siegel, Nadgir, Narayanan, "AI Agents That Matter" (TMLR), arXiv:2407.01502 — [arXiv](https://arxiv.org/html/2407.01502v1)
  - Evaluate agents on an accuracy-vs-cost Pareto frontier.
  - Measure dollar cost rather than proxies like parameter count.
  - Complex agents cost up to 50× more than simple baselines without accuracy gains.
- Krishnamurthy et al. report total query counts (~200K GPT-3.5, ~50K GPT-4). — [arXiv HTML v3](https://arxiv.org/html/2403.15371v3)
- Harris & Slivkins report $ and wall-clock: GPT-4 + code interpreter ~$70 and >6 h vs a few dollars and <5 min without. — [arXiv HTML v3](https://arxiv.org/html/2502.00225v3)
- EVOLvE includes a cost table across models. — [arXiv HTML v2](https://arxiv.org/html/2410.06238v2)

### Inferences
- Keep GPU-seconds and tokens per 1000 decisions. Add the following:
  - Approximate $/1000 decisions, using an L40S hourly cloud price and/or API pricing for equivalent tokens.
  - Latency per decision vs TS (microseconds).
  - A regret-vs-cost Pareto plot with TS, UCB, LLM, LLM+tool, and LLM-prior TS as points.
- The headline message reviewers will want is that LLM+tool matches tuned UCB at ~10^6–10^7× the compute. Say it explicitly.

### Gaps
- None critical.

## 7. Prioritized proposed experiments for Q4 (impact vs effort, ~1 month, single L40S + vLLM)

| # | Experiment | Impact | Effort | Notes |
|---|---|---|---|---|
| 1 | **LLM-prior Thompson/UCB (LOLA-style)**: Gemma zero-shot ranking/CTR as Beta pseudo-count prior with strength n_aux ∈ {0, 10, 100, 1000}. Controls: simulator-derived prior (ρ≈0.44), shuffled prior, uniform. | High: directly tests whether the weak prior helps or hurts, links to LOLA, Alamdari, and Bayley 2026 | Very low (CPU only; prior computed once) | The cleanest new result. Bayley et al. predict harm at high n_aux for a misaligned prior |
| 2 | **Re-analysis of existing logs** (no new runs): GreedyFrac, action coverage, frequency-bias rate, and on tool rounds the agreement of the LLM's choice with argmax-UCB / argmax-P(best) / TS sample. Tool-call rate vs posterior uncertainty over time | High: addresses "is the LLM just executing the tool?" and the knowing-doing gap | Very low | Use K et al. and Schmied et al. metrics |
| 3 | **Always-on stats (EVOLvE AG-style) vs optional tool vs none**, plus a **stats-only-argmax policy** (deterministic argmax of 95% UCB from the tool) as a non-LLM baseline | High: reviewers will ask if the LLM adds value over the tool | Low (one new prompt config, 30 runs × 2 K) | If argmax-UCB-of-tool ≈ LLM+tool, report it honestly as "LLM ≈ wrapper" |
| 4 | **Prompt ablations from the literature**: (a) forced round-robin start (try each arm once or twice; Schmied's strongest fix), (b) suggestive "explore" framing, (c) temperature 0 vs 1, (d) anonymized arms ("Article A–J") vs real headlines | Medium-high: covers K et al.'s grid dimensions and separates prior-anchoring from exploration failure | Low-medium (4 configs × 2 K × 30 runs; ~1 GPU-day at current cost) | (d) is especially informative for a news paper |
| 5 | **Second model family + reasoning toggle**: Qwen3-8B/14B with thinking on/off; gpt-oss-20b (low/medium effort); optionally Llama-3.1-8B | High: addresses generalization and the reasoning-model question | Medium (thinking mode multiplies tokens 5-20×; maybe N=15-20 runs and K=5 only) | Report tokens/$ per config |
| 6 | **Within-family scale**: Gemma-3-4B and 27B (FP8/AWQ) on the main configs | Medium | Medium (27B ~2-3× slower) | Literature shows non-monotone coverage with scale |
| 7 | **Few-shot UCB/TS demonstration trajectories** in context (EVOLvE) | Medium | Low-medium (long contexts raise cost) | Optional if budget remains |
| 8 | **Single API frontier anchor** (e.g. GPT-5-mini or Gemini Flash) on the no-tool and tool configs, N=10 | Medium: shows whether failure is model-capacity-bound | Low engineering, $ cost (~0.7M tokens/run → modest) | Check API terms and budget |
| 9 | **Cost-regret Pareto plot** with $ estimates | Medium: reviewer expectation (Kapoor et al.) | Very low | Use existing measurements |
| 10 | **Headline-prior quality checks**: pairwise winner accuracy and numeric CTR calibration of Gemma vs simulator, to compare with LOLA's 37.85% (GPT-4 zero-shot) | Medium | Very low | Strengthens the weak-prior section |
| 11 | OFT / RLFT on UCB or TS trajectories (LoRA on Gemma-3-4B/12B) | High novelty but out of scope | High (training infra, risk) | Mention as future work, citing EVOLvE and Schmied |

Suggested order: 2 → 1 → 9 → 10 (all in week 1, no GPU) → 3 → 4 → 5 → 6, with 7/8 if time permits.
