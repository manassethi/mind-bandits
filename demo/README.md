# Interactive demo (poster session)

A Streamlit app that lets visitors pick a policy and watch its recommendations and regret evolve,
see the SAE features behind a contextual bandit's choices, and read the LLM agent's stated reasoning.

```bash
python -m venv .venv-demo
.venv-demo/bin/python -m pip install -r demo/requirements.txt   # Windows: .venv-demo\Scripts\python
.venv-demo/bin/streamlit run demo/app.py
```

The demo needs **no GPU, no torch and no Git LFS pull**. It runs on the small bundle in `demo/data/`
(about 11 MB) plus plain-text tables and figures already in `results/`.

## Pages

| Page | Question | What a visitor does |
|---|---|---|
| A/B test vs. bandits | Q1 | Picks a pool of 5 or 10 real MIND headlines, races A/B, ε-greedy, UCB, Thompson sampling and discounted Thompson on shared coin flips; turns on drift or shrinks gaps; presses Play to watch traffic shift and the posterior statistics update |
| Play the bandit | Q1 | Chooses headlines for 20 to 80 readers, then compares with Thompson sampling and UCB on the same coin flips |
| SAE bandit | Q2, Q3 | Runs per-article TS and LinUCB with category, dense Gemma and SAE features on a churning pool; steps through decisions to see the reader, the live articles' scores and uncertainty bonuses, and which labeled SAE features pushed the chosen article up |
| LLM agent | Q4 | Reads Gemma-3-12B transcripts with and without the statistics tool; sees the 30-pool results |
| Results | all | Simulator validation, Q1 table and the final-campaign figures with their captions |

## Modeling choices (and why)

- **Live, but on stored probabilities.** Simulations run in the browser session, but the click
  probabilities come from the same simulator banks the paper uses (Q4 headline pools for the race
  and game; the first 6 of 30 pools of the Q2 test environment for the SAE page). The demo is
  semi-synthetic in exactly the way the paper is.
- **Same policy code.** The race calls `src/bandits/policies.py` directly. The contextual page uses
  `demo/engines.py:churn_run`, a NumPy port of `src/bandits/contextual.py` (same churn schedule,
  features `[1, a, u * a]`, ridge LinUCB with α = 0.3 from `results/q2/tuned.json`), so it runs
  without torch and can record the policy state for explanations.
- **Explanations.** For a chosen article, feature j contributes `θ_a[j] a[j] + θ_u[j] u[j] a[j]` to the
  predicted score. Only differences between live articles change a choice, so the chart shows each
  contribution minus its live-article average, the same centring as the Q3 importance measure.
  SAE labels are Neuronpedia's automated descriptions (Gemma Scope 2, 12B layer 41).
- **Common random numbers.** All policies in a race, and the visitor in the game, see the same
  readers and coin flips, so differences come from choices alone.
- **Demo-only settings, labeled in the app.** Discounted TS defaults to γ = 0.9999 (tuned value
  0.99999) so drift adaptation is visible at 10,000 readers. The game multiplies click rates by up to
  5x so short games produce clicks.

## Checks

`python -m demo.check_engines` (needs the LFS inputs) runs the engines on all 30 stored test pools:

| Setting | Paper | Demo engine |
|---|---|---|
| Q1, K=10, T=10,000: A/B / ε-greedy / UCB / TS / discounted TS | 179±56 / 90±29 / 47±16 / 80±6 / 80±5 | 150±46 / 92±27 / 41±10 / 77±5 / 80±6 |
| Q2, 20,000 readers: per-article TS / dense Gemma L43 LinUCB | 1338±28 / 754±44 | 1343±28 / 786±67 |

The random streams differ from the original runners, so single runs differ, but all means fall
inside the paper's 95% intervals.

## Rebuilding the bundle

`python -m demo.build_demo_data` (after `git lfs pull --include="dataset/processed/**,results/q2/**,results/q4/**"`)
regenerates `demo/data/` from the stored experiment inputs. Nothing is re-fitted.
