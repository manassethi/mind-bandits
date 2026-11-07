# CSE 592 midway report

`paper.tex` is the main document; `paper.bib` contains the references. The supplied ICML 2026 style is retained, with visible authors and a course-report notice. The revised report is six pages, including a separate references page. It combines related points into fuller paragraphs and adds a compact vector concept figure, using the figures inside the published PDFs as design references. The separate future-work heading uses `\section*{Work Before the Final Submission}`.

## Build

From the project root:

```sh
tectonic --keep-logs --keep-intermediates latex/paper.tex
```

Alternatively, open this folder in Overleaf, select `paper.tex` as the main file, and use pdfLaTeX with BibTeX. Both the local Tectonic build and the Codex desktop compiler have been checked. The explicit T1 font encoding preserves the template's Times fonts in both engines. The sources include the required style files, bibliography style, and vector figures.

## Repository and preserved versions

Repository: [manassethi/mind-bandits](https://github.com/manassethi/mind-bandits). The `final` snapshot (f649242) is the most recent code version and matches all 41 local code files used in this report. `main` is based on this snapshot; the prior `main` is preserved at `archive/pre-final-main-2026-11-07`, and the `final` and `experiments/mind-bandits-sae-control` branches are retained. No branch history is rewritten.

## Evidence and scope

The report uses the current saved results in `results/click_model`, `results/q1`, `results/q2`, `results/q3`, and `results/q4`. The older top-level `results.md` describes different experiments and was not used as the quantitative source. No existing model or experiment output was overwritten.

The repository's `test` data cover November 15, 2019: the labeled MIND-large development split, not the hidden-label official benchmark test set. The report calls it the evaluation day. Article cold start in Q2 means a new arrival in the simulated stream, not necessarily absence from model fitting. Human feature-label checks and the interactive demo remain planned work. The saved alternative last-day calibrator was fitted using a refitted model, so it is not a clean held-out calibration experiment.

The additional gap-contraction study completed on Great Lakes, account `mihalcea_owned1`, standard CPU partition, job **65624194** (16 seconds, exit code 0). It uses an isolated staging directory under the existing project, preserving previous remote results. `evidence/gap_sensitivity.json` and `.npz` contain the paired results; the JSON records all per-run regret values, settings, and checks. The initial job 65623998 failed because its expected remote environment cache was absent; the successful run used the exact local caches copied into the isolated staging directory.

The new experiment verifies probability bounds, conservation of each user's mean probability, zero regret when all arms are equal, and replication of the original cached policy runs at full gaps. `evidence/checkpoint_audit.json` additionally checks 50 cached Q1 probabilities against the current calibrated model; the maximum absolute difference is below 5e-8. This is a sample consistency check, not an exhaustive cache audit.

`evidence/audit.json` stores input hashes, constant-predictor baselines, top-two-gap diagnostics, and the contraction results. Existing outputs are unchanged. To rebuild the figures and audit from the repository root:

```sh
.venv/bin/python latex/scripts/build_report_assets.py
```

The script reuses `evidence/gap_sensitivity.json` if present. To rerun the new experiment explicitly:

```sh
.venv/bin/python latex/scripts/gap_sensitivity.py
```

The optional `CS592_PROJECT_ROOT` environment variable selects a staged project root. Figure regeneration needs the repository data and outputs; compiling the report only needs this LaTeX folder. SciPy is used for Student-t intervals, consistent with the existing analyses.

The latest scheduler recheck is recorded in `evidence/cluster_status.json`. The report experiment has completed, with no active news-bandit jobs under either requested account. Only the completed gap-contraction result is included; unrelated jobs from other projects are excluded.

## Writing and references

The prose follows the question-driven comparisons, explicit experimental descriptions, and result-then-interpretation structure in these writing samples:

- [Generalists vs. Specialists: Evaluating Large Language Models for Urdu](https://aclanthology.org/2024.findings-emnlp.426/)
- [WER We Stand: Benchmarking Urdu ASR Models](https://aclanthology.org/2025.coling-main.397/)
- [The Fellowship of the LLMs](https://aclanthology.org/2025.gem-1.4/)

They are writing samples, not unrelated self-citations in the report. Bibliographic metadata and clean source URLs were checked against primary publications. The Gemma Scope 2 citation identifies the technical report and links its official landing page. Author order follows the proposal: Mansi Sethi, Samee Arif, Andrew Koss. The corresponding address is the one publicly listed on Samee Arif's website.

The concept diagram uses original vector artwork with compact panels and a feedback loop, informed by Figure 1 inside the [Child Safety paper](https://aclanthology.org/2026.findings-emnlp.781.pdf) and Figure 1 inside [One Word at a Time](https://arxiv.org/pdf/2604.25921). Rebuild it with `.venv/bin/python latex/scripts/build_concept_figure.py`.
