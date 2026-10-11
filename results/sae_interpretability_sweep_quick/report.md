# Repeated SAE feature ablations

2 fixed article pools and 3 independently fitted policies per pool.

95% intervals average pools within each seed first. They cover fitted-policy/evaluation randomness for this fixed SAE, cohort and set of pools.

| Metric | Mean | 95% interval |
|---|---:|---:|
| top_ctr_drop | 0.300% | -0.671% to 1.270% |
| random_ctr_drop | 0.036% | -0.110% to 0.181% |
| excess_ctr_drop | 0.264% | -0.703% to 1.231% |
| top_choice_change | 29.400% | 10.250% to 48.550% |
| random_choice_change | 13.656% | -7.715% to 35.026% |

CTR drops are absolute percentage points. Positive excess drop means removing top features reduced expected reward more than removing random features.

This is a frozen-policy input sensitivity analysis. It does not show that the units are clean semantic concepts or that article meaning was causally changed.

Feature stability and article examples are in feature_review.md. Full per-policy explanations and random controls are saved in each pool/seed folder.
