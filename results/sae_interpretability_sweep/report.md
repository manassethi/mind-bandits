# Repeated SAE feature ablations

3 fixed article pools and 5 independently fitted policies per pool.

95% intervals average pools within each seed first. They cover fitted-policy/evaluation randomness for this fixed SAE, cohort and set of pools.

| Metric | Mean | 95% interval |
|---|---:|---:|
| top_ctr_drop | 0.692% | 0.385% to 0.999% |
| random_ctr_drop | 0.162% | 0.084% to 0.240% |
| excess_ctr_drop | 0.530% | 0.192% to 0.868% |
| top_choice_change | 32.413% | 27.583% to 37.244% |
| random_choice_change | 16.615% | 13.151% to 20.080% |

CTR drops are absolute percentage points. Positive excess drop means removing top features reduced expected reward more than removing random features.

This is a frozen-policy input sensitivity analysis. It does not show that the units are clean semantic concepts or that article meaning was causally changed.

Feature stability and article examples are in feature_review.md. Full per-policy explanations and random controls are saved in each pool/seed folder.
