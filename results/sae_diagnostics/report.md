# SAE redundancy and length diagnostics

Primary sample: articles present only in the validation catalog; no click labels are used. Policy-pool results are saved separately.

These are descriptive associations. Residual correlation removes a linear fit to effective token length and GPT-2 state norm; it does not prove a causal explanation.

| Feature pair | Raw correlation | After length/norm adjustment | Top-article overlap |
|---|---:|---:|---:|
| 872 / 750 | 0.992 | 0.790 | 80% |
| 439 / 5 | 0.992 | 0.837 | 90% |
| 539 / 228 | 0.992 | 0.938 | 50% |
| 439 / 735 | 0.991 | 0.826 | 90% |
| 750 / 205 | 0.990 | 0.746 | 90% |
| 5 / 992 | 0.990 | 0.833 | 90% |
| 439 / 872 | 0.990 | 0.772 | 80% |
| 872 / 735 | 0.990 | 0.812 | 90% |
| 915 / 735 | 0.990 | 0.839 | 100% |
| 439 / 750 | 0.990 | 0.771 | 90% |

Article-length associations for frequently important features:

- Feature 470: token-length correlation -0.5299574099088942; state-norm correlation 0.918642896955062.
- Feature 439: token-length correlation -0.5501882276985324; state-norm correlation 0.930278217406365.
- Feature 724: token-length correlation 0.6695274621588804; state-norm correlation -0.4800042390585291.

Inspect feature_pairs.csv and feature_length_correlations.csv for all units. Undefined correlations indicate constant/inactive features. High redundancy weakens claims of distinct concepts; length associations can reflect style or content as well as nuisance signals.
