# SAE redundancy and length diagnostics

Primary sample: articles present only in the validation catalog; no click labels are used. Policy-pool results are saved separately.

These are descriptive associations. Residual correlation removes a linear fit to effective token length and GPT-2 state norm; it does not prove a causal explanation.

| Feature pair | Raw correlation | After length/norm adjustment | Top-article overlap |
|---|---:|---:|---:|
| 539 / 228 | 0.996 | 0.966 | 90% |
| 872 / 750 | 0.995 | 0.870 | 90% |
| 750 / 205 | 0.994 | 0.860 | 90% |
| 872 / 205 | 0.993 | 0.836 | 90% |
| 5 / 992 | 0.992 | 0.852 | 100% |
| 439 / 735 | 0.992 | 0.867 | 90% |
| 750 / 5 | 0.992 | 0.825 | 90% |
| 915 / 735 | 0.991 | 0.891 | 100% |
| 549 / 38 | 0.991 | 0.817 | 80% |
| 872 / 735 | 0.991 | 0.834 | 100% |

Article-length associations for frequently important features:

- Feature 470: token-length correlation -0.5246360938973953; state-norm correlation 0.894132306948953.
- Feature 439: token-length correlation -0.5672686664814623; state-norm correlation 0.9143138461903623.
- Feature 724: token-length correlation 0.6872123805212104; state-norm correlation -0.5190545744445723.

Inspect feature_pairs.csv and feature_length_correlations.csv for all units. Undefined correlations indicate constant/inactive features. High redundancy weakens claims of distinct concepts; length associations can reflect style or content as well as nuisance signals.
