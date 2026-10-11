# SAE redundancy and length diagnostics

Primary sample: articles present only in the validation catalog; no click labels are used. Policy-pool results are saved separately.

These are descriptive associations. Residual correlation removes a linear fit to effective token length and GPT-2 state norm; it does not prove a causal explanation.

| Feature pair | Raw correlation | After length/norm adjustment | Top-article overlap |
|---|---:|---:|---:|
| 383 / 741 | 0.985 | 0.966 | 70% |
| 816 / 735 | 0.984 | 0.863 | 90% |
| 816 / 335 | 0.983 | 0.847 | 80% |
| 383 / 143 | 0.982 | 0.959 | 40% |
| 735 / 335 | 0.982 | 0.846 | 90% |
| 443 / 335 | 0.979 | 0.843 | 70% |
| 816 / 936 | 0.976 | 0.806 | 90% |
| 443 / 816 | 0.976 | 0.812 | 70% |
| 443 / 735 | 0.975 | 0.812 | 70% |
| 270 / 735 | 0.974 | 0.814 | 90% |

Article-length associations for frequently important features:

- Feature 383: token-length correlation 0.7344707263813379; state-norm correlation -0.49873874176701405.
- Feature 443: token-length correlation -0.661826322502637; state-norm correlation 0.9237248611850324.
- Feature 816: token-length correlation -0.649073408954989; state-norm correlation 0.9317805715193903.

Inspect feature_pairs.csv and feature_length_correlations.csv for all units. Undefined correlations indicate constant/inactive features. High redundancy weakens claims of distinct concepts; length associations can reflect style or content as well as nuisance signals.
