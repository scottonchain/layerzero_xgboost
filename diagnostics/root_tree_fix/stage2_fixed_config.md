# Stage 2: fixed-configuration sensitivity analysis (LightGBM, seed 42, 62 features, committed hyperparameters)

Not the retuned paper result. Old = b7a9a3f features, corrected = 663e026 features; identical rows and procedure.

| Metric | Original | Corrected | Difference |
|---|---|---|---|
| f1 | 0.7105 | 0.7169 | +0.0064 |
| ap | 0.7650 | 0.7668 | +0.0018 |
| auroc | 0.9672 | 0.9672 | +0.0000 |
| precision | 0.6769 | 0.7158 | +0.0389 |
| recall | 0.7476 | 0.7179 | -0.0297 |
| threshold | 0.4355 | 0.5841 | +0.1486 |

Early-stopping rounds: 1277 -> 1214. Test rows 130,435 (5,463 Sybils).

| Category | n | Sybils | Recall old | new | F1 old | new |
|---|---|---|---|---|---|---|
| IxE | 28,171 | 258 | 0.1977 | 0.1783 | 0.3148 | 0.2921 |
| IxI | 9,967 | 426 | 0.3967 | 0.3732 | 0.4767 | 0.4670 |
| IxL | 92,079 | 4,770 | 0.8086 | 0.7778 | 0.7384 | 0.7472 |
| No provider | 218 | 9 | 0.7778 | 0.7778 | 0.8750 | 0.8750 |

Affected Sybil roots (test): {"affected_sybil_roots": {"old": {"sybils": 4779, "recall": 0.8085373509102323, "mean_prob": 0.7728320535650344}, "new": {"sybils": 4779, "recall": 0.7777777777777778, "mean_prob": 0.7765905473386456}}, "affected_sybil_roots_with_descendants": {"old": {"sybils": 70, "recall": 0.4714285714285714, "mean_prob": 0.4524800194379633}, "new": {"sybils": 70, "recall": 0.38571428571428573, "mean_prob": 0.44192563283272657}}, "affected_sybil_singleton_roots": {"old": {"sybils": 4709, "recall": 0.8135485241027819, "mean_prob": 0.7775941351935958}, "new": {"sybils": 4709, "recall": 0.78360586111701, "mean_prob": 0.781565392107262}}, "unaffected_sybils": {"old": {"sybils": 684, "recall": 0.3216374269005848, "mean_prob": 0.30566837830902016}, "new": {"sybils": 684, "recall": 0.2997076023391813, "mean_prob": 0.3140213762792319}}}

Flips: {"total": 702, "pct_of_test": 0.538199103001495, "to_flagged": 74, "to_unflagged": 628, "sybil": 222, "non_sybil": 480, "in_changed_input_rows": 657, "in_unchanged_input_rows": 45, "sybil_missed_to_caught": 30, "sybil_caught_to_missed": 192, "false_pos_added": 44, "false_pos_removed": 436}
