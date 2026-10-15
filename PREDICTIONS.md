# Pre-registered predictions for the mechanism sweep

Written 2026-09-24, before `configs/sweep.yaml` was executed for the first
time. The result column is filled by `scripts/analyze_mechanism.py` from the
recorded run and by hand where a rule is not mechanical; the prediction text
is not edited after the run. Rules used to fill the column are stated next to
each prediction so that a reader can recompute the verdict.

The table below preserves the original empty result cells. Recorded verdicts are in the [completed scorecard](results/sweep-20260924/mechanism/scorecard.md).

Notation. r is the configured translation-to-width ratio of the `shift_r*`
groups: both peaks sit at +-(0.52 + r * 0.10) with width 0.10, so r = 0 is the
edge of the training centre range and r = 4 places the peaks 0.4 outside it.
s is the width scale of the `width_s*` groups. "Gain" is the paired
difference of normalized MSE-AUC, full minus baseline, so negative favors full.
Intervals are paired task bootstraps (2,000 resamples) over the 60 tasks per
group (12 tasks x 5 training seeds); the training-seed component of
uncertainty is not separately estimated, which is a stated limitation.

| ID | From | Prediction | Test | Rule for the verdict | Result |
|---|---|---|---|---|---|
| H1 | Prop. 2 | Gain of `full` over `fixed` is near zero or positive (full worse) at r <= 0.5 and turns negative (full better) for large r; the gain curve is monotone decreasing in r over the sweep. | Fig. gain vs r | Supported if the r = 3 and r = 4 gains are negative with upper CI < 0 and the r = 0 gain interval contains 0 or is positive; not supported if r = 3, 4 gains are >= 0; inconclusive otherwise. | |
| H2 | Prop. 2 | Across all sweep tasks, task-level MSE-AUC of `fixed` increases with its span residual at z = 0. | Fig. residual vs MSE | Supported if Spearman rho > 0.5 with n >= 100 tasks; not supported if rho < 0.2. | |
| H3 | Prop. 2 | The residual reduction found by the posterior, residual(z=0) - residual(z_hat), predicts the realised gain: tasks with larger reduction show more negative full-minus-fixed AUC. | Fig. reduction vs gain | Supported if Spearman rho < -0.4; not supported if rho > -0.1. | |
| H4 | Prop. 1 | The repaired `linearized` ablation tracks `full` at small r and degrades toward `fixed` as r grows: its AUC minus fixed AUC is at least half of full's at r <= 1 and less than half at r >= 3. `linearized_prior` remains indistinguishable from `fixed` at every r (the degenerate comparator recorded on 2026-09-24). | Fig. gain vs r, table | Supported if both halves hold on group means; not supported if linearized is closer to full than to fixed at r = 4 or if linearized_prior differs from fixed by more than 1e-3 in AUC anywhere. | |
| H5 | Prop. 2 | `oracle_z` is a lower envelope: its AUC is below `full` in every sweep group, and the ratio (full - oracle_z)/(fixed - oracle_z) measures how much of the available geometric gain the posterior recovers; it is below 0.5 for r >= 2. | table | Supported if oracle_z <= full in all 12 groups and the ratio condition holds at r = 2, 3, 4; partial if the envelope holds but the ratio does not. | |
| H6 | GP position | The stationary two-scale GP beats `full` when peaks are wide (s >= 0.7) and loses or ties when peaks are narrow relative to its learned length scales (s <= 0.35), at a budget of 3 to 10 observations. | Fig. GP vs full by width | Supported if the sign of gp minus full AUC flips between s = 1 and s = 0.25 with the s = 0.25 interval excluding 0; not supported if gp beats full at s = 0.25 with CI excluding 0. | |
| H7 | GP position | `gp_basis` (meta-learned basis + GP residual) is at least as good as `gp` in every group; if `full` beats `gp` anywhere but not `gp_basis`, the advantage is explained by the meta-learned basis rather than by geometric adaptation. | table | Reported as a comparison; the conditional statement is the interpretation rule, not a pass/fail. | |
| H8 | Total covariance | The between-component (geometric) share of predictive variance, between/(within+between), increases with r for `full`. Pointwise 95% coverage of `full` is below 0.90 at every r at this training budget, and the validation-chosen inflation factor is above 1.5. | Fig. variance share vs r; table | Supported if the share is higher at r = 4 than at r = 0 in group means and coverage < 0.90 in all shift groups; the factor statement is read from the checkpoints. | |
| H9 | Approximation vs misspecification | `full_rich` (40 VI steps, 32 particles) improves coverage over `full` by less than 0.05 on average, so undercoverage is not primarily approximation error at this budget. | table | Supported if mean coverage(full_rich) - coverage(full) < 0.05 across sweep groups; not supported if >= 0.10. | |
| H10 | Acquisition | In the 2x2 design, the acquisition gain (random minus active AUC) is larger for `full` than for `fixed`, and it grows with r. | Fig. 2x2 | Supported if mean gain(full) > gain(fixed) over the shift groups and gain(full) at r = 4 exceeds gain(full) at r = 0; inconclusive if intervals overlap heavily. | |

Predictions H1 to H5 are the ones the paper's Proposition 1 and 2 speak to
directly. H6 to H10 place the GP result, the undercoverage and the active
learning result inside the same framework. A prediction that fails is
reported as such; the sweep configuration and these rules are not changed
after the run.
