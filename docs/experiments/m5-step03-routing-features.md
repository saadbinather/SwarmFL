# M5 Step 3 — Pre-consultation features and utility predictability

Read with [protocol](m5-step03-protocol.md) and [full routing report](m5-step03-intelligent-routing.md).
No outer model selection or new historical validation/test evaluation.

## Information contract

Base11: origin/candidate one-hots6, own calibrated p/confidence2, candidate domain
width1, availability1, remaining budget1. Uncertainty adds margin/entropy/p(1-p)3;
reliability adds origin/candidate training OOF BCE/Brier/ECE6; history adds one
training-only origin-peer mean gross gain. Enriched21 combines all. Cost is not
a learned feature: subtract known tariff at decision time, giving identical model
predictions under .02/.05/.10. Self/unavailable peers masked; budget0 alwaysSTOP.

Own margin/entropy add nonlinear transforms, not new evidence. Domain width is
public capability metadata; no raw clinical fields, unauthorized summaries,
patient IDs or peer predictions enter router features. No empirical capability
or missing-node variation was induced. Origin-local feature generation is node
local; only compatible shared router weights/scalars leave FL updates.

For training target H, reliability and history derive solely from upstream T\H.
History is a resubstitution summary of that teacher's OOF fusion risks, not an
independent outcome estimate; final assessment history uses cross-fitted T gains.
This intentional train/final summary shift is measured as a limitation. All
utility LABELS themselves use the fully held target-fold pipeline. Final fusers
use T-only specialist OOF, never old validation/test or outer assessment labels.

## Controlled feature comparisons, tariff .02

R2 uses a training-origin/peer mean gain predictor as null (not the assessment
mean); table macro-averages origins/folds. Correlation is descriptive. Positive R2
in some rows does not meet the all-fold/selection gates by itself. Full per-origin
MSE/R2/correlation and cross-cost metrics are in outer_metrics.csv/results.json.

| Fusion | Policy | Objective | Calls | Gain R2 vs train mean | Gain corr |
| --- | --- | --- | --- | --- | --- |
| original | ridge_base | 0.30162 | 0.84626 | 0.06585 | 0.25192 |
| original | mlp_base | 0.30212 | 0.88176 | 0.05915 | 0.25020 |
| original | tree_base | 0.31678 | 0.82374 | -0.02499 | 0.13614 |
| original | ridge_uncertainty | 0.30180 | 0.86441 | 0.06440 | 0.24788 |
| original | mlp_uncertainty | 0.29743 | 0.84439 | 0.06631 | 0.24976 |
| original | tree_uncertainty | 0.31678 | 0.82374 | -0.02499 | 0.13614 |
| original | ridge_reliability | 0.31437 | 0.79554 | -0.15456 | 0.24912 |
| original | mlp_reliability | 0.30212 | 0.88181 | 0.06021 | 0.24675 |
| original | tree_reliability | 0.31664 | 0.82374 | -0.01726 | 0.15048 |
| original | ridge_history | 0.29778 | 0.84260 | 0.06430 | 0.25480 |
| original | mlp_history | 0.30145 | 0.89467 | 0.06042 | 0.24639 |
| original | tree_history | 0.30921 | 0.73488 | 0.00130 | 0.16638 |
| original | ridge_enriched | 0.31365 | 0.75601 | -0.10546 | 0.24447 |
| original | mlp_enriched | 0.30184 | 0.86624 | 0.06976 | 0.25629 |
| original | tree_enriched | 0.30982 | 0.73488 | 0.00242 | 0.17385 |
| logistic | ridge_base | 0.43598 | 0.84834 | 0.06346 | 0.29969 |
| logistic | mlp_base | 0.43513 | 0.92856 | 0.06126 | 0.28954 |
| logistic | tree_base | 0.43858 | 0.85935 | 0.05969 | 0.32120 |
| logistic | ridge_uncertainty | 0.43909 | 0.82922 | 0.06228 | 0.30930 |
| logistic | mlp_uncertainty | 0.43723 | 0.94399 | 0.05122 | 0.28535 |
| logistic | tree_uncertainty | 0.43858 | 0.85935 | 0.05969 | 0.32120 |
| logistic | ridge_reliability | 0.43631 | 0.82648 | 0.03419 | 0.29841 |
| logistic | mlp_reliability | 0.43697 | 0.88576 | 0.06261 | 0.29365 |
| logistic | tree_reliability | 0.43849 | 0.85935 | 0.06583 | 0.32591 |
| logistic | ridge_history | 0.43606 | 0.85016 | 0.06320 | 0.29853 |
| logistic | mlp_history | 0.43772 | 0.88572 | 0.05599 | 0.28215 |
| logistic | tree_history | 0.43690 | 0.94215 | 0.05860 | 0.31761 |
| logistic | ridge_enriched | 0.43729 | 0.81009 | 0.01795 | 0.30582 |
| logistic | mlp_enriched | 0.43625 | 0.90215 | 0.06287 | 0.29643 |
| logistic | tree_enriched | 0.43609 | 0.95133 | 0.07100 | 0.32720 |

## Target variability and upstream model shift

| Fusion | Fold | Origin | Crossfit gain SD | Mean abs teacher gain shift | Own p shift |
| --- | --- | --- | --- | --- | --- |
| original | 0 | 0 | 0.41754 | 0.19273 | 0.06980 |
| original | 0 | 1 | 0.36569 | 0.12098 | 0.05704 |
| original | 0 | 2 | 0.33300 | 0.14738 | 0.13957 |
| original | 1 | 0 | 0.42179 | 0.19270 | 0.06618 |
| original | 1 | 1 | 0.17473 | 0.08414 | 0.07611 |
| original | 1 | 2 | 0.31808 | 0.13594 | 0.11710 |
| original | 2 | 0 | 0.46562 | 0.16603 | 0.10031 |
| original | 2 | 1 | 0.19296 | 0.06175 | 0.04463 |
| original | 2 | 2 | 0.36573 | 0.14393 | 0.11491 |
| logistic | 0 | 0 | 0.19873 | 0.06468 | 0.06980 |
| logistic | 0 | 1 | 0.19711 | 0.09410 | 0.05704 |
| logistic | 0 | 2 | 0.17016 | 0.04383 | 0.13957 |
| logistic | 1 | 0 | 0.14702 | 0.05578 | 0.06618 |
| logistic | 1 | 1 | 0.16152 | 0.04423 | 0.07611 |
| logistic | 1 | 2 | 0.24404 | 0.07105 | 0.11710 |
| logistic | 2 | 0 | 0.27262 | 0.11572 | 0.10031 |
| logistic | 2 | 1 | 0.16039 | 0.06972 | 0.04463 |
| logistic | 2 | 2 | 0.26803 | 0.11019 | 0.11491 |

Teacher shift compares proper held-stack gains with the final fusion's fitting-OOF
resubstitution gains on the same T patients. It combines training size, specialist
and fusion/calibration differences; it is NOT an unbiased noise estimator or an
isolated causal test. No optimistic target was used to fit/select a router.
Confidence-bin counts/mean gains/SD, per-origin/peer/fold, for target and assessment
are saved in target_diagnostics.json. They describe signal and variability without
fitting extra post-hoc policies. Oracle's lower loss bound uses labels unavailable
at query time and cannot establish attainable personalized benefit.

## Interpretation

Existing p/confidence can distinguish some uncertainty regions, but higher
dimensional summaries do not give stable gains. History/reliability vary across
teacher folds as well as candidates, so apparent fitting associations can shift
at deployment. Large gain spread relative to predicted differences and unstable
outer utility scores are consistent with noisy/shifted supervision and limited
context. They do not prove an information-theoretic impossibility or show that
the optimizer is solely responsible. The central/Flower same-architecture control
tests training protocol sensitivity separately; weight change alone is not success.
Fixed peer/always-consult remains a strong baseline and no routing solution is
promoted. Further supervised labels/features and stronger independent fits require
a new protocol, not selecting an outer winner or relaxing gates after these results.
