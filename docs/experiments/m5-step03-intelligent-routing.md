# M5 Step 3 — Intelligent consultation-value routing

Executed 2026-10-14, Windows, **experimental sprint complete; routing is not solved**.
Original production router/fusion, M1–M4/Steps 1–2 checkpoints, reports and pointers
remain unchanged. No historical validation/test metric, Qwen tuning, registry or
generalized fusion system. See [frozen protocol](m5-step03-protocol.md),
[feature report](m5-step03-routing-features.md) and
[execution log](../devlog/2026-10-14-m5-step03.md).

## Findings

At tariff .02 the original graph always consults and reproduces the training-best
peer mapping in these frozen-fusion conditions; higher tariffs can produce STOP.
New value policies produce STOP and
patient-dependent decisions, but the inner-selected pipelines fail the registered
stability gates below. Enrichment is not a reliable repair. Cost/quality, utility
generalization and origin-specific behavior matter more than STOP rate alone.
No candidate is promoted and no claim of statistically proven superiority follows.

The inner-selected residual pipeline has gain R2 .0415/.1003/-.0286 over folds,
and BCE+cost .30589 versus fixed peer .30349. The selected logistic-condition
pipeline has gain R2 .0563/.1134/.0341, objective .43136 versus .43590, but worsens
one fold and origin A mean BCE by .01172. Thus measurable conditional signal in
some conditions does not establish stable action or cost-quality improvement.
Residual origin C mean BCE worsens .01200. The patient-dependence gate also fails:
selected policies vary actions in only1/1/2 origins (residual),3/2/1 (logistic).

Step 2's marginal utility teacher used the same meta-training labels as its fusion.
Here each target fold H is excluded from ALL upstream specialist fits/calibration
and fusion-fitting inputs. Splitting an existing full-T OOF matrix at the fusion
level would still let H labels influence other rows' expert predictions; this
implementation refits those experts inside T\H as well. This removes that target
leakage path, but produces smaller/noisier teacher fits and a teacher/final model
shift. It does not make realized one-label gains noiseless conditional expectations.

## Fixed experiment

Only original364 training patients; outer seed53003, three folds; inner75/25
seed53100+fold. Outer fit/held242/122,243/121,243/121; inner181/61,182/61,182/61.
Three additional target folds within each inner/outer fitting set. Existing
specialists30 epochs, own-feature preprocessing, nested calibration. **288 private
specialist fits**, 48 separately fitted fusion models, uniform valid-action BCE,
40 full-batch Adam.01 steps. Routing comparisons freeze fusion. Original residual
and existing experimental six-input logistic are paired conditions, not jointly
optimized replacements. Logistic's short uniform schedule is NOT Step2's joint
50-epoch schedule; poorer scores here cannot establish a general fusion ranking.

All value models regress untruncated gross BCE reduction; select largest positive
estimated reduction minus tariff, otherwise STOP. Ridge alpha10+fit-only scaling;
MLP21-or-ablation-input→16Tanh→1,80 Adam.01 steps/weight_decay.01; tree depth3,
minleaf30 (regularization for small noisy data). Graph is the original architecture
trained50 steps on soft expected cross-fitted action losses, frozen fusion, not
a historical M2 checkpoint. Fixed peer uses training mean gains; confidence uses
own confidence<.6 and that peer. Controls have the same permitted information
envelope but intentionally restricted decision functions. New routing APIs never
receive y/unqueried probabilities; offline teacher enumeration does.

All costs are fixed .02/.05/.10 and maximum budget1. Random-matched controls use
exact patient call masks and seeded random peers; their complete metrics are in
outer_metrics.csv/results.json. Logical replay budgets are not measured network
traffic. Separate deployed checks below use genuine selected-only Flower queries.
Metrics macro-average origins then folds; dependent views are not independent
patients. Accuracy/F1 threshold.5; ECE10 equal bins. Harm means realized BCE no
better than same-fusion STOP, not medical harm/necessity. Oracle uses outcomes
only retrospectively and is never fitted or deployed. Nothing is selected by
outer scores. All fixed candidates/costs/controls are retained, including failures.

## Outer metrics

### original, tariff 0.02

| Policy | Acc | AUROC | BCE | F1 | BCE+cost | Calls | ECE | Harm/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.80958 | 0.87860 | 0.41820 | 0.83133 | 0.41820 | 0.00000 | 0.09052 | None |
| best_peer | 0.88012 | 0.94989 | 0.28349 | 0.89997 | 0.30349 | 1.00000 | 0.08671 | 0.20036 |
| fixed_value | 0.88012 | 0.94989 | 0.28349 | 0.89997 | 0.30349 | 1.00000 | 0.08671 | 0.20036 |
| confidence | 0.87466 | 0.93587 | 0.31543 | 0.89463 | 0.32446 | 0.45165 | 0.08593 | 0.19533 |
| random_max | 0.88475 | 0.95211 | 0.29073 | 0.90371 | 0.31073 | 1.00000 | 0.09021 | 0.18291 |
| graph | 0.88012 | 0.94989 | 0.28349 | 0.89997 | 0.30349 | 1.00000 | 0.08671 | 0.20036 |
| ridge_base | 0.88012 | 0.94989 | 0.28469 | 0.89997 | 0.30162 | 0.84626 | 0.08728 | 0.18765 |
| ridge_enriched | 0.87466 | 0.94446 | 0.29853 | 0.89577 | 0.31365 | 0.75601 | 0.09118 | 0.20527 |
| mlp_base | 0.88012 | 0.94984 | 0.28449 | 0.89997 | 0.30212 | 0.88176 | 0.08775 | 0.19653 |
| mlp_enriched | 0.88012 | 0.94989 | 0.28452 | 0.89997 | 0.30184 | 0.86624 | 0.08745 | 0.19407 |
| tree_base | 0.87098 | 0.94016 | 0.30030 | 0.89298 | 0.31678 | 0.82374 | 0.08225 | 0.20239 |
| tree_enriched | 0.87186 | 0.94499 | 0.29512 | 0.89352 | 0.30982 | 0.73488 | 0.08203 | 0.19233 |

### original, tariff 0.05

| Policy | Acc | AUROC | BCE | F1 | BCE+cost | Calls | ECE | Harm/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.80958 | 0.87860 | 0.41820 | 0.83133 | 0.41820 | 0.00000 | 0.09052 | None |
| best_peer | 0.88012 | 0.94989 | 0.28349 | 0.89997 | 0.33349 | 1.00000 | 0.08671 | 0.20036 |
| fixed_value | 0.87369 | 0.94497 | 0.29408 | 0.89441 | 0.33297 | 0.77778 | 0.08503 | 0.21865 |
| confidence | 0.87466 | 0.93587 | 0.31543 | 0.89463 | 0.33801 | 0.45165 | 0.08593 | 0.19533 |
| random_max | 0.88475 | 0.95211 | 0.29073 | 0.90371 | 0.34073 | 1.00000 | 0.09021 | 0.18291 |
| graph | 0.87461 | 0.94645 | 0.29002 | 0.89526 | 0.33447 | 0.88889 | 0.08657 | 0.21301 |
| ridge_base | 0.88012 | 0.95025 | 0.28335 | 0.89997 | 0.32296 | 0.79222 | 0.08730 | 0.19333 |
| ridge_enriched | 0.87466 | 0.94454 | 0.29838 | 0.89577 | 0.33439 | 0.72024 | 0.09147 | 0.19716 |
| mlp_base | 0.87920 | 0.94888 | 0.28745 | 0.89915 | 0.32673 | 0.78567 | 0.08723 | 0.20059 |
| mlp_enriched | 0.88012 | 0.94977 | 0.28490 | 0.89997 | 0.32511 | 0.80406 | 0.08671 | 0.19406 |
| tree_base | 0.86547 | 0.93673 | 0.30590 | 0.88827 | 0.34218 | 0.72548 | 0.08128 | 0.21600 |
| tree_enriched | 0.87186 | 0.94499 | 0.29512 | 0.89352 | 0.33186 | 0.73488 | 0.08203 | 0.19233 |

### original, tariff 0.1

| Policy | Acc | AUROC | BCE | F1 | BCE+cost | Calls | ECE | Harm/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.80958 | 0.87860 | 0.41820 | 0.83133 | 0.41820 | 0.00000 | 0.09052 | None |
| best_peer | 0.88012 | 0.94989 | 0.28349 | 0.89997 | 0.38349 | 1.00000 | 0.08671 | 0.20036 |
| fixed_value | 0.87096 | 0.94259 | 0.30016 | 0.89188 | 0.36683 | 0.66667 | 0.08191 | 0.17996 |
| confidence | 0.87466 | 0.93587 | 0.31543 | 0.89463 | 0.36059 | 0.45165 | 0.08593 | 0.19533 |
| random_max | 0.88475 | 0.95211 | 0.29073 | 0.90371 | 0.39073 | 1.00000 | 0.09021 | 0.18291 |
| graph | 0.87278 | 0.94540 | 0.29544 | 0.89390 | 0.36575 | 0.70310 | 0.08108 | 0.17210 |
| ridge_base | 0.87371 | 0.94431 | 0.29981 | 0.89403 | 0.36255 | 0.62733 | 0.08703 | 0.25963 |
| ridge_enriched | 0.87281 | 0.93789 | 0.30877 | 0.89341 | 0.36212 | 0.53346 | 0.08984 | 0.21605 |
| mlp_base | 0.87461 | 0.94563 | 0.29305 | 0.89526 | 0.36000 | 0.66949 | 0.08392 | 0.23119 |
| mlp_enriched | 0.87370 | 0.94445 | 0.29733 | 0.89440 | 0.36033 | 0.63006 | 0.08411 | 0.23314 |
| tree_base | 0.85908 | 0.92473 | 0.33898 | 0.88164 | 0.37852 | 0.39539 | 0.08488 | 0.25095 |
| tree_enriched | 0.86544 | 0.93421 | 0.32117 | 0.88793 | 0.36353 | 0.42359 | 0.08292 | 0.27008 |

### logistic, tariff 0.02

| Policy | Acc | AUROC | BCE | F1 | BCE+cost | Calls | ECE | Harm/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.71727 | 0.88374 | 0.54468 | 0.79097 | 0.54468 | 0.00000 | 0.13697 | None |
| best_peer | 0.82730 | 0.91334 | 0.41590 | 0.87497 | 0.43590 | 1.00000 | 0.15776 | 0.22492 |
| fixed_value | 0.82730 | 0.91334 | 0.41590 | 0.87497 | 0.43590 | 1.00000 | 0.15776 | 0.22492 |
| confidence | 0.81455 | 0.90606 | 0.47612 | 0.86943 | 0.48515 | 0.45165 | 0.19219 | 0.26990 |
| random_max | 0.77967 | 0.90817 | 0.45215 | 0.84786 | 0.47215 | 1.00000 | 0.17339 | 0.29188 |
| graph | 0.82730 | 0.91334 | 0.41590 | 0.87497 | 0.43590 | 1.00000 | 0.15776 | 0.22492 |
| ridge_base | 0.80448 | 0.90738 | 0.41902 | 0.86025 | 0.43598 | 0.84834 | 0.16147 | 0.19331 |
| ridge_enriched | 0.80722 | 0.90892 | 0.42109 | 0.86114 | 0.43729 | 0.81009 | 0.15357 | 0.18709 |
| mlp_base | 0.81632 | 0.91288 | 0.41656 | 0.86697 | 0.43513 | 0.92856 | 0.15601 | 0.19910 |
| mlp_enriched | 0.81177 | 0.91208 | 0.41821 | 0.86548 | 0.43625 | 0.90215 | 0.15500 | 0.19741 |
| tree_base | 0.81444 | 0.90733 | 0.42139 | 0.86070 | 0.43858 | 0.85935 | 0.15239 | 0.17814 |
| tree_enriched | 0.81808 | 0.91817 | 0.41707 | 0.86784 | 0.43609 | 0.95133 | 0.16262 | 0.20300 |

### logistic, tariff 0.05

| Policy | Acc | AUROC | BCE | F1 | BCE+cost | Calls | ECE | Harm/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.71727 | 0.88374 | 0.54468 | 0.79097 | 0.54468 | 0.00000 | 0.13697 | None |
| best_peer | 0.82730 | 0.91334 | 0.41590 | 0.87497 | 0.46590 | 1.00000 | 0.15776 | 0.22492 |
| fixed_value | 0.81996 | 0.90774 | 0.42188 | 0.86845 | 0.46633 | 0.88889 | 0.15353 | 0.21998 |
| confidence | 0.81455 | 0.90606 | 0.47612 | 0.86943 | 0.49870 | 0.45165 | 0.19219 | 0.26990 |
| random_max | 0.77967 | 0.90817 | 0.45215 | 0.84786 | 0.50215 | 1.00000 | 0.17339 | 0.29188 |
| graph | 0.81996 | 0.90774 | 0.42188 | 0.86845 | 0.46633 | 0.88889 | 0.15353 | 0.21998 |
| ridge_base | 0.79538 | 0.88800 | 0.42756 | 0.85190 | 0.46637 | 0.77627 | 0.17036 | 0.19383 |
| ridge_enriched | 0.79538 | 0.89321 | 0.42853 | 0.85190 | 0.46446 | 0.71878 | 0.16652 | 0.17742 |
| mlp_base | 0.79446 | 0.88838 | 0.42612 | 0.85120 | 0.46484 | 0.77444 | 0.16861 | 0.18699 |
| mlp_enriched | 0.79446 | 0.88757 | 0.42689 | 0.85120 | 0.46516 | 0.76534 | 0.16664 | 0.18455 |
| tree_base | 0.81444 | 0.90739 | 0.42324 | 0.86070 | 0.46363 | 0.80792 | 0.15443 | 0.17468 |
| tree_enriched | 0.81262 | 0.90760 | 0.42251 | 0.85924 | 0.46409 | 0.83160 | 0.15356 | 0.17328 |

### logistic, tariff 0.1

| Policy | Acc | AUROC | BCE | F1 | BCE+cost | Calls | ECE | Harm/call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.71727 | 0.88374 | 0.54468 | 0.79097 | 0.54468 | 0.00000 | 0.13697 | None |
| best_peer | 0.82730 | 0.91334 | 0.41590 | 0.87497 | 0.51590 | 1.00000 | 0.15776 | 0.22492 |
| fixed_value | 0.79165 | 0.91841 | 0.45567 | 0.84866 | 0.50012 | 0.44444 | 0.13497 | 0.14876 |
| confidence | 0.81455 | 0.90606 | 0.47612 | 0.86943 | 0.52128 | 0.45165 | 0.19219 | 0.26990 |
| random_max | 0.77967 | 0.90817 | 0.45215 | 0.84786 | 0.55215 | 1.00000 | 0.17339 | 0.29188 |
| graph | 0.79165 | 0.91841 | 0.45567 | 0.84866 | 0.50012 | 0.44444 | 0.13497 | 0.14876 |
| ridge_base | 0.78710 | 0.89450 | 0.44355 | 0.84498 | 0.50119 | 0.57647 | 0.15731 | 0.16774 |
| ridge_enriched | 0.77883 | 0.89795 | 0.44607 | 0.84056 | 0.49952 | 0.53459 | 0.15954 | 0.16380 |
| mlp_base | 0.78710 | 0.89392 | 0.44386 | 0.84508 | 0.50142 | 0.57559 | 0.15678 | 0.19725 |
| mlp_enriched | 0.78983 | 0.90271 | 0.44322 | 0.84725 | 0.49996 | 0.56740 | 0.15100 | 0.18889 |
| tree_base | 0.81444 | 0.90438 | 0.43207 | 0.86070 | 0.50388 | 0.71816 | 0.15956 | 0.19153 |
| tree_enriched | 0.81262 | 0.90099 | 0.43312 | 0.85924 | 0.50521 | 0.72089 | 0.16132 | 0.20574 |

## Nested selected pipeline and numerical gates

### original

| Fold | Inner choice | Objective | Best peer | Improvement | Gain R2 | Varying origins |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | mlp_history | 0.36351 | 0.36553 | 0.00202 | 0.04153 | 1 |
| 1 | mlp_enriched | 0.24654 | 0.24709 | 0.00055 | 0.10034 | 1 |
| 2 | tree_reliability | 0.30762 | 0.29785 | -0.00977 | -0.02863 | 2 |

Gates: `{"every_fold_improvement_at_least_.01": false, "accuracy_drop_at_most_.01": true, "each_origin_mean_bce_degradation_at_most_.01": false, "every_fold_mean_origin_gain_r2_positive": false, "two_patient_dependent_origins_each_fold": false}`.

### logistic

| Fold | Inner choice | Objective | Best peer | Improvement | Gain R2 | Varying origins |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | tree_base | 0.56290 | 0.57370 | 0.01080 | 0.05630 | 3 |
| 1 | tree_reliability | 0.40920 | 0.41419 | 0.00498 | 0.11339 | 2 |
| 2 | ridge_enriched | 0.32197 | 0.31981 | -0.00216 | 0.03412 | 1 |

Gates: `{"every_fold_improvement_at_least_.01": false, "accuracy_drop_at_most_.01": true, "each_origin_mean_bce_degradation_at_most_.01": false, "every_fold_mean_origin_gain_r2_positive": true, "two_patient_dependent_origins_each_fold": false}`.

No automatic promotion. Gain gate uses mean origin R2 per fold relative to its training-origin/peer mean predictor; other gates are as preregistered.

## Per-origin actions, tariff .02

| Fusion | Policy | Origin | BCE | Objective | Calls | [STOP,A,B,C] | Varying folds/3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | graph | 0 | 0.35112 | 0.37112 | 1.00000 | [0, 0, 364, 0] | 0 |
| original | graph | 1 | 0.28752 | 0.30752 | 1.00000 | [0, 242, 0, 122] | 0 |
| original | graph | 2 | 0.21184 | 0.23184 | 1.00000 | [0, 0, 364, 0] | 0 |
| original | best_peer | 0 | 0.35112 | 0.37112 | 1.00000 | [0, 0, 364, 0] | 0 |
| original | best_peer | 1 | 0.28752 | 0.30752 | 1.00000 | [0, 242, 0, 122] | 0 |
| original | best_peer | 2 | 0.21184 | 0.23184 | 1.00000 | [0, 0, 364, 0] | 0 |
| original | ridge_base | 0 | 0.35112 | 0.37112 | 1.00000 | [0, 0, 364, 0] | 0 |
| original | ridge_base | 1 | 0.29049 | 0.30279 | 0.61527 | [140, 144, 0, 80] | 3 |
| original | ridge_base | 2 | 0.21247 | 0.23094 | 0.92350 | [28, 0, 336, 0] | 1 |
| original | ridge_enriched | 0 | 0.35112 | 0.37112 | 1.00000 | [0, 0, 364, 0] | 0 |
| original | ridge_enriched | 1 | 0.29172 | 0.30375 | 0.60136 | [145, 144, 0, 75] | 3 |
| original | ridge_enriched | 2 | 0.25274 | 0.26607 | 0.66667 | [122, 0, 242, 0] | 0 |
| original | mlp_enriched | 0 | 0.35112 | 0.37112 | 1.00000 | [0, 0, 364, 0] | 0 |
| original | mlp_enriched | 1 | 0.29059 | 0.30278 | 0.60965 | [142, 137, 0, 85] | 3 |
| original | mlp_enriched | 2 | 0.21184 | 0.23162 | 0.98907 | [4, 0, 360, 0] | 1 |
| original | tree_enriched | 0 | 0.35430 | 0.37173 | 0.87158 | [47, 0, 317, 0] | 1 |
| original | tree_enriched | 1 | 0.29278 | 0.30191 | 0.45615 | [198, 119, 0, 47] | 3 |
| original | tree_enriched | 2 | 0.23828 | 0.25582 | 0.87691 | [45, 57, 262, 0] | 2 |
| logistic | graph | 0 | 0.42664 | 0.44664 | 1.00000 | [0, 0, 242, 122] | 0 |
| logistic | graph | 1 | 0.42130 | 0.44130 | 1.00000 | [0, 122, 0, 242] | 0 |
| logistic | graph | 2 | 0.39975 | 0.41975 | 1.00000 | [0, 122, 242, 0] | 0 |
| logistic | best_peer | 0 | 0.42664 | 0.44664 | 1.00000 | [0, 0, 242, 122] | 0 |
| logistic | best_peer | 1 | 0.42130 | 0.44130 | 1.00000 | [0, 122, 0, 242] | 0 |
| logistic | best_peer | 2 | 0.39975 | 0.41975 | 1.00000 | [0, 122, 242, 0] | 0 |
| logistic | ridge_base | 0 | 0.43128 | 0.45008 | 0.93989 | [22, 0, 242, 100] | 1 |
| logistic | ridge_base | 1 | 0.42477 | 0.43999 | 0.76087 | [87, 98, 0, 179] | 3 |
| logistic | ridge_base | 2 | 0.40100 | 0.41788 | 0.84426 | [57, 65, 242, 0] | 1 |
| logistic | ridge_enriched | 0 | 0.42969 | 0.44789 | 0.90984 | [33, 0, 242, 89] | 1 |
| logistic | ridge_enriched | 1 | 0.42477 | 0.43999 | 0.76087 | [87, 98, 0, 179] | 3 |
| logistic | ridge_enriched | 2 | 0.40879 | 0.42398 | 0.75956 | [88, 34, 242, 0] | 1 |
| logistic | mlp_enriched | 0 | 0.42664 | 0.44664 | 1.00000 | [0, 0, 242, 122] | 0 |
| logistic | mlp_enriched | 1 | 0.42745 | 0.44453 | 0.85399 | [53, 122, 0, 189] | 1 |
| logistic | mlp_enriched | 2 | 0.40054 | 0.41759 | 0.85246 | [54, 68, 242, 0] | 1 |
| logistic | tree_enriched | 0 | 0.42664 | 0.44664 | 1.00000 | [0, 0, 242, 122] | 0 |
| logistic | tree_enriched | 1 | 0.42591 | 0.44299 | 0.85399 | [53, 256, 0, 55] | 2 |
| logistic | tree_enriched | 2 | 0.39864 | 0.41864 | 1.00000 | [0, 91, 273, 0] | 2 |

## Retrospective oracle lower loss bound

| Fusion | Tariff | Origin | Oracle objective | Oracle STOP |
| --- | --- | --- | --- | --- |
| original | 0.02000 | 0 | 0.23578 | 0.08217 |
| original | 0.02000 | 1 | 0.21954 | 0.40943 |
| original | 0.02000 | 2 | 0.16804 | 0.33257 |
| original | 0.05000 | 0 | 0.26180 | 0.15876 |
| original | 0.05000 | 1 | 0.23581 | 0.48096 |
| original | 0.05000 | 2 | 0.18639 | 0.43698 |
| original | 0.10000 | 0 | 0.30298 | 0.20280 |
| original | 0.10000 | 1 | 0.25830 | 0.64571 |
| original | 0.10000 | 2 | 0.21162 | 0.54688 |
| logistic | 0.02000 | 0 | 0.37993 | 0.15596 |
| logistic | 0.02000 | 1 | 0.36281 | 0.15942 |
| logistic | 0.02000 | 2 | 0.34604 | 0.06851 |
| logistic | 0.05000 | 0 | 0.40512 | 0.16698 |
| logistic | 0.05000 | 1 | 0.38662 | 0.25024 |
| logistic | 0.05000 | 2 | 0.37323 | 0.11523 |
| logistic | 0.10000 | 0 | 0.44622 | 0.20275 |
| logistic | 0.10000 | 1 | 0.42059 | 0.40435 |
| logistic | 0.10000 | 2 | 0.41582 | 0.17556 |

## Cost/loss Pareto comparison

| Fusion | Tariff | Nondominated fixed policies |
| --- | --- | --- |
| original | 0.02000 | mlp_uncertainty, ridge_history, tree_history, confidence, stop |
| original | 0.05000 | mlp_base, ridge_reliability, mlp_reliability, ridge_history, tree_history, ridge_enriched, fixed_value, confidence, stop |
| original | 0.10000 | mlp_base, ridge_uncertainty, mlp_uncertainty, ridge_reliability, mlp_reliability, tree_reliability, ridge_history, mlp_history, tree_history, ridge_enriched, best_peer, confidence, stop |
| logistic | 0.02000 | ridge_base, mlp_base, ridge_reliability, ridge_enriched, mlp_enriched, fixed_value, best_peer, confidence, stop, graph |
| logistic | 0.05000 | ridge_uncertainty, tree_reliability, ridge_enriched, mlp_enriched, tree_enriched, linear_enriched, fixed_value, best_peer, confidence, stop, graph |
| logistic | 0.10000 | ridge_uncertainty, ridge_reliability, mlp_reliability, tree_reliability, ridge_enriched, linear_enriched, fixed_value, best_peer, stop, graph |

This empirical frontier is descriptive and not an additional selection procedure.
Identical predictors can occupy the same point. CSV retains per-fold metrics,
utility MSE/correlation/R2, all matched random controls, calibration and actions;
results.json retains per-patient actions. No budget2 evidence enters comparison.

## Genuine Flower routing-only comparison

| Fusion | Fold | Architecture | Central obj | FL obj | Central calls | FL calls | Weight L2 | Min grad cosine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original | 0 | mlp | 0.36323 | 0.36314 | 0.88798 | 0.88525 | 2.27192 | -0.91919 |
| original | 1 | mlp | 0.24654 | 0.24709 | 0.82094 | 1.00000 | 2.23638 | -0.89850 |
| original | 2 | mlp | 0.29575 | 0.29785 | 0.88981 | 1.00000 | 2.34866 | -0.88344 |
| logistic | 0 | mlp | 0.57154 | 0.57862 | 0.85246 | 0.82240 | 2.28597 | -0.90293 |
| logistic | 1 | mlp | 0.41741 | 0.41428 | 0.85399 | 0.95592 | 2.23319 | -0.85429 |
| logistic | 2 | mlp | 0.31981 | 0.31981 | 1.00000 | 1.00000 | 2.37172 | -0.83721 |

Run **8790470071382857781**, **30 rounds / 90 client updates**, three logical clients,
one CPU Ray worker; wall time **29.859s** including evidence queries,
training and selected consultations. Both fusion conditions selected MLP among
the two eligible neural architectures using INNER scores only. This is not a
claim that MLP beat every centrally evaluated tree/ridge/feature variant.
Central comparator uses identical architecture, seed, target rows and80 pooled
Adam steps. FL uses five rounds×16 origin-local steps, Adam reset each round;
optimizer/averaging schedules differ. Every global array equals its three local
updates within tolerance. Private expert/fusion parameters never enter ArrayRecords.
Gradient cosine conflict and nonzero weight drift demonstrate actual optimization,
not improvement. Contexts/parameters remain incompatible with default serving.

72 actual evidence queries bind probabilities to context/client/IDs and hashes;
central diagnostics initially load trusted local caches, and Flower later acquires
and verifies those exact probabilities. No claim that central preparation itself
was federated. At tariff.02 the FL router selected peers before sending aligned
assessment IDs, peers actually loaded private classifiers and returned only those
probabilities. Masked fusion outputs matched replay within1e-6. These are batched
functional checks, not per-request serving latency/security benchmarks.

## Verification and preservation

Independent verification: `{"private_states": 72, "fusion_states_refitted": 48, "local_router_updates_replayed": 90, "oof_columns_refitted": 72, "utility_rows": 15276, "metric_fields": 83862, "aggregations": 30, "client_updates": 90, "preserved_files": 4166, "changed": [], "success": true}`.
Fresh regression QA: **84 passed, 2 skipped, 2 warnings in 13.31s**. Ruff lint and format (71 files) and pip check
passed; exact commands/return codes/runtimes are in qa/checks.json. Optional old
Qwen/feedback runtime benchmarks are explicitly disabled; this sprint's actual
training-only Flower integration was executed separately. Two Typer warnings
are upstream. Captured pip freeze is environment evidence, not a portable lock.
Nested centralized/preparation time **86.696s**, CPU. The device
audit performed a real RTX A4000 CUDA matrix operation (environment.json); CUDA
was not used to accelerate these tiny repeated training fits. CPU peak CUDA
allocation is0; that is not process RAM, reserved GPU memory or total board usage.
No CPU/GPU equal-work performance comparison or GPU saturation claim.

## Commands and artifacts

```powershell
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.m5_step03 --output artifacts/m5-step03-v1
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.value_federation --output artifacts/m5-step03-v1 --device cpu
.venv/Scripts/python.exe scripts/verify_m5_step03.py artifacts/m5-step03-v1 --refit-oof
.venv/Scripts/python.exe scripts/report_m5_step03.py artifacts/m5-step03-v1
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check --no-cache src tests scripts
.venv/Scripts/python.exe -m ruff format --no-cache --check src tests scripts
.venv/Scripts/python.exe -m pip check
```

For reproduction choose a fresh versioned output directory for BOTH launchers.
Prepared M2 seed42 source is required; no downloads/service/Qwen needed. CPU is
default; optional Flower --device auto/cuda/cpu supports at most one GPU worker.
Windows is executed; Linux substitutes activated python, but WSL is unverified.
Artifacts include immutable protocol copy, preservation manifest, every nested
specialist/fusion state and OOF provenance, fitting/held utility arrays, fitted
routers, inner choices, all assessment metrics/actions, calibration/reliability,
confidence bins, target diagnostics, Pareto/CSV, full Flower message evidence,
all client/global states, gradient/drift audits, verification and QA logs.

## Limitations, failures and recommendations

One364-patient cohort, small overlapping fits, reused training patients from past
development and many comparisons: these are development diagnostics, not an
independent study, statistical superiority or clinical validity. Realized utility
is heavy-tailed BCE from one binary outcome; small target-fold specialists/fusers
can shift calibration and predicted benefits. Reliable-looking mean statistics
can encode teacher-fold identity and shift at final inference. Margins/entropy
are deterministic own-probability transforms; fixed capabilities/availability
have no experimentally identified effect. No raw metadata summary was added
without a new authorization contract. All patient files remain accessible to a
trusted local simulator; predictions/IDs/model updates can disclose information.
No proven privacy, differential privacy or secure aggregation.

Separately fitted logistic in this fixed short schedule is a limited condition,
not evidence invalidating Step2's distinct logistic results. Architecture choice,
target quality, local optimizer and model-size shift remain confounded; this
study does not causally identify a sole failure mechanism. STOP variation is not
automatically useful. The historical M4 observed-only objective was not replaced
with offline full-information targets. No Qwen learning or production migration.

Initial inspection tried nonexistent fusion.py/router.py paths; source lives in
fusion/network.py and routing/graph.py. Initial Ruff findings were ordinary
dictionary iteration/closure binding/style issues, corrected without model
changes. Existing Windows Ray shutdown/deprecated-launcher warnings remain;
successful rounds are established by saved arrays, messages and numerical replay.
QA details and any additional failures are in the execution log.

Priorities for a NEW reviewed protocol: stronger independently cross-fitted
fusion schedules with sample-size/temperature controls; label-efficient utility
estimators and conditional-null analyses; feature-history transport/shift controls;
patient-level uncertainty/repeated new splits and eventually independent cohorts.
Choose future experiments before outcomes, retain simple fixed-peer baselines,
and preserve the historical test. No registry/generalized fusion/Qwen training,
OS/driver changes, publishing or GitHub push. **Stop after Step3.**

## Exact changed files

New experiments/consultation_value.py, experiments/m5_step03.py,
experiments/value_federation.py; scripts/verify_m5_step03.py,
scripts/report_m5_step03.py; tests/test_consultation_value.py; this report,
m5-step03-routing-features.md, m5-step03-protocol.md and the dated devlog.
Updated README.md and docs/status.md, architecture.md, limitations.md,
next-steps.md. All earlier experiment reports/default modules remain unchanged.
