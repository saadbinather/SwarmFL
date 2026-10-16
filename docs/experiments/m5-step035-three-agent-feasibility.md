# CDC three-specialist amendment — routing feasibility

Read [benchmark](m5-step035-three-agent-benchmark.md), [protocol](m5-step035-three-agent-protocol.md)
and [execution log](../devlog/2026-10-14-m5-step035-three-agent.md).
Fixed specialists are A metabolic, B lifestyle, C general health. All results
are development-only; the old21-feature grouping/test boundary is unchanged.

Compare STOP, best training peer with eligibility fallback, positive-net fixed
value, random, confidence<.6, original graph and ridge/MLP base/enriched. All three
frozen fusion conditions and costs .02/.05/.10 are retained. Exact-call-mask random
controls are in complete CSV. Utility R² is relative to the training-origin/peer
mean, across both potential non-self peers including temporarily unavailable ones;
actual choices/oracle/deployment always mask unavailable peers. Gross gain labels
are whole-stack cross-fitted; true outcomes never enter prospective selection.

Macro-average origins then folds (dependent views of the same patients). AP means
average precision; accuracy/F1 threshold.5, ECE10 equal bins. Reported harm is a
realized BCE proxy versus same-fusion STOP, never clinical harm or necessity.

## Full pilot, tariff .02


### original/static

| Policy | Acc | AUROC | AP | BCE | F1 | Objective | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86111 | 0.71071 | 0.27022 | 0.36721 | 0.00821 | 0.36721 | 0.00000 | — |
| best_peer | 0.86300 | 0.79378 | 0.37261 | 0.33452 | 0.15026 | 0.35452 | 1.00000 | — |
| fixed_value | 0.86264 | 0.78973 | 0.36579 | 0.33677 | 0.13776 | 0.35455 | 0.88889 | 0.00000 |
| random | 0.86222 | 0.77579 | 0.34697 | 0.34364 | 0.09634 | 0.36364 | 1.00000 | — |
| confidence | 0.86296 | 0.72872 | 0.32533 | 0.35809 | 0.14868 | 0.36232 | 0.21162 | — |
| graph | 0.86213 | 0.78002 | 0.35062 | 0.34206 | 0.09430 | 0.35539 | 0.66667 | — |
| ridge_base | 0.86304 | 0.78656 | 0.37003 | 0.33821 | 0.15005 | 0.35408 | 0.79329 | 0.00082 |
| ridge_enriched | 0.86218 | 0.78573 | 0.36576 | 0.33885 | 0.12446 | 0.35373 | 0.74396 | 0.00048 |
| mlp_base | 0.86211 | 0.78003 | 0.35078 | 0.34209 | 0.09802 | 0.35545 | 0.66829 | -0.00024 |
| mlp_enriched | 0.86273 | 0.78029 | 0.35963 | 0.34117 | 0.12768 | 0.35472 | 0.67736 | 0.00022 |
| inner_selected | 0.86258 | 0.78599 | 0.36816 | 0.33869 | 0.13916 | 0.35418 | 0.77472 | 0.00057 |

### original/variable

| Policy | Acc | AUROC | AP | BCE | F1 | Objective | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86111 | 0.71071 | 0.27022 | 0.36721 | 0.00821 | 0.36721 | 0.00000 | — |
| best_peer | 0.86238 | 0.78383 | 0.35679 | 0.33973 | 0.12300 | 0.36140 | 0.93733 | — |
| fixed_value | 0.86162 | 0.77076 | 0.33744 | 0.34653 | 0.07923 | 0.35777 | 0.55660 | 0.00000 |
| random | 0.86191 | 0.77451 | 0.34209 | 0.34444 | 0.09378 | 0.36618 | 0.93733 | — |
| confidence | 0.86233 | 0.72532 | 0.31522 | 0.36023 | 0.12163 | 0.36493 | 0.19816 | — |
| graph | 0.86164 | 0.76860 | 0.33561 | 0.34743 | 0.07870 | 0.35808 | 0.53151 | — |
| ridge_base | 0.86191 | 0.77124 | 0.34509 | 0.34614 | 0.10008 | 0.35698 | 0.53647 | 0.00081 |
| ridge_enriched | 0.86160 | 0.77134 | 0.34103 | 0.34636 | 0.08559 | 0.35701 | 0.52498 | 0.00048 |
| mlp_base | 0.86131 | 0.76789 | 0.33261 | 0.34781 | 0.06764 | 0.35799 | 0.51192 | -0.00044 |
| mlp_enriched | 0.86160 | 0.76814 | 0.33545 | 0.34762 | 0.08069 | 0.35783 | 0.51232 | -0.00003 |
| inner_selected | 0.86169 | 0.77132 | 0.34259 | 0.34628 | 0.09505 | 0.35715 | 0.53487 | 0.00057 |

### logistic/static

| Policy | Acc | AUROC | AP | BCE | F1 | Objective | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86133 | 0.71071 | 0.27022 | 0.41357 | 0.00000 | 0.41357 | 0.00000 | — |
| best_peer | 0.86133 | 0.78921 | 0.36156 | 0.36993 | 0.00000 | 0.38993 | 1.00000 | — |
| fixed_value | 0.86133 | 0.78105 | 0.34712 | 0.37183 | 0.00000 | 0.38739 | 0.77778 | 0.00000 |
| random | 0.86133 | 0.73818 | 0.31400 | 0.37598 | 0.00000 | 0.39598 | 1.00000 | — |
| confidence | 0.86133 | 0.53642 | 0.17300 | 0.42324 | 0.00000 | 0.42747 | 0.21162 | — |
| graph | 0.86133 | 0.77870 | 0.34071 | 0.36965 | 0.00000 | 0.38509 | 0.77160 | — |
| ridge_base | 0.86133 | 0.77007 | 0.31226 | 0.35979 | 0.00000 | 0.37493 | 0.75675 | 0.03348 |
| ridge_enriched | 0.86133 | 0.76878 | 0.30947 | 0.35969 | 0.00000 | 0.37477 | 0.75384 | 0.03514 |
| mlp_base | 0.86133 | 0.77488 | 0.32178 | 0.36125 | 0.00000 | 0.37722 | 0.79853 | 0.02864 |
| mlp_enriched | 0.86133 | 0.77256 | 0.31975 | 0.36068 | 0.00000 | 0.37625 | 0.77820 | 0.03048 |
| inner_selected | 0.86133 | 0.77003 | 0.31222 | 0.35982 | 0.00000 | 0.37497 | 0.75724 | 0.03400 |

### logistic/variable

| Policy | Acc | AUROC | AP | BCE | F1 | Objective | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86133 | 0.71071 | 0.27022 | 0.41357 | 0.00000 | 0.41357 | 0.00000 | — |
| best_peer | 0.86133 | 0.73426 | 0.26028 | 0.37509 | 0.00000 | 0.39742 | 0.93733 | — |
| fixed_value | 0.86133 | 0.72491 | 0.26188 | 0.37636 | 0.00000 | 0.39122 | 0.67186 | 0.00000 |
| random | 0.86133 | 0.71579 | 0.25254 | 0.37827 | 0.00000 | 0.40001 | 0.93733 | — |
| confidence | 0.86133 | 0.54442 | 0.17808 | 0.42260 | 0.00000 | 0.42741 | 0.19816 | — |
| graph | 0.86133 | 0.72862 | 0.26201 | 0.37582 | 0.00000 | 0.39188 | 0.70547 | — |
| ridge_base | 0.86133 | 0.73676 | 0.28215 | 0.36697 | 0.00000 | 0.38216 | 0.68137 | 0.03328 |
| ridge_enriched | 0.86133 | 0.73710 | 0.28132 | 0.36718 | 0.00000 | 0.38195 | 0.67093 | 0.03501 |
| mlp_base | 0.86133 | 0.73515 | 0.28007 | 0.36811 | 0.00000 | 0.38341 | 0.69711 | 0.02865 |
| mlp_enriched | 0.86133 | 0.73650 | 0.28067 | 0.36736 | 0.00000 | 0.38271 | 0.68971 | 0.03052 |
| inner_selected | 0.86133 | 0.73829 | 0.28271 | 0.36700 | 0.00000 | 0.38198 | 0.68066 | 0.03382 |

### mean/static

| Policy | Acc | AUROC | AP | BCE | F1 | Objective | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86073 | 0.71071 | 0.27022 | 0.36598 | 0.01351 | 0.36598 | 0.00000 | — |
| best_peer | 0.86133 | 0.79313 | 0.37038 | 0.34406 | 0.00128 | 0.36406 | 1.00000 | — |
| fixed_value | 0.86078 | 0.77268 | 0.33860 | 0.35005 | 0.01131 | 0.36116 | 0.55556 | 0.00000 |
| random | 0.86133 | 0.77570 | 0.34756 | 0.35258 | 0.00128 | 0.37258 | 1.00000 | — |
| confidence | 0.86133 | 0.72678 | 0.32442 | 0.36071 | 0.00128 | 0.36494 | 0.21162 | — |
| graph | 0.86078 | 0.77268 | 0.33860 | 0.35005 | 0.01131 | 0.36116 | 0.55556 | — |
| ridge_base | 0.86116 | 0.76759 | 0.34412 | 0.35102 | 0.00159 | 0.36003 | 0.45064 | 0.00028 |
| ridge_enriched | 0.86098 | 0.77251 | 0.33607 | 0.34933 | 0.00253 | 0.36024 | 0.54569 | 0.00074 |
| mlp_base | 0.86076 | 0.76052 | 0.31964 | 0.35389 | 0.01131 | 0.36060 | 0.33578 | -0.00043 |
| mlp_enriched | 0.86078 | 0.76157 | 0.32872 | 0.35350 | 0.01131 | 0.36067 | 0.35869 | -0.00005 |
| inner_selected | 0.86078 | 0.76623 | 0.33307 | 0.35172 | 0.00850 | 0.36027 | 0.42713 | -0.00009 |

### mean/variable

| Policy | Acc | AUROC | AP | BCE | F1 | Objective | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86073 | 0.71071 | 0.27022 | 0.36598 | 0.01351 | 0.36598 | 0.00000 | — |
| best_peer | 0.86127 | 0.78270 | 0.35149 | 0.34835 | 0.00127 | 0.37003 | 0.93733 | — |
| fixed_value | 0.86082 | 0.76155 | 0.32050 | 0.35338 | 0.01228 | 0.36142 | 0.42284 | 0.00000 |
| random | 0.86127 | 0.77327 | 0.33797 | 0.35299 | 0.00127 | 0.37473 | 0.93733 | — |
| confidence | 0.86127 | 0.72373 | 0.31048 | 0.36231 | 0.00127 | 0.36701 | 0.19816 | — |
| graph | 0.86078 | 0.76163 | 0.32322 | 0.35342 | 0.01163 | 0.36229 | 0.44314 | — |
| ridge_base | 0.86096 | 0.76225 | 0.32536 | 0.35292 | 0.00221 | 0.36091 | 0.41393 | 0.00025 |
| ridge_enriched | 0.86098 | 0.76151 | 0.32163 | 0.35335 | 0.00472 | 0.36141 | 0.41284 | 0.00072 |
| mlp_base | 0.86082 | 0.76090 | 0.32023 | 0.35354 | 0.01228 | 0.36145 | 0.41771 | -0.00045 |
| mlp_enriched | 0.86082 | 0.75659 | 0.31666 | 0.35493 | 0.01228 | 0.36165 | 0.34650 | -0.00021 |
| inner_selected | 0.86091 | 0.76184 | 0.32461 | 0.35326 | 0.00853 | 0.36124 | 0.41411 | -0.00011 |

## Inner-selected pipelines and unchanged gates

Logistic fusion's selected pipeline achieves objective
0.381983 versus always-best peer
0.397418 and cost-aware fixed value
0.391216. Its fold utility R²
.0420/.0321/.0274 remains below the unchanged .05 gate, despite positive static
gains and all-three-origin variation. These are real gains in this fixed fusion
condition, not fulfillment of all acceptance criteria. Its absolute BCE/objective
is still worse than the original residual condition. Fresh three-agent fusion has
fewer training masks/optimizer updates at the unchanged epoch schedule; differences
can reflect fusion calibration/capacity, not solely a better clinical consultation.

Original residual selected objective 0.357150
versus fixed value 0.357775; mean
0.361237 versus fixed value
0.361420. Their extra gains over cost-aware
fixed controls are small and R²/peer-personalization gates fail. No candidate
passes all frozen pilot gates; no promotion,50k expansion or Step4 follows.

| Size | Fusion | Inner choices | Pass | Fold objective gains | Fold R2 | Varying origins |
| --- | --- | --- | --- | --- | --- | --- |
| small | original | ['mlp_base', 'ridge_base', 'ridge_base'] | False | [0.008335, 0.007809, 0.008405] | [-0.018469, 0.003206, 0.000462] | [0, 1, 1] |
| small | logistic | ['ridge_base', 'ridge_enriched', 'mlp_enriched'] | False | [0.009169, 0.007566, 0.012736] | [-0.019924, -0.000674, -0.028461] | [0, 1, 1] |
| small | mean | ['ridge_enriched', 'ridge_enriched', 'ridge_base'] | False | [0.007312, 0.008142, 0.006741] | [-0.001516, -5e-05, 3e-05] | [0, 1, 2] |
| pilot | original | ['ridge_base', 'ridge_enriched', 'ridge_base'] | False | [0.003962, 0.003386, 0.005402] | [0.000717, 0.000258, 0.000749] | [1, 1, 1] |
| pilot | logistic | ['ridge_enriched', 'ridge_base', 'ridge_base'] | False | [0.018985, 0.016749, 0.010571] | [0.042015, 0.032091, 0.027368] | [3, 3, 3] |
| pilot | mean | ['ridge_base', 'ridge_enriched', 'mlp_base'] | False | [0.008905, 0.00789, 0.009584] | [0.000136, 0.000408, -0.000872] | [2, 1, 1] |

All gates including origin-BCE/accuracy/static criteria are saved in gates_{small,pilot}.json. No automatic promotion.

## Cost and fitting-size sensitivity

| Size | Fusion | Tariff | Policy | BCE | Objective | Calls | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | original | 0.02000 | best_peer | 0.36043 | 0.38274 | 0.93733 | — |
| small | original | 0.02000 | fixed_value | 0.37287 | 0.37499 | 0.13204 | 0.00000 |
| small | original | 0.02000 | inner_selected | 0.37181 | 0.37456 | 0.15791 | -0.00493 |
| small | original | 0.05000 | best_peer | 0.36043 | 0.41620 | 0.93733 | — |
| small | original | 0.05000 | fixed_value | 0.37695 | 0.37695 | 0.00000 | 0.00000 |
| small | original | 0.05000 | inner_selected | 0.37696 | 0.37697 | 0.00016 | -0.00493 |
| small | original | 0.10000 | best_peer | 0.36043 | 0.47196 | 0.93733 | — |
| small | original | 0.10000 | fixed_value | 0.37695 | 0.37695 | 0.00000 | 0.00000 |
| small | original | 0.10000 | inner_selected | 0.37695 | 0.37695 | 0.00000 | -0.00493 |
| small | logistic | 0.02000 | best_peer | 0.45324 | 0.47497 | 0.93733 | — |
| small | logistic | 0.02000 | fixed_value | 0.45106 | 0.46589 | 0.61198 | 0.00000 |
| small | logistic | 0.02000 | inner_selected | 0.44967 | 0.46515 | 0.64570 | -0.01635 |
| small | logistic | 0.05000 | best_peer | 0.45324 | 0.50756 | 0.93733 | — |
| small | logistic | 0.05000 | fixed_value | 0.45701 | 0.48802 | 0.53093 | 0.00000 |
| small | logistic | 0.05000 | inner_selected | 0.45662 | 0.48812 | 0.54055 | -0.01635 |
| small | logistic | 0.10000 | best_peer | 0.45324 | 0.56188 | 0.93733 | — |
| small | logistic | 0.10000 | fixed_value | 0.48532 | 0.51614 | 0.30054 | 0.00000 |
| small | logistic | 0.10000 | inner_selected | 0.48125 | 0.51389 | 0.30649 | -0.01635 |
| small | mean | 0.02000 | best_peer | 0.35722 | 0.37890 | 0.93733 | — |
| small | mean | 0.02000 | fixed_value | 0.36285 | 0.37068 | 0.41438 | 0.00000 |
| small | mean | 0.02000 | inner_selected | 0.36338 | 0.37150 | 0.40757 | -0.00051 |
| small | mean | 0.05000 | best_peer | 0.35722 | 0.41141 | 0.93733 | — |
| small | mean | 0.05000 | fixed_value | 0.37456 | 0.37456 | 0.00000 | 0.00000 |
| small | mean | 0.05000 | inner_selected | 0.37365 | 0.37435 | 0.01682 | -0.00051 |
| small | mean | 0.10000 | best_peer | 0.35722 | 0.46560 | 0.93733 | — |
| small | mean | 0.10000 | fixed_value | 0.37456 | 0.37456 | 0.00000 | 0.00000 |
| small | mean | 0.10000 | inner_selected | 0.37456 | 0.37457 | 0.00018 | -0.00051 |
| pilot | original | 0.02000 | best_peer | 0.33973 | 0.36140 | 0.93733 | — |
| pilot | original | 0.02000 | fixed_value | 0.34653 | 0.35777 | 0.55660 | 0.00000 |
| pilot | original | 0.02000 | inner_selected | 0.34628 | 0.35715 | 0.53487 | 0.00057 |
| pilot | original | 0.05000 | best_peer | 0.33973 | 0.39391 | 0.93733 | — |
| pilot | original | 0.05000 | fixed_value | 0.36096 | 0.36624 | 0.13204 | 0.00000 |
| pilot | original | 0.05000 | inner_selected | 0.36234 | 0.36641 | 0.10175 | 0.00057 |
| pilot | original | 0.10000 | best_peer | 0.33973 | 0.44810 | 0.93733 | — |
| pilot | original | 0.10000 | fixed_value | 0.36721 | 0.36721 | 0.00000 | 0.00000 |
| pilot | original | 0.10000 | inner_selected | 0.36721 | 0.36721 | 0.00000 | 0.00057 |
| pilot | logistic | 0.02000 | best_peer | 0.37509 | 0.39742 | 0.93733 | — |
| pilot | logistic | 0.02000 | fixed_value | 0.37636 | 0.39122 | 0.67186 | 0.00000 |
| pilot | logistic | 0.02000 | inner_selected | 0.36700 | 0.38198 | 0.68066 | 0.03382 |
| pilot | logistic | 0.05000 | best_peer | 0.37509 | 0.43090 | 0.93733 | — |
| pilot | logistic | 0.05000 | fixed_value | 0.38522 | 0.40509 | 0.39185 | 0.00000 |
| pilot | logistic | 0.05000 | inner_selected | 0.37404 | 0.39876 | 0.48284 | 0.03382 |
| pilot | logistic | 0.10000 | best_peer | 0.37509 | 0.48671 | 0.93733 | — |
| pilot | logistic | 0.10000 | fixed_value | 0.39619 | 0.41457 | 0.18502 | 0.00000 |
| pilot | logistic | 0.10000 | inner_selected | 0.38996 | 0.40949 | 0.20889 | 0.03382 |
| pilot | mean | 0.02000 | best_peer | 0.34835 | 0.37003 | 0.93733 | — |
| pilot | mean | 0.02000 | fixed_value | 0.35338 | 0.36142 | 0.42284 | 0.00000 |
| pilot | mean | 0.02000 | inner_selected | 0.35326 | 0.36124 | 0.41411 | -0.00011 |
| pilot | mean | 0.05000 | best_peer | 0.34835 | 0.40254 | 0.93733 | — |
| pilot | mean | 0.05000 | fixed_value | 0.36598 | 0.36598 | 0.00000 | 0.00000 |
| pilot | mean | 0.05000 | inner_selected | 0.36571 | 0.36604 | 0.00793 | -0.00011 |
| pilot | mean | 0.10000 | best_peer | 0.34835 | 0.45673 | 0.93733 | — |
| pilot | mean | 0.10000 | fixed_value | 0.36598 | 0.36598 | 0.00000 | 0.00000 |
| pilot | mean | 0.10000 | inner_selected | 0.36594 | 0.36595 | 0.00009 | -0.00011 |

## Per-origin behavior at .02, variable services

| Fusion | Origin | BCE | Objective | Calls | R2 | [STOP,A,B,C] | Minority peer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | A | 0.34723 | 0.34930 | 0.08653 | 0.00050 | [13702, 0, 1, 1297] | 0.00099 |
| original | B | 0.35366 | 0.37217 | 0.89320 | 0.00028 | [1602, 11973, 0, 1425] | 0.10637 |
| original | C | 0.33795 | 0.34998 | 0.62487 | 0.00095 | [5627, 9373, 0, 0] | 0.00000 |
| logistic | A | 0.35757 | 0.37151 | 0.54638 | 0.03823 | [6804, 0, 3477, 4719] | 0.21484 |
| logistic | B | 0.38067 | 0.39783 | 0.80406 | 0.01260 | [2939, 10340, 0, 1721] | 0.14352 |
| logistic | C | 0.36277 | 0.37660 | 0.69154 | 0.05065 | [4627, 8969, 1404, 0] | 0.13561 |
| mean | A | 0.34854 | 0.34857 | 0.00120 | -0.00036 | [14982, 0, 4, 14] | 0.07407 |
| mean | B | 0.36252 | 0.38022 | 0.86947 | -0.00064 | [1958, 11943, 0, 1099] | 0.08426 |
| mean | C | 0.34871 | 0.35492 | 0.37167 | 0.00067 | [9425, 5573, 2, 0] | 0.00034 |

## Diversity and historical context

| Size | Prediction corr | Error corr | Probability disagreement | Threshold disagreement |
| --- | --- | --- | --- | --- |
| small | 0.22918 | 0.93470 | 0.09295 | 0.01680 |
| pilot | 0.27272 | 0.93981 | 0.09025 | 0.00511 |

Retrospective age/sex/health group losses and peer gain vectors are saved in diversity.json; these groups never enter routing features.
| Fusion | Historical6 baseline (same3 origins) | New3 baseline |
| --- | --- | --- |
| original | 0.35858 | 0.36140 |
| logistic | 0.36694 | 0.39742 |
| mean | 0.36921 | 0.37003 |

Historical numbers are read from saved development results, not a new six-agent
model evaluation. Although origins and patients match, the six-agent action set,
fusion architecture/fitting objective/mask count differ: this is descriptive,
not a controlled causal claim that fewer specialists improve quality.

## Genuine three-client Flower check

| Pilot fold0/original | Acc | AUROC | AP | BCE | Objective | Calls | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| federated | 0.86180 | 0.77408 | 0.34836 | 0.34459 | 0.36459 | 0.87753 | -0.06546 |
| central | 0.86120 | 0.76366 | 0.33228 | 0.35004 | 0.36010 | 0.50840 | -0.00118 |

Run **421462273542764912**, exactly **3 logical clients**, three FedAvg
rounds/**9 actual updates**, one CPU worker. Server work
**10.944s** excludes startup, includes full evidence queries/training/
selected-only batched consultations. Shared-router drift **1.120558**.
Only compatible37→32Tanh→1 router arrays/scalars are aggregated. Private experts
and fusion stay frozen. Numeric OOF evidence is full-information diagnostic
traffic, not production privacy. Actual selected aligned-ID queries load private
classifiers; masked fusion matches replay. Central comparison uses the same
architecture/random initialization and six pooled epochs versus3×2 local epochs
with Adam resets; optimizer schedules differ. Weight updates are not superiority.

## Interpretation and stop boundary

All three-agent criteria are assessed without relaxed thresholds. Logistic gains
and partial utility signal, where present, must be distinguished from meeting the
full R²/stability gate. STOP rate or forced availability fallback alone is not
clinical personalization. See all fixed candidates and per-origin/cost rows in
outer_metrics.csv; no outer winner is selected or default pointer promoted.

Limitations: reused development cohort/known prior results; only the user-fixed
domains; one frozen optimizer schedule; shared patients and overlapping fits;
imbalanced survey target; teacher/final calibration shift; unavailable outcomes
only as retrospective diagnostics; public availability/coverage is simulated.
No external/clinical validity, authenticated consent, filesystem isolation,
secure aggregation or proven privacy. New three-agent source is an explicit CDC
adapter, not arbitrary registration, a Qwen integration or Step4. All old checkpoints,
reports and frozen test bytes remain untouched. Stop after this amendment.
