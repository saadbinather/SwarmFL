# M5 Step 3.5 — Routing feasibility on larger CDC data

Read [benchmark/data protocol](m5-step035-benchmark.md), [frozen preregistration](m5-step035-protocol.md)
and [execution log](../devlog/2026-10-14-m5-step035.md). All results are development-only.

## Controls and permitted information

STOP; best training-selected peer with eligibility fallback; positive-net fixed
value; random max; confidence<.6; existing graph architecture6clients/one-hop;
ridge/MLP each base/enriched. Frozen original/logistic/mean fusion, maxone peer,
costs .02/.05/.10. Best peer ranks mean CROSS-FITTED training gain, not assessment
labels. Ridgealpha100/fitting-only standardization; MLP32Tanh,20epochs Adam.003,
weight_decay.01. Graph15epochs on fixed cross-fitted action risks; same architecture
as existing routing, fresh six-client weights. Graph raw scores are not calibrated
BCE gains; its subtract-price comparison is a restricted control.
Utility R2 scores all five potential non-self peers, including temporarily
unavailable peers in retrospective full-information diagnostics. Actual action
objectives, oracle bounds and deployments strictly mask unavailable specialists.

Base own p/confidence, identities, public candidate width/overlap/availability/
budget; enriched entropy/variance, training-only own/candidate reliability and
public21-field coverage. Routing inference APIs have no labels, hidden peer
predictions, patient identifiers or clinical feature values. Labels create only
offline cross-fitted targets and retrospective metrics. Reliability for target H
uses T\H OOF, not H labels. No in-sample fusion target teacher.

STATIC: all covered/available, uniform tariffs. VARIABLE: independent exogenous
lifestyle/general-health coverage outages,80% service availability, heterogeneous
recipient/congestion cost. Origin fields stay present. Missing required coverage
makes that specialist unavailable; experts are never artificially damaged or given
untrained missing inputs. Schedules depend on row IDs/seed, not clinical outcomes.
All comparisons see identical masks/prices/evidence budget. Changing service
eligibility alone can cause peer variation; static results are essential.

## Matched outer results at tariff .02

Macro-average origins then folds; AP=average precision, ECE10 equal-width bins;
F1/accuracy threshold .5. Complete all-cost/per-origin metrics and exact-mask random
controls are in outer_metrics.csv; actions.npz retains per-patient decisions.


### small/original/static

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86052 | 0.68931 | 0.24340 | 0.37599 | 0.02882 | 0.37599 | 0.00000 | — |
| best_peer | 0.85996 | 0.76707 | 0.33035 | 0.34857 | 0.10034 | 0.36857 | 1.00000 | — |
| fixed_value | 0.85981 | 0.75238 | 0.31221 | 0.35437 | 0.08909 | 0.36881 | 0.72222 | -0.00000 |
| random | 0.86002 | 0.73440 | 0.28940 | 0.36135 | 0.05081 | 0.38135 | 1.00000 | — |
| confidence | 0.86000 | 0.70887 | 0.28787 | 0.36891 | 0.09342 | 0.37339 | 0.22397 | — |
| graph | 0.86053 | 0.71128 | 0.26780 | 0.36816 | 0.05836 | 0.37371 | 0.27778 | — |
| ridge_base | 0.86024 | 0.75015 | 0.32062 | 0.35527 | 0.11004 | 0.36809 | 0.64109 | 0.00007 |
| ridge_enriched | 0.86006 | 0.75590 | 0.32149 | 0.35340 | 0.09584 | 0.36692 | 0.67612 | 0.00137 |
| mlp_base | 0.86017 | 0.70838 | 0.26441 | 0.36946 | 0.04799 | 0.37407 | 0.23050 | -0.00132 |
| mlp_enriched | 0.86042 | 0.71682 | 0.27570 | 0.36591 | 0.05699 | 0.37305 | 0.35693 | -0.00178 |
| inner_selected | 0.86047 | 0.74123 | 0.30764 | 0.35816 | 0.10303 | 0.36917 | 0.55049 | -0.00016 |

### small/original/variable

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86052 | 0.68931 | 0.24340 | 0.37599 | 0.02882 | 0.37599 | 0.00000 | — |
| best_peer | 0.85994 | 0.76324 | 0.32434 | 0.35022 | 0.09156 | 0.37360 | 0.99942 | — |
| fixed_value | 0.85999 | 0.73449 | 0.29046 | 0.36159 | 0.06301 | 0.37022 | 0.44558 | -0.00000 |
| random | 0.86002 | 0.73488 | 0.29026 | 0.36116 | 0.05143 | 0.38183 | 0.99942 | — |
| confidence | 0.85996 | 0.70811 | 0.28466 | 0.36944 | 0.08583 | 0.37477 | 0.22388 | — |
| graph | 0.86049 | 0.70706 | 0.26191 | 0.36987 | 0.05251 | 0.37432 | 0.22196 | — |
| ridge_base | 0.86004 | 0.73631 | 0.30052 | 0.36091 | 0.08375 | 0.37017 | 0.49657 | 0.00012 |
| ridge_enriched | 0.86004 | 0.74016 | 0.30356 | 0.35965 | 0.07855 | 0.36940 | 0.51860 | 0.00134 |
| mlp_base | 0.85994 | 0.71437 | 0.27161 | 0.36825 | 0.05278 | 0.37250 | 0.26392 | -0.00123 |
| mlp_enriched | 0.86019 | 0.71851 | 0.27519 | 0.36667 | 0.06405 | 0.37309 | 0.33408 | -0.00182 |
| inner_selected | 0.86016 | 0.73039 | 0.29444 | 0.36301 | 0.07865 | 0.37099 | 0.43017 | -0.00008 |

### small/logistic/static

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86133 | 0.68931 | 0.24340 | 0.41950 | 0.00000 | 0.41950 | 0.00000 | — |
| best_peer | 0.86133 | 0.75229 | 0.30938 | 0.37291 | 0.00000 | 0.39291 | 1.00000 | — |
| fixed_value | 0.86133 | 0.74931 | 0.30677 | 0.37422 | 0.00000 | 0.39311 | 0.94444 | 0.00000 |
| random | 0.86133 | 0.71924 | 0.27497 | 0.37810 | 0.00000 | 0.39810 | 1.00000 | — |
| confidence | 0.86133 | 0.54039 | 0.16509 | 0.42202 | 0.00000 | 0.42650 | 0.22397 | — |
| graph | 0.86133 | 0.73139 | 0.28635 | 0.37591 | 0.00000 | 0.39591 | 1.00000 | — |
| ridge_base | 0.86133 | 0.72555 | 0.26360 | 0.37146 | 0.00000 | 0.38721 | 0.78730 | 0.02085 |
| ridge_enriched | 0.86133 | 0.72518 | 0.26325 | 0.37103 | 0.00000 | 0.38648 | 0.77268 | 0.02163 |
| mlp_base | 0.86133 | 0.74025 | 0.28049 | 0.37274 | 0.00000 | 0.39081 | 0.90333 | 0.01515 |
| mlp_enriched | 0.86133 | 0.73654 | 0.27368 | 0.37204 | 0.00000 | 0.38956 | 0.87597 | 0.01709 |
| inner_selected | 0.86133 | 0.72510 | 0.26309 | 0.37141 | 0.00000 | 0.38707 | 0.78300 | 0.02096 |

### small/logistic/variable

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86133 | 0.68931 | 0.24340 | 0.41950 | 0.00000 | 0.41950 | 0.00000 | — |
| best_peer | 0.86133 | 0.74877 | 0.30355 | 0.37309 | 0.00000 | 0.39591 | 0.99942 | — |
| fixed_value | 0.86133 | 0.73683 | 0.28955 | 0.37588 | 0.00000 | 0.39421 | 0.94233 | 0.00000 |
| random | 0.86133 | 0.71956 | 0.27464 | 0.37794 | 0.00000 | 0.39861 | 0.99942 | — |
| confidence | 0.86133 | 0.54036 | 0.16547 | 0.42192 | 0.00000 | 0.42699 | 0.22388 | — |
| graph | 0.86133 | 0.73288 | 0.28623 | 0.37570 | 0.00000 | 0.39441 | 0.99921 | — |
| ridge_base | 0.86133 | 0.72096 | 0.26132 | 0.37251 | 0.00000 | 0.38652 | 0.77203 | 0.02076 |
| ridge_enriched | 0.86133 | 0.72089 | 0.26059 | 0.37220 | 0.00000 | 0.38603 | 0.75417 | 0.02132 |
| mlp_base | 0.86133 | 0.71301 | 0.25894 | 0.37585 | 0.00000 | 0.39029 | 0.88535 | 0.01507 |
| mlp_enriched | 0.86133 | 0.71190 | 0.25647 | 0.37579 | 0.00000 | 0.38961 | 0.85194 | 0.01686 |
| inner_selected | 0.86133 | 0.72023 | 0.26039 | 0.37257 | 0.00000 | 0.38642 | 0.76266 | 0.02088 |

### small/mean/static

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86026 | 0.68931 | 0.24340 | 0.37463 | 0.02603 | 0.37463 | 0.00000 | — |
| best_peer | 0.86142 | 0.77069 | 0.33677 | 0.35353 | 0.00438 | 0.37353 | 1.00000 | — |
| fixed_value | 0.86024 | 0.74089 | 0.29718 | 0.36203 | 0.02540 | 0.37091 | 0.44444 | 0.00000 |
| random | 0.86132 | 0.73600 | 0.29095 | 0.36497 | 0.00111 | 0.38497 | 1.00000 | — |
| confidence | 0.86142 | 0.70871 | 0.29264 | 0.36823 | 0.00438 | 0.37271 | 0.22397 | — |
| graph | 0.86048 | 0.71167 | 0.26845 | 0.36813 | 0.02422 | 0.37369 | 0.27778 | — |
| ridge_base | 0.86130 | 0.74289 | 0.30528 | 0.36096 | 0.00423 | 0.37062 | 0.48327 | 0.00063 |
| ridge_enriched | 0.86118 | 0.75834 | 0.31953 | 0.35656 | 0.00913 | 0.37154 | 0.74905 | -0.00052 |
| mlp_base | 0.86046 | 0.70232 | 0.25735 | 0.37140 | 0.02391 | 0.37335 | 0.09774 | -0.00069 |
| mlp_enriched | 0.86047 | 0.71091 | 0.26678 | 0.36940 | 0.02269 | 0.37264 | 0.16235 | -0.00110 |
| inner_selected | 0.86132 | 0.74660 | 0.30918 | 0.35947 | 0.00439 | 0.37055 | 0.55404 | 0.00022 |

### small/mean/variable

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86026 | 0.68931 | 0.24340 | 0.37463 | 0.02603 | 0.37463 | 0.00000 | — |
| best_peer | 0.86133 | 0.76581 | 0.32756 | 0.35510 | 0.00345 | 0.37809 | 0.99942 | — |
| fixed_value | 0.86047 | 0.73407 | 0.28893 | 0.36363 | 0.02428 | 0.37134 | 0.41321 | 0.00000 |
| random | 0.86133 | 0.73638 | 0.29110 | 0.36486 | 0.00127 | 0.38553 | 0.99942 | — |
| confidence | 0.86133 | 0.70750 | 0.28703 | 0.36885 | 0.00345 | 0.37411 | 0.22388 | — |
| graph | 0.86047 | 0.70781 | 0.26322 | 0.36937 | 0.02485 | 0.37382 | 0.22196 | — |
| ridge_base | 0.86126 | 0.73379 | 0.29080 | 0.36352 | 0.00298 | 0.37123 | 0.41669 | 0.00065 |
| ridge_enriched | 0.86112 | 0.74259 | 0.30240 | 0.36120 | 0.00979 | 0.37177 | 0.55862 | -0.00045 |
| mlp_base | 0.86050 | 0.70628 | 0.26077 | 0.37042 | 0.02233 | 0.37273 | 0.14580 | -0.00070 |
| mlp_enriched | 0.86049 | 0.70613 | 0.26173 | 0.37046 | 0.02375 | 0.37284 | 0.14599 | -0.00110 |
| inner_selected | 0.86126 | 0.73571 | 0.29466 | 0.36283 | 0.00408 | 0.37147 | 0.46121 | 0.00026 |

### pilot/original/static

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86131 | 0.70973 | 0.26208 | 0.36777 | 0.00127 | 0.36777 | 0.00000 | — |
| best_peer | 0.86171 | 0.79246 | 0.36407 | 0.33553 | 0.12663 | 0.35553 | 1.00000 | — |
| fixed_value | 0.86171 | 0.79246 | 0.36407 | 0.33553 | 0.12663 | 0.35553 | 1.00000 | 0.00000 |
| random | 0.86106 | 0.75849 | 0.31736 | 0.35089 | 0.05447 | 0.37089 | 1.00000 | — |
| confidence | 0.86184 | 0.73494 | 0.32366 | 0.35709 | 0.11757 | 0.36217 | 0.25367 | — |
| graph | 0.86129 | 0.78352 | 0.35340 | 0.34016 | 0.10034 | 0.35682 | 0.83333 | — |
| ridge_base | 0.86167 | 0.78520 | 0.36103 | 0.33863 | 0.12333 | 0.35512 | 0.82442 | 0.00030 |
| ridge_enriched | 0.86168 | 0.78604 | 0.36159 | 0.33846 | 0.11940 | 0.35457 | 0.80587 | 0.00043 |
| mlp_base | 0.86130 | 0.72716 | 0.27881 | 0.36225 | 0.01310 | 0.36447 | 0.11111 | -0.00378 |
| mlp_enriched | 0.86128 | 0.72517 | 0.27603 | 0.36305 | 0.00475 | 0.36527 | 0.11111 | -0.00362 |
| inner_selected | 0.86178 | 0.78584 | 0.36171 | 0.33844 | 0.12451 | 0.35481 | 0.81875 | 0.00031 |

### pilot/original/variable

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86131 | 0.70973 | 0.26208 | 0.36777 | 0.00127 | 0.36777 | 0.00000 | — |
| best_peer | 0.86102 | 0.78725 | 0.35318 | 0.33798 | 0.10930 | 0.36098 | 0.99942 | — |
| fixed_value | 0.86072 | 0.77342 | 0.33330 | 0.34442 | 0.08189 | 0.35796 | 0.70299 | 0.00000 |
| random | 0.86072 | 0.75909 | 0.31583 | 0.35052 | 0.05199 | 0.37119 | 0.99942 | — |
| confidence | 0.86129 | 0.73235 | 0.31628 | 0.35862 | 0.10243 | 0.36458 | 0.25353 | — |
| graph | 0.86101 | 0.77091 | 0.33642 | 0.34576 | 0.08109 | 0.35907 | 0.66458 | — |
| ridge_base | 0.86096 | 0.77479 | 0.34272 | 0.34403 | 0.09636 | 0.35808 | 0.73159 | 0.00029 |
| ridge_enriched | 0.86087 | 0.77283 | 0.33856 | 0.34505 | 0.08376 | 0.35758 | 0.65659 | 0.00040 |
| mlp_base | 0.86078 | 0.73354 | 0.28923 | 0.35960 | 0.02479 | 0.36907 | 0.68103 | -0.00377 |
| mlp_enriched | 0.86083 | 0.73298 | 0.28831 | 0.35982 | 0.02350 | 0.36906 | 0.66796 | -0.00361 |
| inner_selected | 0.86096 | 0.77415 | 0.34183 | 0.34435 | 0.09432 | 0.35787 | 0.70544 | 0.00031 |

### pilot/logistic/static

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86120 | 0.70972 | 0.26167 | 0.40461 | 0.00544 | 0.40461 | 0.00000 | — |
| best_peer | 0.86151 | 0.79251 | 0.36548 | 0.34287 | 0.00378 | 0.36287 | 1.00000 | — |
| fixed_value | 0.86151 | 0.79251 | 0.36548 | 0.34287 | 0.00378 | 0.36287 | 1.00000 | 0.00000 |
| random | 0.86128 | 0.75708 | 0.31231 | 0.35609 | 0.00096 | 0.37609 | 1.00000 | — |
| confidence | 0.86151 | 0.67167 | 0.26890 | 0.39527 | 0.00378 | 0.40035 | 0.25367 | — |
| graph | 0.86139 | 0.77906 | 0.34684 | 0.34745 | 0.00157 | 0.36745 | 1.00000 | — |
| ridge_base | 0.86119 | 0.77032 | 0.30756 | 0.34894 | 0.00482 | 0.36568 | 0.83692 | 0.00182 |
| ridge_enriched | 0.86119 | 0.76411 | 0.29622 | 0.35091 | 0.00482 | 0.36661 | 0.78519 | 0.00299 |
| mlp_base | 0.86136 | 0.78361 | 0.34857 | 0.34591 | 0.00456 | 0.36580 | 0.99464 | 0.00195 |
| mlp_enriched | 0.86123 | 0.78296 | 0.34103 | 0.34612 | 0.00479 | 0.36579 | 0.98358 | 0.00246 |
| inner_selected | 0.86122 | 0.77322 | 0.31485 | 0.34848 | 0.00479 | 0.36552 | 0.85200 | 0.00337 |

### pilot/logistic/variable

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86120 | 0.70972 | 0.26167 | 0.40461 | 0.00544 | 0.40461 | 0.00000 | — |
| best_peer | 0.86146 | 0.78553 | 0.35285 | 0.34492 | 0.00332 | 0.36810 | 0.99942 | — |
| fixed_value | 0.86141 | 0.77595 | 0.33883 | 0.34856 | 0.00205 | 0.36746 | 0.99928 | 0.00000 |
| random | 0.86124 | 0.75717 | 0.31001 | 0.35599 | 0.00096 | 0.37665 | 0.99942 | — |
| confidence | 0.86146 | 0.67017 | 0.26420 | 0.39624 | 0.00332 | 0.40226 | 0.25353 | — |
| graph | 0.86138 | 0.77501 | 0.33705 | 0.34891 | 0.00173 | 0.36780 | 0.99854 | — |
| ridge_base | 0.86118 | 0.76150 | 0.29684 | 0.35217 | 0.00451 | 0.36794 | 0.82089 | 0.00187 |
| ridge_enriched | 0.86119 | 0.75490 | 0.28603 | 0.35441 | 0.00482 | 0.36916 | 0.76916 | 0.00305 |
| mlp_base | 0.86129 | 0.76629 | 0.32500 | 0.35175 | 0.00315 | 0.36932 | 0.99221 | 0.00198 |
| mlp_enriched | 0.86128 | 0.76827 | 0.32089 | 0.35072 | 0.00452 | 0.36837 | 0.97434 | 0.00245 |
| inner_selected | 0.86124 | 0.76215 | 0.30013 | 0.35251 | 0.00436 | 0.36822 | 0.84396 | 0.00336 |

### pilot/mean/static

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86069 | 0.70972 | 0.26167 | 0.36699 | 0.01225 | 0.36699 | 0.00000 | — |
| best_peer | 0.86133 | 0.78928 | 0.36130 | 0.34497 | 0.00112 | 0.36497 | 1.00000 | — |
| fixed_value | 0.86106 | 0.77179 | 0.33423 | 0.34997 | 0.00645 | 0.36219 | 0.61111 | 0.00000 |
| random | 0.86126 | 0.75635 | 0.31222 | 0.35682 | 0.00206 | 0.37682 | 1.00000 | — |
| confidence | 0.86133 | 0.73174 | 0.32144 | 0.36004 | 0.00112 | 0.36511 | 0.25367 | — |
| graph | 0.86120 | 0.77669 | 0.34477 | 0.34832 | 0.00537 | 0.36285 | 0.72656 | — |
| ridge_base | 0.86124 | 0.76863 | 0.33873 | 0.35027 | 0.00127 | 0.36121 | 0.54726 | 0.00053 |
| ridge_enriched | 0.86107 | 0.77424 | 0.33308 | 0.34877 | 0.00485 | 0.36111 | 0.61694 | 0.00079 |
| mlp_base | 0.86069 | 0.70972 | 0.26167 | 0.36699 | 0.01225 | 0.36699 | 0.00000 | -0.00493 |
| mlp_enriched | 0.86069 | 0.70972 | 0.26167 | 0.36699 | 0.01225 | 0.36699 | 0.00000 | -0.00481 |
| inner_selected | 0.86107 | 0.77317 | 0.33552 | 0.34895 | 0.00485 | 0.36075 | 0.58999 | 0.00126 |

### pilot/mean/variable

| Policy | Acc | AUROC | AP | BCE | F1 | BCE+cost | Calls | Gain R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.86069 | 0.70972 | 0.26167 | 0.36699 | 0.01225 | 0.36699 | 0.00000 | — |
| best_peer | 0.86131 | 0.78463 | 0.35147 | 0.34653 | 0.00112 | 0.36920 | 0.99942 | — |
| fixed_value | 0.86100 | 0.75634 | 0.30998 | 0.35438 | 0.00805 | 0.36278 | 0.44052 | 0.00000 |
| random | 0.86129 | 0.75642 | 0.31045 | 0.35678 | 0.00207 | 0.37745 | 0.99942 | — |
| confidence | 0.86131 | 0.72988 | 0.31489 | 0.36081 | 0.00112 | 0.36662 | 0.25353 | — |
| graph | 0.86111 | 0.76428 | 0.32395 | 0.35207 | 0.00664 | 0.36364 | 0.57792 | — |
| ridge_base | 0.86122 | 0.75860 | 0.32030 | 0.35364 | 0.00143 | 0.36243 | 0.47063 | 0.00053 |
| ridge_enriched | 0.86103 | 0.76113 | 0.31419 | 0.35321 | 0.00455 | 0.36247 | 0.49229 | 0.00076 |
| mlp_base | 0.86069 | 0.70972 | 0.26167 | 0.36699 | 0.01225 | 0.36699 | 0.00000 | -0.00493 |
| mlp_enriched | 0.86069 | 0.71138 | 0.26296 | 0.36664 | 0.01225 | 0.36691 | 0.02207 | -0.00480 |
| inner_selected | 0.86103 | 0.76115 | 0.31803 | 0.35303 | 0.00455 | 0.36215 | 0.48183 | 0.00124 |

## Frozen scale gates and across-fold stability

The larger pilot provides **limited cost-aware STOP benefit, not robust learned
peer personalization**. With mean fusion, inner-selected objective
0.362145 beats always-best peer
0.369197, but the stronger cost-aware fixed-value
baseline is already 0.362780. Residual fusion's
selected objective 0.357874 is close to its
fixed-value baseline 0.357956. Mean-fusion gain
R2 stays around .0003–.0026 by fold, below .05, and only two origins/fold satisfy
the peer-variation gate. Larger data alone does not establish the requested
feasibility in this context. This does NOT prove personalized routing impossible
with richer authorized local observations, different honest experts or new data.

### Matched fitting-size effects (variable services, tariff .02)

| Fitting size | Fusion | Selected BCE | Selected objective | Best peer objective | Gain vs fixed | Utility R2 | Calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small | original | 0.36301 | 0.37099 | 0.37360 | 0.00261 | -0.00008 | 0.43017 |
| pilot | original | 0.34435 | 0.35787 | 0.36098 | 0.00310 | 0.00031 | 0.70544 |
| small | logistic | 0.37257 | 0.38642 | 0.39591 | 0.00949 | 0.02088 | 0.76266 |
| pilot | logistic | 0.35251 | 0.36822 | 0.36810 | -0.00012 | 0.00336 | 0.84396 |
| small | mean | 0.36283 | 0.37147 | 0.37809 | 0.00662 | 0.00026 | 0.46121 |
| pilot | mean | 0.35303 | 0.36215 | 0.36920 | 0.00705 | 0.00124 | 0.48183 |

| Size | Fusion | Pass | Inner choices | Fold objective gains | Fold R2 | Varying origins |
| --- | --- | --- | --- | --- | --- | --- |
| small | original | False | ['ridge_base', 'ridge_base', 'mlp_base'] | [0.004418, 0.002323, 0.0011] | [0.000258, 0.000506, -0.001015] | [2, 5, 3] |
| small | logistic | False | ['ridge_enriched', 'ridge_base', 'ridge_base'] | [0.009409, 0.012196, 0.006871] | [0.022509, 0.02201, 0.018109] | [6, 6, 6] |
| small | mean | False | ['ridge_enriched', 'ridge_enriched', 'ridge_base'] | [0.005023, 0.008063, 0.006785] | [-0.001357, 0.00062, 0.001504] | [3, 4, 3] |
| pilot | original | False | ['ridge_base', 'ridge_enriched', 'ridge_base'] | [0.002611, 0.004191, 0.002508] | [0.00025, 0.000459, 0.000215] | [5, 5, 6] |
| pilot | logistic | False | ['ridge_enriched', 'mlp_enriched', 'ridge_enriched'] | [0.000288, 0.001483, -0.002131] | [0.004334, 0.001044, 0.004702] | [6, 6, 6] |
| pilot | mean | False | ['ridge_base', 'ridge_enriched', 'ridge_enriched'] | [0.005942, 0.006561, 0.008653] | [0.000289, 0.000851, 0.002567] | [2, 2, 2] |

Gate: >=.005 objective gain EVERY fold at.02; mean utilityR2>=.05 EVERY fold;
accuracy drop<=.005; no origin mean BCE worsening>.01; >=5% minority peer calls
in>=3origins each fold; positive static-scenario gain each fold. No acceptance
criterion is relaxed. No automatic promotion or outer-selected winner. All gates,
including per-origin/accuracy failures, are retained in gates_{small,pilot}.json.
Service-induced variation is not necessarily clinical personalization, even when
the peer-count gate passes. Useful STOP must improve cost-quality, not STOP alone.

## Cost-quality tradeoffs and fitting-size sensitivity

| Size | Fusion | Tariff | Policy | BCE | Paid cost | Objective | Calls | Minority peer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | original | 0.02000 | stop | 0.37599 | 0.00000 | 0.37599 | 0.00000 | 0.00000 |
| small | original | 0.02000 | best_peer | 0.35022 | 0.02338 | 0.37360 | 0.99942 | 0.22728 |
| small | original | 0.02000 | inner_selected | 0.36301 | 0.00798 | 0.37099 | 0.43017 | 0.07890 |
| small | original | 0.05000 | stop | 0.37599 | 0.00000 | 0.37599 | 0.00000 | 0.00000 |
| small | original | 0.05000 | best_peer | 0.35022 | 0.05846 | 0.40868 | 0.99942 | 0.22728 |
| small | original | 0.05000 | inner_selected | 0.37569 | 0.00018 | 0.37587 | 0.00445 | 0.00000 |
| small | original | 0.10000 | stop | 0.37599 | 0.00000 | 0.37599 | 0.00000 | 0.00000 |
| small | original | 0.10000 | best_peer | 0.35022 | 0.11691 | 0.46713 | 0.99942 | 0.22728 |
| small | original | 0.10000 | inner_selected | 0.37599 | 0.00000 | 0.37599 | 0.00000 | 0.00000 |
| small | logistic | 0.02000 | stop | 0.41950 | 0.00000 | 0.41950 | 0.00000 | 0.00000 |
| small | logistic | 0.02000 | best_peer | 0.37309 | 0.02282 | 0.39591 | 0.99942 | 0.22227 |
| small | logistic | 0.02000 | inner_selected | 0.37257 | 0.01384 | 0.38642 | 0.76266 | 0.36069 |
| small | logistic | 0.05000 | stop | 0.41950 | 0.00000 | 0.41950 | 0.00000 | 0.00000 |
| small | logistic | 0.05000 | best_peer | 0.37309 | 0.05704 | 0.43014 | 0.99942 | 0.22227 |
| small | logistic | 0.05000 | inner_selected | 0.37966 | 0.02180 | 0.40146 | 0.56527 | 0.21314 |
| small | logistic | 0.10000 | stop | 0.41950 | 0.00000 | 0.41950 | 0.00000 | 0.00000 |
| small | logistic | 0.10000 | best_peer | 0.37309 | 0.11409 | 0.48718 | 0.99942 | 0.22227 |
| small | logistic | 0.10000 | inner_selected | 0.39640 | 0.01800 | 0.41441 | 0.26104 | 0.13589 |
| small | mean | 0.02000 | stop | 0.37463 | 0.00000 | 0.37463 | 0.00000 | 0.00000 |
| small | mean | 0.02000 | best_peer | 0.35510 | 0.02299 | 0.37809 | 0.99942 | 0.22382 |
| small | mean | 0.02000 | inner_selected | 0.36283 | 0.00864 | 0.37147 | 0.46121 | 0.07203 |
| small | mean | 0.05000 | stop | 0.37463 | 0.00000 | 0.37463 | 0.00000 | 0.00000 |
| small | mean | 0.05000 | best_peer | 0.35510 | 0.05747 | 0.41258 | 0.99942 | 0.22382 |
| small | mean | 0.05000 | inner_selected | 0.37438 | 0.00012 | 0.37449 | 0.00296 | 0.07047 |
| small | mean | 0.10000 | stop | 0.37463 | 0.00000 | 0.37463 | 0.00000 | 0.00000 |
| small | mean | 0.10000 | best_peer | 0.35510 | 0.11495 | 0.47005 | 0.99942 | 0.22382 |
| small | mean | 0.10000 | inner_selected | 0.37448 | 0.00010 | 0.37459 | 0.00153 | 0.03816 |
| pilot | original | 0.02000 | stop | 0.36777 | 0.00000 | 0.36777 | 0.00000 | 0.00000 |
| pilot | original | 0.02000 | best_peer | 0.33798 | 0.02300 | 0.36098 | 0.99942 | 0.22763 |
| pilot | original | 0.02000 | inner_selected | 0.34435 | 0.01352 | 0.35787 | 0.70544 | 0.10126 |
| pilot | original | 0.05000 | stop | 0.36777 | 0.00000 | 0.36777 | 0.00000 | 0.00000 |
| pilot | original | 0.05000 | best_peer | 0.33798 | 0.05750 | 0.39548 | 0.99942 | 0.22763 |
| pilot | original | 0.05000 | inner_selected | 0.36519 | 0.00225 | 0.36745 | 0.05720 | 0.00979 |
| pilot | original | 0.10000 | stop | 0.36777 | 0.00000 | 0.36777 | 0.00000 | 0.00000 |
| pilot | original | 0.10000 | best_peer | 0.33798 | 0.11500 | 0.45297 | 0.99942 | 0.22763 |
| pilot | original | 0.10000 | inner_selected | 0.36777 | 0.00000 | 0.36777 | 0.00000 | 0.00000 |
| pilot | logistic | 0.02000 | stop | 0.40461 | 0.00000 | 0.40461 | 0.00000 | 0.00000 |
| pilot | logistic | 0.02000 | best_peer | 0.34492 | 0.02318 | 0.36810 | 0.99942 | 0.22763 |
| pilot | logistic | 0.02000 | inner_selected | 0.35251 | 0.01572 | 0.36822 | 0.84396 | 0.21180 |
| pilot | logistic | 0.05000 | stop | 0.40461 | 0.00000 | 0.40461 | 0.00000 | 0.00000 |
| pilot | logistic | 0.05000 | best_peer | 0.34492 | 0.05796 | 0.40287 | 0.99942 | 0.22763 |
| pilot | logistic | 0.05000 | inner_selected | 0.37028 | 0.02081 | 0.39108 | 0.53305 | 0.23639 |
| pilot | logistic | 0.10000 | stop | 0.40461 | 0.00000 | 0.40461 | 0.00000 | 0.00000 |
| pilot | logistic | 0.10000 | best_peer | 0.34492 | 0.11591 | 0.46083 | 0.99942 | 0.22763 |
| pilot | logistic | 0.10000 | inner_selected | 0.39517 | 0.00793 | 0.40310 | 0.12584 | 0.08496 |
| pilot | mean | 0.02000 | stop | 0.36699 | 0.00000 | 0.36699 | 0.00000 | 0.00000 |
| pilot | mean | 0.02000 | best_peer | 0.34653 | 0.02266 | 0.36920 | 0.99942 | 0.21925 |
| pilot | mean | 0.02000 | inner_selected | 0.35303 | 0.00911 | 0.36215 | 0.48183 | 0.05770 |
| pilot | mean | 0.05000 | stop | 0.36699 | 0.00000 | 0.36699 | 0.00000 | 0.00000 |
| pilot | mean | 0.05000 | best_peer | 0.34653 | 0.05666 | 0.40319 | 0.99942 | 0.21925 |
| pilot | mean | 0.05000 | inner_selected | 0.36690 | 0.00004 | 0.36694 | 0.00084 | 0.05214 |
| pilot | mean | 0.10000 | stop | 0.36699 | 0.00000 | 0.36699 | 0.00000 | 0.00000 |
| pilot | mean | 0.10000 | best_peer | 0.34653 | 0.11332 | 0.45985 | 0.99942 | 0.21925 |
| pilot | mean | 0.10000 | inner_selected | 0.36695 | 0.00002 | 0.36698 | 0.00033 | 0.08333 |

## Diversity and patient-group consultation benefit

| Size | Mean error corr | Threshold disagreement | Retrospective private oracle BCE |
| --- | --- | --- | --- |
| small | 0.95160 | 0.01016 | 0.21827 |
| pilot | 0.95002 | 0.00468 | 0.21186 |

In the full pilot mean prediction correlation is .37470 and mean absolute probability disagreement .08010.
High .95002 error correlation partly reflects the shared binary outcome and imbalance; it is not conditional independence analysis.


diversity.json contains all pairwise correlations/disagreements, specialist
metrics, retrospective best-specialist counts and groupwise consultation gains
by Age bands1–8/9–13, Sex0/1, GenHlth1–3/4–5. These are post-decision analyses,
never router inputs or tuning data. Labels can make hindsight 'best' identities
vary even when predictable utility does not; do not mistake that for a learned
signal. Origin-specific selected actions/utility/harm/calibration appear in CSV.
Different domains can be predictive without providing novel conditional evidence.

### Retrospective group comparisons, full pilot/variable services/.02

| Fusion | Group | Selected objective | Fixed objective | Calls |
| --- | --- | --- | --- | --- |
| original | age_1_8 | 0.25694 | 0.26090 | 0.66726 |
| original | age_9_13 | 0.46526 | 0.46744 | 0.74629 |
| original | health_1_3 | 0.29955 | 0.30381 | 0.69090 |
| original | health_4_5 | 0.64092 | 0.63838 | 0.77597 |
| original | sex_0 | 0.33258 | 0.33639 | 0.69945 |
| original | sex_1 | 0.38958 | 0.39180 | 0.71298 |
| logistic | age_1_8 | 0.26639 | 0.26521 | 0.91466 |
| logistic | age_9_13 | 0.47656 | 0.47752 | 0.76957 |
| logistic | health_1_3 | 0.30783 | 0.30509 | 0.88346 |
| logistic | health_4_5 | 0.66125 | 0.67391 | 0.65217 |
| logistic | sex_0 | 0.34438 | 0.34173 | 0.85299 |
| logistic | sex_1 | 0.39811 | 0.40116 | 0.83269 |
| mean | age_1_8 | 0.26091 | 0.27068 | 0.42758 |
| mean | age_9_13 | 0.46982 | 0.47399 | 0.53950 |
| mean | health_1_3 | 0.30041 | 0.30949 | 0.46952 |
| mean | health_4_5 | 0.66175 | 0.65895 | 0.54166 |
| mean | sex_0 | 0.33723 | 0.34464 | 0.47954 |
| mean | sex_1 | 0.39337 | 0.39998 | 0.48468 |

Group means macro-average folds/origins; group sizes and peer gain vectors are in diversity.json. No group result changes a policy.

## Genuine Flower compatibility, verification and decision

| First pilot outer fold | Acc | AUROC | AP | BCE | Objective | Calls | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| federated | 0.86167 | 0.77417 | 0.34840 | 0.34382 | 0.36283 | 0.99933 | -0.02877 |
| central | 0.86167 | 0.76787 | 0.34220 | 0.34667 | 0.36083 | 0.75333 | -0.00077 |

Flower run **11629480563570813115**,3actual FedAvg rounds/**18 local updates**, six
logical ClientApps on one CPU actor. Routing-only drift **2.114021**,
server work **19.227s** including queries/consultations, excluding
startup. Central same43→32Tanh→1 architecture/same initialization,six pooled
epochs; FL3×2local epochs with optimizer resets. These are not identical global
optimizer schedules. Only compatible numeric router arrays/scalars are aggregated.
24 real evidence queries bind prepared probabilities/IDs/hashes; actual selected
aligned-ID queries load private models and masked fusion matches offline replay.
This does not establish learned routing improvement, generalized multi-agent
federation, Qwen integration or production readiness. Full-information preparation
and central diagnosis are explicitly trusted-local; later queries do not make
earlier preparation federated. Private experts/fusion remain frozen during FL.

Verification: `{"private_models": 1152, "private_models_refitted": 12, "fusion_states": 144, "assessment_metric_fields": 309744, "aggregations": 3, "client_updates_replayed": 18, "preserved_files": 4995, "changed": [], "frozen_test_evaluations": 0, "frozen_test_hash_unchanged": true, "success": true}`. QA logs/checks.json record pytest/Ruff/pip
commands and results; old optional Qwen/feedback historical-validation benchmarks
are skipped, while this new six-client integration is actually executed.
Fresh regression **97 passed, 2 skipped, 2 warnings in 8.41s**; Ruff lint/format78files and pip check passed.
The144 verified fusion states include48 stateless mean snapshots and96 trained
neural/logistic checkpoints. Verification replays every private prediction and
refits12 calibrated models; it does not claim all1152 training fits were rerun.

Scale gate passed: **False**; expanded50k:
**False**. Larger data may improve specialist fits and fusion,
but whether it improves useful consultation selection must be judged on these
paired gates/metrics, not accuracy alone. No retrospective threshold/cost/model
changes are made to force success. Interpret pilot and small results as fitting-
size sensitivity, not evidence that dataset size alone caused earlier failures.
The frozen test remains untouched; a confirmatory test requires a future frozen
protocol and authorization, not this development sprint.

Advanced GAT/bandit/ranker work is not justified merely by weight changes or
oracle headroom. Prioritize transferable pre-query signals, stable expert/fusion
calibration and independent replication of useful value prediction before that
complexity. Service-aware deterministic baselines remain essential. All results
are exploratory development evidence, not clinical validity or proven privacy.
No default model change, registry, Qwen training, push or publishing. STOP after3.5.
