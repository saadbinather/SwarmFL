# M5 Step 3.6 — Context-aware intelligent routing

## Frozen protocol (written before fitting)

Scope: exactly three CDC specialists, A metabolic, B lifestyle, C general health;
existing 15k pilot only. Source artifacts/m5-step035-three-agent-v1. Preserve
all earlier reports/models, original 21-field duplicate groups, contexts, private
calibrators and frozen test bytes. Never parse/evaluate frozen-test labels.
No registry, Step 4, Qwen training or default-pointer changes.

Fusion first: within every pilot inner context, refit residual/logistic from their
original cold seeds with identical nine valid STOP/one-peer masks, Adam .003,
batch4096, clip5, FP32, same random permutations. Save epochs15 and60, training
loss/gradient curves and held calibration/BCE by origin/action. Fifteen epochs
must reproduce previous fits numerically. Select among the four fusion choices
using inner-held uniform-action BCE only (fixed order residual15,logistic15,
residual60,logistic60 breaks ties). Freeze that choice before outer assessment.
No fitted recalibration is selected on outer labels. Calibration is audited,
not assumed repaired by convergence. Refit the chosen schedule from cold seeds
inside each of three upstream target complements and the final outer context.
This preserves whole-stack cross-fitted consultation utility targets. No target
label enters specialist, calibration or fusion fitting for its own target.

Local context authorization: each origin may use only its declared fields:
A BMI/HighBP/HighChol/CholCheck; B Smoker/PhysActivity/Fruits/Veggies/
HvyAlcoholConsump; C GenHlth/MentHlth/PhysHlth/DiffWalk. Use existing own-p/public
enriched37 plus a fixed zero-padded13-field union and13 origin-presence flags
(63 inputs). Standardization fits routing-training rows only. No additional
fields/embeddings, patient IDs, labels or unqueried peer predictions enter an
inference API. Raw local fields remain origin-local in Flower; only shared arrays,
scalar metrics, routing decisions and authorized numerical predictions may travel.
Trusted simulator file access is not filesystem isolation or privacy proof.

Candidates, all fixed before results: STOP, fitting-best peer with eligible
fallback, cost-aware origin/peer-mean fixed value, existing graph, enriched37
ridge and MLP; clinical63 ridge, MLP, and pairwise two-stage MLP. Ridge alpha100;
MLP32Tanh; routing Adam.003/weight_decay.01,20epochs,batch4096,clip5. Fit-only
scaling for every candidate. Baseline and clinical MLP schedules match. Pairwise
network has expected-gain and ranking heads; expected-gain MSE plus .1 soft
pairwise logistic loss (gain-difference temperature .1). Rank eligible peers by
rank score minus actual cost, then consult only if selected peer's expected gain
exceeds its actual cost; otherwise STOP. The ordinary value MLP uses the same
two-stage decision with expected-gain ordering. Every inference masks self and
ineligible peers. Costs .02/.05/.10 and both original static/variable schedules,
same old-slot mapping/prices. Exact-call random controls preserve each candidate's
call mask; they never inspect outcomes to choose peers.

Inner routing selection among clinical ridge, clinical MLP and clinical pairwise
uses variable-service macro-origin BCE+cost at .02; freeze names before outer
evaluation. Baseline winner is separately inner-selected from nonclinical ridge/
MLP. Report all candidates, costs, origins, calibration, decisions, utility R²,
paired objective differences and peer variation, not only selected favorable rows.
R² retains the original training-origin/peer-mean null and both potential peers.
Clinical feature access is a new user authorization; gates remain unchanged.

Separate observed-only contextual bandit: each context's training patients appear
once per origin in a deterministic shuffled sequence. Frozen fusion and existing
target-specific OOF risks supply ONLY the sampled action's outcome after sampling.
Uniform behavior over STOP/eligible peers with exact logged propensity; no
counterfactual gain/STOP reward or full-information utility target enters bandit
fitting. Train three-action linear expected-loss heads with selected-action IPS
squared loss (clipped at5,20epochs,Adam.003) on origin-local clinical features and
public metadata, then choose minimum predicted BCE+known cost with masks.
Separate full-information routing and bandit selections/claims. Held assessment
may evaluate actions after they are fixed; full-information oracles remain only
retrospective diagnostics. Log partial evidence, outcomes, propensities and update
provenance; replay every observed loss. This is retrospective offline logged
bandit learning, not verified clinical online outcomes or an unbiased causal claim.

Scale gate, fixed before results: original all-fold best-peer objective gain>=.005,
positive static gains, macro utility R²>=.05 every fold, >=3 origins with minority
peer>=.05 each fold, mean accuracy drop<=.005, no origin mean BCE degradation>.01,
all verification. Add meaningful improvement over cost-aware fixed value>=.005
EVERY fold at .02 and over the separately inner-selected metadata baseline>0
EVERY fold. All must pass before any 50k authorization decision. This step runs
the 15k study and reports the gate; expansion is a separate fresh preregistered
reproduction, not automatic tuning or promotion. If any gate fails, no50k.

Flower after numerical validation: exactly3 clients,one CPU actor,three FedAvg
rounds/two local epochs of compatible clinical expected-gain MLP, first pilot
outer context only. Run only if compatibility, finite local gradients, no hidden
features and numerical checks pass; this is not conditioned on positive scientific
results. Save/replay every local/global update. Origin-local routing queries must
return actions without raw features, then selected peer aligned-ID queries return
private predictions for masked fusion. No handwritten aggregation substitute.

Use existing venv, device auto/cuda/cpu with actual CUDA operation audit. No OS,
driver, WSL, Docker or dependency changes. Run pytest, Ruff and independent
numerical/provenance/preservation checks. Stop after3.6. Whether richer context,
GAT or bandits merit further tests must follow actual evidence, not gate changes.

## Execution results

### Interpretation

Inner-selected clinical routing has fold improvements over cost-aware fixed value of -0.000560, -0.000836, -0.000634 (positive is better). The unchanged/strengthened full scale gate **FAILS**. No production model is promoted. These are reused development patients, not independent clinical validation or evidence of privacy.

**Answer on this pilot: no robust useful personalization beyond cost-aware fixed routing is demonstrated.** Inner-selected clinical objective is .356859 versus fixed value .356182 and metadata .356186. Clinical utility R² by fold is -.006698/.001156/-.001535. Some individual candidate aggregates are favorable, but choosing those using outer assessment would be post-selection.

Whether context adds value must be judged against the separately inner-selected metadata reference, not just the weak always-call baseline. Fusion schedules and routing choices below were frozen using inner-held results, before outer assessment. No acceptance threshold was changed after evaluation.

### Fusion convergence and calibration audit

Each row is the inner context's uniform nine-action held metric, not outer model selection. Matched schedules use identical mask count, batches, seeds, optimizer and learning rate. ECE is 10 equal bins; calibration is estimated, not guaranteed. Training curve entries are batch means, while held metrics average complete origin/action predictions. A 60-epoch loss reduction does not establish complete convergence.

| Fold | Fusion | Train BCE | Gradient norm | Inner uniform BCE | Inner ECE | Brier | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | original15 | 0.347802 | 0.023937 | 0.376101 | 0.017551 | 0.114280 | False |
| 0 | logistic15 | 0.389065 | 0.097354 | 0.407422 | 0.057449 | 0.123612 | False |
| 0 | original60 | 0.345757 | 0.028620 | 0.373492 | 0.016942 | 0.113741 | True |
| 0 | logistic60 | 0.366034 | 0.024456 | 0.388223 | 0.050145 | 0.118688 | False |
| 1 | original15 | 0.360156 | 0.025790 | 0.339808 | 0.024189 | 0.102593 | False |
| 1 | logistic15 | 0.400798 | 0.103628 | 0.383609 | 0.075980 | 0.114072 | False |
| 1 | original60 | 0.357649 | 0.031659 | 0.339339 | 0.028436 | 0.102467 | True |
| 1 | logistic60 | 0.383058 | 0.022353 | 0.363350 | 0.063540 | 0.108536 | False |
| 2 | original15 | 0.351854 | 0.028128 | 0.368839 | 0.015228 | 0.112776 | False |
| 2 | logistic15 | 0.396068 | 0.099998 | 0.411592 | 0.071029 | 0.125645 | False |
| 2 | original60 | 0.349729 | 0.030849 | 0.367743 | 0.014309 | 0.112639 | True |
| 2 | logistic60 | 0.370872 | 0.021190 | 0.387179 | 0.054705 | 0.118253 | False |

Per-action/origin metrics, every epoch and 15-epoch historical tensor differences are in each context's fusion_audit.json/curve files. Final and target-complement fusions are fresh fits; private experts and calibrators are unchanged. This study does not claim that longer optimization isolates calibration from model capacity or OOF/final probability shift.

### Frozen nested choices

| Fold | Fusion | Metadata router | Clinical router |
| --- | --- | --- | --- |
| 0 | original60 | mlp_meta | mlp_clinical |
| 1 | original60 | ridge_meta | ridge_clinical |
| 2 | original60 | mlp_meta | pair_clinical |

### Outer pilot metrics, variable services / tariff .02

Macro-average origins within each fold, then folds. PR-AUC is average precision. Folds/origins share patients/fitting sets; these means are not independent replicates or formal significance tests. All exact per-fold/origin/scenario/cost rows remain in CSV.

| Policy | Accuracy | AUROC | PR-AUC | BCE | F1 | BCE+cost | Calls | ECE | Brier | Utility R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stop | 0.860889 | 0.710147 | 0.266605 | 0.366553 | 0.015691 | 0.366553 | 0.000000 | 0.012900 | 0.110480 | not defined |
| best_peer | 0.862667 | 0.785076 | 0.358611 | 0.338259 | 0.127376 | 0.359934 | 0.937334 | 0.011128 | 0.103326 | not defined |
| fixed_value | 0.861400 | 0.773510 | 0.339330 | 0.344368 | 0.091275 | 0.356182 | 0.572309 | 0.011880 | 0.104903 | -0.000000 |
| graph | 0.861489 | 0.769399 | 0.335790 | 0.346128 | 0.090746 | 0.356775 | 0.531511 | 0.012659 | 0.105323 | not defined |
| ridge_meta | 0.862067 | 0.771642 | 0.344427 | 0.345029 | 0.100002 | 0.355499 | 0.516889 | 0.011689 | 0.104765 | 0.000493 |
| mlp_meta | 0.861578 | 0.770978 | 0.339406 | 0.345506 | 0.092314 | 0.356095 | 0.527573 | 0.012453 | 0.105064 | 0.000070 |
| ridge_clinical | 0.861422 | 0.770808 | 0.340002 | 0.345584 | 0.093937 | 0.355640 | 0.497578 | 0.011747 | 0.104982 | 0.000907 |
| mlp_clinical | 0.861133 | 0.769537 | 0.334256 | 0.346050 | 0.079673 | 0.356804 | 0.528506 | 0.012563 | 0.105343 | -0.002633 |
| pair_clinical | 0.860911 | 0.768890 | 0.333386 | 0.346510 | 0.076943 | 0.356702 | 0.503573 | 0.012402 | 0.105423 | -0.002284 |
| bandit_observed | 0.862378 | 0.766902 | 0.339555 | 0.346459 | 0.086545 | 0.362221 | 0.641352 | 0.011740 | 0.105284 | not defined |
| inner_metadata | 0.861867 | 0.770440 | 0.341686 | 0.345745 | 0.097477 | 0.356186 | 0.519907 | 0.012163 | 0.105000 | 0.000241 |
| inner_clinical | 0.861133 | 0.769297 | 0.334957 | 0.346357 | 0.082640 | 0.356859 | 0.513217 | 0.012067 | 0.105335 | -0.002359 |

### Fold stability and acceptance

Gains are paired objective differences within each fold using the same frozen fusion and evidence budget. Utility R² uses the historical fitting-origin/peer-mean null, not an assessment-fitted null.

| Fold | Best-peer gain | Fixed-value gain | Metadata gain | Static gain | Utility R² | Varying origins | Accuracy delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.003129 | -0.000560 | -0.000141 | -0.002146 | -0.006698 | 1 | -0.001200 |
| 1 | 0.003221 | -0.000836 | -0.000133 | 0.001501 | 0.001156 | 1 | -0.001066 |
| 2 | 0.002875 | -0.000634 | -0.001743 | -0.008543 | -0.001535 | 1 | -0.002334 |

| Gate | Passed |
| --- | --- |
| best_peer_gain_.005_each_fold | False |
| fixed_value_gain_.005_each_fold | False |
| metadata_gain_positive_each_fold | False |
| static_gain_positive_each_fold | False |
| utility_r2_.05_each_fold | False |
| three_varying_origins_each_fold | False |
| accuracy_drop_.005 | True |
| no_origin_bce_degradation_.01 | False |

### Selected clinical decisions by origin, scenario and cost

| Scenario | Cost | Origin | Objective | BCE | AUROC | PR-AUC | Calls | R² | [STOP,A,B,C] | Minority peer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| static | 0.020000 | A | 0.348449 | 0.343947 | 0.773740 | 0.344954 | 0.225113 | -0.005636 | [11623, 0, 0, 3377] | 0.000000 |
| static | 0.020000 | B | 0.371108 | 0.351140 | 0.761865 | 0.321266 | 0.998400 | -0.000002 | [24, 9985, 0, 4991] | 0.000000 |
| static | 0.020000 | C | 0.348918 | 0.335034 | 0.789001 | 0.373299 | 0.694165 | 0.000166 | [4587, 10413, 0, 0] | 0.000000 |
| static | 0.050000 | A | 0.349316 | 0.349286 | 0.764572 | 0.323162 | 0.000600 | -0.005636 | [14991, 0, 0, 9] | 0.000000 |
| static | 0.050000 | B | 0.393935 | 0.375992 | 0.680623 | 0.254878 | 0.358853 | -0.000002 | [9617, 5383, 0, 0] | 0.000000 |
| static | 0.050000 | C | 0.356239 | 0.355652 | 0.744905 | 0.290347 | 0.011732 | 0.000166 | [14824, 176, 0, 0] | 0.000000 |
| static | 0.100000 | A | 0.349310 | 0.349310 | 0.764527 | 0.322909 | 0.000000 | -0.005636 | [15000, 0, 0, 0] | 0.000000 |
| static | 0.100000 | B | 0.393642 | 0.393642 | 0.623539 | 0.194308 | 0.000000 | -0.000002 | [15000, 0, 0, 0] | 0.000000 |
| static | 0.100000 | C | 0.356707 | 0.356707 | 0.742374 | 0.282597 | 0.000000 | 0.000166 | [15000, 0, 0, 0] | 0.000000 |
| variable | 0.020000 | A | 0.349808 | 0.348611 | 0.765643 | 0.326478 | 0.049457 | -0.006792 | [14258, 0, 4, 738] | 0.001807 |
| variable | 0.020000 | B | 0.370318 | 0.350745 | 0.761823 | 0.321930 | 0.920804 | -0.000103 | [1188, 11914, 0, 1898] | 0.136713 |
| variable | 0.020000 | C | 0.350450 | 0.339715 | 0.780426 | 0.356462 | 0.569391 | -0.000182 | [6459, 8483, 58, 0] | 0.005671 |
| variable | 0.050000 | A | 0.349322 | 0.349314 | 0.764513 | 0.322894 | 0.000200 | -0.006792 | [14997, 0, 3, 0] | 0.000000 |
| variable | 0.050000 | B | 0.391005 | 0.377002 | 0.686893 | 0.254287 | 0.343075 | -0.000103 | [9854, 5146, 0, 0] | 0.000000 |
| variable | 0.050000 | C | 0.356367 | 0.355406 | 0.745354 | 0.291750 | 0.023864 | -0.000182 | [14642, 358, 0, 0] | 0.000000 |
| variable | 0.100000 | A | 0.349310 | 0.349310 | 0.764527 | 0.322909 | 0.000000 | -0.006792 | [15000, 0, 0, 0] | 0.000000 |
| variable | 0.100000 | B | 0.393642 | 0.393642 | 0.623539 | 0.194308 | 0.000000 | -0.000103 | [15000, 0, 0, 0] | 0.000000 |
| variable | 0.100000 | C | 0.356707 | 0.356707 | 0.742374 | 0.282597 | 0.000000 | -0.000182 | [15000, 0, 0, 0] | 0.000000 |

### Separate observed-only bandit

Uniform actions were written to bandit_selection.json before outcome lookup. Each event contains only own/selected probabilities, sampled risk, true training label and exact propensity. The four global action heads include a masked self head; three actions are valid per origin when both peers are available. Fitting uses selected-loss IPS regression with cap5; valid inverse propensities are at most3, so clipping is inactive. No unchosen action BCE or STOP counterfactual enters its loss. It uses retrospective cross-fitted diagnoses, not prospective clinical outcomes, and is not a full-information router. Bandit action loss is evaluated only after held actions are fixed. Full-information baselines have more supervision and are not equal-feedback comparisons.

| Fold | Observed records | Minimum propensity | Mean observed loss | Outer objective | Fixed-value objective |
| --- | --- | --- | --- | --- | --- |
| 0 | 30000 | 0.333333 | 0.355292 | 0.362758 | 0.358287 |
| 1 | 29997 | 0.333333 | 0.355079 | 0.364250 | 0.356564 |
| 2 | 30003 | 0.333333 | 0.357063 | 0.359654 | 0.353695 |

### Flower, numerical verification, runtime and QA

Real Flower run **18290472637018263396**, exactly3 clients, one CPU actor, three FedAvg rounds/nine local updates. Shared clinical MLP63→32Tanh→1 only; drift 1.638164; server work 17.689s excludes startup. Origin-local route queries expose actions/own probabilities, not raw local features. Selected aligned-ID consultations use actual frozen private models; masked fusion replays within2e-6. Central preparation remains trusted-local, full-information diagnostic training; this is not a private service.

```json
{
  "verified": true,
  "fusion_outcome_replays": 60,
  "metric_fields": 27864,
  "bandit_observations": 162084,
  "aggregations": 3,
  "local_update_replays": 9,
  "preserved_files": 8058,
  "changed": [],
  "test_hash_unchanged": true,
  "test_evaluations": 0
}
```

Neural device cuda, NVIDIA RTX A4000, actual CUDA audit checksum 3680.0. Pipeline time 138.768s; peak torch allocated 36176384 bytes. This excludes CUDA contexts/reserved/board memory and historical private training. No GPU saturation or speedup claim.

Actual regression: **113 passed,2 skipped,2 upstream Typer warnings in19.23s**. Ruff lint/format83files and pipcheck passed. Optional historical Qwen/feedback runtime benchmarks are disabled to preserve their old evaluation boundaries; this step's new three-client Flower integration actually ran. Pip freeze is environment evidence, not a portable dependency lock.

Fresh QA commands/return codes:

```json
{
  "pytest": {
    "command": [
      "<venv-python>",
      "-m",
      "pytest",
      "-q"
    ],
    "returncode": 0,
    "seconds": 20.79015040000013,
    "log": "artifacts/m5-step036-v1/qa/pytest.log"
  },
  "ruff": {
    "command": [
      "<venv-python>",
      "-m",
      "ruff",
      "check",
      "--no-cache",
      "src",
      "tests",
      "scripts"
    ],
    "returncode": 0,
    "seconds": 0.07815579999987676,
    "log": "artifacts/m5-step036-v1/qa/ruff.log"
  },
  "format": {
    "command": [
      "<venv-python>",
      "-m",
      "ruff",
      "format",
      "--no-cache",
      "--check",
      "src",
      "tests",
      "scripts"
    ],
    "returncode": 0,
    "seconds": 0.06754150000051595,
    "log": "artifacts/m5-step036-v1/qa/format.log"
  },
  "dependencies": {
    "command": [
      "<venv-python>",
      "-m",
      "pip",
      "check"
    ],
    "returncode": 0,
    "seconds": 2.8500739999999496,
    "log": "artifacts/m5-step036-v1/qa/dependencies.log"
  }
}
```

### Limitations and research direction


No fitting, Flower, test or numerical verification failure was suppressed. Scientific acceptance gates fail as reported. Initial Ruff caught unused imports, a dict-construction style issue and a stale variable name in the enhanced verifier; these were corrected before its successful execution. No model setting, gate or numerical tolerance was relaxed. Windows Ray experimental/deprecated-launcher warnings remain; no OS/driver intervention was used.

The registered evidence does **not** justify50k expansion or a GAT. Richer context may help some fold/metric/condition, but must not be called robust personalization when any stability/fixed-value/utility gate fails. There are only two peers to rank and no new inter-agent topology observation; graph complexity alone is not evidence of additional signal.

A bandit continuation needs repeated behavior-policy seeds, genuine delayed verified outcomes, action coverage/ESS and comparison against equal-feedback controls. One uniform logged simulation does not demonstrate online learning or that a more complex bandit will beat supervised/fixed routing. Further context use must retain origin authorization and feature-domain completeness.

Other limitations: reused development cohort; class imbalance; cross-sectional survey target and unmodeled survey weights; dependent origin views; noisy single-outcome BCE gains; teacher/final shift; fixed training schedules/capacities; pairwise rank/gate errors; simulated public availability/prices; no external validation, secure isolation, authentication or privacy proof. Raw features are origin-local in Flower but accessible to the trusted simulator. Predictions, identifiers and updates can disclose information. New context is explicitly authorized for this experiment, not a general consent/serving mechanism.

### Commands and exact changed files

```powershell
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_context --device cuda
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_context --flower
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_context --verify
.venv/Scripts/python.exe -m fed_med_agents.experiments.cdc_context --report
```

New files: experiments/cdc_context.py under src/fed_med_agents; tests/test_cdc_context.py; this protocol/report; docs/devlog/2026-10-14-m5-step036.md. Updated README/status/architecture/limitations/next-steps only. Previous implementation/report files stay unchanged. Detailed CSV/actions/curves/checkpoints/partial bandit records/Flower/provenance/QA live under ignored artifacts/m5-step036-v1. Frozen-test evaluations0. No GitHub push, Qwen training, registry, default promotion or Step4. **Stop after3.6.**
