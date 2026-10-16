# M5 Step 2 preregistered protocol — 2026-10-14

This protocol is fixed before new results. No historical validation/test outcomes
will be evaluated, tuned on, or used to initialize a candidate policy. Outputs:
artifacts/m5-step02-v1. Historical artifacts/checkpoints/reports remain immutable.

## Data and evaluation

Use only the original 364 training patients. Recover raw age by inverting the
stored affine transform and rounding to original integer age; refit age scaling
within every specialist fit. Other source fields are fixed binary encodings.
No old specialist probability or trained shared weight initializes these studies.

Three stratified outer folds, shuffle seed 20261014, give disjoint assessment
patients. Within each outer fitting portion, one stratified 80/20 split (seed
9100+fold) supplies fitting and selection patients. For BOTH inner fitting and
outer refitting, independently generate three-fold specialist OOF predictions
(seed 4200+fold), with an additional 80/20 fit/calibration holdout in each fold.
Fit final experts on the fitting patients; fit their temperature from fitting-only
raw OOF logits. Assessment/selection predictions come from those final experts.
Specialists train 30 epochs using existing M2 helpers. All provenance IDs persist.
These are nested partitions of one cohort, not independent hospitals/cohorts.

Select candidate names only by inner-selection macro-origin BCE + .02*actual
calls/patient, tie-breaking in declared candidate order. Refit all fixed
candidates on outer fitting patients, then report all outer results without
selecting on them. Report the inner-selected pipeline as a separate row. Outer
assessment labels never enter a loss, fitting/calibration routine or selection.
Five rounds fixed; no best-round selection. Threshold remains .5.

## Phase B: optimization, in declared candidate order

1. `avg10`: original five-round, ten-local-epoch Adam .01 reset baseline.
2. `avg1`, `avg2`, `avg5`: identical five rounds with 1/2/5 local epochs.
3. `persistent2`: two epochs, origin-local Adam moments persist across global
   broadcasts; reset only at a new variant/context. Incoming parameters are loaded
   before each local optimization. It is a distinct optimizer protocol, not an
   assertion that retained moments are globally consistent.
4. `prox2`: installed Flower FedProx, mu=.01, two epochs, reset Adam.
5. Central pooled and interleaved references at ten epochs, same initialization,
   batches, resets and total origin-loss computations. Pooled makes one mean-loss
   step per batch; interleaved makes three successive origin-specific steps.

All shared learning: batch64, gradient clip5, FP32. Save each client/global state,
steps, gradient norms, parameter/update cosines, device/time/memory and messages'
record keys. FedAvg/FedProx arithmetic is independently replayed, three origins
required. No private weights/features in update messages. Specialists remain frozen
through all shared training. Smaller local epochs have fewer gradient computations;
report that confound rather than calling it a compute-matched superiority test.

## Phase C: routing

Keep avg10 soft expected BCE + .02*call as baseline. `cost10` and `cost30` change
only the training consultation coefficient to .1 and .3. Report each candidate
under evaluation tariffs .02/.1/.3; primary selection tariff remains .02.

`value` first pretrains the ORIGINAL fusion uniformly over all valid actions for
three genuine Flower rounds of five epochs, with router frozen. Freeze fusion,
then fit compatible router scores for five rounds/two epochs to regress fitting
OOF realized stop-loss minus peer-loss (targets clipped [-3,3]). Its deployed
STOP score is zero, peer scores are estimated benefit minus .02. Targets may use
training labels and exhaustive fitting OOF evidence; deployment inputs are only
origin identity/probability/confidence and public availability/domains. The teacher
fusion's targets are in-sample at the meta level and can overfit; outer nested
assessment is the protection, not a claim these targets are independent outcomes.

A deterministic utility-bin control uses that same pretrained fusion. Four fixed
confidence bins [0,.25,.5,.75,1], minimum eight fitting records/bin, otherwise
origin-wide fallback; estimate each eligible peer's mean gain on fitting data,
consult the best iff estimated gain > .02. No assessment label or peer score
enters prospective selection. Include STOP, seeded random and cyclic controls,
maximum one peer, and same-action-mask fusion comparisons.

Report STOP/calls, action counts, entropy, expected-soft versus hard BCE, BCE,
accuracy/AUROC/F1/Brier/ECE10, cost and retrospective selected-call harm versus
same-fusion STOP. No label-aware oracle is a deployable policy.

## Phase D: fusion

Baseline original 12→24→1 residual fusion; arithmetic mean on IDENTICAL selected
evidence; `logistic` (masked logits and presence, six inputs, linear output,
no origin identity); `small` (original residual fusion with width8); `value`'s
separately pretrained original fusion. Logistic/small train jointly with the
original soft router through five ten-epoch Flower rounds. Compare both native
actions and avg10's fixed action masks to separate policy/evidence changes from
fusion changes. These fixed three-client experimental classes are not a flexible
fusion framework or production replacement. Identity/capacity changes in logistic
are confounded and will not be presented as isolated causal effects.

## Phase E: verified feedback

Keep original M4 pipeline untouched. Audit its observed-only replay at BOTH actual
local optimizer steps, with clipping/ESS/gradient/reward-age diagnostics. Then use
the existing M4 Apps on outer fold 0's fitting-only experts and its outer assessment
patients (all drawn from original training partition) as a new guarded holdout.
Two fresh matched 120-patient streams: baseline two local replay steps and optional
eight steps, all other M4 settings fixed (seed42, max budget1, epsilon.2, delay3,
interval60, lr.0003, window128, same initialization/patient order/random draws).
No unconsulted specialist outcome enters a replay ledger. Report realized call
rates separately; matching maximum budgets does not imply equal realized calls.
The guard uses this training-only holdout; thus its scores are development scores,
not independent final assessment for the feedback study. No historical validation
or test metric is generated. Do not interpret two small streams as deployed learning.

## Promotion and limitations

No automatic default replacement. A candidate could be recommended for review only
if the nested inner-selected pipeline improves the primary objective by >=.01 on
EVERY outer fold versus avg10, aggregate accuracy drops <=.01, and no origin's
mean BCE degrades >.01, with all routing/budget/isolation/aggregation/regression
checks passing. Report all candidates even when they fail. These gates are
conservative development criteria, not significance or clinical safety claims.
Keep baseline if evidence is mixed; no tuning after outer results. GPU optional;
at most one GPU Ray actor; CPU supported. Stop after Step 2, no registry/SLM changes.
