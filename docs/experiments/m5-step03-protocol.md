# M5 Step 3 preregistered routing protocol — 2026-10-14

Frozen before implementation/results. New artifacts: artifacts/m5-step03-v1.
Only the original 364 training patients; no old validation/test metric or label.
Three stratified outer folds seed 53003; within each outer fit, stratified 75/25
inner fit/selection seed 53100+fold. Report all fixed candidates; select names only
on inner macro-origin BCE+.02*calls, separately for original/logistic fusion.
Costs .02/.05/.10, threshold .5, maximum one eligible peer. No outer selection.

Each inner/outer fitting context T gets fresh specialist OOF and final fits using
existing 30-epoch CPU helpers. To generate utility labels split T into three
stratified target folds H. Independently prepare specialists ONLY on T\H, with
their own three-fold OOF and inner calibration. Fit each fusion on T\H OOF,
uniformly over STOP and the two eligible peers; assess on H using final T\H
experts. Thus neither fusion fitting features nor any upstream fit/calibration
uses H labels. Save provenance, held action risks and untruncated BCE gains.
Final frozen fusion uses full T specialist OOF, never assessment labels. Fusion:
original residual or existing experimental logistic; 40 full-batch Adam .01 steps,
seed 53200+fold, uniform action loss, clip5, no checkpoint selection.

All inference APIs accept only own p and declared prequery metadata. Candidate
inputs: base origin/candidate one-hots, own p/confidence, public domain width,
availability and remaining budget; uncertainty adds margin/entropy/p(1-p);
reliability adds own/candidate OOF BCE/Brier/ECE; history adds training-only mean
incremental utility/origin-peer. Enriched combines uncertainty/reliability/history.
History/reliability for target H use T\H only (history is a resubstitution
summary on that teacher's fitting OOF, explicitly noisy); inference uses T only.
No raw feature summaries added: no new authorization protocol. Constant capability
flags and deterministic uncertainty transforms add no independent information.
Costs are subtracted at decision time from estimated gross gain, not targets.

Models: original contextual graph trained frozen-fusion expected action loss on
cross-fitted action losses (50 Adam .01 pooled steps); per-origin fixed best peer
from mean cross-fitted gross gains (also cost-aware positive-only variant);
confidence<.6 then best fixed peer; ridge regression alpha10 with fit-only standard
scaling; MLP 16 Tanh 1, Adam .01 weight_decay .01, 80 pooled steps, raw features;
tree depth3/min_leaf30 seed42 (existing sklearn). Value models fitted on all six
origin/candidate examples per patient, shared parameters. Ridge/MLP/tree compare
base, +uncertainty, +reliability, +history, enriched. No hyperparameter search.
Controls STOP, always best peer, seeded random maximum, and random with the exact
learned call mask for each value/graph policy. Tie break first candidate; STOP if
best net gain <=0. All policies have the same permitted information envelope;
graph/threshold/fixed controls intentionally use restricted functions of it.

Report all costs, per-fold/origin accuracy/AUROC/BCE/F1/ECE/Brier, call/STOP/actions,
conditional harm vs same-fusion STOP, utility MSE/correlation/R2 vs training-mean
predictor, oracle regret, action variation and cost/loss Pareto membership. Oracle
is retrospective only. Compare target gain variability with attainable prediction.

Optional genuine Flower comparison is executed for the inner-selected neural
value architecture (linear or MLP; choose by average inner score, tie linear),
using enriched context and each outer frozen-fusion condition. Five rounds,
16 full-batch Adam .01 local steps reset per round, clip5, at most one CPU worker.
Central pooled SAME architecture, initialization, 80 steps/80 origin-mean losses;
MLP matches central model above, linear reference additionally fitted in torch.
Only compatible router parameters aggregated; save all local/global arrays,
gradients/cosines/drift, predictions and payload keys. Query training evidence
through Flower; full-information offline caches disclosed, not selected-only
serving traffic. Do not aggregate private experts or frozen fusion.

Numerical gates (development, no automatic promotion): inner-selected value
pipeline must beat always-best peer primary by >=.01 in EVERY outer fold, aggregate
accuracy drop <=.01, each origin mean BCE degradation <=.01, gain R2 positive
against training-origin/peer mean in every fold, and patient-dependent decisions
in >=2 origins/fold. Useful STOP additionally needs lower primary than always
consult at .02, not just more STOP. Require all isolation/aggregation tests.
Report failures without relaxing gates. No claim of clinical/privacy validity.
Stop after Step 3; production pointers unchanged regardless of outcome.
