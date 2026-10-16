# M5 Step 2 — federated optimization repair

Executed 2026-10-14, Windows/Python 3.11/Flower 1.39/PyTorch CUDA.
Only the original **364 training patients** enter any new fitting, calibration,
selection or assessment. Three stratified outer assessment folds; each outer
fitting set has an independent 80/20 inner fit/selection split. Experts are
refit/calibrated within every context, and fusion/router fitting uses target-label
excluding specialist OOF probabilities. Source age was reconstructed by inverting
its old affine transform/rounding and then refit inside each new fitting subset.
The historical 78 validation and 78 test patients are not newly evaluated.

Original Step 1 historical validation reference: accuracy/BCE .93162/.19573;
identical-evidence mean .95014/.26517; random+.same fusion .93305/.19798.
Those used different patients/fits and are **not** paired with the new table.
The paired original-protocol baseline below is `avg10` (five rounds, ten epochs,
Adam .01 reset, cost .02). Metrics macro-average three dependent origins and three
outer folds; AUROC/F1 are means, not pooled patient metrics or significance tests.
Calibration is ten equal-width ECE and Brier, threshold fixed .5. Folds partition
one cohort and fitting sets overlap. No independent clinical validity is claimed.

[Preregistered protocol](m5-step02-protocol.md) fixes candidates, data partitions,
cost tariffs and promotion gates before results. All five terminal rounds are
reported without outer-based round/model selection. Costs .02/.1/.3 are BCE units
per actual selected call. Primary objective is BCE+.02*calls. Availability is
matched, all three domains present, maximum one peer; this does not identify the
effect of variable capability/availability. No default/serving pointer is changed.

## Original protocol versus every candidate/control

| New nested variant | Accuracy | AUROC | BCE | F1 | ECE10 | Brier | Calls | BCE+.02 calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg10 | 0.88367 | 0.95430 | 0.27260 | 0.90335 | 0.06769 | 0.08281 | 1.00000 | 0.29260 |
| avg1 | 0.87360 | 0.94333 | 0.30078 | 0.89687 | 0.08305 | 0.09325 | 1.00000 | 0.32078 |
| avg2 | 0.88276 | 0.95333 | 0.27494 | 0.90375 | 0.06933 | 0.08395 | 1.00000 | 0.29494 |
| avg5 | 0.88275 | 0.95516 | 0.26911 | 0.90329 | 0.07090 | 0.08171 | 1.00000 | 0.28911 |
| persistent2 | 0.88184 | 0.95452 | 0.27381 | 0.90271 | 0.06884 | 0.08355 | 1.00000 | 0.29381 |
| prox2 | 0.88184 | 0.95478 | 0.27249 | 0.90293 | 0.06991 | 0.08328 | 1.00000 | 0.29249 |
| cost10 | 0.86998 | 0.94395 | 0.30296 | 0.89056 | 0.08105 | 0.09533 | 0.66667 | 0.31629 |
| cost30 | 0.81135 | 0.88390 | 0.39951 | 0.84488 | 0.08393 | 0.12911 | 0.00000 | 0.39951 |
| logistic | 0.90743 | 0.96214 | 0.23599 | 0.92402 | 0.07283 | 0.07168 | 1.00000 | 0.25599 |
| small | 0.89005 | 0.95028 | 0.27392 | 0.90936 | 0.06434 | 0.08223 | 1.00000 | 0.29392 |
| value | 0.88549 | 0.94689 | 0.28629 | 0.90659 | 0.06469 | 0.08673 | 0.93388 | 0.30497 |
| utility_bins | 0.88093 | 0.94738 | 0.28581 | 0.90229 | 0.07234 | 0.08682 | 0.81314 | 0.30207 |
| central_pooled | 0.88096 | 0.95218 | 0.27438 | 0.90055 | 0.08074 | 0.08534 | 0.77778 | 0.28993 |
| central_interleaved | 0.88646 | 0.95556 | 0.26095 | 0.90411 | 0.07790 | 0.08062 | 0.83196 | 0.27759 |
| stop | 0.79851 | 0.87214 | 0.42588 | 0.82267 | 0.08676 | 0.13422 | 0.00000 | 0.42588 |
| random | 0.87728 | 0.94140 | 0.31363 | 0.89747 | 0.08055 | 0.09646 | 1.00000 | 0.33363 |
| cyclic | 0.87634 | 0.94771 | 0.29386 | 0.89742 | 0.07123 | 0.08845 | 1.00000 | 0.31386 |


Controls STOP/random/cyclic use the same avg10 fusion. Utility bins use the
uniform-pretrained original fusion. All inference routing uses only own probability,
origin identity/confidence and public domains/availability; no label or unqueried
peer probability enters selection. Complete OOF evidence is deliberately used in
full-information fitting; assessment replays masked evidence from Flower-acquired
probabilities. These logical inference call budgets are not selected-only deployed
traffic measurements. No raw training features/private specialist weights are in
shared update messages. This is not a proven privacy guarantee.
## Optimization interpretation

Reduced local work is not an unconditional repair: one epoch underfits relative
to avg10, two epochs are close, and five epochs slightly reduce mean BCE. FedProx
mu=.01 slightly improves over plain two epochs here; persistent Adam also changes
results only modestly. These are descriptive paired outcomes, not reliable
superiority. Compare per-fold/per-origin JSON rather than selecting the best mean.

FedProx is installed Flower 1.39 FedProx: configure_train injects proximal-mu=.01;
clients add mu/2 times squared distance from received weights. Its server arithmetic
remains weighted FedAvg and is independently checked. Persistent Adam keeps only
each origin's moments across parameter broadcasts, with a fresh optimizer at every
context/variant. Stale local moments are not globally averaged. Unit tests verify
zero initial proximal gradient, later divergence and actual persistent step counts.

Central pooled: 60/round in original Step 1 size, but **40/round** at these fitting
sizes (ceil(n/64)=4, ten epochs). It averages three origin losses per step: 120
origin-loss evaluations. Interleaved uses the same 120 origin-loss evaluations but
120 optimizer steps. Each ten-epoch client takes 40 steps; three clients total120,
then one aggregate. Both central schedules reset each round. Pooled BCE is not
uniformly better than FL on these new folds; the Step 1 pattern is not universal.
Interleaved central has more global optimizer steps and remains confounded by
that difference. Conflicting local objectives are observed, not proof that FedAvg
arithmetic alone is wrong.

Local statistics below average all outer-fold client rounds. Steps are total
origin-loss computations per outer fold across three clients (not FLOP counts).
Cosines are A/B, A/C, B/C averaged across rounds/folds; they are not clinical
correlations. Proximal gradient is zero at the global broadcast for the origin
gradient diagnostic; local updates include it.
| Variant | FL rounds | Origin-loss computations/fold | Mean router local L2 | Mean fusion local L2 | Fusion update cosine | Origin fusion gradient cosine | Mean FL seconds/fold |
| --- | --- | --- | --- | --- | --- | --- | --- |
| avg10 | 5 | 600 | 1.77587 | 1.57436 | -0.16338 | -0.27682 | 6.54523 |
| avg1 | 5 | 60 | 0.48210 | 0.47459 | 0.04833 | -0.11826 | 1.79600 |
| avg2 | 5 | 120 | 0.87085 | 0.81811 | -0.07916 | -0.26036 | 2.43261 |
| avg5 | 5 | 300 | 1.48722 | 1.35279 | -0.17153 | -0.34329 | 3.94210 |
| persistent2 | 5 | 120 | 0.74192 | 0.75063 | -0.05699 | -0.27525 | 2.63099 |
| prox2 | 5 | 120 | 0.62435 | 0.70707 | -0.10458 | -0.25728 | 2.34602 |
## Nested selection and promotion decision

Candidate names were selected only on the INNER selection objective, with declared
order as tie-break. Outer metrics did not change the selected name.

| Outer fold | Inner-selected pipeline | Outer objective | avg10 objective | Improvement |
| --- | --- | --- | --- | --- |
| 0 | logistic | 0.21379 | 0.24911 | 0.03532 |
| 1 | avg10 | 0.31123 | 0.31123 | 0.00000 |
| 2 | logistic | 0.27148 | 0.31744 | 0.04597 |


Selected-pipeline aggregate: {'accuracy': 0.9129371208357795, 'auroc': 0.9608583977131617, 'binary_cross_entropy': 0.24549914509275705, 'f1': 0.9289222428721678, 'ece': 0.06535588819814825, 'brier': 0.07150728089941873, 'calls': 1.0, 'primary': 0.2654991450927571}.
Predeclared promotion gates: `{'each_fold_primary_improvement_at_least_.01': False, 'accuracy_gate': True, 'per_origin_bce_gate': True, 'automatic_promotion': False, 'original_retained': True}`. **Original retained; no model promoted.**
In particular the unchanged avg10 selection on one fold provides zero improvement,
so the all-fold >=.01 gate fails even though other folds improve. The fixed logistic
candidate is promising, but choosing it after inspecting all outer scores would
override the nested selection protocol. No such override occurred. No significance
or medical-safety inference is attached to this gate.
## Feedback: actual two-step versus eight-step replay

The ORIGINAL M4 Apps/pipeline are unchanged. Both new 120-patient streams use outer
fold0 fitting-only experts and its original-training-only holdout, same cold
initialization, patient order, random draws, delayed true labels, epsilon.2, max
budget1, lr.0003, window128, triggers60/120/123. The only changed training setting
is two versus eight local replay steps. Source training IDs are remapped local
positions with original IDs retained; no original held-out labels are evaluated.
The guard uses a reused outer-training-study assessment holdout: these are guarded
development scores, not independent feedback generalization estimates.

Frozen/adaptive policies share maximum budgets; realized counts are reported, not
assumed equal. Frozen predictions/actions across both trials are independently
checked identical. Every label is a verified training source outcome revealed
after prediction, never an SLM acknowledgement or guessed peer outcome. Only
actually consulted evidence reaches local replay. The offline reference was fit
on all fitting outcomes and is excluded from claims of adaptive improvement.
| Local steps | Policy | Accuracy | Macro-origin AUROC | BCE | F1 | Calls | Harm/calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | frozen | 0.82500 | 0.89235 | 0.37726 | 0.85106 | 0.71667 | 0.48837 |
| 2 | adaptive | 0.82500 | 0.89235 | 0.37638 | 0.85106 | 0.71667 | 0.47674 |
| 8 | frozen | 0.82500 | 0.89235 | 0.37726 | 0.85106 | 0.71667 | 0.48837 |
| 8 | adaptive | 0.82500 | 0.89235 | 0.37576 | 0.85106 | 0.71667 | 0.48837 |

### Actual feedback learning curves and versions

| Local steps | Label cutoff | Candidate | Approved pointer | Guard BCE | Update L2 | Adaptive stream BCE | Frozen stream BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 60 | v001 | v001 | 0.37927 | 0.00856 | 0.39849 | 0.39849 |
| 2 | 120 | v002 | v002 | 0.37922 | 0.00929 | 0.38483 | 0.38573 |
| 2 | 123 | v003 | v003 | 0.37903 | 0.00915 | 0.37638 | 0.37726 |
| 8 | 60 | v001 | v001 | 0.37905 | 0.03427 | 0.39849 | 0.39849 |
| 8 | 120 | v002 | v002 | 0.37849 | 0.03627 | 0.38420 | 0.38573 |
| 8 | 123 | v003 | v003 | 0.37874 | 0.03585 | 0.37576 | 0.37726 |

These are six additional genuine Flower rounds and 18 client updates, separate from the 348 main aggregations. Every version/update is retained. Labels from the historical validation/test partitions never enter these curves.

Actual local-step audit (including steps AFTER the first optimizer update):

| Steps | Flower run | Approved | Audited local steps | Min ESS | Min propensity | Max clipped fraction | Max preclip norm | Runtime seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 12734747006543126497 | v003 | 18 | 18.97583 | 0.28566 | 0.00000 | 0.24284 | 148.97791 |
| 8 | 5037714295706925219 | v003 | 72 | 18.95417 | 0.28226 | 0.00000 | 0.24284 | 143.22704 |

Historical M4 replay: 42 actual local steps reconstructed; maximum clipped fraction 0.00000. Local reconstructed checkpoints agree within 2e-5 (CPU/CUDA tolerance). Full per-step ESS, gradients, loss/reward age, action frequencies and model-version ages persist in feedback_audit.json.

The maximum entropy for three valid actions is ln(3)≈1.09861. Entropy stays high at actual local steps; there is no entropy penalty in the M4 objective. Logged reward cost is recorded BCE plus actual consultation cost (the router surrogate also subtracts its fixed .5 baseline).

| Replay | Min entropy | Max entropy | Min mean logged cost | Max mean logged cost |
| --- | --- | --- | --- | --- |
| historical_local_steps | 1.08529 | 1.09691 | 0.31188 | 0.68840 |
| steps2 | 1.08389 | 1.09672 | 0.33296 | 0.43865 |
| steps8 | 1.08247 | 1.09687 | 0.33293 | 0.43865 |

More replay steps increase movement/work; any small BCE changes must not be
called meaningful adaptation or deployed continuous learning. Sparse observed
actions, high entropy, stale reward versions and small learning rate remain
structural limitations. Matched prefixes do not reveal unconsulted counterfactuals.
No independent delay/entropy/learning-rate/regularization sweep was preregistered;
those causal hypotheses remain unresolved. The original full M4 protocol remains
the historical baseline, not overwritten or silently retuned.
Every stream label is eventually observed after the drain (120/120). Here
feedback sparsity means one action's evidence out of three valid actions, not
manufactured missing diagnoses. Two and eight steps tie on accuracy/AUROC/F1;
the small BCE difference is descriptive and does not establish meaningful learning.
## Flower, isolation and reproduction evidence

Main Flower run **11975360845880475050**: **348** actual
FedAvg/FedProx aggregations, **1044** client updates,
six contexts and 234 final per-origin metric
replays independently verified. Three clients per round, same compatible state
architecture within a variant. Different fusion architectures are never mixed.
All new private expert hashes match their pre-FL hashes; all
**1177** historical artifact/report hashes match.
No old checkpoint/report/policy pointer was overwritten. Candidate arrays, client
audits, optimizer persistence, step counts and payload keys are saved under
`artifacts/m5-step02-v1/runs`. `flower_evidence.json` binds probabilities/IDs to actual queries.
`verification.json` checks disjoint nested IDs/OOF provenance, inner-only selection,
all weighted aggregates, freeze semantics and recorded-device metric replay.

CPU specialist preparation; CUDA shared optimization with at most one full-GPU
Ray actor serving three logical clients; CPU feedback simulations. Environment
audit ran an actual CUDA matrix operation. Torch peak memory is not total board
memory. Timings include diagnostic instrumentation; no GPU speed/saturation claim.
Actual device **NVIDIA RTX A4000**, CUDA checksum
**3680.0**, maximum local torch allocated memory
**18222592 bytes**. Sum of main strategy runtimes **298.924 s**, excluding private
preparation, framework startup/evidence acquisition and central controls. Per-variant
and feedback runtimes are recorded separately. CPU/GPU speed was not compared on
an identical workload; no acceleration claim follows.
Windows Ray shutdown diagnostics and deprecated Python simulation launcher remain.

```powershell
python -m fed_med_agents.experiments.m5_step02 --prepare-only --output artifacts/my-m5-step02
python -m fed_med_agents.experiments.m5_step02 --device cuda --output artifacts/my-m5-step02
python -m fed_med_agents.experiments.repair_feedback --output artifacts/my-m5-step02
python scripts/verify_m5_step02.py artifacts/my-m5-step02
python scripts/report_m5_step02.py --output artifacts/my-m5-step02
python -m pytest -q
python -m ruff check --no-cache src tests scripts
python -m ruff format --no-cache --check src tests scripts
python -m pip check
```

Use the venv Python and a fresh versioned path. Original execution prepared the
same frozen contexts with `repair_data.prepare_nested` after separately hashing
prior artifacts. The CLI prepare-only entry records hashes/copies the protocol
for future reproductions. Historical Step 1 is preserved, not regenerated.

`qa/pytest.log`:

```text
    from click.utils import get_text_stream as get_text_stream

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
74 passed, 2 skipped, 2 warnings in 7.11s
```

`qa/ruff.log`:

```text
All checks passed!
```

`qa/format.log`:

```text
65 files already formatted
```

`qa/dependencies.log`:

```text
No broken requirements found.
```
## Failures, limitations and unchanged scope

Two initial unit tests failed because a standalone Flower Message lacked the
process TaskIdentity normally supplied by its runtime. A test-only fixture now
sets/restores that identity; no model/acceptance criteria were loosened. Existing
Ruff cache instability is avoided with --no-cache. Actual Flower training completed
without failed client updates; Windows Ray warnings persist.
The first report-generation attempt shadowed the top-level results dictionary
with an NPZ reader and failed while writing summary.json. Renamed the local
reader and regenerated reports successfully; no training results changed.

One cohort, three small nested assessment folds, a single prescribed optimizer
seed per fold and many fixed exploratory comparisons: no significance or external
validity. Logistic changes capacity, identity/confidence features and residual
form together; it does not isolate one cause. Smaller local epochs use less work,
so they are not compute-matched. Pretraining/value regression changes objective,
schedule and supervision. Its teacher targets are meta-level in-sample training
quantities, not independent clinical outcomes. No historical validation/test label
was used to choose configurations. Stronger claims require independent data and
preregistered follow-up. No dynamic registry, variable-client framework, generalized
fusion API, new SLM, Qwen fine-tuning, publishing or GitHub push. Stop after Step 2.
