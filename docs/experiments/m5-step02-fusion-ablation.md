# M5 Step 2 — fixed fusion ablations

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
## Identical-evidence fusion comparison

The following uses avg10's exact action mask for EVERY fusion, preventing extra
peer evidence from explaining differences. Original residual12→24→1, slim12→8→1,
linear6→1 masked logits/presence without origin identity, uniform-pretrained
original residual, and arithmetic mean. Logistic improves BCE/Brier and accuracy
here, but ECE does not improve consistently. Reduced hidden capacity alone is not
a clear repair. Separately pretrained original fusion plus value routing does not
beat the baseline primary objective. Its shorter pretraining and different router
objective confound attribution to staging alone.

Logistic removes confidence, origin identity and residual form together; results
do not establish overreliance on identity or excessive flexibility as the isolated
cause. All fitting inputs are specialist OOF predictions. Threshold remains .5;
BCE and accuracy/ECE optimize different properties, so arithmetic mean is not
unconditionally better. Correlated expert errors and fold/final calibration shift
remain plausible contributors; no validation/test recalibration occurred.

Experimental logistic/slim states require their explicit variant classes and cannot be loaded into the default original model. They are compatible across the three clients within their own trial, not interchangeable production checkpoints. No serving adapter or default pointer changed.
| Fusion, fixed avg10 mask | accuracy | auroc | binary_cross_entropy | f1 | ece | brier | calls | primary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg10 | 0.88367 | 0.95430 | 0.27260 | 0.90335 | 0.06769 | 0.08281 | 1.00000 | 0.29260 |
| logistic | 0.90287 | 0.96430 | 0.23265 | 0.92021 | 0.07829 | 0.07134 | 1.00000 | 0.25265 |
| small | 0.88091 | 0.95241 | 0.27312 | 0.90175 | 0.06989 | 0.08253 | 1.00000 | 0.29312 |
| value | 0.88367 | 0.95291 | 0.27601 | 0.90480 | 0.06833 | 0.08410 | 1.00000 | 0.29601 |
| arithmetic_mean | 0.90927 | 0.96194 | 0.30298 | 0.92656 | 0.13044 | 0.08913 | 1.00000 | 0.32298 |

### Frozen expert distribution/calibration diagnostics

Final-on-fit probabilities below are resubstitution diagnostics ONLY, never fusion fitting or model selection inputs. The shift combines fold sample size, specialist refitting and calibration; it does not isolate a cause.

| Fold | Origin | Mean abs(OOF−final fit) | OOF BCE | Final-fit BCE (optimistic) | Assessment BCE | OOF ECE | Assessment ECE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0.05368 | 0.55089 | 0.51712 | 0.50662 | 0.06315 | 0.07250 |
| 0 | 1 | 0.04342 | 0.33423 | 0.28686 | 0.31819 | 0.06587 | 0.07516 |
| 0 | 2 | 0.11518 | 0.37776 | 0.22595 | 0.31721 | 0.05397 | 0.07678 |
| 1 | 0 | 0.09982 | 0.63189 | 0.54355 | 0.49626 | 0.07644 | 0.12723 |
| 1 | 1 | 0.05456 | 0.32277 | 0.26608 | 0.35568 | 0.03708 | 0.06874 |
| 1 | 2 | 0.12358 | 0.43262 | 0.25019 | 0.28682 | 0.06466 | 0.07249 |
| 2 | 0 | 0.07791 | 0.56278 | 0.47288 | 0.60668 | 0.09610 | 0.08802 |
| 2 | 1 | 0.06424 | 0.35434 | 0.30124 | 0.29832 | 0.05084 | 0.08794 |
| 2 | 2 | 0.10548 | 0.47122 | 0.25270 | 0.33593 | 0.08666 | 0.09337 |

Signed probability-error correlations are associations, not conditional independence or causal evidence of redundancy.

| Fold | A/B error corr | A/C error corr | B/C error corr | B/C threshold disagreement |
| --- | --- | --- | --- | --- |
| 0 | 0.55043 | 0.62130 | 0.45929 | 0.19672 |
| 1 | 0.61611 | 0.55864 | 0.52950 | 0.18182 |
| 2 | 0.64026 | 0.62796 | 0.67123 | 0.23140 |
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
