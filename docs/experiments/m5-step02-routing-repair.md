# M5 Step 2 — STOP and routing repair

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
## STOP and usefulness interpretation

At the primary .02 tariff the selected pipelines still always consult; useful
individualized STOP is not established. Raising training cost to .1/.3 increases
STOP but predictive BCE worsens. At their higher assigned evaluation tariffs,
fewer calls can give a better cost-aware objective; compare the whole tradeoff,
not STOP count alone. No inference uses label-aware best actions. Always-consult
is competitive among the tested low-cost policies, not proven Bayes-optimal.

Marginal-value regression predicts fitting-only realized loss reduction from
pre-query context. Inference sets STOP value zero and subtracts .02 from eligible
peer estimates. Uniform-pretrained fusion remains frozen. The simple utility bins
use fixed confidence bins and fitting-only mean gains. Both can STOP, but fail
to improve the primary mean objective over avg10. They do not establish calibrated
probabilities of correctness or useful prospective value estimates.

Entropy/hard-soft quantities use normalized raw scores. For value regression these
are diagnostics, not its MSE fitting objective. For deterministic controls they
describe the underlying avg10 score distribution, not sampling randomness of the
control. Harm is selected-action realized BCE >= same-fusion STOP BCE; it is not
clinical harm. Availability/capability flags are constant here: their predictive
value cannot be identified. Correlations below are retrospective assessment
diagnostics only and never feed deployed choices.
| Policy | STOP fraction | Calls | Score entropy | Hard-soft BCE | Harm/calls | Objective .02 | Objective .1 | Objective .3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| avg10 | 0.00000 | 1.00000 | 0.00209 | -0.00002 | 0.23444 | 0.29260 | 0.37260 | 0.57260 |
| cost10 | 0.33333 | 0.66667 | 0.10157 | 0.00640 | 0.16889 | 0.31629 | 0.36963 | 0.50296 |
| cost30 | 1.00000 | 0.00000 | 0.00000 | -0.00000 | undefined | 0.39951 | 0.39951 | 0.39951 |
| value | 0.06612 | 0.93388 | 1.09661 | -0.05343 | 0.19024 | 0.30497 | 0.37968 | 0.56646 |
| utility_bins | 0.18686 | 0.81314 | 1.04525 | -0.06229 | 0.16465 | 0.30207 | 0.36712 | 0.52975 |
| stop | 1.00000 | 0.00000 | 0.00209 | 0.15327 | undefined | 0.42588 | 0.42588 | 0.42588 |
| random | 0.00000 | 1.00000 | 0.00209 | 0.04102 | 0.29411 | 0.33363 | 0.41363 | 0.61363 |
| cyclic | 0.00000 | 1.00000 | 0.00209 | 0.02125 | 0.31060 | 0.31386 | 0.39386 | 0.59386 |

### Action counts [STOP,A,B,C]

| Fold | Policy | Origin | Counts |
| --- | --- | --- | --- |
| 0 | avg10 | 0 | [0, 0, 122, 0] |
| 0 | avg10 | 1 | [0, 0, 0, 122] |
| 0 | avg10 | 2 | [0, 0, 122, 0] |
| 0 | cost10 | 0 | [0, 0, 122, 0] |
| 0 | cost10 | 1 | [122, 0, 0, 0] |
| 0 | cost10 | 2 | [0, 0, 122, 0] |
| 0 | cost30 | 0 | [122, 0, 0, 0] |
| 0 | cost30 | 1 | [122, 0, 0, 0] |
| 0 | cost30 | 2 | [122, 0, 0, 0] |
| 0 | value | 0 | [0, 0, 59, 63] |
| 0 | value | 1 | [0, 0, 0, 122] |
| 0 | value | 2 | [0, 57, 65, 0] |
| 0 | utility_bins | 0 | [0, 0, 122, 0] |
| 0 | utility_bins | 1 | [62, 26, 0, 34] |
| 0 | utility_bins | 2 | [0, 9, 113, 0] |
| 1 | avg10 | 0 | [0, 0, 121, 0] |
| 1 | avg10 | 1 | [0, 0, 0, 121] |
| 1 | avg10 | 2 | [0, 0, 121, 0] |
| 1 | cost10 | 0 | [0, 0, 121, 0] |
| 1 | cost10 | 1 | [121, 0, 0, 0] |
| 1 | cost10 | 2 | [0, 0, 121, 0] |
| 1 | cost30 | 0 | [121, 0, 0, 0] |
| 1 | cost30 | 1 | [121, 0, 0, 0] |
| 1 | cost30 | 2 | [121, 0, 0, 0] |
| 1 | value | 0 | [0, 0, 121, 0] |
| 1 | value | 1 | [0, 0, 0, 121] |
| 1 | value | 2 | [0, 0, 121, 0] |
| 1 | utility_bins | 0 | [0, 0, 121, 0] |
| 1 | utility_bins | 1 | [74, 0, 0, 47] |
| 1 | utility_bins | 2 | [0, 0, 121, 0] |
| 2 | avg10 | 0 | [0, 0, 121, 0] |
| 2 | avg10 | 1 | [0, 0, 0, 121] |
| 2 | avg10 | 2 | [0, 0, 121, 0] |
| 2 | cost10 | 0 | [0, 0, 121, 0] |
| 2 | cost10 | 1 | [121, 0, 0, 0] |
| 2 | cost10 | 2 | [0, 0, 121, 0] |
| 2 | cost30 | 0 | [121, 0, 0, 0] |
| 2 | cost30 | 1 | [121, 0, 0, 0] |
| 2 | cost30 | 2 | [121, 0, 0, 0] |
| 2 | value | 0 | [0, 0, 121, 0] |
| 2 | value | 1 | [55, 21, 0, 45] |
| 2 | value | 2 | [17, 65, 39, 0] |
| 2 | utility_bins | 0 | [0, 0, 63, 58] |
| 2 | utility_bins | 1 | [68, 5, 0, 48] |
| 2 | utility_bins | 2 | [0, 0, 121, 0] |

### Confidence and retrospective best-peer gain

| Fold | Origin | Pearson confidence/gain | Fraction gain > .02 |
| --- | --- | --- | --- |
| 0 | 0 | -0.43777 | 0.90984 |
| 0 | 1 | -0.43953 | 0.45902 |
| 0 | 2 | -0.22044 | 0.46721 |
| 1 | 0 | -0.64013 | 0.95041 |
| 1 | 1 | -0.37836 | 0.35537 |
| 1 | 2 | -0.41361 | 0.44628 |
| 2 | 0 | -0.19684 | 0.90083 |
| 2 | 1 | -0.61874 | 0.40496 |
| 2 | 2 | -0.53840 | 0.60331 |
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
