# Verified SwarmFL research results

This document consolidates verified results without redistributing private data,
checkpoints, or bulky run evidence. Values retain their recorded precision. The
primary public provenance index is [`docs/status.md`](../docs/status.md); detailed
methods are in [`docs/experiments/`](../docs/experiments/). Local `artifacts/...`
references identify machine-readable evidence that is intentionally gitignored.

## A. Framework architecture

SwarmFL separates independently fitted private specialists from shared
collaboration components:

1. A specialist owns its preprocessing, private checkpoint, feature contract, and
   scalar binary-probability output. Private specialist weights are not federated.
2. `SpecialistRegistry` manages immutable model/task/output/feature declarations,
   availability, cost, authorization, and logical lifecycle. Logical registration
   is distinct from joining the fixed physical Flower simulation.
3. Fixed, contextual MLP, and graph-inspired routing policies choose STOP or at
   most one eligible peer under the configured cost budget.
4. Genuine Flower messages carry an aligned opaque identifier or, for explicitly
   authorized new-patient inference, only the selected peer's required fields.
   Unselected specialists receive no consultation.
5. Dynamic fusion supports arithmetic mean, training-OOF reliability weighting,
   and a permutation-invariant DeepSets model over the observed evidence set.
6. Predictions and delayed outcomes are separately signed and verified. Only
   released, authenticated feedback can enter shared learning.
7. Flower FedAvg optimizes compatible shared router/fusion parameters. It does not
   transmit or average private specialist models.
8. Frozen validation gates precede immutable version activation. Compatible pairs
   can be rejected, activated, recovered after restart, and rolled back exactly.

See [`docs/architecture.md`](../docs/architecture.md) and
[`docs/feedback-learning.md`](../docs/feedback-learning.md).

## B. CDC diabetes development research

The CDC work uses the UCI 891 diabetes health indicators survey: 253,680 records,
21 public fields, and a binary diabetes indicator. The frozen 50,861-row test
container and ID file have never been opened or evaluated in the later research
sequence. The working development pool contains 202,819 rows. Grouped boundaries
keep identical 21-feature records together, with upstream specialist fitting,
calibration, routing-target construction, inner validation, and assessment kept
separate within each recorded protocol.

The final three-specialist amendment fixes these heterogeneous views:

| Specialist | View | Required fields |
|---|---|---|
| A | Metabolic | BMI, HighBP, HighChol, CholCheck |
| B | Lifestyle | Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump |
| C | General health | GenHlth, MentHlth, PhysHlth, DiffWalk |

The amendment reused 576 byte-identical private model/calibrator files only after
feature, identity, exclusion, alignment, and hash checks. Shared fusion/router
states were freshly initialized. Sources:
[`three-agent protocol`](../docs/experiments/m5-step035-three-agent-protocol.md) and
[`benchmark`](../docs/experiments/m5-step035-three-agent-benchmark.md).

### Routing supervision scale

Macro objective is binary cross-entropy plus consultation cost 0.02 on the same
variable-service development assessments for each row.

| Routing records | Fixed | Contextual MLP | Graph-inspired | MLP utility R² |
|---:|---:|---:|---:|---:|
| 15,000 | 0.350173 | 0.350328 | 0.350800 | 0.000158 |
| 50,000 | 0.350173 | 0.349980 | 0.350704 | 0.000611 |
| 100,000 | 0.350173 | 0.350096 | 0.350173 | 0.000262 |
| 160,000 | 0.350173 | 0.350110 | 0.350173 | 0.000003 |

At 160,000 records, the MLP's 0.000063 aggregate objective difference from fixed
is paired with utility R² 0.000003; all sizes failed the preregistered meaningful-
gain, utility, and all-origin variation gates. These values do not establish
learned-routing superiority. Assessment cohorts can overlap across repeated
splits, and this varies routing supervision rather than total pipeline data.
Source: [`docs/status.md`](../docs/status.md), M5 Step 5; local machine-readable
evidence: `artifacts/m5-step05-v1/summary.json`.

### Dynamic fusion on identical ABC evidence

Assessment below is BCE over the same 19,815 development patients and observed
A/B/C evidence set.

| Existing residual | Mean | Reliability-weighted | Central DeepSets | Flower DeepSets |
|---:|---:|---:|---:|---:|
| 0.329449 | 0.345023 | 0.344550 | 0.320283 | 0.321116 |

The old residual was trained on nine STOP/one-peer masks, so ABC is outside its
training-mask support. New set models trained all 12 origin/subset masks. Capacity,
training protocol, and compute are not matched; therefore the table is a
development diagnostic, not a clean superiority comparison. Training used 162,850
private OOF rows, validation 20,154, and assessment 19,815. Source:
[`docs/status.md`](../docs/status.md), M5 Step 6; local evidence:
`artifacts/m5-step06-v1/summary.json`.

Earlier UCI 529 work used 520 records with a fixed 364/78/78 split and established
the initial three-view Flower mechanics. Its retrospective M1 shared ensemble
recorded accuracy 0.9230769, AUROC 0.9347222, F1 0.9375, and BCE 0.3919495 on 78
held-out records. The small reused cohort is not comparable to the CDC studies and
is not independent clinical evidence. Source:
[`milestone-01.md`](../docs/experiments/milestone-01.md).

## C. Continual federated learning

Three sequential retrospective cycles used disjoint feedback and validation groups.
Each cycle released 516 verified origin records (172 per origin); each component
consumed observations once. Objective is BCE plus consultation cost.

| Cycle | Verified records | Previous | Fixed | Active | Router candidate | Fusion candidate |
|---:|---:|---:|---:|---:|---|---|
| 1 | 516 | 0.348792 | 0.349695 | 0.348702 | Rejected | Activated |
| 2 | 516 | 0.368325 | 0.367517 | 0.368325 | Rejected | Rejected |
| 3 | 516 | 0.327500 | 0.330738 | 0.327500 | Rejected | Rejected |

The genuine Flower run used three physical clients, six FedAvg aggregates, and 18
local component updates. The active pair moved from `(r000,f000)` to `(r000,f001)`
after cycle 1, retained that pair in cycles 2–3, then exactly rolled back to
`(r000,f000)`. The cycle populations differ, so scores are not a learning curve;
the small cycle-1 fusion gain does not establish sustained improvement. Source:
[`docs/status.md`](../docs/status.md), M5 Step 7; machine-readable source:
`artifacts/m5-step07-v1/summary.json`.

## D. External specialist validation — Step 8A

Step 8A used 600 generated six-feature records split before fitting, three external
sklearn specialists (two logistic pipelines and one shallow tree), and three real
CPU Flower clients. It verified 360 decisions/fusions and 324 signed origin-feedback
records. Two FedAvg rounds produced six local shared updates.

| Candidate comparison | Objective |
|---|---:|
| Previous/incumbent pair | 0.608611 |
| Fixed routing reference | 0.608611 |
| Router candidate | 0.608611 |
| Matched mean fusion | 0.607400 |
| DeepSets fusion candidate | 0.608733 |

The router candidate passed engineering gates by matching, not improving upon,
incumbent/fixed behavior and was activated as `r001`. The fusion candidate was
rejected. Compatible `r001/f000` activation and exact rollback to `r000/f000`
passed. Run ID: `11798885370385872129`. Replay preserved 12,881 historical hashes
without opening or hashing the frozen CDC test. Sources:
[`milestone-05.md`](../docs/experiments/milestone-05.md) and local
`artifacts/m5-step08a-v1/{results,verification,execution}.json`.

This demonstrates external adapter extensibility in one trusted synthetic process,
not transfer to unseen institutions or distributions.

## E. Installed framework validation — Steps 8B/8C

The `swarmfl` 0.1.0a1 wheel was installed in a fresh Python 3.12 environment outside
the source repository. Imports resolved from installed `fed_med_agents`. The
synthetic command connected three distinct CPU Flower nodes, performed exactly one
selected peer call per case, authenticated delayed feedback, released one verified
record, approved `v001`, and rolled back to `v000`. The custom-specialist example
ran without core edits.

Release QA: **196 passed, 4 skipped, 0 failed**; Ruff, build, Twine, `pip check`,
public import, archive-content, and clean-wheel checks passed. This is packaging and
integration evidence, not predictive generalization evidence. Sources: local
`artifacts/m5-step08b-v1/execution.json` and
`artifacts/m5-step08c-v1/execution.json`.

## F. Limitations and public reproducibility

- The frozen CDC test has not been evaluated; no public claim uses it.
- There is no independent hospital validation or established clinical utility.
- Selected-only messages, signatures, opaque IDs, and authorization metadata are
  not confidentiality proofs. There is no differential privacy, secure aggregation,
  HIPAA/GDPR claim, or adversarial multi-hospital isolation.
- Learned routing superiority and sustained continual improvement are unproven.
- Registry membership is dynamic logically; demonstrations retain fixed physical
  Flower clients.
- CDC data, private checkpoints, Qwen weights, generated artifacts, and frozen
  evaluation material are not distributed. Private CDC studies are therefore not
  fully reproducible from a public clone.
- Flower 1.40.0's simulation extra pins Ray 2.55.1. SwarmFL accepts no untrusted Ray
  Data/WebDataset input; the residual dependency risk is documented in
  [`SECURITY.md`](../SECURITY.md).

The supported public reproduction is synthetic:

```bash
python -m pip install dist/swarmfl-0.1.0a1-py3-none-any.whl
cd /a/directory/outside/the/source/repository
swarmfl-quickstart --output demo --feedback
python /path/to/SwarmFL/examples/custom_specialist.py
```

After PyPI publication, the first command becomes `python -m pip install swarmfl`.
