# Milestone 4 — verified feedback-driven federated learning

Executed on Windows with three real Flower 1.39 ClientApps, one GPU actor and
CPU server orchestration/CUDA model operations. No WSL execution is claimed.
Main Flower run ID: `11671572409874175043`. Output: `artifacts/milestone04`.
Runtime: 478.052 seconds; device `cuda`.

## Protocol and leakage controls

364 unique training patients, rotating origin, seed 42,
label delay 3 episodes, scheduled updates every 60 episodes plus final drain.
Adaptive and matched frozen routing/fusion are identically initialized without
stream-label information. M2 full-information is a separately marked offline
reference fitted on these outcomes, not an unseen-outcome baseline. Frozen nested
OOF experts exclude the target label; prior fits can include other patients later
in the stream, so this is retrospective cross-fitting rather than temporal validation.
No new historical-test evaluation or test-driven tuning occurred.

Predictions and signed verified outcomes are separate origin-local records. True
training labels are revealed only after recorded decisions; neither risk nor SLM
acknowledgement is a label. Only selected evidence is present in replay. Client
gradients optimize clipped IPS cost policy loss and weighted observed-evidence
fusion BCE. Flower aggregates only compatible shared arrays and aggregate metrics.
Qwen, private specialists, preprocessing and temperatures remain frozen.
See [exact objective/assumptions](../feedback-learning.md).

## Actual streaming results

| Policy | Accuracy | Macro-origin AUROC | F1 | BCE | Calls/patient | Unnecessary fraction of calls |
|---|---:|---:|---:|---:|---:|---:|
| adaptive | 0.788462 | 0.858307 | 0.826966 | 0.449557 | 0.675824 | 0.434959 |
| frozen | 0.788462 | 0.857944 | 0.826966 | 0.450215 | 0.673077 | 0.436735 |
| offline | 0.884615 | 0.935422 | 0.903670 | 0.328534 | 1.000000 | 0.170330 |

AUROC is a mean of origin-local AUROCs, not pooled AUROC. Accuracy/F1/BCE aggregate
exact local confusion counts/loss sums. Unnecessary means selected-call realized
BCE >= same-model origin-only fusion BCE; this is not clinical necessity.
One reused stream cannot establish superiority. The offline reference has prior
full-information supervision and a different protocol. No positive clinical
improvement claim is made from these development metrics.

## Actual updates and validation curve

Initial validation mean-origin BCE: 0.33888923.
Final approved validation BCE: 0.33827713.

| Cutoff | Version | Approved | Validation BCE | Shared update L2 | Changed validation routes |
|---:|---|---|---:|---:|---:|
| 60 | v001 | True | 0.33879258 | 0.00716594 | 0.012821 |
| 120 | v002 | True | 0.33874727 | 0.00850876 | 0.012821 |
| 180 | v003 | True | 0.33865772 | 0.00824636 | 0.012821 |
| 240 | v004 | True | 0.33863375 | 0.00718441 | 0.000000 |
| 300 | v005 | True | 0.33851158 | 0.00731921 | 0.000000 |
| 360 | v006 | True | 0.33839444 | 0.00725402 | 0.000000 |
| 367 | v007 | True | 0.33827713 | 0.00724780 | 0.000000 |

Approved version `v007`; 21 local updates.
Initial-to-approved router L2 0.02146328, fusion L2
0.03749849, combined L2 0.04320659.
All private specialist hashes remained unchanged. A later real Flower coordinator
request loaded `v007`, origin
1, patient ID 282,
selected mask `[[True, False, False]]`, risk
`[0.31685060262680054]`.

Full config, per-origin validation metrics, cumulative stream curves, clipped
ratios, gradient norms, runtime and torch peak allocation are in metrics.json,
versions/vNNN.json and client_N/update_R.json. Learning curves are preserved as
learning_curve.json and learning_curve.csv. The approval guard accepts BCE within
0.02 of best seen and update L2 <= 0.1; it can permit small degradation and is not
medical safety. All candidates remain immutable; rollback was unit tested.

Consultation logical JSON: 955503 bytes.
Shared tensor upload/download estimate: 103824 bytes.
These exclude transport/serialization overhead and feedback/control messages.

## Verification and commands

```powershell
python -m fed_med_agents.experiments.milestone04 --device cuda --output artifacts/milestone04
python -m fed_med_agents.feedback.verify artifacts/milestone04
python scripts/report_milestone04.py --checks --real-qwen
```

Actual QA: `52 passed, 2 warnings in 80.11s (0:01:20)`. Ruff lint/format and pip check passed. Fresh CPU integration
output `artifacts/milestone04-integration-20261014-130323`; run ID `10082744410246933231`,
24 patients, early insufficient-data trigger skipped, then one three-client
update approved and used by a later actual peer consultation. Saved verification
replayed authentication, true source labels, delayed cutoff joins, training IDs,
selected-only Flower traces, numerical predictions and FedAvg against all three
client shared checkpoints. Test IDs were read only to assert exclusion; test labels
were not evaluated. Qwen is optional for tabular tests; the separate regression
output is `artifacts/milestone04-qwen-regression-20261014-130323`.

## Real frozen-Qwen approved-policy tool

Flower run `10659713594225044832` used an unlabeled, validation-derived
synthetic dictionary with origin B. Real Qwen called `get_routing_recommendation, get_specialist_capabilities, request_specialist_prediction, finalize_prediction`, loaded
approved policy `v007`, consulted
`['A']` through Flower and returned risk
`0.2134048044681549` to B. Latency 1.988s,
zero tool failures. Independent verification reproduced numerical fusion and
authorized subsets/return hops. All classification metrics are null: this request
has no verified new-patient diagnosis. Its acknowledgement was not training data.
Qwen remains frozen; this demonstrates use of the updated neural policy, not SLM learning.

The first attempt failed before Flower because the prior Ollama endpoint was not
running (Windows connection refused). Failure logs remain in artifacts/milestone04-slm*.
Restarted the existing portable Ollama 0.40.4/model hidden, without downloading,
installing an OS service or changing drivers. Successful output is
artifacts/milestone04-slm-approved; Ollama GPU residency was 1,169,382,440 bytes.

```powershell
python -m fed_med_agents.experiments.milestone03 --device cpu --request-file artifacts/milestone04/slm-request.json --policy-registry artifacts/milestone04/versions --output artifacts/milestone04-slm-approved
python -m fed_med_agents.agents.verify artifacts/milestone04-slm-approved
```

Use a fresh output path to retain prior evidence. The tabular framework requires
no Qwen service. The separately requested Qwen demonstration does.

## Failures, scope and unfinished work

First warm-start smoke failed in the parent console with Windows cp1252
UnicodeEncodeError; UTF-8 stdout fixed it, with failed log retained. The successful
warm-start smoke is engineering evidence only, excluded from the controlled
comparison. A first cold CUDA stream was interrupted at 120 episodes by tool/chat
session termination and preserved under artifacts/milestone04-interrupted; it has
no final metrics. The complete run used hidden background execution with preserved
driver logs and identical frozen configuration. Windows Ray/deprecated-launcher
warnings remain. No OS/driver changes or publishing occurred.

Simulation diagnoses are authenticated trusted-source training labels, not newly
collected clinical outcomes. Clipped/stale replay is biased; no arbitrary SLM
propensity is invented. External diagnosis verification, production scheduling,
secure isolation/transport, concurrent approval writers and independent future
cohort evaluation remain open. Qwen fine-tuning is not implemented.
