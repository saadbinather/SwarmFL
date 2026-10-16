# Verified feedback and adaptive federation

M4 separates a prediction from a diagnosis. `feedback.events.PredictionEvent`
contains a request/patient pseudonym, origin, public domain availability, selected
action/peers, exact behavior propensity, score, **observed** probabilities (None
for unconsulted peers), risk, origin-only fusion risk, cost, policy/model versions
and UTC timestamp. It contains no feature values or label. `VerifiedOutcome`
contains a separately received binary diagnosis, authority, timestamp and the
digest of the original prediction. An acknowledgement is never an outcome.

`FeedbackLedger` validates strict schemas, HMAC signatures, origin/patient
authorization, association/digest, delayed arrival and duplicates. Prediction and
outcome files are separate, signed JSONL in the origin's local directory. Replay
joins only verified outcomes received by the training cutoff; pending records
and future outcomes are excluded. The simulation authority reads the original
**training** label locally only after the decision has been persisted. No label
is derived from a classifier or Qwen. Deployment must replace that source with
an authenticated verified-diagnosis service. Local simulator keys/files are
trusted and readable by the operator; HMAC is integrity evidence within that
trust model, not clinical verification, secure transport or proven privacy.

## What learns

Each origin optimizes a compatible `CollaborativeModel`: contextual routing MLP
and directed edge logits, plus masked probability fusion. Private classifiers,
their calibration/preprocessing and Qwen remain frozen. Clients receive shared
weights/configuration and return only shared ArrayRecord weights and aggregate
MetricRecord values. Client-side replay, labels, feature rows and private models
are absent from training replies. The coordinator already sees the selected
probabilities to perform fusion; updates can still disclose information.

For a logged action a, the router computes pi(a|context) using own probability,
origin and public availability only. The behavior distribution is
`mu = 0.8*pi + 0.2*uniform(valid actions)`, including the empty action. Self and
unavailable actions have probability zero. The selected action's exact mu is
logged before consultation; no peer prediction or label enters action selection.

The clipped weight is `w = stop_gradient(min(pi(a)/mu(a), 2))`. The minimized
router surrogate is `0.1 * mean(w * (logged_BCE + actual_cost - 0.5) * log(pi(a)))`.
This is a cost-minimizing score-function gradient with an action-independent
fixed baseline. Fusion minimizes `mean(w * BCE(current_fusion(observed evidence), label))`.
Only the performed action is scored against its observed outcome; there is no
M2-style enumeration of unconsulted probabilities. Missing candidates remain
masked. The two losses are summed; each update uses two full local replay steps,
Adam 0.0003, gradient norm clipping 0.5 and at most 128 latest verified records per
origin. Minimum four records per origin are required; otherwise the server skips
the trigger and keeps the approved policy. Optimizers reset per update.

Clipping biases IPS. Replay mixes model/fusion versions; a logged reward belongs
to the fusion deployed at the time, not a counterfactual newly updated fusion.
No causal or unbiased off-policy evaluation is claimed. Small batches, stale
rewards, FedAvg interference, fixed learning rates and soft stochastic training
versus deterministic validation/inference can degrade routing. There is no
validated reward for arbitrary SLM-selected actions: their action propensity is
unknown. Do not invent that propensity or automatically train this objective on
SLM choices. This benchmark's behavior policy is the audited neural distribution.

## Reproducible sequence and versions

The diabetes adapter uses the existing 364 fixed training patients once, shuffled
with seed 42. Origin rotates A/B/C, so feedback rows assigned to clients are
distinct episodes from one aligned cohort, not independent hospitals. Origin and
selected peer evidence are obtained via **real Flower queries** from existing
nested OOF predictions. Their fitting/calibration excludes the target patient's
label. These frozen cross-fitted experts are retrospective features, not a
strict chronological refit of specialists: their fits can include other patients
that occur later in the simulated stream.

Adaptive and matched frozen routing/fusion start from the same seeded random
initialization, independent of stream labels. They do **not** warm-start from
M2 weights fitted on future stream outcomes. Paired policies use the same
per-episode uniform variate but perform their own selected peer consultations;
neither can inspect the other's evidence. M2's original full-information model
is a separately labeled optimistic offline reference, already trained on this
cohort's OOF outcomes. It is not an unseen-data baseline. Earlier warm-start smoke
artifacts are engineering checks only and are excluded from the final comparison.

Labels arrive after three episode positions. At each interval (default 60), after
due labels arrive, one genuine Flower FedAvg round trains all three logical
clients. Final label drain is followed by an explicit final trigger. `--triggers`
replaces that schedule with specified episode cutoffs; it may intentionally skip
updates due to insufficient feedback. Future labels are unavailable to replay.
The historical test labels never participate in fitting, selection or new metrics.

Every candidate is saved immutably under `versions/vNNN.pt` with parent, cutoff,
client metrics, weight delta and validation scores. Approval requires finite
compatible weights, update L2 <= 0.1 and mean-origin validation BCE within 0.02
of the best validation BCE so far. This is a predeclared development guard that
can accept small degradation; it is not evidence of improvement or medical safety.
Rejected candidates remain saved; `approved.json` stays on the previous policy.
`PolicyRegistry.rollback(version)` atomically returns to a retained eligible
checkpoint. Integrity hashes are checked before load. No history is erased.

M3 `--policy-registry artifacts/milestone04/versions` loads the latest approved
policy per request. The typed `get_routing_recommendation` SLM tool reloads that
version and returns masked action scores/selected names, never a fabricated
diagnosis. It is unavailable when a registry is not configured. Qwen may use the
recommendation, but its frozen weights are not trained and recommendation use
does not establish reasoning quality. Final risk comes from the loaded fusion.

## Metrics and audit

Streaming accuracy, F1, BCE and mean-origin AUROC are computed in origin-local
ledgers, then aggregate scalars are returned. AUROC is the unweighted macro mean
of defined origin AUROCs, **not** pooled AUROC. Counts/TP/FP/FN/BCE sums permit
exact global accuracy/F1/BCE. Reused cumulative curves are not independent tests.
"Unnecessary" means a performed consultation's realized labeled BCE is no lower
than the same checkpoint's origin-only fusion BCE. It is a retrospective loss
proxy on selected actions, not clinical necessity or an estimate for unselected
actions. No extra peer is queried to compute it.

Validation evaluates all 78 validation patients at each origin through selected
Flower queries. These dependent views share patients and prior M2/M3 development
history; no independent generalization or formal significance claim follows.
Routing changes are the fraction of origin/patient validation decisions differing
from initialization. Communication reports logical consultation JSON bytes and
shared tensor upload/download bytes, excluding transport/serialization overhead.
The verifier checks signatures, training membership, label source, delayed replay,
actual selected consultation traces, numerical prediction replay, all-client
updates and FedAvg against saved client shared checkpoints. It uses test IDs only
to assert exclusion and never reads historical test labels.
