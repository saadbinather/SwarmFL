# Scientific and implementation limitations

## Step 8B release-candidate limitations

- A 2026-10-16 lookup found no live `swarmfl` PyPI page, but the name is not
  reserved or guaranteed until authenticated publication.
- Git history begins with the local Step 8C initialization; no commit exists yet.
- Apache-2.0 is approved by the owner and installed with complete text and PEP 639
  metadata. Canonical repository URLs target
  `https://github.com/saadbinather/SwarmFL`; no maintainer identity is inferred.
- The Hugging Face card covers generated synthetic adapters, not a clinical checkpoint.
- Python 3.11 is locally testable; 3.12/3.13 remain CI targets.
- Flower 1.40.0 is pinned for release packaging; historical reports retain the
  Flower versions actually used for their runs.
- Flower's simulation extra pins Ray 2.55.1, affected by PYSEC-2026-2273. The
  vulnerable `ray.data.read_webdataset` decoder is not imported or called by
  SwarmFL, and the supported demo accepts no WebDataset/TAR input, but the affected
  package remains installed. The owner accepts this residual high-severity risk
  for the trusted-local research alpha only; a compatible upstream pin is preferred.
- PyTorch currently constrains setuptools below 82. The environment therefore
  retains PYSEC-2026-3447, whose attack is malicious Unicode-normalized filenames
  bypassing `MANIFEST.in` exclusions on macOS. Release builds use an isolated
  setuptools >=83 backend and independently inspect both archives.
- The quickstart tests transport/API boundaries, not model quality or security.
- Authorization, signatures, opaque IDs, selected-only messages, and minimized logs
  are not privacy proofs; disclosures through predictions, IDs, timing, metadata,
  logs, and model updates remain possible.

Step 8A is a deterministic synthetic engineering validation, not clinical or
external predictive validation. Its 30-day event horizon is simulated; cases are
independent generated rows, the three feature views come from one aligned dataset,
and all private sklearn models run on one trusted machine. Registration proves an
external adapter can satisfy the current scalar binary-probability contracts. It
does not prove performance on unseen specialists, arbitrary tasks, multiclass
outputs, hospitals, modalities or distributions.

The bootstrap router always consulted one peer in the feedback and validation
populations, so the measured STOP rate is zero. STOP itself is functionally tested
with no transport. The accepted router changed weights but produced the same held
actions/metrics as incumbent and fixed control; its gate result is not an accuracy
gain. The fusion candidate was rejected. One 120-case synthetic validation set is
too small for scientific comparison, and no configuration was tuned after seeing
it. Controlled approval fixtures validate mechanics only.

All communication is local trusted simulation. Observed process-subtree listeners
were loopback-only, but this does not test firewall rules, authentication, TLS,
remote deployment or adversarial isolation. Application JSON byte counts omit
Flower/Ray serialization and network framing. Latencies are batched whole-message
wall times; first-client startup dominates the maximum. Windows Ray remains
experimental, its Python launcher is deprecated in Flower 1.39, and shutdown stack
diagnostics remain in the successful run log.

Authorization and feature permissions are trusted caller metadata. Opaque IDs,
predictions, costs, signatures, checkpoint hashes and model updates may disclose
information. Signed feedback provides integrity under local simulation keys, not
medical correctness, confidentiality, consent, key management or nonrepudiation.
There is no differential privacy, secure aggregation, encrypted transport,
cryptographic hospital isolation, durable global budget, process-safe distributed
transaction or exactly-once crash recovery. Logical unregister/availability does
not physically remove a Flower client.

The frozen CDC test container and ID file were not opened, loaded, rehashed or
evaluated. Historical hashes exclude those files by construction. CDC checkpoint
metadata/shared states were read only to exercise incompatible identity rejection;
CDC specialists, routers, fusions and defaults were not retrained or changed.

Step 7 is retrospective continual-learning simulation. CDC has no genuine arrival
or delayed-outcome timestamps. Feedback/validation group allocation and synthetic
episodes are frozen before the run; signed outcomes come from a distinct simulator
key/authority reading development diagnoses locally. This is not independent
clinical adjudication, deployment consent or prospective online evidence. Original
private expert/router/fusion training excludes these groups, but historical
selection/evaluation already reused their source partitions. Cross-origin views
share patients; record counts are not independent patient counts.

The deterministic existing router logs propensity1. It has no exploration or
support for unchosen actions, so observed-only IPS updates cannot establish
unbiased off-policy comparisons or discover unseen peer rewards. No counterfactual
predictions/rewards enter fitting. Fusion learns only observed selected sets;
STOP/singleton cases give no fusion-parameter signal. Weighted reliability remains
frozen. Validation populations differ by cycle; falling raw metrics across cycles
would not establish improvement. Compare candidates only within their own matched
cycle population, and fusion only on identical evidence. Small approvals are
engineering gate outcomes, not statistical or clinical superiority.

Controller context construction, authorization and independent outcome verification
are application responsibilities. Local signatures bind persisted records but do
not prove private-inference honesty or protect against a compromised trusted host.
Labels/raw local fields remain outside prediction/consultation messages; model
updates/losses/predictions/identifiers can still disclose sensitive information.
No privacy proof, secure aggregation or remote multi-hospital isolation is added.
All three physical clients remain connected; registry outages are logical views.

Recovery supports signed ledger/context/consumption validation and atomic local
active-pair replacement under a single writer. It is not a distributed transaction
manager, concurrent registry snapshot, durable remote queue or crash-proof FedAvg
resume service. A crash before ledger persistence may leave a signed orphan;
requests fail closed rather than automatically repeating a consultation. A crash
after aggregation but before consumption commit requires explicit reconciliation;
no exactly-once end-to-end federation claim. Active pointer is authoritative;
transition intent files may include an uncommitted interrupted write. No fsync
power-loss guarantee/key rotation/persistent global budget is provided. The caller
must not activate an independent legacy approved pointer expecting the controller
to follow it. Long-lived ledgers currently load their local history into memory.

All candidate versioning is research-only. No private specialist/Qwen retraining,
new routing algorithm, arbitrary model compatibility, production promotion or
frozen-test evaluation is authorized by this milestone. Next-stage cleanup is
explicitly deferred. Earlier milestone limitations follow in their original scope.


Step 6 supports scalar binary probabilities with exactly matching task/label/output
contracts, not arbitrary models, heterogeneous output spaces, multiclass fusion,
new modalities or demonstrated unseen-specialist generalization. Only A/B/C are
used. The reusable set network has no count-specific parameter dimension, but
new identities/checkpoints require explicit declarations and new validation.
Hashed identity coordinates preserve differentiation, not clinical semantics.
Reliability is one global OOF-BCE estimate per specialist, not contextual or
clinically calibrated reliability. It is normalized only over actual evidence.

The API trusts its application to supply real consultation receipts and origin
inference. FusionEvidence/checkpoint hashes and registry admission are not signed
proof of prediction provenance or patient consent. The generic tensor training
adapter is trusted; public fuse() performs identity/schema/authorization checks.
It has no consultation budget side effects and cannot prove prior budget spend;
ConsultationSession governs actual dispatch. Stale/unavailable observed evidence
is rejected, not retroactively erased from previous fused results. Runtime
registry updates are not an atomic transaction spanning a whole multi-peer case.
Current supported public fuse() is single-case; tensor batching requires a trusted
aligned adapter. Extreme probabilities are epsilon-clipped inside tensor/logit
math; public mean and singleton retain raw endpoints. Floating point reductions
are invariant within tolerance, not guaranteed bitwise for arbitrary tensor order.

The study reuses Step 5 development split0 and frozen expert/calibration files.
Train162850/validation20154/assessment 19,815 boundaries remain group-disjoint;
this is not a fresh external cohort or untouched clinical benchmark. All held
outcomes are diagnostics; no model/schedule was chosen using assessment scores.
Central20epoch and Flower3round/2localepoch fitting have different optimizer/reset
and step budgets. Their results cannot establish aggregation superiority. Existing
residual60 was trained on 9 STOP/one-peer masks, while new set models see12 masks;
fullABC historical comparisons are outside old trained-mask support. Singletons
are deliberately exact in new strategies; historical residual may change them.
Different evidence subsets need separate reporting and incur different call costs.
No routing optimization, causal improvement or clinical validity is established.

Flower uses3 fixed physical nodes and one CPU actor, with dynamic logical
registration only. Local OOF caches/labels are trusted prepared files; no new
federated private-expert training or distributed filesystem isolation is implied.
Predictions/IDs/shared updates can leak information; no privacy/security proof,
secure aggregation, production consent or physical join/leave is implemented.
Only new fusion state is aggregated; historical private/fusion/routing states
are preserved by hashes. Research version approvals use fixed development gates;
no production default is changed. CUDA figures measure torch allocation only,
excluding context/reserved/board memory; Flower CPU RSS is unmeasured. Reported
latency is local tensor fusion including encoding, excluding private inference,
registry/session checks, transport and network latency.

Earlier milestone limitations follow unchanged in scope.



Step 5 scientific outcome: the MLP has small aggregate improvements over matched
fixed value at 50k/100k/160k, but all sizes miss the frozen meaningful-gain, utility
R² and all-origin variation gates. The larger-data consistency checks fail. The
real feedback candidate worsens validation objective and is rejected. These are
negative/limited findings under fixed schedules, not proof that useful routing
or continual learning is impossible. No outer winner or production model is chosen.

Step 5 is a newly authorized exploratory development study, independent of the
failed Step 3.6 expansion gate. Three repeated group-held splits overlap across
replicates; fold/origin views are not independent patients or external validation.
Within each split, assessment and validation groups are excluded from every fit.
The common upstream pool is roughly 160k, even for a 15k router. Sizes measure the
number of routing-supervision examples, not total expert/fusion data budgets.
Larger sizes have more optimizer steps under fixed epochs, and initialization
seeds differ by size; data quantity is not isolated from compute/initialization.
Fresh experts/calibrators/fusion avoid historical split leakage; old checkpoints
stay unchanged. Final fusion is a fixed 60-epoch residual, not proven optimally
calibrated/converged. Teacher/final shift, noisy single-label utility targets,
survey imbalance/weights and absence of clinical validation remain.

The graph wrapper preserves the existing architecture but predicts action scores,
not calibrated gains; its explicit cost adjustment and soft training differ from
the MLP value objective. It has no additional clinical input and is not GAT. The
observed-only feedback exercise is a small training-cohort simulation, not a
large-scale bandit comparison or evidence of automatic continual improvement.
Its96 patients produce288 origin observations, and initial supervised weights have
already used their OOF labels. Feedback predictions use target-fold experts/fusion
that exclude those patients, but this is not a genuinely new prospective cohort.
The demonstration uses static services/flat.02 registry costs, while validation
and the matched size study include variable tariffs. It does not establish
performance under online service drift. The signed outcome authority is the simulator reading known training diagnoses,
not real delayed clinical adjudication. Clipped IPS is biased and there are no
confidence bounds, equal-feedback bandit controls or deployment evidence.

Registry state/permissions and local context snapshots are trusted. The generic
router checks ID order/dimensions and admission; the application must bind the
correct task/checkpoints/context adapter and enforce origin-local authorization.
CDC's checked adapter enforces its own field boundaries. Raw local contexts are
not sent in Flower messages; predictions/IDs/model updates still disclose
information. Frozen model files are verified unchanged, not physically sandboxed.
Routing arrays can indirectly encode labels; no secure aggregation, differential
privacy, authenticated remote consent or multi-host isolation is claimed.
Version approval is a research pointer under trusted metadata, not permission to
replace the production router. Insufficient/rejected feedback retains the previous
version. CPU Flower peak RSS was not measured; CUDA figures are torch allocation,
excluding contexts, reserved memory and total board use. Runtime overlaps local QA
and CPU integration work; no GPU saturation or speedup comparison is claimed.

The following statements retain the scope and outcomes of earlier steps.

Step 4 provides logical in-memory registration, not physical Flower membership
changes, persistence, multi-process consistency, signed capability/consent claims,
external authentication or a privacy proof. Three physical CDC clients stay active;
one CPU actor serves them. The existing three-input fusion/checkpoint slot map is
fixed and cannot accept a fourth evidence dimension without a separate compatible
model. Only A/B/C are demonstrated; the reusable registry currently validates
binary-probability tasks, not arbitrary multimodal output fusion.

Metadata feature/case authorization governs supported registry dispatch; trusted
central callers and historical benchmark handlers still have local file/transport
access. It is not an adversarial deployment boundary. In-flight calls admitted
before an availability change are not cancelled. Sessions reserve attempted-call
budgets and costs per case/batch, not global/persistent patient quotas. No automatic
retry, stale-node failover, durable membership recovery or hot model replacement
is provided. A recipient validates actual checkpoint/features and aligned IDs;
registry declarations alone do not prove model correctness or privacy. Generic
new-patient subset support does not authorize aligned training-row transfer; CDC
accepts aligned inference only. Prediction/identifier/update disclosure can leak
information. No routing accuracy experiment, retraining, calibration change or
frozen-test evaluation belongs to this architecture step.

Step3.6 tests authorized raw origin-local fields, not every possible representation
or embedding. Inner-selected clinical routing loses to both cost-aware fixed value
and the inner-selected metadata control in all3outer folds. Utility RÂ²/stability
gates fail; no robust clinical personalization,50k justification or GAT benefit is
established. A clinical ridge candidate has a small favorable aggregate comparison,
but choosing it on outer scores would be post-selection. All candidates remain
reported. Matched longer fusion fitting improves inner loss, especially logistic,
but does not prove complete convergence, repair calibration or remove teacher/
final shift. The reused pilot/overlapping fits/dependent origins are not new
independent clinical evidence.

The separately logged bandit observes one sampled action per patient/origin;
its162,084 records aggregate overlapping nested contexts, not162,084 independent
patients. It uses retrospective simulated outcomes and uniform propensities,
not deployed delayed verified diagnoses or equal-feedback comparison with the
full-information controls. No unselected reward enters fitting. One behavior
seed/linear action heads and imbalanced noisy losses limit conclusions; no
bandit-superiority or unbiased off-policy clinical claim. Inverse-propensity cap5
is inactive for uniform valid actions here. Raw clinical features are excluded
from Flower traffic, but trusted centralized diagnostics can read local files.
Predictions, identifiers and updates can disclose information. All frozen-test
bytes and historical checkpoints/reports remain preserved; test evaluations0.

The CDC three-agent amendment is development reuse, not a blinded new cohort.
Membership A metabolic/B lifestyle/C general health is explicitly user-fixed,
not selected on outer metrics. Private models are compatible byte-identical
copies; shared fusion/router weights are fresh. Original 21-feature duplicate
grouping and test boundaries remain unchanged; this does not exclude every
13-feature lookalike or identify unique individuals. Removing specialists changes
action sets, origin averaging, numerical dimensions and examples per fusion epoch;
six-versus-three score/time differences are not a controlled causal agent-count
or speedup estimate, especially because private fits are reused.

All numeric gates remain unchanged: >=3 varying origins now means all three.
Logistic routing's stable cost-quality gains and utility RÂ² .027â€“.042 do not meet
the .05 gate. Its absolute quality is worse than the residual-fusion condition;
improvements can reflect conditional fusion/calibration errors, not better
clinical peer knowledge. Fixed epochs with nine masks yield fewer optimizer
updates than the six-agent 36-mask run. No post-outcome schedule repair or gate
relaxation. Static/variable service schedules map the old slots exactly, but
availability fallback still induces variation that is not patient personalization.
Privacy, simulated heterogeneity, full-information utility supervision, survey
imbalance, dependent views, calibration shift and fixed-capacity limitations
from the historical CDC study below still apply. The test is not evaluated;
no promotion, 50k expansion, Qwen training or dynamic registry is implemented.

M5 Step 3.5 is a new UCI891 development benchmark, not a clinical validation.
The binary label combines prediabetes and diabetes; it is cross-sectional survey
status, not future disease incidence. BRFSS sampling/weights are not modeled.
Age is an ordinal band, not years. Identical-feature grouping prevents exact
feature duplicates crossing boundaries, but does not identify repeated people
with changed answers. Test is frozen by label-independent hash and never evaluated.
Pilot15kdevelopment rows yield ~10kouter fitting rows, compared with2500 on identical
assessments; this is fitting-size sensitivity, not a causal dataset-vs-architecture
comparison against529. Six honest domain models can still have limited conditional
complementarity. All use one fixed capacity/schedule, not a claim of optimal models.

Whole-stack utility cross-fitting and grouped calibration prevent known label
leakage, but teacher/final sample-size/calibration shift and one-outcome noisy BCE
gains remain. Bigger fits improve private predictors without meeting utility RÂ²/
personalization/scale gates. Mean-fusion routing offers some cost-aware STOP savings;
cost-aware fixed value captures most of that improvement. RÂ² is relative to a
training-origin/peer-mean null, not an assessment-fitted mean. Utility RÂ² scores
all potential non-self peers; action/oracle/deployment paths still exclude
temporarily unavailable specialists. Peer variation under
random availability is not automatically clinical personalization. Public domain
outages only disable specialists whose required fields are absent; models are not
deliberately weakened. Raw authorized local fields were not explored as router
inputs, so these results do not prove richer pre-query context cannot work.

Prediction correlation (~.375) differs from error correlation (~.950), which also
reflects shared labels/class imbalance. Threshold disagreement is small because
positives are rare; probability differences remain. Retrospective best-peer/group
statistics depend on observed outcomes and cannot serve as oracle routing. Group
analyses are descriptive dependent views, not demographic fairness conclusions.
The original neural, logistic and mean fusion conditions are frozen finite models,
not optimal Bayes fusion. Many fixed comparisons and three overlapping fitting
sets do not establish statistical/clinical superiority or external generalization.

The fixed-six Flower check really aggregates router arrays and consults selected
aligned IDs, but is not a working generalized federation, dynamic registry or Qwen
integration. Full-information preparation remains local and supervised; subsequent
evidence queries do not make it distributed/private retroactively. Central/FL
optimizer schedules differ. CUDA neural training and CPU histogram/Flower work are
distinguished; torch allocation excludes runtime contexts/board usage. No proven
privacy, driver/OS change,50kexpansion, default promotion or advanced routing.

M5 Step 3 removes meta-target leakage by refitting the complete specialist/fusion
stack inside target-fold complements. It still uses the same original training
cohort and overlapping small nested fits, not independent clinical data. Realized
BCE utility has one outcome per patient, substantial teacher/final shift and
variable calibration. Some utility-prediction R2 is positive, but action/cost gains
fail preregistered stability/origin gates; routing is not solved. Enrichment can
encode teacher-fold differences, and its training-only history resubstitution
summary differs from final cross-fitted history. Deterministic uncertainty
transforms add no new observations; constant capability flags have no identified
effect. No raw local risk summary/feature disclosure was added without a new
authorization design. All permitted information is available to the experimental
methods, while graph/fixed/confidence controls intentionally use restricted parts.

Step 3 fusion is uniformly pretrained for 40 full-batch steps, unlike Step 2's
joint schedules. Logistic's poorer scores in this condition cannot refute Step 2
or isolate fusion capacity/calibration causes. Central and Flower use the same
neural architecture/initialization but different optimizer reset/averaging
schedules. Genuine weight changes, negative gradient cosines and selected-only
consultation checks demonstrate functionality, not federated improvement. Central
preparation reads trusted local probability caches; later Flower queries verify
them, not retroactively make preparation federated. Replay call rates are logical
budgets. Batched live checks are not deployment latency. CPU training and a real
CUDA audit are distinguished; no acceleration/saturation claim. Historical
validation/test metrics and serving pointers are untouched; no privacy guarantee.

M5 Step 2 uses nested assessment drawn exclusively from the original training
partition; it does not create an independent cohort. Three outer fitting sets
overlap and origin views share patients. Many fixed candidates are explored with
one optimizer seed/fold, so no statistical significance is established. The
preregistered selected-pipeline gate failed and production defaults remain intact.
Logistic's gains confound capacity, residual, confidence and identity changes;
reduced capacity alone is not confirmed as the cause. Smaller local epochs also
use less compute; central optimizer schedules differ. Higher costs can justify
more STOP at those tariffs, but no useful personalized low-cost STOP improvement
is established by these candidates. Value targets use a meta-level in-sample
teacher on specialist OOF evidence and can overfit.

Step 2 feedback experiments use original-training-only experts and an outer-fold
guard holdout already used in the study. Their guarded scores are development
metrics, not independent feedback generalization. Labels are complete after
drain; missing action counterfactuals, delayed labels, high entropy and old rewards
remain limitations. Eight steps versus two is not a causal isolation of each of
those factors. Frozen/adaptive maximum budgets match; realized calls are measured.
These are simulated verified dataset outcomes, not deployed patient diagnoses.
Historical validation/test patients were not newly evaluated. No privacy or
clinical-validity guarantee, new SLM or generalized registry/fusion system is added.

M5 Step 1 adds diagnostic evidence, not a model improvement or independent
evaluation. [Full report](experiments/m5-step01-routing-diagnostics.md): local
origin gradients/updates conflict and centralized references reduce validation
BCE, but averaging, optimizer resets and local drift are not independently
identified causes. Random-initialization component freezes are restricted
controls. Always-consult routing occurs in all nine final seed/origin views;
fixed-peer routing occurs in eight (seed 44 origin A uses B and C). Statistical overfitting
and a causal calibration explanation remain unproven. Full diagnostic caches
are acquired via Flower then masked for replay, so logical policy budgets are
not deployed selected-only traffic. Identical-evidence fusion rankings vary by
metric. Sparse M4 replay mixes reward versions, but cap 2 is inactive at audited
parent states; theoretical clipping bias is not evidence that it caused this
run's small gains. Three seeds reuse the same 78 validation patients. CPU/CUDA
near-tie ranking can change AUROC despite very close probabilities. Historical
test evidence is retained, with no new historical-test metrics or tuning.

1. UCI 529 has 520 rows and only 78 test patients. Three model/fold seeds share the same fixed patient split; variability is optimization/fold variability, not external validity or split uncertainty. Origin-specific results are dependent views of the same patients and must not be pooled as independent observations. M1 test results were already known; this is not a newly blinded test cohort. M2 fitting/selection did not use those labels or tune configuration on test outcomes; regression reruns retained the frozen configuration.
2. This is one patient-aligned feature partition, with training labels at all clients. There are no independent hospitals, no complete vertical-FL protocol, and no external patient alignment mechanism. `case-XXXX` is a row pseudonym and does not establish anonymity.
3. Specialist columns and model parameters stay separate in runtime communication, but all files are accessible to a local trusted simulator. No filesystem isolation, differential privacy, secure aggregation, formal cryptographic confidentiality, or proven privacy guarantee is implemented. Predictions, IDs, hashes, gradients/model updates, logs, and saved artifacts can disclose information.
4. Training explores all OOF candidates and sends their probabilities through Flower to each origin's local training cache. This training cost/disclosure is larger than the deployed one-consultation budget. This is a full-information supervised objective, not online learning from selectively observed outcomes.
5. OOF specialist fitting uses an inner calibration holdout, reducing fold training data. Final specialists train on all training rows and use an OOF-fitted temperature. The distribution shift between fold outcomes and final inference can harm fusion. Calibration is estimated, not guaranteed; confidence is `2|pâˆ’0.5|`, not a validated probability of correctness.
6. Edge logits and contextual edge biases are not uniquely identifiable; adjacency alone does not explain decisions. The actual finite action scores and selected actions are reported, without invented natural-language reasons. Enumerating actions is practical for three clients but combinatorial for large networks.
7. The learned objective uses soft expected action loss; inference uses argmax. That mismatch, local-origin specialization, FedAvg interference, small-data overfitting, and weak fusion can degrade results. Experiments establish functioning training/communications, not superiority. Configuration was frozen before test evaluation; no test-driven tuning is reported.
8. Learned/random/confidence policies share the same selected fusion checkpoint and maximum budget of one. Confidence routing uses threshold 0.6 and a fixed cyclic destination. Random routing always uses the permitted budget; confidence may abstain. Full-evidence mean/all consultation use two peers and more information. The fixed-mean and fixed-all rows deliberately implement the same predictor via the consultation protocol and therefore are not independent algorithms.
9. Historical M1 masked-model metrics are on the same test IDs but use its original weights, training protocol, seed and torch 2.14.1+cpu. M2 calibrated specialists/fusion use torch 2.11.0+cu128. Cross-milestone differences are descriptive, not a controlled causal attribution.
10. Missing-node/domain/record paths are unit tested. Complete training requires all three clients. Missing-origin numeric 0.5 is an explicit flagged fallback, not a medical risk assessment. Large-scale failure/retry/security behavior is not tested.
11. Device audit confirms real CUDA execution on an RTX A4000. PyTorch peak allocated memory excludes CUDA contexts, reserved allocator memory, other processes and total board usage. nvidia-smi samples include desktop activity and cannot attribute utilization solely to this workload. Tiny tabular workloads and Flower orchestration are overhead dominated; no saturation claim is made.
12. Windows Ray support remains experimental and emits shutdown access-violation diagnostics despite successful completed Flower rounds. WSL/Ubuntu execution remains unverified. The M2 Python simulation launcher is deprecated in Flower 1.39; migration to the newer CLI with explicit CUDA runtime dependencies is a next task. AMP has a CUDA gradient/numerical smoke test, not a full multi-seed AMP benchmark.
13. Consultation inference times are whole-batch wall time divided by 78 patients, including real Flower round trips, model loading, GPU synchronization and coordinator work. They are not single-request deployment latency. Logical JSON sizes exclude Flower transport overhead.
14. M1/M2 had no SLMs. M3 adds frozen-Qwen coordinator and specialist interfaces with real typed tool dispatch. Learned multi-hop control, production serving, disease plugin registry, published packages, dataset deduplication and external clinical validation remain unimplemented.
15. Captured `pip freeze` files are environment evidence and include platform-specific/editable paths; they are not portable cross-platform lockfiles. Install from pyproject and the explicitly tested CUDA wheel instructions rather than blindly replaying a Windows lockfile on Linux. Budget zero has no routing gradient by construction; the router-change assertion is disabled for that configuration, while fusion still trains.
16. M3 evaluates 24 stratified validation records, not externally collected patients. These validation labels previously selected the M2 fusion checkpoint, so predictive metrics are descriptive development results, not an independent held-out estimate. No new historical-test evaluation or test-driven tuning occurred. Protocol/schema development used validation observations; the initial invalid-action run is preserved and reported.
17. The Qwen2.5-1.5B-Instruct Q4_K_M model uses frozen pretrained weights and constrained JSON action generation. This is real SLM tool dispatch but is not native Ollama message.tool_calls. Small-model decisions can violate prompts, decline complete inputs or vary despite temperature 0. Dynamic capability/budget schemas and application checks constrain execution; they do not prove autonomous reasoning quality. Logs or acknowledgements are not SLM training.
18. The initial unconstrained coordinator repeatedly attempted self-consultation; those requests were blocked. A constrained-schema rerun had an aligned client decline and emitted no prediction, rather than a fabricated risk. The stricter budget-matched rerun succeeded for every request. All outcomes are reported; robustness is not established by one successful rerun. A one-dimensional binary-risk fusion score is not a language-model diagnosis.
19. New-patient inference sends only explicitly authorized recipient fields, but the feature authorization map is trusted caller-provided metadata and not a signed/authenticated consent mechanism. Intake at A receives the user-supplied full case; only local required fields reach its classifier. Raw features are not sent to the SLM. Network authentication, secure transport, filesystem isolation and deployed consent management remain open work.
20. All M3 policies share SLM client interfaces. Latency includes their overhead plus real individual Flower round trips and the final acknowledgement. p50/p95/p99 are distributions of separate patient requests, not M2-style batch averages. The first request after warmup can still include actor/model loading; sequential policy order and KV caching can affect timing. Ollama GPU residency includes its runtime/model buffers and differs from torch peak allocation; total-board samples include desktop processes.
21. Full required classifier inputs are mandatory. No missingness-aware classifier was trained. Unknown categories, nonfinite values, unsupported fields and unauthorized subsets are rejected; numeric age validation does not establish out-of-distribution clinical validity. No new-patient external calibration or clinical deployment validation has been performed.
22. A fresh Windows Ray integration rerun failed before patient processing: default Plasma object storage requested approximately 4.377 GB and CreateFileMapping failed with Windows error 1450. M3 now explicitly caps Ray object storage at 256 MiB for small JSON messages; the fresh real-Qwen test passed after this change. The preserved failure log documents the issue. This reduces startup allocation pressure but does not establish that all Windows Ray failures are resolved; worker shutdown diagnostics persist.
23. M4 feedback is real UCI training labels with simulated delay, not newly collected clinical diagnoses. Adaptive/frozen shared policies start independently of stream labels. Frozen specialist OOF fits exclude each target patient but can use other patients appearing later in the stream: this is not strict temporal validation. M2's full-information reference already used these outcomes and is an optimistic, distinct training protocol.
24. M4 signatures authenticate trusted local simulation issuers and bind outcomes to requests. Keys/files/labels remain readable by the operator. This is not deployed clinician authentication, patient consent, secure transport, isolation or a proven privacy guarantee; deployment needs a verified-diagnosis service.
25. Selected-action clipped IPS replay is biased by clipping and mixes policies/fusion reward versions. The objective has no unconsulted outcomes. Arbitrary Qwen choices have unknown propensities and cannot replace the benchmark's audited neural behavior policy. Feedback-based neural adaptation is not language-model learning; Qwen and specialists stay frozen.
26. M4 uses one 364-patient stream and the reused 78-patient validation partition. Origin AUROCs are macro-averaged, not pooled. Validation views share patients; cumulative curves/replay reuse records. No external validity, statistical superiority or clinical improvement is established.
27. M4 approval permits validation BCE up to 0.02 above the best seen and limits update L2 to 0.1. This is a development guard, not medical safety. Rollback restores a checkpoint pointer, not external consequences. "Unnecessary consultation" is a selected-action realized BCE proxy against origin-only fusion, not clinical necessity.
28. Complete M4 training requires three responsive clients and compatible checkpoints. Explicit triggers are experiment episode cutoffs, not deployed background scheduling. Large-scale retry/recovery, key rotation, operational approval roles and concurrent checkpoint writers remain unimplemented. Logical byte counts exclude Flower transport overhead.
