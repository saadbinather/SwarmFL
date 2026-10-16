# Current implementation status — 2026-10-16

M5 **Step 8C finalization and research organization is locally complete**.
Apache-2.0 is owner-approved with complete license text and PEP 639 metadata. The
public research index and verified-results summary are intentional additions to
the approved allowlist. Canonical metadata
now targets `https://github.com/saadbinather/SwarmFL`. Local Git is initialized on
`main` with that `origin`; no commit has been created, pushed, uploaded, or
published. The audited proposed first commit has 139 allowlisted files (the approved
137 paths plus the two reviewed `research/` documents) and excludes
artifacts, environments, distributions, datasets, checkpoints, secrets, caches,
local paths, and generated outputs.

Flower 1.40.0 brings cryptography 50.0.2, eliminating the Step 8B cryptography
advisories. Isolated builds use setuptools 84. Flower still pins Ray 2.55.1
(PYSEC-2026-2273); SwarmFL does not use its vulnerable WebDataset decoder, but the
installed-package risk is explicitly accepted by the owner for trusted-local alpha
use only; an upstream fix remains preferred. Full regression:
**196 passed, 4 skipped**; Ruff, build, Twine, and dependency checks pass. A fully
isolated Python 3.12 wheel install outside the repository completed the real
three-client Flower quickstart, signed feedback, rollback, and external adapter.

No bundled third-party code restriction was found. GitHub-hosted Linux CI has not
run. Exact evidence is in the Step 8C devlog and ignored execution artifact.

M5 **Step 8B local release preparation is complete with a CONDITIONAL GO**.
Distribution `swarmfl` 0.1.0a1 preserves the `fed_med_agents` import package.
The wheel/sdist and Twine checks pass; a wheel-installed external-directory CPU
quickstart completed with three real Flower clients, selected-only calls, signed
feedback, version approval, and rollback. QA: **196 passed, 4 skipped**; Ruff and
dependency consistency checks pass. Final archive audit found no artifacts,
datasets, checkpoints, secrets, or machine paths. No publish, upload, push, commit,
frozen-test access, or historical artifact mutation occurred.

Publication is conditional on license/ownership approval, real repository/security
URLs, restored Git history checks, PyPI name confirmation, and resolution or formal
risk acceptance of Flower-constrained cryptography/Ray advisories. Windows Flower
simulation succeeds but uses a deprecated launcher and emits experimental Ray
shutdown diagnostics. Hugging Face has an unpublished synthetic card only; CDC and
Qwen assets are excluded. Full evidence is in the Step 8B devlog and ignored
`artifacts/m5-step08b-v1/execution.json`.

M5 **Step 8A final framework validation is complete**. A self-contained external
synthetic adapter added three heterogeneous sklearn specialists without modifying
the framework core: A logistic pipeline, B shallow tree and C second logistic
pipeline, each with a distinct two-feature view. The machine-readable protocol
was frozen before fitting at `artifacts/m5-step08a-v1/protocol.json`; the first
incomplete startup attempt and subsequent verification correction are retained.

Real Flower run **11798885370385872129** used exactly three CPU clients, two FedAvg
rounds and six local shared updates. It exercised STOP, one-peer selection, three
selected-only feedback batches, authorized new-patient fields, aligned ID-only
consultation, availability, unregister/re-register, budget/cost enforcement and
timeout failure without fallback. Application JSON totals were 54,922 request and
104,729 reply bytes; shared router/fusion states were 704/3,140 bytes. Logical
membership did not add or remove physical Flower nodes.

The router candidate passed by matching incumbent/fixed macro objective .608611;
the DeepSets candidate failed against incumbent .608611 and mean .607400 with
objective .608733. The accepted compatible r001/f000 pair was activated and exact
rollback restored r000/f000. Controlled fixtures separately verify rejection,
approval, restart recovery, incompatible activation and rollback. These gates do
not establish predictive superiority.

Independent replay passed 600 split/data records,360 decisions/fusions,324 signed
verified outcomes,6 local optimizations,2 averages and648 per-component consumption
links. Historical CDC and all prior evidence remain unchanged by 12,881 hashes;
the frozen CDC test files were never opened, loaded or rehashed. QA: **195 passed,
3 skipped**, two upstream warnings; Ruff and pip check pass. No new Markdown report
or devlog, core module, production default, CDC checkpoint, Qwen weight, package,
CI or release artifact was added. **Stop after Step 8A; Step 8B is not started.**

Earlier milestone statuses below retain historical scope and findings.

M5 **Step 7 continual-learning engineering is complete**. The reusable controller
connects selected-only Flower consultation, approved routing/dynamic fusion,
M4 signed delayed feedback, independent router/fusion updates, validation and
atomic compatible pair activation/rollback. No new routing algorithm/private
training, dependencies, specialists or production promotion.

| Cycle | Verified records | Previous objective | Fixed objective | Active objective | Router update | Fusion update |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 516 | 0.348792 | 0.349695 | 0.348702 | Rejected | Activated |
| 2 | 516 | 0.368325 | 0.367517 | 0.368325 | Rejected | Rejected |
| 3 | 516 | 0.327500 | 0.330738 | 0.327500 | Rejected | Rejected |

Each cycle uses 192 new feedback patients / 3 origin views; every tenth outcome is
missing, yielding 516 fresh verified origin records/cycle (172/origin). Across
cycles: 576 distinct feedback patients, 1,728 predictions, 1,548 verified origin records,
180 missing. Each component consumes observations independently once. Validation
cohorts are 1,024 patients/cycle with nonoverlapping duplicate groups, excluded from
feedback and all initial expert/router/fusion fitting. Historical source partition
reuse and synthetic chronology prevent any prospective clinical-evidence claim.
The tiny cycle 1 fusion gate gain does not establish generalizable improvement;
different cycle populations are not a learning trend. All router candidates and
fusion cycles 2/3 are rejected without relaxed thresholds.

Final Flower run 16942607461224652031:3 physical clients,1 GPU worker,6 FedAvg
aggregates/18 CUDA local updates across 3 sequential cycles. Exactly 9 selected-only
private batch consultations; no private weights/raw feature rows/labels in public
consultation/prediction payloads. Both components keep separate checkpoints.
Actual research activation r000/f000->r000/f001, followed by exact rollback to
r000/f000. Restart reauthenticates ledger/context/consumption; missing outcomes
never train. Crash-mid-round exactly-once federation and physical client churn
remain unsupported.

Final replay: 1,728 decisions/fusions, 9 private batches,18 local optimizations,
6 aggregates, 756 scalar metrics; all 9,169 historical hashes unchanged; frozen test
unevaluated. QA 179 passed, 2 skipped, 2 upstream Typer warnings; Ruff 94 files and
pipcheck passed. Existing registry Flower integration reran separately in new
output. Server work 57.815s, measured peak local torch allocation 18,530,304 bytes;
CUDA RTX A4000 actual updates audited. No memory/saturation/clinical claim.

Detailed protocol, curves, metrics, provenance, gates, source changes, commands
and limitations:artifacts/m5-step07-v1/execution.json and summary.json.
No new Markdown devlogs/reports were created. **Stop after Step 7.** Earlier
milestone statuses below retain historical scope and findings.


M5 **Step 6 dynamic fusion is complete**. New reusable mean, training-OOF-reliability
weighted mean and lightweight DeepSets accept changing registered prediction sets
with stable identities and exact contracts. No router/private expert/Qwen training,
new agents, dependencies, physical membership changes or production promotion.
Existing dense residual/logistic/masked mean APIs/checkpoints remain unchanged.

| Evidence | Existing residual | Mean | Weighted | Set central | Set Flower |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 0.342116 | 0.342054 | 0.342054 | 0.342054 | 0.342054 |
| B | 0.392883 | 0.392435 | 0.392435 | 0.392435 | 0.392435 |
| C | 0.352827 | 0.352185 | 0.352185 | 0.352185 | 0.352185 |
| AB | 0.338775 | 0.353562 | 0.352991 | 0.338618 | 0.340119 |
| AC | 0.320793 | 0.332021 | 0.331996 | 0.321110 | 0.322210 |
| BC | 0.350729 | 0.361817 | 0.361454 | 0.350698 | 0.352385 |
| ABC | 0.329449 | 0.345023 | 0.344550 | 0.320283 | 0.321116 |

Assessment BCE above uses the same frozen19,815 development patients and separately
reports evidence subsets. New singleton fallback is exact; historical residual
can transform it. Existing residual60 trained 9 STOP/one-peer masks; ABC is an
out-of-training-mask diagnostic. New shared set models trained all 12 origin/subset
masks. It is not a matched-compute/capacity superiority comparison. Full AUROC,
PR-AUC, calibration, per-origin and latency evidence remains in ignored artifacts.
Training 162,850 private OOF rows, validation 20,154, assessment 19,815; existing
Step 5 grouped boundaries retained. No held labels fit reliability/shared weights.

Final genuine Flower run5515635167392953571:3 physical clients,1 CPU actor,
3 rounds/9 fusion-only updates; drift3.348779. Actual A/B->A/B/C->A/C->A/B/C->A/B
lifecycle, selected-only private inference and 18 fusion outputs pass. Shared
contracts checked before training; private classifiers/calibrators and routers
remain frozen. Research candidates pass fixed guard gates and versions advance
v000->v001->v002, then actually rollback v000; no production promotion.

CUDA central fit 52.512s; pipeline 58.152s; peak allocation 32,013,312 bytes.
Replay passes 12private predictions,1 central/9 local optimizer reproductions,
3averages,840 scalar metrics and 18 lifecycle fusion outputs. All 9,048 historical
hashes match and frozen test evaluations 0. QA:169 passed,2 skipped,2 upstream
Typer warnings; Ruff91files and pipcheck passed. Fresh historical registry Flower
regression passed. QA results/failures/exact changed files
are recorded in artifacts/m5-step06-v1/execution.json; no new Markdown reports.
**Stop after Step 6.** Previous statuses below retain historical scope/outcomes.


M5 **Step 5 trainable routing and continual-feedback engineering is complete**.
Reusable fixed/contextual MLP/existing graph policies share registry-aware routing,
OOF supervised fit, observed-only update, serialization and M4 version management.
Exactly A/B/C; no new bandit family, GAT, Qwen training or production default change.
New development-only scale authorization is separate from old failed expansion
gates. Protocol frozen before fitting under artifacts/m5-step05-v1/protocol.json.

| Routing rows | Fixed objective | MLP objective | Graph objective | MLP utility R² |
| ---: | ---: | ---: | ---: | ---: |
| 15,000 | 0.350173 | 0.350328 | 0.350800 | 0.000158 |
| 50,000 | 0.350173 | 0.349980 | 0.350704 | 0.000611 |
| 100,000 | 0.350173 | 0.350096 | 0.350173 | 0.000262 |
| 160,000 | 0.350173 | 0.350110 | 0.350173 | 0.000003 |

Variable/.02 assessment macro above uses three repeated group-held splits with
independent fitting/validation/assessment within each split. Upstream pools
162,850/162,768/162,776; guards 20,154/19,991/20,077; assessments 19,815/20,060/19,966.
Nested routing prefixes keep whole duplicate groups:15k/50k are 15,023/50,004 in
split 2 and exact requested counts elsewhere. All size comparisons share each
split's cold-fit upstream evidence/fusion. This is routing-supervision scaling,
not a total pipeline data-budget comparison. The three assessment sets can overlap
across replicates; no independent-clinical-cohort claim.

All sizes fail meaningful-gain/R²/all-origin diversity gates. Small aggregate MLP
gains at 50k/100k/160k are reported without outer selection;100k/160k do not improve
15k objective and R² consistently across splits. At160k, MLP accuracy.862818,
AUROC.785127, PR-AUC.373214, calls.559161; fixed accuracy.862586, AUROC.786404,
PR-AUC.369236, calls.580984. Full per-origin/cost/scenario/calibration/action results
and fitting timings are in ignored metrics/results/summary evidence.

Full study pipeline 871.066s; neural fits used CUDA, peak allocated 664,392,704 bytes, matrix audit 3680.
Flower 8165990254838736479:3 physical clients, 3 rounds/9 updates, 288 observations of
96 training patients, routing drift .113123. Live B outage masks decisions; actual
selected private inference/fusion, signed delayed ledger replay, insufficient-
feedback retention, immutable versions and rejected approval are exercised.
Feedback validation objective previous.347985/fixed.348201/updated.356184: reject.
Actual pointer stays v000; unit tests separately advance and roll back distinct
eligible versions. No production promotion or automatic improvement claim.

QA: 154 passed, 2 skipped,2upstream Typer warnings; Ruff 88 files and pipcheck passed.
Historical registry Flower regression freshly reran in the new output only.
Numerical replay passed 144 private predictions, 12 fusion outcomes, 20,106 scalar
metric fields, 9 local optimizations and 3 FedAvg averages. All 8,614 historical
hashes and 222 prepared-file hashes match; frozen test evaluations 0.

Execution/failures/exact changed files:artifacts/m5-step05-v1/execution.json.
No new report/devlog Markdown was created, per user instruction. **Stop Step 5.**
The following status remains historical evidence for earlier steps.


M5 **Step 4 dynamic specialist registry** adds disease-independent immutable
declarations, register/unregister/discovery, availability/cost updates, compatible
task/label/output/model/feature validation, eligibility and budgeted authorized
consultations. The supported CDC Flower coordinator resolves selected recipients
through registry bindings and checks actual adapter/checkpoint/case identities.
The inference-only lifecycle demonstration uses exactly A/B/C and existing saved
private models/fusion/graph/fixed-value data. No training or accuracy improvement
claim; no frozen-test evaluation or model/default change.

Inference-only Flower run **5093394496059785868** verified six logical lifecycle
transitions, five actual private predictions, four fusion replays and six
pre-dispatch rejections. Three physical clients remain connected throughout;
training updates0. Fresh regression: **144 passed, 2 skipped**, Ruff lint/format
85files and pipcheck passed. All8,390 prior artifact/report hashes match; frozen
test bytes unchanged. First runtime attempt failed on missing psutil; corrected
to standard-library Python allocation tracing, with failure evidence preserved.

Logical membership is dynamic; physical Flower membership stays fixed at three.
The saved fusion remains three-input. The registry is not a persistent authenticated
serving/consent system, and historical benchmark handlers remain separate. New
implementation/tests are the only new source files; per-user instruction there
are no new milestone protocols, benchmarks, feasibility reports or devlogs.
Detailed execution, failed attempts, numerical checks, QA and preservation hashes:
artifacts/m5-step04-v1/execution.json. Stop after Step 4. Earlier milestone status
below is retained historical evidence.

M5 **Step 3.6 is complete experimentally; richer context is not validated as a
useful personalization improvement** on this pilot. Exactly A metabolic, B
lifestyle and C general health remain fixed. The 15-epoch fusion audit exactly
reproduces prior fits; inner selection prefers residual60 in all folds. Authorized
local raw fields are compared with matched metadata models, two-stage expected
gain/ranking, pairwise learning-to-rank and separately sampled observed-only
bandit feedback. No hidden peer predictions enter prospective context.

Inner-selected clinical BCE+cost **.356859** versus fixed value **.356182**;
fold improvements **-.000560/-.000836/-.000634**. Utility RÂ²
**-.006698/.001156/-.001535** and only one varying origin/fold fail the frozen
gates. Clinical routing also loses to the inner-selected metadata reference in
each fold. Individual candidate averages are retained, not promoted as outer
winners. Observed-only bandit objective **.362221** is not evidence of improvement.
No50k expansion, GAT, default promotion, registry, Qwen training or Step4.

Actual Flower run **18290472637018263396**, exactly3 clients,3FedAvg rounds/9
updates and6 selected-peer query batches. Independent replay verifies60 fusion
outcome arrays,27,864 metric fields,162,084 sampled observations,all9 local
updates/3aggregations. All8,058 preserved hashes match; frozen-test evaluations 0.
Fresh QA **113passed,2skipped**, Ruff lint/format83files and pipcheck passed.
[Full protocol/results](experiments/m5-step036-context-routing.md),
[execution log](devlog/2026-10-14-m5-step036.md). Stop after3.6; earlier status below
is retained milestone history.

The requested CDC amendment is complete: **exactly three specialists**, A metabolic,
B lifestyle and C general health. All 576 compatible nested private checkpoint
files are byte-identically reused; all three-agent fusion and routing models are
fresh fits. Original context/target/group boundaries, frozen test bytes and
numerical acceptance gates remain unchanged. No historical UCI529 model changes.

Pilot variable-service objective at .02, selected versus always-best/fixed-value:
original **.357150/.361401/.357775**; logistic **.381983/.397418/.391216**;
mean **.361237/.370030/.361420**. Logistic's gains are real in this fixed fusion
condition, but its fold utility RÂ² **.04201/.03209/.02737** misses the unchanged
.05 gate. Other conditions also fail; no promotion, 50k expansion or Step4.

Actual Flower run **421462273542764912**, three clients, three rounds/nine updates.
New verification replays 576 private models, 144 fusion states, 154,872 metric
fields and all nine local updates; all 96 neural/logistic fusion fits are
independently reproduced from cold seeds. A separate fresh historical six-agent
replay passed without overwriting its prior summary. All **6,776** preserved
artifact/report hashes match. QA **105 passed, 2 skipped**, Ruff lint/format
(81 files) and pip check passed. Reports:
[amended benchmark](experiments/m5-step035-three-agent-benchmark.md),
[feasibility](experiments/m5-step035-three-agent-feasibility.md),
[protocol](experiments/m5-step035-three-agent-protocol.md),
[execution log](devlog/2026-10-14-m5-step035-three-agent.md).
Below is preserved historical status, including the earlier six-agent CDC study.

M5 **Step 3.5 is complete as a separate CDC development benchmark**. Official
UCI891253,680Ã—21 binary target is verified;50,861 new test records are frozen and
unevaluated. A15k development pilot compares roughly10k versus2500 fitting rows
with the same grouped assessments and six legitimate, independently calibrated
specialists. Identical-feature groups cannot cross any fitting/calibration/
selection/assessment boundary. Whole-stack utility cross-fitting prevents target
labels entering upstream specialist or fusion fitting.

Full-pilot variable-service/.02 mean-fusion selected objective .362145 versus
always-best-peer .369197; cost-aware fixed value already .362780. Mean-fusion gain
RÂ² .00029/.00085/.00257 fails the frozen gate, and peer variation is concentrated
in only two origins/fold. Other fusion conditions also fail their gates. Larger
fitting sets improve specialist BCE but do not establish robust personalized peer
selection here. **Stop scaling: no50k, advanced router, registry or promotion.**

Genuine fixed-six Flower run11629480563570813115 completed3FedAvg rounds/18router
updates and30selected-only batch queries. Verification replays1,152 private
model predictions,144fusion states,309,744metric fields and all18local updates;
12calibrated models independently refitted. All4,995prior hashes match, frozen
test hash unchanged. QA **97passed,2skipped**; Ruff lint/format78files and pip
check passed. Old optional historical-validation runtime benchmarks stay disabled;
the new six-client integration genuinely executed. Reports:
[benchmark](experiments/m5-step035-benchmark.md),
[feasibility](experiments/m5-step035-routing-feasibility.md),
[protocol](experiments/m5-step035-protocol.md),
[execution log](devlog/2026-10-14-m5-step035.md).
The following sections retain prior milestone history and immutable evidence.

M5 **Step 3 is complete experimentally; routing is not solved**. Fully cross-fitted
specialist-to-fusion utility targets and new nested folds use only the original
364 training patients. Frozen residual/logistic conditions compare graph,
training-best peer, confidence, ridge, MLP and shallow-tree routing with five
feature sets and three costs. The inner-selected residual objective .30589 is
worse than fixed peer .30349; logistic-condition .43136 improves mean .43590 but
fails fold/origin stability gates. No default or pointer is promoted. These are
new folds/fits, not a direct before/after comparison with earlier scores.

Real Flower run 8790470071382857781 completed 30 routing-only rounds/90 updates,
with real selected-peer consultations and masked prediction replay. All 4,166
prior artifact/report hashes match. Independent verification refits all 72 OOF
columns and checks private predictions, upstream exclusions, utility labels,
metrics, fusion states, compatible arrays and local updates. Fresh full regression:
**84 passed,2 skipped**, Ruff lint/format71files and pip check passed. The optional
skipped integrations would reopen old runtime benchmarks; new training-only
Flower integration was actually run. See
[routing report](experiments/m5-step03-intelligent-routing.md),
[features](experiments/m5-step03-routing-features.md),
[protocol](experiments/m5-step03-protocol.md) and
[execution log](devlog/2026-10-14-m5-step03.md) for complete evidence and fresh QA.
Stop after Step 3. The following Step 2/M1â€“M4 descriptions are retained history.

M5 **Step 2 is complete as separate experiments; the original default is retained**.
Three nested outer folds use only the original 364 training patients, with inner
selection and independently refitted/calibrated OOF experts. No new historical
validation/test evaluation or tuning occurred. Genuine Flower evidence verifies
348 FedAvg/FedProx aggregations and 1,044 client updates, plus two 120-patient
observed-feedback streams. All 1,177 preserved artifact/report hashes match.
Final regression QA: **74 passed, 2 optional runtime tests skipped**; Ruff lint/
format (65 files) and pip check passed. All 18 OOF columns were independently
refitted/reproduced, and private-checkpoint assessment predictions match.

New nested avg10 accuracy/BCE: .88367/.27260; logistic fusion .90743/.23599,
and .90287/.23265 on exactly avg10's evidence. Logistic is promising, but capacity,
identity/confidence removal and residual changes are confounded. The inner-selected
pipeline fails the preregistered all-fold improvement gate, so no model is promoted.
One epoch underfits; five epochs modestly improve mean BCE; FedProx/persistent Adam
show no decisive gain. Higher costs increase STOP but worsen the primary .02-cost
objective. More feedback steps change BCE only slightly, with no accuracy gain.

Read [optimization](experiments/m5-step02-optimization-repair.md),
[routing](experiments/m5-step02-routing-repair.md),
[fusion](experiments/m5-step02-fusion-ablation.md),
[protocol](experiments/m5-step02-protocol.md) and
[execution log](devlog/2026-10-14-m5-step02.md). Dynamic registry, variable-client
routing, generalized fusion and Qwen fine-tuning are not implemented. Stop after
Step 2 until further explicit instructions.

## Historical Step 1 and M1â€“M4 status

M5 **Step 1 diagnostics only** is implemented separately from production models.
Three seeded training/validation ablations reproduced the historical M2 final
shared weights exactly. Joint training beat both random-initialization component
freezes in validation BCE. Both centralized diagnostic references improved BCE
relative to Flower training in every seed, including matched global steps;
opposing origin gradients/local fusion updates support local-objective conflict,
without identifying averaging alone as the cause. Joint policies always consulted;
eight of nine seed/origin views used a fixed peer, while seed 44 origin A used
both B and C. Identical-evidence mean comparisons differ
by metric. Default M4 replay clipping was inactive at inspected parent states;
feedback is sparse and contains older reward versions, while policy changes are
small. Staleness is not established as their cause. No historical test reevaluation,
default model replacement, new specialist registry or Qwen training occurred.
Read the [complete M5 Step 1 report](experiments/m5-step01-routing-diagnostics.md)
and [execution log](devlog/2026-10-14-m5-step01.md) for exact results, fresh checks,
reproduction evidence and limitations. Independent continuation replay verified
84 states and 45 Flower aggregations; fresh QA passed 54 tests with two optional
runtime integrations skipped. Earlier full integration QA passed 56. All 628
preserved artifact hashes match. Step 2 results are recorded above.

M1â€“3 code paths, checkpoints and historical reports remain preserved. M4 is implemented and genuinely executed: signed prediction/outcome separation, delayed verified training feedback, selected-action routing/fusion learning, three-client Flower FedAvg, validation approval, immutable versions and rollback. Qwen and private classifiers remain frozen.

The M4 CUDA stream processed 364 unique training patients, with seven approved rounds and 21 local updates. Independent audit verified 1,092 policy events, real delayed source labels, selected-only consultations, numerical predictions and FedAvg against all client shared checkpoints. Initial-to-approved shared L2 was 0.04321. Adaptive/frozen accuracy both 0.7885; BCE 0.44956 versus 0.45021. Validation mean BCE 0.338889â†’0.338277. These small descriptive changes do not establish superiority. No new historical-test evaluation occurred.

Real frozen Qwen invoked get_routing_recommendation on approved v007, selected A from origin B, performed the authorized Flower consultation and returned trained numerical fusion to B. Independent replay confirmed the score. This synthetic API request had no label and all classification metrics remained null; its acknowledgement never entered feedback training. Qwen GPU residency was approximately 1.089 GiB under Ollama 0.40.4. A fresh 24-episode CPU integration skipped an insufficient-data trigger, approved one real three-client update and used it in a subsequent consultation. All 52 tests passed, including separate M3 Qwen regression; Ruff lint/format and pip check passed.

Read [M4 experiment](experiments/milestone-04.md), [feedback protocol](feedback-learning.md), [execution log](devlog/2026-10-14-milestone-04.md), architecture, limitations and roadmap. Existing M1/M2/M3 reports retain historical results. Remaining issues include retrospective OOF/reused validation, biased/stale replay, deployed outcome verification/consent, privacy/security, frozen-SLM reliability, operational scheduling/concurrency, deprecated launcher/Windows diagnostics and independent clinical validation. No production serving, SLM training, publication or GitHub push.
