# Milestone 3 experiment — 2026-10-14

Milestone 3 is implemented and genuinely executed: frozen Qwen coordinator and client roles selected typed actions; Flower relayed private-classifier consultations; saved trained fusion produced a verifiable probability; the result returned to the origin over Flower. M1/M2 training and artifacts are retained. This is not a claim of clinical superiority, privacy or language-model learning.

## Runtime and learning distinction

Actual runtime: Ollama 0.40.4, `qwen2.5:1.5b-instruct`, digest `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`, 1.5B, Q4_K_M. Portable runtime and downloaded weights are local under gitignored artifacts/ollama; service uses loopback 127.0.0.1:11435. The same model instance serves server/A/B/C roles sequentially with isolated per-patient histories. No fine-tuning, optimizer, model updates or purported continuous learning of Qwen occurred. Conversation records are logs/state. The neural routing/fusion weights were learned by M2 Flower FedAvg and are now read-only at inference.

Ollama's constrained JSON schema generates tool actions; this implementation does not claim native message.tool_calls. Strict Pydantic validation plus application destination, authorization, patient and budget checks govern execution. The model sees field names/capabilities/status/origin score; clinical feature values and true labels never enter its prompts. Client backend tools perform the actual numerical inference. The coordinator cannot supply a final probability: finalize_prediction accepts no score and invokes trained fusion.

Settings: temperature 0, seed 42, 4096 context, 192 output tokens, keep-alive 10 minutes, one parallel model request, max 8 coordinator tool steps, one malformed-action/provider retry, HTTP inference timeout 45 seconds, Flower request timeout 90 seconds. Generation schemas restrict prediction arguments to eligible unconsulted peers, and remove predictions after budget exhaustion. The numerical model is M2 seed-42 `shared_best.pt`, budget one, validation-selected round 5. Hash `143ecbb2a4a1c68cd1e456a90255b0879bf1d60c2480fb7c4eb5ae16447009bc` was unchanged. Classifiers apply their saved training-fitted scaling/temperature; required fields must all exist, without missingness imputation. All artifact inference was performed using saved weights rather than retraining.

## Data and experimental protocol

24 validation patients (stratified subset of the original 78, sampling seed 42), 15 positives and 9 negatives. Same IDs, fields, permissions, classifier/fusion checkpoints, origin A and maximum budget one for every policy. All three client roles were exercised (origin A, peer B chosen by SLM/learned policy, peer C by domain control or SLM). No test-set inference or test-driven tuning occurred in M3. These are reused **validation development results**, not blinded clinical estimates: validation labels already selected the M2 numerical checkpoint. Initial protocol fixes were guided by invalid tool requests, not predictive labels.

New-patient requests reconstruct raw age from the saved M1 scaler and round to the original integer age; other features retain binary encoding. They are supplied dictionaries built from validation records, not externally collected patients. Full requests authorize A's 3, B's 5 and C's 8 fields. Peer envelopes carry only the selected recipient's required/authorized supplied fields, with no aligned lookup. Aligned requests carry an ID and no feature values. The data feeder submits intake to A through the server; A's real response starts coordination. No private training feature rows or specialist weights are transferred. Authorization is trusted metadata, not deployed authenticated consent.

The --request-file CLI also accepts an unlabeled new-patient PatientRequest dictionary, bypassing benchmark dataset loading and alignment IDs. A separate real run (`12038324033062771851`) consumed a synthetic dictionary borrowed from validation data under a new unrelated patient ID, consulted B, returned 0.1673954725265503, and was independently replayed. It reported null accuracy/AUROC/F1/BCE because no true label was supplied. This demonstrates the API contract without claiming an external clinical cohort.

Policies: SLM uses real Qwen actions; learned_router uses the M2 neural policy before peer evidence; feature_domain chooses the largest eligible complete feature group (C when all fields are available). All policies share the same SLM client interfaces and frozen fusion, so timing isolates server-policy overhead. Primary stricter comparison grants controls each patient's actual SLM consultation count as the budget; controls never receive SLM peer probabilities to select their destination. In the retained strict run all policies consumed exactly one call per patient, so both configured and realized budgets match. A separate common-maximum-budget run permits SLM abstention and is also reported.

## Strict budget-matched comparison

| Mode / policy | Accuracy | AUROC | F1 | BCE | Coverage | Valid routing | Calls | Failures | p50 s | p95 s | p99 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned / feature_domain | 0.9167 | 0.9704 | 0.9333 | 0.2202 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6050 | 0.7083 | 0.7843 |
| aligned / learned_router | 0.9583 | 0.9704 | 0.9677 | 0.2016 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.7062 | 0.8104 | 0.8114 |
| aligned / slm | 0.9167 | 0.9704 | 0.9333 | 0.2149 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.1009 | 1.3590 | 1.3694 |
| new_patient / feature_domain | 0.9167 | 0.9704 | 0.9333 | 0.2202 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6052 | 0.7065 | 0.7823 |
| new_patient / learned_router | 0.9583 | 0.9704 | 0.9677 | 0.2016 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6077 | 0.7923 | 0.8106 |
| new_patient / slm | 0.9583 | 0.9852 | 0.9677 | 0.1893 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.6538 | 1.8090 | 1.8194 |

Accuracy/F1/BCE/AUROC use non-null final predictions; coverage and failed-predictions-counted-wrong accuracy are stored separately. Strict run coverage is 100%, valid routing 100%, zero tool failures for all policies. A language agent did not improve accuracy over learned routing in new-patient mode and was worse in aligned mode. Higher new-patient AUROC/lower BCE here are descriptive differences on 24 reused validation cases, not evidence of clinical improvement or statistical significance. No policy was tuned to maximize these scores.

Times are real separate patient requests, from origin intake through classifier/SLM, peer Flower round trips, fusion and final return acknowledgement. They are not divided batch times. Rows/policies run sequentially, with runtime warmup before measurement; first actor/classifier load, KV caching and fixed ordering can affect latency. Model warmup time is recorded separately. No simultaneous benchmark/Qwen integration run was used for these timings.

## Common maximum-budget comparison and observed failures

| Mode / policy | Accuracy | AUROC | F1 | BCE | Coverage | Valid routing | Calls | Failures | p50 s | p95 s | p99 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aligned / feature_domain | 0.9167 | 0.9704 | 0.9333 | 0.2202 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6038 | 0.7062 | 0.7841 |
| aligned / learned_router | 0.9583 | 0.9704 | 0.9677 | 0.2016 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.7064 | 0.8109 | 0.8113 |
| aligned / slm | 0.9130 | 0.9683 | 0.9286 | 0.2569 | 0.9583 | 1.0000 | 0.9583 | 0.0000 | 1.2677 | 1.4038 | 1.4093 |
| new_patient / feature_domain | 0.9167 | 0.9704 | 0.9333 | 0.2202 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6047 | 0.7053 | 0.7053 |
| new_patient / learned_router | 0.9583 | 0.9704 | 0.9677 | 0.2016 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.7061 | 0.7111 | 0.7116 |
| new_patient / slm | 0.9583 | 0.9852 | 0.9677 | 0.1939 | 1.0000 | 1.0000 | 0.9583 | 0.0000 | 1.6476 | 1.8188 | 1.8350 |

The earlier constrained common-budget run had one aligned client-agent decline, explicitly producing no prediction, and SLM abstention on some requests. Its original metric counter counted coordinator tool rejections but did not count that client decline; the current implementation increments client-failure counters. Predictive metrics on its 23 successful aligned outputs should not be compared without the 95.83% coverage / failed-prediction-as-wrong score. The strict later run had no missing predictions. Temperature zero does not guarantee identical role choices across mode/cache/backend contexts.

Before capability-scoped schemas, the initial archived run generated 47 prohibited self-consultation attempts: new-patient routing validity 0.0417, rejected tool calls 23; aligned validity 0.0000, rejected calls 24. These never reached specialist inference. Preserved artifacts/milestone03-initial includes all raw evidence. The fix constrained allowed arguments and retained independent runtime checks; it did not replace Qwen's choice with hardcoded responses. Other fixed checks handle nonexistent tools, false scores, invalid fields, unavailable clients, malformed output/retries and timeouts.

## Genuine A → coordinator → B → fusion → A trace

Demo uses only A+B supplied fields, with C ineligible, to verify the specifically requested path. Two balanced validation patients yield six individual requests over three policies; these tiny demo metrics are functional evidence only.

Actual server-generated actions:

```json
[
  {
    "name": "list_specialists",
    "arguments": {}
  },
  {
    "name": "get_specialist_capabilities",
    "arguments": {
      "specialist": "A"
    }
  },
  {
    "name": "get_specialist_capabilities",
    "arguments": {
      "specialist": "B"
    }
  },
  {
    "name": "request_specialist_prediction",
    "arguments": {
      "specialist": "B"
    }
  },
  {
    "name": "finalize_prediction",
    "arguments": {}
  }
]
```

Actual B tool response (no clinical feature values):

```json
{
  "request_id": "simulated-val-0499-new_patient-slm",
  "patient_id": "simulated-val-0499",
  "specialist": "B",
  "status": "ok",
  "probability": 0.22547264397144318,
  "confidence": 0.5490547120571136,
  "missing_features": [],
  "received_feature_names": [
    "polydipsia",
    "polyphagia",
    "polyuria",
    "sudden_weight_loss",
    "weakness"
  ],
  "agent_action": "run_local_inference",
  "provider_seconds": 0.158803700000135,
  "error": null
}
```

Actual final result:

```json
{
  "request_id": "simulated-val-0499-new_patient-slm",
  "patient_id": "simulated-val-0499",
  "origin": "A",
  "status": "ok",
  "probability": 0.1673954725265503,
  "consulted": [
    "B"
  ],
  "tool_failures": 0,
  "routing_valid": true,
  "latency_seconds": 1.7870352999998431,
  "policy": "slm"
}
```

Communication trace includes real client_A/client_B Qwen role outputs, classifier tool outputs and a matching final request/patient acknowledgement at A. Independent verifier replayed each successful numerical score from recorded private probabilities/masks and the frozen fusion checkpoint within 2e-6; it did not trust an SLM-generated score. Trace files omit raw feature values; manifests containing simulated clinical inputs remain gitignored.

## Flower runs and GPU evidence

| Run | In-memory Flower ID | Requests | Recomputed scores | Tabular device |
|---|---|---:|---:|---|
| A+B demo | 8624846614071447944 | 6 | 6 | cuda |
| Common maximum budget | 16867145065210701218 | 144 | 143 | cuda |
| Strict budget-matched | 18185838803515003331 | 144 | 144 | cuda |
| Fresh genuine integration test | 13332066215018280165 | 6 | 6 | cpu |

These are actual inference simulations, not new federated training rounds or persistent CLI histories. The separate M2 five-round FedAvg workflow remains available. Latest integration ran fresh real Qwen calls with CPU tabular inference and real Flower communication, confirming CPU fallback while SLM inference stayed on GPU.

A fresh QA integration rerun initially failed before patient processing: Ray's default Plasma allocation was approximately 4.377 GB and Windows CreateFileMapping returned error 1450. The failure logs are preserved under artifacts/milestone03/qa. M3 now caps Ray object storage at 256 MiB for these small JSON payloads; the subsequent fresh genuine test passed. This is an allocation-pressure mitigation, not proof that experimental Windows Ray is universally reliable. Shutdown diagnostics remain. Benchmark timings above were recorded before this startup-only memory cap.

Actual model GPU allocation from `/api/ps`: 1169382440 bytes (1.0891 GiB), context 4096. Ollama serve.log reports **29/29 layers offloaded to GPU**, CUDA compute 8.6 and NVIDIA RTX A4000. Real generated actions and model residency were verified, not just cuda.is_available. During strict run 131 nvidia-smi samples observed board utilization mean 33.27%, range 0–93%, board memory 2517–3127 MiB. These board readings include desktop processes and CUDA contexts; they are not a workload-only VRAM or saturation measurement. Tabular audit confirms torch 2.11.0+cu128 / CUDA 12.8; Ollama is a separate native runtime. No driver/OS modifications or concurrent GPU clients.

## Commands, tests and engineering evidence

```powershell
python scripts/bootstrap_ollama.py
python -m fed_med_agents.experiments.milestone03 --patients 2 --modes new_patient --demo --output artifacts/milestone03-demo
python -m fed_med_agents.experiments.milestone03 --patients 24 --modes new_patient aligned
python -m fed_med_agents.experiments.milestone03 --patients 24 --modes new_patient aligned --match-realized-budget --output artifacts/milestone03-budget-matched
$env:SWARMFL_REAL_QWEN='1'
python -m pytest -q
python -m fed_med_agents.agents.verify artifacts/milestone03-demo artifacts/milestone03 artifacts/milestone03-budget-matched artifacts/milestone03-integration
python scripts/report_milestone03.py
```

Windows commands used .venv/Scripts/python.exe. Exact QA commands/output are saved in artifacts/milestone03/qa. Model binaries/weights use artifacts/ollama, never source control. Re-run instructions and platform differences are in README. The native service was launched hidden with CREATE_NO_WINDOW and is not an installed OS/autostart service.

```text
....................................                                     [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\typer\__init__.py:24
  <venv>/site-packages/typer/__init__.py:24: DeprecationWarning: 'click.utils.get_binary_stream' is deprecated and will be removed in Click 9.0.
    from click.utils import get_binary_stream as get_binary_stream

.venv\Lib\site-packages\typer\__init__.py:25
  <venv>/site-packages/typer/__init__.py:25: DeprecationWarning: 'click.utils.get_text_stream' is deprecated and will be removed in Click 9.0.
    from click.utils import get_text_stream as get_text_stream

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
36 passed, 2 warnings in 25.07s
```

Ruff check, Ruff format check and pip check exited zero. The genuine integration test opts in via SWARMFL_REAL_QWEN=1 and launches a fresh six-request Flower/Qwen CPU-tabular demo; ordinary installations skip it unless configured. Unit tests use a deterministic mock only to test malformed output, permissions, unavailable fields, budgets, histories and failure conditions. Saved-evidence tests recompute real benchmark fusion, prove only authorized recipient subsets, and confirm final hops. Original M1/M2 tests remain passing. The independent verifier passed all four retained functional runs, including the explicit no-prediction case rather than inventing a score.

## Unfinished work

No SLM fine-tuning/feedback policy learning, production service, external new-patient calibration, independent cohort validation, secured consent/transport, multihop autonomy, new missingness-aware models or clinical benefit claim. Prompt/schema and role reliability require further training/validation-only development. The frozen generic role prompt sometimes declines complete records; initial/protocol failures are preserved. Existing deprecated Flower launcher, Windows Ray shutdown diagnostics and unverified Ubuntu/WSL remain. See limitations and roadmap. Nothing was published or pushed.
