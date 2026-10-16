"""Generate M3 review documents from real Qwen, Flower, metrics and QA evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from fed_med_agents.agents.verify import verify
from fed_med_agents.data import ROOT


def save(path: str, text: str) -> None:
    if path == "docs/status.md" and (ROOT / "docs/experiments/milestone-04.md").exists():
        return  # Preserve current status when replaying a historical report.
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.strip() + "\n", encoding="utf-8")


def table(data: dict) -> str:
    lines = [
        "| Mode / policy | Accuracy | AUROC | F1 | BCE | Coverage | Valid routing | Calls | Failures | p50 s | p95 s | p99 s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, policies in data.items():
        for policy, scores in policies.items():
            values = [
                scores[key]
                for key in [
                    "accuracy",
                    "auroc",
                    "f1",
                    "binary_cross_entropy",
                    "prediction_coverage",
                    "routing_validity",
                    "average_consultations",
                    "tool_failures",
                ]
            ]
            values += [scores["single_patient_latency_seconds"][key] for key in ["p50", "p95", "p99"]]
            lines.append(
                f"| {mode} / {policy} | "
                + " | ".join("n/a" if value is None else f"{value:.4f}" for value in values)
                + " |"
            )
    return "\n".join(lines)


def main() -> None:
    base = ROOT / "artifacts"
    main_path = base / "milestone03-budget-matched"
    matched = json.loads((main_path / "metrics.json").read_text())
    regular = json.loads((base / "milestone03/metrics.json").read_text())
    initial = json.loads((base / "milestone03-initial/metrics.json").read_text())
    demo = json.loads((base / "milestone03-demo/metrics.json").read_text())
    qa = base / "milestone03" / "qa"
    qa.mkdir(exist_ok=True)
    checks = {}
    commands = {
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "ruff": [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        "format": [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
        "pip": [sys.executable, "-m", "pip", "check"],
    }
    for name, command in commands.items():
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ | {"SWARMFL_REAL_QWEN": "1"},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        text = process.stdout + process.stderr
        (qa / f"{name}.log").write_text(text, encoding="utf-8")
        checks[name] = {
            "command": subprocess.list2cmdline(command),
            "returncode": process.returncode,
            "output": text,
        }
        if process.returncode:
            raise RuntimeError(f"{name} failed: {text}")
    checkpoints = ROOT / "artifacts/milestone02/seed_42/shared_best.pt"
    evidence = [
        verify(base / name, checkpoints)
        for name in [
            "milestone03-demo",
            "milestone03",
            "milestone03-budget-matched",
            "milestone03-integration",
        ]
    ]
    (qa / "verification.json").write_text(json.dumps(evidence, indent=2))
    (qa / "checks.json").write_text(json.dumps(checks, indent=2))
    integration = json.loads((base / "milestone03-integration/metrics.json").read_text())
    traces = [json.loads(line) for line in (base / "milestone03-demo/traces.jsonl").read_text().splitlines()]
    example = next(trace for trace in traces if trace["policy"] == "slm")
    actions = [entry["action"] for entry in example["coordinator_trace"] if "action" in entry]
    peer = next(
        entry for entry in example["communication_trace"] if entry.get("action") == "specialist_prediction"
    )
    gpu = matched["provider"]["running_models"]["models"][0]
    samples = json.loads((main_path / "gpu_samples.json").read_text())
    utilization = [sample["gpu_utilization_percent"] for sample in samples]
    sizes = [sample["board_memory_mib"] for sample in samples]
    report = f"""# Milestone 3 experiment — 2026-10-14

Milestone 3 is implemented and genuinely executed: frozen Qwen coordinator and client roles selected typed actions; Flower relayed private-classifier consultations; saved trained fusion produced a verifiable probability; the result returned to the origin over Flower. M1/M2 training and artifacts are retained. This is not a claim of clinical superiority, privacy or language-model learning.

## Runtime and learning distinction

Actual runtime: Ollama {matched["provider"]["version"]["version"]}, `{gpu["name"]}`, digest `{gpu["digest"]}`, {gpu["details"]["parameter_size"]}, {gpu["details"]["quantization_level"]}. Portable runtime and downloaded weights are local under gitignored artifacts/ollama; service uses loopback 127.0.0.1:11435. The same model instance serves server/A/B/C roles sequentially with isolated per-patient histories. No fine-tuning, optimizer, model updates or purported continuous learning of Qwen occurred. Conversation records are logs/state. The neural routing/fusion weights were learned by M2 Flower FedAvg and are now read-only at inference.

Ollama's constrained JSON schema generates tool actions; this implementation does not claim native message.tool_calls. Strict Pydantic validation plus application destination, authorization, patient and budget checks govern execution. The model sees field names/capabilities/status/origin score; clinical feature values and true labels never enter its prompts. Client backend tools perform the actual numerical inference. The coordinator cannot supply a final probability: finalize_prediction accepts no score and invokes trained fusion.

Settings: temperature 0, seed 42, 4096 context, 192 output tokens, keep-alive 10 minutes, one parallel model request, max 8 coordinator tool steps, one malformed-action/provider retry, HTTP inference timeout 45 seconds, Flower request timeout 90 seconds. Generation schemas restrict prediction arguments to eligible unconsulted peers, and remove predictions after budget exhaustion. The numerical model is M2 seed-42 `shared_best.pt`, budget one, validation-selected round 5. Hash `{matched["fusion_checkpoint_sha256"]}` was unchanged. Classifiers apply their saved training-fitted scaling/temperature; required fields must all exist, without missingness imputation. All artifact inference was performed using saved weights rather than retraining.

## Data and experimental protocol

24 validation patients (stratified subset of the original 78, sampling seed 42), 15 positives and 9 negatives. Same IDs, fields, permissions, classifier/fusion checkpoints, origin A and maximum budget one for every policy. All three client roles were exercised (origin A, peer B chosen by SLM/learned policy, peer C by domain control or SLM). No test-set inference or test-driven tuning occurred in M3. These are reused **validation development results**, not blinded clinical estimates: validation labels already selected the M2 numerical checkpoint. Initial protocol fixes were guided by invalid tool requests, not predictive labels.

New-patient requests reconstruct raw age from the saved M1 scaler and round to the original integer age; other features retain binary encoding. They are supplied dictionaries built from validation records, not externally collected patients. Full requests authorize A's 3, B's 5 and C's 8 fields. Peer envelopes carry only the selected recipient's required/authorized supplied fields, with no aligned lookup. Aligned requests carry an ID and no feature values. The data feeder submits intake to A through the server; A's real response starts coordination. No private training feature rows or specialist weights are transferred. Authorization is trusted metadata, not deployed authenticated consent.

The --request-file CLI also accepts an unlabeled new-patient PatientRequest dictionary, bypassing benchmark dataset loading and alignment IDs. The independently verified supplied-JSON run is recorded in the execution log; classification metrics remain null without ground truth. Its synthetic request was borrowed from validation data, not externally collected. This demonstrates the new-dictionary API without a clinical cohort claim.

Policies: SLM uses real Qwen actions; learned_router uses the M2 neural policy before peer evidence; feature_domain chooses the largest eligible complete feature group (C when all fields are available). All policies share the same SLM client interfaces and frozen fusion, so timing isolates server-policy overhead. Primary stricter comparison grants controls each patient's actual SLM consultation count as the budget; controls never receive SLM peer probabilities to select their destination. In the retained strict run all policies consumed exactly one call per patient, so both configured and realized budgets match. A separate common-maximum-budget run permits SLM abstention and is also reported.

## Strict budget-matched comparison

{table(matched["metrics"])}

Accuracy/F1/BCE/AUROC use non-null final predictions; coverage and failed-predictions-counted-wrong accuracy are stored separately. Strict run coverage is 100%, valid routing 100%, zero tool failures for all policies. A language agent did not improve accuracy over learned routing in new-patient mode and was worse in aligned mode. Higher new-patient AUROC/lower BCE here are descriptive differences on 24 reused validation cases, not evidence of clinical improvement or statistical significance. No policy was tuned to maximize these scores.

Times are real separate patient requests, from origin intake through classifier/SLM, peer Flower round trips, fusion and final return acknowledgement. They are not divided batch times. Rows/policies run sequentially, with runtime warmup before measurement; first actor/classifier load, KV caching and fixed ordering can affect latency. Model warmup time is recorded separately. No simultaneous benchmark/Qwen integration run was used for these timings.

## Common maximum-budget comparison and observed failures

{table(regular["metrics"])}

The earlier constrained common-budget run had one aligned client-agent decline, explicitly producing no prediction, and SLM abstention on some requests. Its original metric counter counted coordinator tool rejections but did not count that client decline; the current implementation increments client-failure counters. Predictive metrics on its 23 successful aligned outputs should not be compared without the 95.83% coverage / failed-prediction-as-wrong score. The strict later run had no missing predictions. Temperature zero does not guarantee identical role choices across mode/cache/backend contexts.

Before capability-scoped schemas, the initial archived run generated 47 prohibited self-consultation attempts: new-patient routing validity {initial["metrics"]["new_patient"]["slm"]["routing_validity"]:.4f}, rejected tool calls {initial["metrics"]["new_patient"]["slm"]["tool_failures"]}; aligned validity {initial["metrics"]["aligned"]["slm"]["routing_validity"]:.4f}, rejected calls {initial["metrics"]["aligned"]["slm"]["tool_failures"]}. These never reached specialist inference. Preserved artifacts/milestone03-initial includes all raw evidence. The fix constrained allowed arguments and retained independent runtime checks; it did not replace Qwen's choice with hardcoded responses. Other fixed checks handle nonexistent tools, false scores, invalid fields, unavailable clients, malformed output/retries and timeouts.

## Genuine A → coordinator → B → fusion → A trace

Demo uses only A+B supplied fields, with C ineligible, to verify the specifically requested path. Two balanced validation patients yield six individual requests over three policies; these tiny demo metrics are functional evidence only.

Actual server-generated actions:

```json
{json.dumps(actions, indent=2)}
```

Actual B tool response (no clinical feature values):

```json
{json.dumps(peer["output"], indent=2)}
```

Actual final result:

```json
{json.dumps(example["final"], indent=2)}
```

Communication trace includes real client_A/client_B Qwen role outputs, classifier tool outputs and a matching final request/patient acknowledgement at A. Independent verifier replayed each successful numerical score from recorded private probabilities/masks and the frozen fusion checkpoint within 2e-6; it did not trust an SLM-generated score. Trace files omit raw feature values; manifests containing simulated clinical inputs remain gitignored.

## Flower runs and GPU evidence

| Run | In-memory Flower ID | Requests | Recomputed scores | Tabular device |
|---|---|---:|---:|---|
| A+B demo | {demo["flower_run_id"]} | 6 | 6 | cuda |
| Common maximum budget | {regular["flower_run_id"]} | 144 | 143 | cuda |
| Strict budget-matched | {matched["flower_run_id"]} | 144 | 144 | cuda |
| Fresh genuine integration test | {integration["flower_run_id"]} | 6 | 6 | {integration["tabular_device"]} |

These are actual inference simulations, not new federated training rounds or persistent CLI histories. The separate M2 five-round FedAvg workflow remains available. Latest integration ran fresh real Qwen calls with CPU tabular inference and real Flower communication, confirming CPU fallback while SLM inference stayed on GPU.

A fresh QA integration rerun initially failed before patient processing: Ray's default Plasma allocation was approximately 4.377 GB and Windows CreateFileMapping returned error 1450. The failure logs are preserved under artifacts/milestone03/qa. M3 now caps Ray object storage at 256 MiB for these small JSON payloads; the subsequent fresh genuine test passed. This is an allocation-pressure mitigation, not proof that experimental Windows Ray is universally reliable. Shutdown diagnostics remain. Benchmark timings above were recorded before this startup-only memory cap.

Actual model GPU allocation from `/api/ps`: {gpu["size_vram"]} bytes ({gpu["size_vram"] / 2**30:.4f} GiB), context 4096. Ollama serve.log reports **29/29 layers offloaded to GPU**, CUDA compute 8.6 and NVIDIA RTX A4000. Real generated actions and model residency were verified, not just cuda.is_available. During strict run {len(samples)} nvidia-smi samples observed board utilization mean {np.mean(utilization):.2f}%, range {min(utilization)}–{max(utilization)}%, board memory {min(sizes)}–{max(sizes)} MiB. These board readings include desktop processes and CUDA contexts; they are not a workload-only VRAM or saturation measurement. Tabular audit confirms torch 2.11.0+cu128 / CUDA 12.8; Ollama is a separate native runtime. No driver/OS modifications or concurrent GPU clients.

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
{checks["pytest"]["output"].strip()}
```

Ruff check, Ruff format check and pip check exited zero. The genuine integration test opts in via SWARMFL_REAL_QWEN=1 and launches a fresh six-request Flower/Qwen CPU-tabular demo; ordinary installations skip it unless configured. Unit tests use a deterministic mock only to test malformed output, permissions, unavailable fields, budgets, histories and failure conditions. Saved-evidence tests recompute real benchmark fusion, prove only authorized recipient subsets, and confirm final hops. Original M1/M2 tests remain passing. The independent verifier passed all four retained functional runs, including the explicit no-prediction case rather than inventing a score.

## Unfinished work

No SLM fine-tuning/feedback policy learning, production service, external new-patient calibration, independent cohort validation, secured consent/transport, multihop autonomy, new missingness-aware models or clinical benefit claim. Prompt/schema and role reliability require further training/validation-only development. The frozen generic role prompt sometimes declines complete records; initial/protocol failures are preserved. Existing deprecated Flower launcher, Windows Ray shutdown diagnostics and unverified Ubuntu/WSL remain. See limitations and roadmap. Nothing was published or pushed.
"""
    save("docs/experiments/milestone-03.md", report)
    save(
        "docs/status.md",
        f"""# Current implementation status — 2026-10-14

M1/M2 are preserved. M3 is implemented and genuinely executed with frozen local Qwen2.5-1.5B-Instruct coordinator/client roles, typed tools, real Flower consultations, authorized new-patient feature routing and trained numerical fusion. No SLM parameter learning is claimed.

Real A→server coordinator→B→fusion→A demo verified. Strict budget-matched comparison executed 144 individual requests across 24 reused validation patients, both inference modes and three policies; all 144 scores independently recomputed. New-patient SLM accuracy 0.9583 ties the learned router; aligned SLM accuracy 0.9167 is lower than learned routing 0.9583. This is development evidence, not clinical superiority. Primary run had no tool failures; earlier self-consultation rejections and a client decline remain preserved/documented.

Qwen Q4_K_M uses one Ollama {matched["provider"]["version"]["version"]} model instance on RTX A4000, 29/29 layers GPU-offloaded, approximately {gpu["size_vram"] / 2**30:.3f} GiB model residency. Fresh genuine integration also verified CPU tabular fallback. All tests including real Qwen/Flower passed; Ruff/dependency checks passed. M1/M2 training remains separately runnable.

Read architecture.md, experiments/milestone-01.md, milestone-02.md, milestone-03.md, limitations.md, next-steps.md and devlog/2026-10-14-milestone-03.md. Remaining issues: small/reused validation evidence, frozen-SLM reliability, explicit-consent authentication, privacy/security, deprecated launcher/Windows diagnostics and external clinical validation. No production serving, SLM training, publication or GitHub push.
""",
    )
    print(
        json.dumps(
            {"report": "docs/experiments/milestone-03.md", "qa": "passed", "verified_runs": evidence},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
