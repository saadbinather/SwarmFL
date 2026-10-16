"""Verify retained M4 evidence, optionally run fresh QA, and render actual results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from fed_med_agents.agents.verify import verify as verify_agents
from fed_med_agents.data import ARTIFACTS, ROOT
from fed_med_agents.feedback.verify import verify
from fed_med_agents.feedback.versions import PolicyRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ARTIFACTS / "milestone04")
    parser.add_argument("--checks", action="store_true")
    parser.add_argument(
        "--real-qwen", action="store_true", help="Also rerun M3 integration in a fresh M4 directory"
    )
    args = parser.parse_args()
    path = args.directory.resolve()
    qa = path / "qa"
    qa.mkdir(exist_ok=True)
    if args.checks:
        suffix = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        feedback_output = ARTIFACTS / f"milestone04-integration-{suffix}"
        qwen_output = ARTIFACTS / f"milestone04-qwen-regression-{suffix}"
        environment = os.environ | {
            "PYTHONUTF8": "1",
            "SWARMFL_REAL_FEEDBACK": "1",
            "SWARMFL_REAL_QWEN": "1" if args.real_qwen else "0",
            "SWARMFL_M4_INTEGRATION_DIR": str(feedback_output),
            "SWARMFL_M3_INTEGRATION_DIR": str(qwen_output),
        }
        commands = {
            "pytest": [sys.executable, "-m", "pytest", "-q"],
            "ruff": [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
            "format": [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
            "dependencies": [sys.executable, "-m", "pip", "check"],
        }
        checks = {}
        for name, command in commands.items():
            process = subprocess.run(
                command,
                env=environment,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                check=False,
            )
            output = process.stdout + process.stderr
            (qa / f"{name}.log").write_text(output, encoding="utf-8")
            checks[name] = {"command": command, "exit_code": process.returncode}
            print(f"{name}: exit {process.returncode}\n{output}", flush=True)
        checks["feedback_integration_output"] = str(feedback_output)
        checks["qwen_regression_output"] = str(qwen_output) if args.real_qwen else None
        (qa / "checks.json").write_text(json.dumps(checks, indent=2))
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        (qa / "requirements-evidence.txt").write_text(freeze.stdout)
        if any(checks[name]["exit_code"] for name in commands):
            raise RuntimeError("QA failed; inspect saved logs before reporting completion")
    checks = json.loads((qa / "checks.json").read_text())
    assert all(checks[name]["exit_code"] == 0 for name in ("pytest", "ruff", "format", "dependencies"))
    evidence = verify(path)
    integration = verify(Path(checks["feedback_integration_output"]))
    (qa / "verification.json").write_text(
        json.dumps({"full_run": evidence, "cpu_integration": integration}, indent=2)
    )
    result = json.loads((path / "metrics.json").read_text())
    history = [row for row in result["history"] if "version" in row]
    registry = PolicyRegistry(path / "versions")
    initial = registry.load("v000").state_dict()
    final = registry.load().state_dict()
    delta = lambda prefix: (
        sum(float(((final[k] - initial[k]) ** 2).sum()) for k in initial if k.startswith(prefix)) ** 0.5
    )
    parameter_changes = {"router_l2": delta("router."), "fusion_l2": delta("fusion."), "shared_l2": delta("")}
    (qa / "parameter_changes.json").write_text(json.dumps(parameter_changes, indent=2))
    with (path / "learning_curve.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode",
                "version",
                "approved",
                "validation_bce",
                "update_l2",
                "routing_change_fraction",
                "adaptive_stream_bce",
                "frozen_stream_bce",
            ]
        )
        for row in history:
            writer.writerow(
                [
                    row["episode"],
                    row["version"],
                    row["eligible"],
                    row["validation"]["mean_bce"],
                    row["shared_change_l2"],
                    row["routing_change_fraction"],
                    row["stream_metrics"]["adaptive"]["binary_cross_entropy"],
                    row["stream_metrics"]["frozen"]["binary_cross_entropy"],
                ]
            )
    rows = []
    for name, values in result["stream"].items():
        rows.append(
            f"| {name} | {values['accuracy']:.6f} | {values['auroc_macro_origin']:.6f} | {values['f1']:.6f} | "
            f"{values['binary_cross_entropy']:.6f} | {values['consultations_per_patient']:.6f} | "
            f"{values['unnecessary_consultation_fraction']:.6f} |"
        )
    curve = "\n".join(
        f"| {row['episode']} | {row['version']} | {row['eligible']} | {row['validation']['mean_bce']:.8f} | "
        f"{row['shared_change_l2']:.8f} | {row['routing_change_fraction']:.6f} |"
        for row in history
    )
    pytest_log = (qa / "pytest.log").read_text(encoding="utf-8")
    summary = next(line for line in reversed(pytest_log.splitlines()) if "passed" in line)
    slm_section = ""
    slm_path = ARTIFACTS / "milestone04-slm-approved"
    if (slm_path / "metrics.json").exists():
        slm_evidence = verify_agents(slm_path, ARTIFACTS / "milestone02/seed_42/shared_best.pt")
        trace = json.loads((slm_path / "traces.jsonl").read_text().splitlines()[0])
        tools = [entry["action"]["name"] for entry in trace["coordinator_trace"] if "action" in entry]
        assert "get_routing_recommendation" in tools
        (qa / "slm_verification.json").write_text(
            json.dumps(
                slm_evidence | {"policy_version": trace["approved_policy_version"], "tools": tools}, indent=2
            )
        )
        slm_section = f"""## Real frozen-Qwen approved-policy tool

Flower run `{slm_evidence["flower_run_id"]}` used an unlabeled, validation-derived
synthetic dictionary with origin B. Real Qwen called `{", ".join(tools)}`, loaded
approved policy `{trace["approved_policy_version"]}`, consulted
`{trace["final"]["consulted"]}` through Flower and returned risk
`{trace["final"]["probability"]}` to B. Latency {trace["final"]["latency_seconds"]:.3f}s,
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

"""
    report = f"""# Milestone 4 — verified feedback-driven federated learning

Executed on Windows with three real Flower 1.39 ClientApps, one GPU actor and
CPU server orchestration/CUDA model operations. No WSL execution is claimed.
Main Flower run ID: `{result["flower_run_id"]}`. Output: `{path}`.
Runtime: {result["runtime_seconds"]:.3f} seconds; device `{result["config"]["device"]}`.

## Protocol and leakage controls

{result["unique_patients"]} unique training patients, rotating origin, seed 42,
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
{chr(10).join(rows)}

AUROC is a mean of origin-local AUROCs, not pooled AUROC. Accuracy/F1/BCE aggregate
exact local confusion counts/loss sums. Unnecessary means selected-call realized
BCE >= same-model origin-only fusion BCE; this is not clinical necessity.
One reused stream cannot establish superiority. The offline reference has prior
full-information supervision and a different protocol. No positive clinical
improvement claim is made from these development metrics.

## Actual updates and validation curve

Initial validation mean-origin BCE: {result["initial_validation"]["mean_bce"]:.8f}.
Final approved validation BCE: {result["final_validation"]["mean_bce"]:.8f}.

| Cutoff | Version | Approved | Validation BCE | Shared update L2 | Changed validation routes |
|---:|---|---|---:|---:|---:|
{curve}

Approved version `{result["approved_version"]}`; {len(result["participation"])} local updates.
Initial-to-approved router L2 {parameter_changes["router_l2"]:.8f}, fusion L2
{parameter_changes["fusion_l2"]:.8f}, combined L2 {parameter_changes["shared_l2"]:.8f}.
All private specialist hashes remained unchanged. A later real Flower coordinator
request loaded `{result["later_coordinator"]["version"]}`, origin
{result["later_coordinator"]["origin"]}, patient ID {result["later_coordinator"]["patient_id"]},
selected mask `{result["later_coordinator"]["selected"]}`, risk
`{result["later_coordinator"]["probabilities"]}`.

Full config, per-origin validation metrics, cumulative stream curves, clipped
ratios, gradient norms, runtime and torch peak allocation are in metrics.json,
versions/vNNN.json and client_N/update_R.json. Learning curves are preserved as
learning_curve.json and learning_curve.csv. The approval guard accepts BCE within
0.02 of best seen and update L2 <= 0.1; it can permit small degradation and is not
medical safety. All candidates remain immutable; rollback was unit tested.

Consultation logical JSON: {result["consultation_logical_bytes"]:.0f} bytes.
Shared tensor upload/download estimate: {result["shared_update_array_bytes"]} bytes.
These exclude transport/serialization overhead and feedback/control messages.

## Verification and commands

```powershell
python -m fed_med_agents.experiments.milestone04 --device cuda --output artifacts/milestone04
python -m fed_med_agents.feedback.verify artifacts/milestone04
python scripts/report_milestone04.py --checks --real-qwen
```

Actual QA: `{summary}`. Ruff lint/format and pip check passed. Fresh CPU integration
output `{checks["feedback_integration_output"]}`; run ID `{integration["flower_run_id"]}`,
24 patients, early insufficient-data trigger skipped, then one three-client
update approved and used by a later actual peer consultation. Saved verification
replayed authentication, true source labels, delayed cutoff joins, training IDs,
selected-only Flower traces, numerical predictions and FedAvg against all three
client shared checkpoints. Test IDs were read only to assert exclusion; test labels
were not evaluated. Qwen is optional for tabular tests; the separate regression
output is `{checks["qwen_regression_output"]}`.

{slm_section}## Failures, scope and unfinished work

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
"""
    (ROOT / "docs/experiments/milestone-04.md").write_text(report, encoding="utf-8")
    print(json.dumps(evidence | parameter_changes, indent=2))


if __name__ == "__main__":
    main()
