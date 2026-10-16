"""Render reviewable Markdown experiment records from actual saved evidence."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from fed_med_agents.data import ROOT
from fed_med_agents.evaluation.verify import verify


def write(path: str, content: str) -> None:
    if path == "docs/status.md" and (ROOT / "docs/experiments/milestone-03.md").exists():
        return  # Preserve the later milestone's current-status document.
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content.strip() + "\n", encoding="utf-8")


def table(rows: list[tuple[str, list[float]]]) -> str:
    headers = "| Model/policy | Accuracy | AUROC | F1 | BCE | Calls/patient | Collaboration % | Amortized ms |\n|---|---:|---:|---:|---:|---:|---:|---:|"
    return (
        headers
        + "\n"
        + "\n".join(
            "| " + name + " | " + " | ".join(f"{value:.4f}" for value in values) + " |"
            for name, values in rows
        )
    )


KEYS = [
    "accuracy",
    "auroc",
    "f1",
    "binary_cross_entropy",
    "average_consultations",
    "collaboration_percent",
    "amortized_latency_ms",
]


def main() -> None:
    base = ROOT / "artifacts" / "milestone02"
    results = [json.loads((base / f"seed_{seed}" / "metrics.json").read_text()) for seed in [42, 43, 44]]
    baseline = json.loads((ROOT / "artifacts/milestone01/federated_metrics.json").read_text())
    baseline_specialists = json.loads((ROOT / "artifacts/milestone01/specialists_metrics.json").read_text())
    cpu = json.loads((ROOT / "artifacts/milestone02-cpu/seed_42/metrics.json").read_text())
    audit = json.loads((base / "environment.json").read_text())
    qa = base / "qa"
    qa.mkdir(exist_ok=True)
    checks = {}
    for name, args in {
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "ruff": [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        "ruff-format": [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
        "pip-check": [sys.executable, "-m", "pip", "check"],
    }.items():
        process = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
        )
        output = process.stdout + process.stderr
        (qa / f"{name}.log").write_text(output, encoding="utf-8")
        checks[name] = {
            "command": subprocess.list2cmdline(args),
            "exit_code": process.returncode,
            "output": output,
        }
        if process.returncode:
            raise RuntimeError(f"{name} failed: {output}")
    verification = [verify(base / f"seed_{seed}") for seed in [42, 43, 44]] + [
        verify(ROOT / "artifacts/milestone02-cpu/seed_42")
    ]
    (qa / "verification.json").write_text(json.dumps(verification, indent=2))
    (qa / "checks.json").write_text(json.dumps(checks, indent=2))
    summary = json.loads((base / "summary.json").read_text())
    rows = []
    std_rows = []
    for origin in range(3):
        values = [[r["test"]["none"][str(origin)][key] for key in KEYS] for r in results]
        rows.append((f"M2 calibrated specialist {'ABC'[origin]}", np.mean(values, axis=0)))
    for policy, title in {
        "fixed_mean": "M2 fixed specialist mean (all evidence)",
        "all": "Fixed all-peer consultation (same mean)",
        "random": "Random routing, max budget 1",
        "confidence": "Confidence routing, max budget 1",
        "learned": "Learned routing + fusion, origin A",
    }.items():
        values = np.array([[r["test"][policy]["0"][key] for key in KEYS] for r in results])
        rows.append((title, values.mean(axis=0)))
        std_rows.append((title, values.std(axis=0, ddof=1)))
    all_seed_rows = [
        (
            f"seed {r['seed']}, {policy}, origin {'ABC'[origin]}",
            [r["test"][policy][str(origin)][key] for key in KEYS],
        )
        for r in results
        for policy in ["none", "fixed_mean", "random", "confidence", "learned"]
        for origin in range(3)
    ]
    adjacency = "\n\n".join(
        f"Seed {r['seed']}, selected round {r['selected_round']} (rows origin, columns destination A/B/C):\n\n```text\n"
        + "\n".join("[" + ", ".join(f"{value:.6f}" for value in row) + "]" for row in r["adjacency"])
        + "\n```"
        for r in results
    )
    runtime_rows = []
    peaks = []
    for r in results + [cpu]:
        location = (
            base / f"seed_{r['seed']}"
            if r["device"] == "cuda"
            else ROOT / "artifacts/milestone02-cpu/seed_42"
        )
        prep = json.loads((location / "preparation.json").read_text())
        round_reports = [json.loads(path.read_text()) for path in location.glob("client_*/round_*.json")]
        peak = max(
            [prep["peak_allocated_bytes"], r["server_peak_allocated_bytes"]]
            + [p["peak-vram-bytes"] for p in round_reports]
        )
        peaks.append(peak)
        runtime_rows.append(
            f"| {r['seed']} | {r['device']} | {prep['runtime_seconds']:.3f} | {r['runtime_seconds']:.3f} | {sum(p['runtime-seconds'] for p in round_reports):.3f} | {peak / 2**20:.3f} |"
        )
    samples = json.loads((base / "gpu_samples.json").read_text())
    utilizations = [sample["gpu_utilization_percent"] for sample in samples]
    sample_text = f"{len(samples)} nvidia-smi samples: board GPU utilization mean {np.mean(utilizations):.2f}%, range {min(utilizations)}–{max(utilizations)}%. Board memory range {min(s['board_memory_mib'] for s in samples)}–{max(s['board_memory_mib'] for s in samples)} MiB. These include desktop/background processes."
    learned = np.mean([r["test"]["learned"]["0"]["accuracy"] for r in results])
    random = np.mean([r["test"]["random"]["0"]["accuracy"] for r in results])
    full = np.mean([r["test"]["fixed_mean"]["0"]["accuracy"] for r in results])
    own = np.mean([r["test"]["none"]["0"]["accuracy"] for r in results])
    experiment = f"""# Milestone 2 experiment — 2026-10-14

Implemented and executed learned directed routing, shared prediction fusion, three-client Flower FedAvg, and genuine server-mediated consultations. This establishes working mechanics; it does not establish superiority or privacy.

## Reconstructing the experiment

Read [architecture](../architecture.md) for equations, data provenance, payloads and exact implemented modules, [limitations](../limitations.md) for caveats, and [execution log](../devlog/2026-10-14-milestone-02.md) for commands/failures. Frozen configuration: seeds 42/43/44; immutable split seed 42; 364 training, 78 validation, 78 test; 4 OOF folds with nested 80/20 fit/calibration; specialists 30 epochs, batch 32, Adam 0.01; temperature 80 Adam steps, learning rate 0.05, bounded exp(logT) with logT in [-2.3,2.3]; Flower 5 rounds, 10 local shared epochs, batch 64, Adam 0.01, clip norm 5; maximum 1 consultation; cost 0.02 BCE units per call; FP32. Select the trained round with smallest mean-origin validation BCE, then freeze and evaluate test. No hyperparameter, seed, cost, or threshold was selected using test outcomes.

The objective is `mean_i sum_a softmax(score_i)[a] * (BCE(fusion_i,a, y_i) + 0.02 * |a|)`. The router's inputs contain no unconsulted probability. At deployment it selects argmax action, obtains only selected peer predictions via Flower, and applies masked fusion. Training exploration obtains all OOF outcomes, so deployment call counts below do not describe training traffic. Final models and test predictions are saved for all origins/policies.

## Mean test metrics across three configured seeds

{table(rows)}

Sample standard deviations (ddof=1), same fixed patient split:

{table(std_rows)}

The primary learned row originates at A. Specialist B/C rows are their own local inference, not consultations from A. Fixed mean/all receive all three calibrated specialist probabilities and require two peer consultations. Random and learned have the same maximum budget of one, use the same selected fusion checkpoint, and use one call per A-origin patient in the observed runs; confidence can abstain. Fixed mean and fixed all are deliberately the same predictor, not two independent algorithms. No learned all-evidence fusion ablation is claimed.

Learned A-origin accuracy change: {learned - own:+.4f} versus A alone; {learned - random:+.4f} versus matched-maximum-budget random; {learned - full:+.4f} versus full-evidence fixed mean; {learned - baseline["test"]["mean_probability_ensemble"]["accuracy"]:+.4f} versus historical M1 shared-model mean. Comparisons against full evidence and historical M1 are not evidence-matched causal claims. Individual seeds can favor different policies. ECE is saved in per-seed metrics; temperature calibration does not guarantee correctness confidence.

## Historical masked-model baseline on identical held-out IDs

| M1 view | Accuracy | AUROC | F1 | BCE |
|---|---:|---:|---:|---:|
"""
    for name, scores in baseline["test"].items():
        experiment += "| " + name + " | " + " | ".join(f"{scores[key]:.4f}" for key in KEYS[:4]) + " |\n"
    experiment += f"""
These values were verified by loading the preserved M1 checkpoint. M1 trained a masked-input shared classifier, not the new calibrated specialists/router/fusion. Its original runtime was torch 2.14.1+cpu. No new five-round M1 training run is claimed during M2.

## Learned adjacency and transparent decisions

{adjacency}

The matrices are normalized edge priors, not observed consultation frequencies. Actual scores include contextual network output. Example from seed 42 (not a verbal rationale):

```json
{json.dumps(results[0]["routing_examples"][0], indent=2)}
```

Action ordering is empty/A/B/C. Self-action A is masked (null represents −∞). The selected B score exceeds C and the stop score using only origin probability/confidence and public availability. Other seeds can select C. No private candidate probability or true label was available to this decision.

## Real Flower evidence

| Seed | Device | In-memory Flower run ID | Selected round | Verified client updates | Consultation events |
|---|---|---|---:|---:|---:|
"""
    for r, evidence in zip(results + [cpu], verification, strict=True):
        experiment += f"| {r['seed']} | {r['device']} | {r['flower_run_id']} | {r['selected_round']} | {evidence['client_updates']} | {evidence['consultation_events']} |\n"
    experiment += f"""
Each run received all three shared updates in each of five rounds (15 updates), with zero training failures. Router and fusion parameter changes were independently verified. Private checkpoint hashes remained unchanged. The checker verifies every OOF patient's excluded fitting/calibration IDs, train-only preprocessing, client feature width, lack of val/test labels in client files, training-only exploration IDs, request association and test occurring after all validation queries. In-memory run IDs identify the executed simulations; they are not persistent CLI SuperLink run histories. Full traces/logs: `artifacts/milestone02/seed_{{42,43,44}}/consultations.jsonl`, `flower.log`, `metrics.json`, and `client_*/round_*.json`.

## GPU, timing, and CPU fallback

Device: {audit["device_name"]}; physical VRAM {audit["vram_bytes"] / 2**30:.3f} GiB; Python {audit["python"]}; PyTorch {audit["torch"]}; CUDA wheel runtime {audit["cuda_build"]}; CUDA available {audit["cuda_available"]}; real tensor checksum {audit["tensor_operation_checksum"]}. Original CPU-only build was replaced only inside the venv. Driver/OS were unchanged. One full-GPU Ray actor sequentially executes the three logical clients; no concurrent GPU clients were enabled. Shared training supports opt-in AMP, but experiments used FP32. CUDA AMP gradients passed a numerical smoke test.

| Seed | Device | Specialist/OOF prep seconds | Server workflow seconds | Sum local shared train seconds | Max torch allocated MiB |
|---|---|---:|---:|---:|---:|
{chr(10).join(runtime_rows)}

Preparation includes dataset fetching, fitting and calibration. Server workflow includes Flower setup queries, exploration, shared training, validation, and all test policies; it is not kernel-only time. The latest GPU rerun reused prepared OOF files to validate the final communication changes. {sample_text}

CPU seed 42 completed all five rounds and all consultations; learned A-origin accuracy {cpu["test"]["learned"]["0"]["accuracy"]:.4f}, AUROC {cpu["test"]["learned"]["0"]["auroc"]:.4f}, F1 {cpu["test"]["learned"]["0"]["f1"]:.4f}, BCE {cpu["test"]["learned"]["0"]["binary_cross_entropy"]:.4f}. The tiny dataset and measured preparation/round-trip overhead provide no reason to claim GPU saturation or throughput advantage. Torch allocated memory excludes CUDA contexts/reserved memory. Inference times are amortized batch wall times, not single-patient deployment latency.

## Tests and checks

```text
{checks["pytest"]["output"].strip()}
```

Ruff check, Ruff format check, and pip check exited 0. Saved evidence verification passed all four runs. Tests cover general graph dimensions/budget/self mask, unavailable domains/nodes, router and fusion gradients on CPU/CUDA, specialist isolation, checkpoint/ArrayRecord roundtrip, unavailable/incorrect patient and destination replies, real Message transport adapter, heldout-cache rejection, training-only OOF preprocessing/calibration and label perturbation, unconsulted evidence exclusion, CUDA AMP finite gradients, explicit CUDA failure, and actual three-client participation/provenance. On machines without CUDA, CUDA-specific tests skip explicitly. Saved-artifact verification test skips only when experiments have not been run.

## Outstanding limitations and next work

Learned routing/fusion did not outperform fixed full evidence. The A-origin stop action was never selected: observed deployment collaboration rate is 100%. Budget/cost sensitivity, independent split variability, validation-driven improvements, an all-evidence learned fusion control, authenticated transport, privacy defenses, and persistent standalone serving remain undone. Windows Ray shutdown diagnostics and deprecated Python launcher remain documented. No SLMs, multi-hop routing, package publication, GitHub push or OS installation occurred.

## Per-seed per-origin metrics

{table(all_seed_rows)}

All Markdown report paths are relative to the repository root. Raw numeric source artifacts remain gitignored; this report embeds the main quantitative evidence so an independent reviewer does not require conversation history.
"""
    write("docs/experiments/milestone-02.md", experiment)
    m1 = f"""# Milestone 1 retrospective — verified 2026-10-14

Evidence source: preserved `artifacts/milestone01/` copied before M2 changes. Existing recorded CLI run `{baseline["verification"].get("flower_run_id", "9761394553627465436")}`; Flower {baseline["verification"]["flower_version"]}; original torch {baseline["verification"]["torch_version"]} CPU; Python {baseline["verification"]["python_version"]}; {baseline["verification"]["platform"]}. Original logs record five rounds with three client replies and zero failures. Received participation and saved audits show 15 updates. Initial/final shared L2 change {baseline["verification"]["shared_weight_l2_change"]:.8f}. Checkpoint re-evaluation at the start of M2 reproduced mean accuracy 0.9230769, AUROC 0.9347222, F1 0.9375 and BCE 0.3919495. This is a historical evidence audit and checkpoint replay, not a claimed new M1 training run.

UCI 529 was fetched via ucimlrepo, with 520 rows, 16 expected columns, 320 positive/200 negative labels. Stratified 70/15/15 split seed 42: 364/78/78. Train-fitted age scaling and fixed binary mappings. A gets age/gender/obesity, B gets five primary symptoms, C gets eight remaining symptoms; all share aligned patient IDs and labels. Three independent specialists (`feature width→24→1`) train 30 epochs on CPU. Only the separate shared `32→24→1` model (16 zero-masked values plus 16 availability masks) is federated, using Flower's current ServerApp/ClientApp, ArrayRecord and FedAvg for five rounds, 10 local epochs each. Adam 0.01, batch 32, BCEWithLogitsLoss, fixed seeds, no optimizer aggregation. Mean-probability fusion is untrained.

| Model | Accuracy | AUROC | F1 | BCE |
|---|---:|---:|---:|---:|
"""
    for client, record in baseline_specialists.items():
        m1 += (
            f"| Private specialist {client} | "
            + " | ".join(f"{record['test'][key]:.4f}" for key in KEYS[:4])
            + " |\n"
        )
    for client, record in baseline["test"].items():
        m1 += f"| Shared {client} | " + " | ".join(f"{record[key]:.4f}" for key in KEYS[:4]) + " |\n"
    m1 += """
Four initial tests passed before M2, covering split/statistic leakage, feature isolation/alignment, compatible shared shapes/private parameter separation, and reply payload content. CPU inference replay and dependency checks succeeded. No CUDA operation was possible in the original CPU-only torch wheel; nvidia-smi nevertheless detected the A4000. Existing logs included experimental Windows Ray shutdown diagnostics. Ubuntu/WSL was not installed or verified.

The simulator's shared dataset file was readable by all clients; raw training rows were absent from update messages, but that is not a privacy guarantee. The cohort is not three hospitals, AUROC/accuracy are not clinical validation, and the shared C view was worse than private C. Original M1 scripts remain runnable. M2 source formatting/import cleanup was nonbehavioral; the torch requirement was widened to permit the tested CUDA build, so new M1 reruns may differ slightly from historical values.
"""
    write("docs/experiments/milestone-01.md", m1)
    write(
        "docs/status.md",
        f"""# Current implementation status — 2026-10-14

Milestone 1 preserved and audited; Milestone 2 implemented and executed on Windows CPU and RTX A4000 CUDA. No SLMs, publication or GitHub push.

Verified: train-only nested OOF specialists; compatible directed contextual routing/fusion; five genuine Flower FedAvg rounds per seed; real structured server-mediated consultations; all 3 clients participate; specialist checkpoints unchanged; routing before peer evidence; missing-information handling; saved checkpoints and provenance; CPU/CUDA tests. Main configured seeds 42/43/44 use the same 78 held-out patients, with validation-only checkpoint selection.

Learned A-origin mean accuracy {learned:.4f}; fixed full-evidence mean accuracy {full:.4f}; random maximum-budget-matched accuracy {random:.4f}; historical M1 shared mean 0.9231. Learned collaboration is functional but has not established improvement over stronger baselines. Maximum measured torch allocated VRAM across reported runs {max(peaks) / 2**20:.3f} MiB, excluding runtime contexts/board usage. No GPU saturation claim.

Read [architecture](architecture.md), [M1 retrospective](experiments/milestone-01.md), [M2 experiment](experiments/milestone-02.md), [limitations](limitations.md), [roadmap](next-steps.md), [execution log](devlog/2026-10-14-milestone-02.md), and root AGENTS.md. The experiment report includes exact run IDs, metric tables, adjacency, commands/evidence links, test results and GPU observations. Remaining substantive issues: accuracy/deployment budget tradeoffs, fold-to-final calibration shift, Windows Ray diagnostics, deprecated M2 launcher, external validation, security/privacy and full adapter generalization.
""",
    )
    print(
        json.dumps(
            {
                "reports_written": [
                    "docs/status.md",
                    "docs/experiments/milestone-01.md",
                    "docs/experiments/milestone-02.md",
                ],
                "qa": "passed",
                "summary_runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
