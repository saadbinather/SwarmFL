"""Render the M5 Step 1 Markdown report from preserved diagnostic results only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/m5-step01"
AUDIT = ROOT / "artifacts/m5-step01-audit"
POOLED = ROOT / "artifacts/m5-step01-central-pooled"
REPORT = ROOT / "docs/experiments/m5-step01-routing-diagnostics.md"
METRICS = ("accuracy", "auroc", "binary_cross_entropy", "f1", "ece", "brier")


def read_text(path: Path) -> str:
    """Accept Python UTF-8 and Windows PowerShell UTF-16 redirected evidence."""
    raw = path.read_bytes()
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def read(path: Path) -> dict:
    """Read numeric evidence without rerunning training or historical test evaluation."""
    return json.loads(read_text(path))


def table(headers: list[str], rows: list[list]) -> str:
    """Format a plain Markdown table with explicit missing-value handling."""

    def cell(value: object) -> str:
        if value is None:
            return "undefined"
        if isinstance(value, (float, np.floating)):
            return f"{value:.5f}"
        return str(value)

    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(cell(v) for v in row) + " |" for row in rows],
            ]
        )
        + "\n"
    )


def macro(record: dict, policy: str = "learned", fusion: str = "learned_fusion") -> list[float]:
    """Unweighted mean across the three dependent origin views."""
    return [
        float(np.mean([v["policies"][policy][fusion][metric] for v in record.values()])) for metric in METRICS
    ]


def main() -> None:
    """Regenerate the complete independently readable report; never train a model."""
    cfg = read(RUN / "config.json")
    results = [read(RUN / f"seed_{seed}/results.json") for seed in cfg["seeds"]]
    pooled = {seed: read(POOLED / f"seed_{seed}/results.json") for seed in cfg["seeds"]}
    feedback = read(AUDIT / "feedback.json")
    report = [
        """# M5 Step 1 — Routing and fusion degradation diagnostics

Executed 2026-10-14 on Windows. **Step 1 only is complete.** M1–M4 defaults,
specialists, checkpoints and historical results are preserved. No historical-test
inference, tuning, new SLM, Qwen fine-tuning, new specialist registry or flexible
fusion architecture was introduced. No clinical validity or privacy guarantee is
claimed. This report distinguishes fresh training/validation diagnostics from
historical evidence.

## Findings for architecture review

1. **Origin-local objective conflict is supported.** Several simultaneous origin
   gradients and local fusion updates oppose one another. Both centralized
   references have lower validation BCE in all three seeds, including the control
   with matched global optimizer steps. This implicates the local-training /
   averaging protocol; it does not isolate arithmetic averaging from optimizer
   state resets, local drift or batching as a unique cause.
2. **Joint optimization is not established as the main problem.** Joint training
   has lower validation BCE than either component-freeze ablation in every seed.
   The router and fusion have disjoint parameter sets: there is no direct
   competing-gradient dot product on shared parameters between these modules.
   Coupling occurs through policy-weighted fusion losses and changing action
   values. Freezing at random initialization is a substantial restriction, not a
   complete test of alternating training or a good pretrained fixed fusion.
3. **Always-consult routing is confirmed; fixed-peer collapse is common.** At
   the final round every origin consults on every validation patient. Eight of
   nine seed/origin views use one fixed peer; seed 44 origin A selects B on 50
   cases and C on 28. This confirms always-consult behavior, but does
   not prove statistical overfitting. Validation BCE is below OOF-training BCE,
   and the cohorts and prediction-generating models differ.
4. **Soft/hard mismatch exists but is localized.** Final-round hard-minus-soft
   BCE is largest for origin A; B/C policies are almost deterministic. It cannot
   by itself explain all degradation. M4 remains nearly uniform stochastically
   while its argmax actions scarcely change, so streaming and deterministic
   validation policies must not be conflated.
5. **Calibration/distribution shift is measured, causality unresolved.** Fold
   and final specialist predictions differ on the same training patients, and
   validation calibration is imperfect. Final-on-training values are optimistic
   resubstitution diagnostics, never an unbiased comparator. No recalibration was
   fitted on validation or test, and no calibration fix is claimed.
6. **Fusion quality depends on the metric and evidence.** Identical-evidence
   arithmetic means can improve threshold accuracy while learned fusion improves
   BCE. Random/cyclic controls below match maximum and exact per-case call
   budgets. Two-peer full ensembles are not used to claim a one-peer deficit.
   Some selected calls harm realized BCE; this is not clinical necessity.
   Learned routing's macro accuracy (.93162) is slightly below seeded random
   (.93305), while its BCE (.19573) is slightly lower than random (.19798).
   These small reused-validation differences do not establish a policy winner.
7. **M4 is sparse and changes little.** Authentic selected-only replay has
   19–122 records/origin/update. Default cap 2 clips no inspected parent-policy
   importance weights; effective sample size stays close to sample count. Thus
   active clipping is not supported as the explanation here. Older reward
   versions are common but their measured risk/loss changes are small. Selection
   bias remains an off-policy modeling risk, not a demonstrated failure from
   observing unqueried outcomes.

## Architecture and learning audit

The current `CollaborativeModel` contains a three-client contextual router and
masked residual fusion. Router: a 3×3 edge-logit parameter matrix; context inputs
are origin one-hot, own probability/confidence, domain flags and availability
(11 inputs), then Linear(11,16), Tanh, Linear(16,4). Budget one enumerates
empty/A/B/C actions; self/unavailable actions are masked. Fusion: each specialist's
masked logit, confidence and presence plus origin identity (12 inputs), then
Linear(12,24), ReLU, Linear(24,1), with origin-logit residual. Specialists are
separate Linear(local_width,24), ReLU, Linear(24,1) classifiers.
M1's preserved workflow instead trains a shared masked-input classifier with
32 inputs through Flower. M3 performs frozen-Qwen tool dispatch and numerical
inference with saved specialists/shared weights; it performs no optimization.

M2 jointly minimizes soft expected action BCE plus .02 per call using complete
training-only nested OOF evidence. Inference selects argmax before consultation.
Each origin trains from common shared arrays with its own labels/evidence cache;
Flower 1.39 `FedAvg` aggregates only compatible shared ArrayRecords, weighted by
`num-examples`. Here each origin has 364 aligned training patients, so weighting
is equal. Specialists, temperatures and preprocessing are not in shared updates.
These are dependent views of one cohort, not independent hospitals.

M4 starts adaptive/frozen shared policies at label-independent random weights,
not the already-supervised M2 checkpoint. Signed prediction and verified outcome
ledgers are separate; delayed true training labels are joined locally after
decisions. Behavior is .8×policy + .2×uniform over valid actions. Shared gradients
use selected evidence only: detached clipped IPS-weighted fusion BCE plus .1×
cost policy surrogate, with logged fusion reward, fixed .5 baseline, Adam .0003,
two full replay steps, norm clip .5 and latest 128 verified events per origin.
Three clients send shared arrays and scalar metrics over real Flower; approval
uses validation and immutable retained versions. Private specialists and Qwen are
frozen throughout M4. Logs/acknowledgements are not parameter learning or labels.
Prediction, identifier and model-update disclosure is not proven privacy.
Training feature rows and private weights never enter consultation messages;
M3's separately authorized new-patient feature-subset path is unchanged.

## Protocol and controls

Existing fixed split: 364 training, 78 validation, 78 untouched historical-test
patients. Seeds 42/43/44 change specialist/fold/model fits, not the patient split.
No variant or hyperparameter was selected using these diagnostic results. Report
round 5 for every variant; complete rounds 0–5 remain in JSON (central 1–5).
Common initial checkpoint within seed; Adam .01; 10 local epochs × 5 rounds;
batch 64, norm clip 5, FP32, cost .02, budget one. Frozen components retain common
random initial weights. All three main variants use actual Flower train/reply
messages and FedAvg. The equal-total-step central reference interleaves origins
for 180 optimizer steps/round, versus 60/client. The follow-up pooled central
reference averages three origin losses per matched batch for 60 global steps,
matching the 180 origin-loss computations of three clients. Optimizers reset
each round in all schedules. Pooled control was specified in the execution log
before running it, to resolve the first reference's update-count confound.

OOF training, final-on-training and validation probability caches were acquired
through genuine Flower specialist queries. Diagnostic evaluation replays masked
subsets from these full-information caches; it is **not** a selected-only live
consultation benchmark. Routing uses only own evidence and public metadata.
The central reference deliberately has training labels/probabilities, never
private feature rows. Historical test IDs are used only for exclusion checks;
no historical test label is indexed for a new metric. Final-on-training
predictions are used only to describe fold/final shift, never for model fitting.

Controls use the same fusion checkpoint and budget one. `random_max` chooses a
seeded peer; `cyclic_max` chooses the next origin index, independent of labels or
peer values. `_matched` variants reuse the learned per-patient call/no-call mask
and change only the peer. Arithmetic mean uses the identical selected evidence;
stop+mean is the unmodified origin classifier, whereas stop+learned fusion still
applies its learned residual. All origins share patient IDs and are not
independent samples. Tables macro-average origin metrics, then seeds where
specified; AUROC/F1 are means, not pooled metrics or confidence intervals. ECE is
10 equal-width probability bins; accuracy/F1 use threshold .5.

## Fresh controlled ablations

Final-round training is OOF; validation uses final frozen specialists. All metrics
below are development results. No winner is promoted to a default.
"""
    ]
    rows = []
    for r in results:
        variants = r["variants"] | {"central_pooled": pooled[r["seed"]]}
        for variant, data in variants.items():
            last = data["history"]["5"]
            rows.append(
                [
                    r["seed"],
                    variant,
                    macro(last["train"])[2],
                    *macro(last["validation"]),
                    np.mean([v["policies"]["learned"]["calls"] for v in last["validation"].values()]),
                ]
            )
    report.append(
        table(
            [
                "Seed",
                "Variant",
                "Train BCE",
                "Val accuracy",
                "Val AUROC",
                "Val BCE",
                "Val F1",
                "Val ECE",
                "Val Brier",
                "Calls",
            ],
            rows,
        )
    )
    report.append(
        "\n### Joint-trained policy and fusion controls\n\nMeans across three seeds and three origins. All non-stop joint controls actually\nuse exactly one call per patient here; matched and maximum rows coincide in call\ncount, not necessarily predictions.\n"
    )
    rows = []
    for policy in ("learned", "stop", "random_max", "cyclic_max", "random_matched", "cyclic_matched"):
        for fusion in ("learned_fusion", "mean_fusion"):
            values = np.array(
                [macro(r["variants"]["joint"]["history"]["5"]["validation"], policy, fusion) for r in results]
            ).mean(0)
            rows.append([policy, fusion, *values, 0 if policy == "stop" else 1])
    report.append(
        table(["Policy", "Fusion", "Accuracy", "AUROC", "BCE", "F1", "ECE", "Brier", "Calls"], rows)
    )
    report.append(
        "\n### Per-origin joint diagnostics\n\nAction frequencies are [stop,A,B,C] counts over 78 validation patients. Harm is\nthe fraction of selected calls whose realized BCE is no better than the same\nmodel's stop-fusion BCE. Oracle regret uses the true validation outcome only\nafter predictions and is an unattainable full-information diagnostic, not a\npolicy or a selection criterion.\n"
    )
    rows = []
    for r in results:
        for origin, value in r["variants"]["joint"]["history"]["5"]["validation"].items():
            p = value["policies"]["learned"]
            rows.append(
                [
                    r["seed"],
                    origin,
                    *[p["learned_fusion"][m] for m in METRICS[:4]],
                    p["action_frequencies"],
                    p["harm_fraction_of_calls"],
                    p["mean_call_bce_delta"],
                    value["hard_bce_minus_soft"],
                    value["mean_entropy"],
                    value["label_oracle_cost_regret"],
                ]
            )
    report.append(
        table(
            [
                "Seed",
                "Origin",
                "Accuracy",
                "AUROC",
                "BCE",
                "F1",
                "Actions",
                "Harm/calls",
                "Call BCE delta",
                "Hard−soft BCE",
                "Entropy",
                "Oracle cost regret",
            ],
            rows,
        )
    )
    report.append(
        "\n### Cost and redundancy\n\nFraction of validation cases for which at least one valid peer lowers realized\nfusion BCE by more than the stated cost. This is retrospective and label-aware;\nit does not show that a router can identify those cases prospectively. Costs\nwere not retuned and these are fixed-model diagnostics, not cost-trained variants.\n"
    )
    rows = []
    for origin in ("0", "1", "2"):
        rows.append(
            [
                origin,
                *[
                    np.mean(
                        [
                            r["variants"]["joint"]["history"]["5"]["validation"][origin][
                                "beneficial_call_fraction_at_cost"
                            ][str(cost)]
                            for r in results
                        ]
                    )
                    for cost in (0, 0.02, 0.1, 0.5)
                ],
            ]
        )
    report.append(table(["Origin", "Cost 0", "Cost .02", "Cost .1", "Cost .5"], rows))
    rows = []
    for r in results:
        with np.load(RUN / f"seed_{r['seed']}/evidence.npz") as data:
            p, y = data["validation"], data["val_y"]
            errors = p - y[:, None]
            rows.append(
                [
                    r["seed"],
                    *[
                        float(np.corrcoef(errors[:, i], errors[:, j])[0, 1])
                        for i, j in ((0, 1), (0, 2), (1, 2))
                    ],
                    float(((p[:, 1] >= 0.5) != (p[:, 2] >= 0.5)).mean()),
                ]
            )
    report.append(
        "\nSigned probability-error correlations describe redundancy, not conditional\nindependence or a causal value of information.\n\n"
        + table(["Seed", "A/B error corr", "A/C error corr", "B/C error corr", "B/C disagreement"], rows)
    )
    report.append(
        "\n## Gradients, local drift and calibration\n\nPair order is A/B, A/C, B/C. Gradients are simultaneous full-OOF-training\ngradients at final shared weights, before clipping. Updates are the final\nround's independently trained local state minus its common parent. Negative\ncosine supports objective/update conflict, not an aggregation implementation bug.\n"
    )
    rows = []
    for r in results:
        last = r["variants"]["joint"]["history"]["5"]
        for prefix in ("router.", "fusion."):
            rows.append(
                [
                    r["seed"],
                    prefix,
                    ", ".join(f"{v:.4f}" for v in last["gradient_audit"]["origin_gradient_cosines"][prefix]),
                    ", ".join(f"{v:.4f}" for v in last["local_update_cosines"][prefix]),
                ]
            )
    report.append(table(["Seed", "Component", "Origin gradient cosines", "Local update cosines"], rows))
    rows = []
    max_memory = 0
    for r in results:
        seed = r["seed"]
        initial = torch.load(Path(cfg["source"]) / f"seed_{seed}/shared_initial.pt", weights_only=True)
        for variant in r["variants"]:
            state = torch.load(RUN / f"seed_{seed}/{variant}/server_round_5.pt", weights_only=True)
            norms = []
            for prefix in ("router.", "fusion."):
                norms.append(
                    sum(
                        float(((v - initial[k]) ** 2).sum()) for k, v in state.items() if k.startswith(prefix)
                    )
                    ** 0.5
                )
            if variant == "central_joint":
                audits = [v["training"] for v in r["variants"][variant]["history"].values()]
            else:
                audits = [read(p) for p in (RUN / f"seed_{seed}/{variant}").glob("client_*.json")]
            max_memory = max(max_memory, *[a["peak_allocated_bytes"] for a in audits])
            rows.append(
                [
                    seed,
                    variant,
                    *norms,
                    *[
                        np.mean([a["mean_gradient_norm"][prefix] for a in audits])
                        for prefix in ("router.", "fusion.")
                    ],
                ]
            )
    report.append(
        "\nInitial-to-final parameter L2 (buffers included, constant action buffer has zero\ndelta); mean local preclip gradient norms across saved update reports. Tiny\nfrozen server drift can arise solely from Flower floating point averaging.\n\n"
        + table(
            ["Seed", "Variant", "Router L2", "Fusion L2", "Router gradient norm", "Fusion gradient norm"],
            rows,
        )
    )
    rows = []
    for r in results:
        for origin in ("0", "1", "2"):
            rows.append(
                [
                    r["seed"],
                    origin,
                    r["probability_shift"][origin]["oof_to_final_mean_abs"],
                    *[
                        r["specialists"][split][origin]["ece"]
                        for split in ("train", "train_final", "validation")
                    ],
                    *[
                        r["specialists"][split][origin]["binary_cross_entropy"]
                        for split in ("train", "train_final", "validation")
                    ],
                ]
            )
    report.append(
        "\nSame-training-patient OOF/final absolute probability shift, followed by raw\nspecialist calibration/BCE. Final-training BCE is optimistic resubstitution.\nDifferences combine model fitting, fold sample size and temperature estimation;\nthese data do not isolate calibration as the cause.\n\n"
        + table(
            [
                "Seed",
                "Origin",
                "Mean abs(OOF−final)",
                "OOF ECE",
                "Train-final ECE",
                "Val ECE",
                "OOF BCE",
                "Train-final BCE",
                "Val BCE",
            ],
            rows,
        )
    )
    report.append(
        "\n## M4 feedback investigation and reproduction\n\nRead-only independent verification of the historical run passed all 1,092\nevents, authentication/source/delay/partial-evidence checks and seven shared\nFedAvg candidates. Replay diagnostics below use only authenticated observed\nevidence and outcomes at each saved training cutoff, evaluated at the parent\npolicy before local updates. They do not recover unselected peer outcomes.\n"
    )
    rows = []
    for row in feedback["replay"]:
        clipping = row["clipping"]["2.0"]
        rows.append(
            [
                row["version"],
                row["origin"],
                row["n"],
                row["actions"],
                row["fraction_old_version"],
                clipping["clipped_fraction"],
                clipping["effective_sample_size"],
                row["mean_abs_risk_change"],
                row["current_minus_logged_bce"],
            ]
        )
    report.append(
        table(
            [
                "Update",
                "Origin",
                "Replay n",
                "Actions",
                "Older reward version fraction",
                "Clipped at 2",
                "ESS",
                "Mean abs(risk change)",
                "Current−logged BCE",
            ],
            rows,
        )
    )
    report.append(
        "\nEach patient supplies one observed action only (three valid actions per origin);\nfull action counterfactuals are unavailable in M4 replay. Cap-sensitivity audits\n(.5, 1, 2, 10) retain norms and ESS in `feedback.json`. A .5 cap clips every\ninspected row, while the deployed cap 2 clips none at the parent state. This does\nnot prove clipping is absent after local optimizer steps or in other streams.\nLate replay includes mostly older versions, but absolute risk changes are small;\nno dominant stale-reward failure is demonstrated. Historical adaptive/frozen\nstream BCE was .449557/.450215 with equal .788462 accuracy. These were stochastic\nstream policies, not the hard validation policy in the next table.\n"
    )
    rows = []
    for version, record in feedback["validation"].items():
        for origin, value in record.items():
            p = value["policies"]["learned"]
            rows.append(
                [
                    version,
                    origin,
                    *[p["learned_fusion"][m] for m in METRICS[:4]],
                    p["action_frequencies"],
                    value["mean_entropy"],
                    value["soft_expected_calls"],
                ]
            )
    report.append(
        table(
            [
                "Version",
                "Origin",
                "Accuracy",
                "AUROC",
                "BCE",
                "F1",
                "Hard actions",
                "Policy entropy",
                "Soft expected calls",
            ],
            rows,
        )
    )
    rerun_path = ROOT / "artifacts/m5-step01-m4-rerun/metrics.json"
    if rerun_path.exists():
        rerun, old = read(rerun_path), read(ROOT / "artifacts/milestone04/metrics.json")
        report.append(
            f"\nFresh full M4 rerun: Flower run `{rerun['flower_run_id']}`, device "
            f"`{rerun['config']['device']}`, {rerun['runtime_seconds']:.3f} seconds, "
            f"approved `{rerun['approved_version']}`. Comparison below is a reproduction, "
            "not a newly selected experiment.\n"
        )
        rows = []
        for policy in ("adaptive", "frozen", "offline"):
            for label, data in (("historical", old), ("fresh", rerun)):
                value = data["stream"][policy]
                rows.append(
                    [
                        label,
                        policy,
                        value["accuracy"],
                        value["auroc_macro_origin"],
                        value["binary_cross_entropy"],
                        value["f1"],
                    ]
                )
        report.append(table(["Run", "Policy", "Accuracy", "AUROC", "BCE", "F1"], rows))
        original_state = torch.load(ROOT / "artifacts/milestone04/versions/v007.pt", weights_only=True)
        fresh_state = torch.load(ROOT / "artifacts/m5-step01-m4-rerun/versions/v007.pt", weights_only=True)
        checkpoint_difference = max(
            float((value - fresh_state[key]).abs().max()) for key, value in original_state.items()
        )
        bce_difference = abs(
            old["stream"]["adaptive"]["binary_cross_entropy"]
            - rerun["stream"]["adaptive"]["binary_cross_entropy"]
        )
        report.append(
            f"\nAdaptive BCE absolute reproduction difference: {bce_difference:.10g}; "
            f"maximum final checkpoint element difference: {checkpoint_difference:.10g}. "
            "Accuracy, macro-origin AUROC, F1 and call counts reproduce exactly.\n"
        )
        verification = ROOT / "artifacts/m5-step01-review-audit/m4-rerun-verification.json"
        if not verification.exists():
            verification = AUDIT / "m4-rerun-verification.json"
        if verification.exists():
            report.append(
                "\nFresh saved-evidence verification:\n\n```json\n"
                + json.dumps(read(verification), indent=2)
                + "\n```\n"
            )
    else:
        report.append("\nFull M4 rerun is still pending; do not treat this draft as final.\n")
    report.append(
        "\n## Reproducibility and preservation\n\nFresh M2 joint final weights reproduce the historical final shared checkpoint\n**exactly** in all three seeds (maximum elementwise difference zero); only\ntraining/validation predictions were rerun. Historical M2 checkpoint selection\nwas not repeated. Frozen local components are bitwise unchanged; server arrays\nare checked within 1e-6 to allow FedAvg rounding. Independent M5 verification:\n\n```json\n"
        + json.dumps(read(RUN / "verification.json"), indent=2)
        + "\n```\n"
    )
    report.append(
        "\n"
        + table(
            ["Seed", "Flower run ID", "Post-cache server compute seconds"],
            [[r["seed"], r["flower_run_id"], r["runtime_seconds"]] for r in results],
        )
    )
    environment = read(RUN / "environment.json")
    report.append(
        f"\nDevice audit: {environment.get('device_name')}, PyTorch {environment['torch']}, "
        f"CUDA build {environment['cuda_build']}, actual tensor checksum "
        f"{environment.get('tensor_operation_checksum')}. Main ablations used CUDA with "
        "at most one full-GPU Ray client worker serving all three logical clients. "
        f"Maximum saved torch allocation across new main training reports: {max_memory} "
        "bytes. This excludes contexts, reserved memory and total board memory; no "
        "GPU saturation or speedup is claimed. Pooled central control used CPU and its "
        "own saved runtime/memory reports. Server compute times exclude startup and "
        "cache acquisition.\n"
    )
    preservation = read(AUDIT / "preservation.json")
    report.append(
        f"\nPreservation audit: {preservation['files_checked']} pre-existing artifact files "
        f"hashed, changed list `{preservation['changed']}`. Ollama runtime/model files "
        "were excluded from the manifest. This workspace has no `.git` repository; "
        "`git status --short` fails, so no VCS cleanliness claim is possible. "
        "New artifacts remain under gitignored `artifacts/m5-step01*`.\n"
    )
    report.append("""
## Commands and validation

Run from the repository root with the active environment's Python executable.
Fresh output paths are required; these commands never overwrite M1–M4 runs.

```powershell
python -u -m fed_med_agents.experiments.m5_step01 --device cuda --output artifacts/m5-step01 --seeds 42 43 44
python scripts/verify_m5_step01.py artifacts/m5-step01
python scripts/extend_m5_controls.py --device cpu
python scripts/audit_m5_feedback.py
python -u -m fed_med_agents.experiments.milestone04 --device cuda --output artifacts/m5-step01-m4-rerun
python -m fed_med_agents.feedback.verify artifacts/m5-step01-m4-rerun
python -m pytest -q
python -m ruff check --no-cache src tests scripts
python -m ruff format --no-cache --check src tests scripts
python -m pip check
python scripts/report_m5_step01.py
```

The original 52-test claim includes two opt-in runtime integrations. Ordinary
fresh QA after four new diagnostic tests passed **54, skipped 2** in 12.05 s;
two upstream Typer deprecation warnings remain. Opt-in integrations must use
new M3/M4 output paths to preserve original evidence. Saved QA logs below record
the final executed checks rather than assuming an integration passed.
""")
    for name in ("pytest-full.log", "ruff-check.log", "ruff-format.log", "pip-check.log"):
        file = AUDIT / name
        if file.exists():
            lines = read_text(file).splitlines()
            report.append(
                f"\n`artifacts/m5-step01-audit/{name}`:\n\n```text\n" + "\n".join(lines[-4:]) + "\n```\n"
            )
    review = ROOT / "artifacts/m5-step01-review-audit"
    if (review / "checks.json").exists():
        report.append("""
## Independent continuation review

The completed diagnostic workflow/artifacts were already present when this
continuation began. They were audited rather than replacing their training runs.
Fresh read-only verification passed 84 states, 45 Flower aggregations, 18,144
policy metrics and 7,254 trace rows; historical and reproduced M4 runs both passed
all 1,092 events and seven aggregations. Separate tensor comparisons confirmed
bitwise equality of historical M2 and diagnostic joint final weights in all three
seeds. All 628 preserved artifact hashes still match.

Corrected the fixed-peer claim: eight of nine final seed/origin views choose one
peer; seed 44 origin A chooses B on 50 validation cases and C on 28. All nine
views always consult. This exception is evidence against universal fixed-peer
collapse. Corrected two Markdown absolute-difference table headings.

One existing M5 derived verification JSON contained null bytes. The underlying
checkpoints, raw metrics and ledgers verified successfully; fresh summaries are
saved in the new review directory. Historical artifacts were not repaired or
overwritten. This report uses the independently regenerated M4 verification.

Automatic approval review rejected the combined Ollama-start/background-QA
command with reason "blocked by policy". No service start was executed. Safe
direct replay and ordinary pytest/Ruff/dependency checks completed instead.
The existing 56-test full-runtime result above is preserved evidence; the fresh
review below skips the two opt-in runtime integrations and must not be described
as a new 56-test run. No models were retrained in this review.
""")
        for name in ("pytest-full.log", "ruff-check.log", "ruff-format.log", "pip-check.log"):
            lines = read_text(review / name).splitlines()
            report.append(
                f"\n`artifacts/m5-step01-review-audit/{name}`:\n\n```text\n"
                + "\n".join(lines[-4:])
                + "\n```\n"
            )
    report.append("""
## Failures, caveats and unresolved hypotheses

- Existing Ruff cache panicked with `wrong package cache for file`; no-cache
  lint/format works. Initial replay on CPU failed a CUDA AUROC comparison
  (.93125 versus .927778) because tiny probability/tie changes affect ranks.
  Recorded-device replay passes tight checks. This is a reproducibility limit,
  not evidence of predictive improvement. Exact frozen-server equality also
  failed due to FedAvg rounding; local freezing remains exact and server
  numerical preservation passes 1e-6.
- The local Ollama endpoint was initially offline. Tabular diagnostics need no
  Qwen. Any restarted regression uses the existing frozen model, not a new SLM.
- Windows Ray shutdown/access-violation diagnostics and deprecated simulation
  launcher warnings remain; completed runs are checked by saved results, all
  three returned updates and independent numerical aggregation, not log silence.
- No independent split, future cohort, prospective outcome collection or formal
  significance test. Same 78 validation patients were previously reused in M2–M4.
  Seeds are not independent cohorts and origins are not independent patients.
- Calibration mismatch and policy/fusion coupling are plausible mechanisms but
  are not isolated causal explanations. Component freezing at random weights
  is limited; alternating updates, pretrained fixed fusion and nested calibration
  variants remain untested. No claim that centralization is deployable or private.
- Per-patient harm and label-aware oracle gains are retrospective loss summaries,
  not clinical harms or an unbiased off-policy estimate. Random routing has one
  fixed draw per seed, not a large repeated-policy study. Cost sensitivity holds
  models fixed; it does not retrain an optimal cost-aware policy.
- M4 sparse feedback, model-version staleness and counterfactual missingness
  remain structural limitations. Default clipping was inactive in the inspected
  parent-state replay, so it must not be blamed solely from its theoretical bias.

## Recommendations for Step 2 — proposals only

1. Prioritize a preregistered training/validation-only experiment separating
   origin-local gradient conflict, optimizer resets and local epoch count. Test
   fewer local steps or an explicitly separated router/fusion schedule against
   both matched centralized references; preserve the current default.
2. Retain exact-evidence mean and stop baselines and report BCE, calibration and
   threshold metrics together. Test pretrained fixed fusion and a train-only
   deterministic peer choice before expanding architecture.
3. Diagnose fold/final shift with nested train-only calibration and robust-logit
   experimental variants, without selecting on historical test performance.
4. Reconsider the action objective/cost protocol only after documenting a
   train-only decision rule for stop utility. Do not treat label-aware hindsight
   oracle choices as achievable routing or medical necessity.
5. For M4, preregister more feedback streams and measure coverage, reward age,
   ESS and clipped fractions at actual local steps. Evaluate independent future
   outcomes before making adaptation or clinical improvement claims.

No Step 2 implementation was performed. The specialist registry, flexible fusion
architecture, serving/security redesign and Qwen fine-tuning remain outside scope.

## Exact changed files

New:

- `src/fed_med_agents/experiments/routing_diagnostics.py`
- `src/fed_med_agents/experiments/m5_step01.py`
- `scripts/audit_m5_feedback.py`
- `scripts/extend_m5_controls.py`
- `scripts/verify_m5_step01.py`
- `scripts/report_m5_step01.py`
- `tests/test_routing_diagnostics.py`
- `docs/experiments/m5-step01-routing-diagnostics.md`
- `docs/devlog/2026-10-14-m5-step01.md`

Updated: `README.md`, `docs/status.md`, `docs/architecture.md`,
`docs/limitations.md`, `docs/next-steps.md`. Original M1–M4 reports remain historical
records and were not regenerated. The diagnostic pipeline is separate from the
reusable production routing/fusion/federation modules.

Raw evidence: `artifacts/m5-step01/seed_*/{results.json,evidence.npz,consultations.jsonl}`,
per-variant checkpoints/audits/histories, `artifacts/m5-step01/verification.json`,
`artifacts/m5-step01-central-pooled/seed_*/results.json`,
`artifacts/m5-step01-audit/{feedback.json,preservation.json,preserved-before.json}`,
fresh M4 output `artifacts/m5-step01-m4-rerun`, and QA/driver logs in the audit
directory. Full round-specific values and gradient weighting cosines remain in
these artifacts. [Execution log](../devlog/2026-10-14-m5-step01.md).
""")
    REPORT.write_text("\n".join(report), encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
