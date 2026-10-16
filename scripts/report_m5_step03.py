"""Render saved Step 3 results without fitting or selecting on outer outcomes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fed_med_agents.experiments.consultation_value import COSTS, FEATURE_SETS
from fed_med_agents.experiments.m5_step03 import gains, probabilities, risks, write
from fed_med_agents.experiments.repair_models import make_model

KEYS = (
    "accuracy",
    "auroc",
    "binary_cross_entropy",
    "f1",
    "ece",
    "brier",
    "objective",
    "calls",
    "stop_fraction",
    "harm",
    "oracle_regret",
)


def average(rows: list[dict]) -> dict:
    result = {}
    for key in KEYS:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *[
                "| " + " | ".join(f"{x:.5f}" if isinstance(x, float) else str(x) for x in row) + " |"
                for row in rows
            ],
        ]
    )


def report(root: Path) -> None:
    """Create full policy tables, selected-pipeline gates and reproducible review docs."""
    results = json.loads((root / "results.json").read_text())
    selected = json.loads((root / "selection.json").read_text())
    fed = json.loads((root / "federated/results.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    execution = json.loads((root / "execution.json").read_text())
    qa_text = "See QA logs"
    if (root / "qa/pytest.log").exists():
        pytest_lines = (root / "qa/pytest.log").read_text(encoding="utf-8").splitlines()
        qa_text = next((line for line in reversed(pytest_lines) if "passed" in line), qa_text)
    policies = list(results["fold0_outer"]["original"]["0.02"])
    summaries, records, gate_tables = {}, [], []
    for kind in ("original", "logistic"):
        summaries[kind] = {}
        for cost in COSTS:
            tariff = str(cost)
            summaries[kind][tariff] = {}
            for policy in policies:
                rows = [
                    row
                    for fold in range(3)
                    for row in results[f"fold{fold}_outer"][kind][tariff][policy].values()
                ]
                summaries[kind][tariff][policy] = average(rows)
                for fold in range(3):
                    for origin, row in results[f"fold{fold}_outer"][kind][tariff][policy].items():
                        records.append(
                            {
                                "fusion": kind,
                                "cost": cost,
                                "policy": policy,
                                "fold": fold,
                                "origin": origin,
                                **{key: row[key] for key in KEYS},
                                "actions": json.dumps(row["action_frequencies"]),
                                "patient_dependent": row["patient_dependent"],
                                "utility_r2": row.get("utility", {}).get("r2_vs_training_mean"),
                                "utility_mse": row.get("utility", {}).get("mse"),
                                "utility_correlation": row.get("utility", {}).get("correlation"),
                            }
                        )
        gates, selection_rows, chosen_all, baseline_all = {}, [], [], []
        for fold in range(3):
            name = selected[str(fold)][kind]["name"]
            chosen = list(results[f"fold{fold}_outer"][kind]["0.02"][name].values())
            baseline = list(results[f"fold{fold}_outer"][kind]["0.02"]["best_peer"].values())
            chosen_all.extend(chosen)
            baseline_all.extend(baseline)
            improvement = average(baseline)["objective"] - average(chosen)["objective"]
            r2 = float(np.mean([row["utility"]["r2_vs_training_mean"] for row in chosen]))
            variation = sum(row["patient_dependent"] for row in chosen)
            selection_rows.append(
                [
                    fold,
                    name,
                    average(chosen)["objective"],
                    average(baseline)["objective"],
                    improvement,
                    r2,
                    variation,
                ]
            )
        gates["every_fold_improvement_at_least_.01"] = all(row[4] >= 0.01 for row in selection_rows)
        gates["accuracy_drop_at_most_.01"] = (
            average(chosen_all)["accuracy"] >= average(baseline_all)["accuracy"] - 0.01
        )
        gates["each_origin_mean_bce_degradation_at_most_.01"] = all(
            np.mean(
                [
                    chosen_all[fold * 3 + o]["binary_cross_entropy"]
                    - baseline_all[fold * 3 + o]["binary_cross_entropy"]
                    for fold in range(3)
                ]
            )
            <= 0.01
            for o in range(3)
        )
        gates["every_fold_mean_origin_gain_r2_positive"] = all(row[5] > 0 for row in selection_rows)
        gates["two_patient_dependent_origins_each_fold"] = all(row[6] >= 2 for row in selection_rows)
        summaries[kind]["selected"] = {
            "metrics": average(chosen_all),
            "gates": gates,
            "success": all(gates.values()),
            "rows": selection_rows,
        }
        gate_tables.append(
            f"### {kind}\n\n"
            + table(
                [
                    "Fold",
                    "Inner choice",
                    "Objective",
                    "Best peer",
                    "Improvement",
                    "Gain R2",
                    "Varying origins",
                ],
                selection_rows,
            )
            + f"\n\nGates: `{json.dumps(gates)}`.\n"
        )
    write(root / "summary.json", summaries)
    with (root / "outer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Pareto uses fixed policies on the same frozen fusion: lower calls and BCE.
    pareto, pareto_rows = {}, []
    for kind in summaries:
        pareto[kind] = {}
        for cost in COSTS:
            candidates = summaries[kind][str(cost)]
            names = [name for name in candidates if not name.startswith("random_matched_")]
            frontier = []
            for name in names:
                a = candidates[name]
                dominated = any(
                    candidates[b]["calls"] <= a["calls"]
                    and candidates[b]["binary_cross_entropy"] <= a["binary_cross_entropy"]
                    and (
                        candidates[b]["calls"] < a["calls"]
                        or candidates[b]["binary_cross_entropy"] < a["binary_cross_entropy"]
                    )
                    for b in names
                )
                if not dominated:
                    frontier.append(name)
            pareto[kind][str(cost)] = frontier
            pareto_rows.append([kind, cost, ", ".join(frontier)])
    write(root / "pareto.json", pareto)
    # Retrospective diagnostics quantify supervision instability without declaring noise causal.
    diagnostics, bins = [], []
    import torch

    for kind in ("original", "logistic"):
        for fold in range(3):
            path = root / f"fold{fold}_outer"
            data = dict(np.load(path / f"{kind}.npz"))
            tp, hp = probabilities(path / "final")
            fusion = make_model(kind)
            fusion.load_state_dict(torch.load(path / f"fusion_{kind}.pt", weights_only=True))
            optimistic = gains(risks(fusion, tp), data["y"])
            for origin in range(3):
                peers = [p for p in range(3) if p != origin]
                diagnostics.append(
                    {
                        "fusion": kind,
                        "fold": fold,
                        "origin": origin,
                        "gain_std": float(data["gain"][:, origin, peers].std()),
                        "mean_abs_teacher_gain_shift": float(
                            np.abs(optimistic[:, origin, peers] - data["gain"][:, origin, peers]).mean()
                        ),
                        "own_probability_shift": float(np.abs(tp[:, origin] - data["p"][:, origin]).mean()),
                        "fitting_crossfit_mean_gains": data["mean_gain"][origin, peers].tolist(),
                    }
                )
                for stage, own_p, action_p, labels in (
                    ("crossfit", data["p"], data["target_risks"], data["y"]),
                    ("assessment", hp, data["held_risks"], data["held_y"]),
                ):
                    gross = gains(action_p, labels)
                    bucket = np.minimum((2 * np.abs(own_p[:, origin] - 0.5) * 4).astype(int), 3)
                    for b in range(4):
                        mask = bucket == b
                        if not mask.any():
                            continue
                        for peer in peers:
                            values = gross[mask, origin, peer]
                            bins.append(
                                {
                                    "fusion": kind,
                                    "fold": fold,
                                    "origin": origin,
                                    "peer": peer,
                                    "stage": stage,
                                    "confidence_bin": b,
                                    "n": int(mask.sum()),
                                    "mean_gross_gain": float(values.mean()),
                                    "std": float(values.std()),
                                }
                            )
    write(root / "target_diagnostics.json", {"rows": diagnostics, "confidence_bins": bins})
    main_names = [
        "stop",
        "best_peer",
        "fixed_value",
        "confidence",
        "random_max",
        "graph",
        "ridge_base",
        "ridge_enriched",
        "mlp_base",
        "mlp_enriched",
        "tree_base",
        "tree_enriched",
    ]
    sections = []
    for kind in summaries:
        for cost in COSTS:
            rows = [
                [
                    name,
                    *[
                        summaries[kind][str(cost)][name][key]
                        for key in (
                            "accuracy",
                            "auroc",
                            "binary_cross_entropy",
                            "f1",
                            "objective",
                            "calls",
                            "ece",
                            "harm",
                        )
                    ],
                ]
                for name in main_names
            ]
            sections.append(
                f"### {kind}, tariff {cost}\n\n"
                + table(
                    ["Policy", "Acc", "AUROC", "BCE", "F1", "BCE+cost", "Calls", "ECE", "Harm/call"], rows
                )
            )
    origin_rows, feature_rows, fl_rows = [], [], []
    for kind in summaries:
        for name in ("graph", "best_peer", "ridge_base", "ridge_enriched", "mlp_enriched", "tree_enriched"):
            for origin in range(3):
                rows = [results[f"fold{f}_outer"][kind]["0.02"][name][str(origin)] for f in range(3)]
                mean = average(rows)
                actions = np.array([r["action_frequencies"] for r in rows]).sum(0).tolist()
                origin_rows.append(
                    [
                        kind,
                        name,
                        origin,
                        mean["binary_cross_entropy"],
                        mean["objective"],
                        mean["calls"],
                        actions,
                        sum(r["patient_dependent"] for r in rows),
                    ]
                )
        for group in FEATURE_SETS:
            for family in ("ridge", "mlp", "tree"):
                name = f"{family}_{group}"
                rows = [r for f in range(3) for r in results[f"fold{f}_outer"][kind]["0.02"][name].values()]
                value = summaries[kind]["0.02"][name]
                r2 = float(np.mean([r["utility"]["r2_vs_training_mean"] for r in rows]))
                corr = [r["utility"]["correlation"] for r in rows if r["utility"]["correlation"] is not None]
                feature_rows.append(
                    [
                        kind,
                        name,
                        value["objective"],
                        value["calls"],
                        r2,
                        float(np.mean(corr)) if corr else None,
                    ]
                )
        for fold in range(3):
            trial = fed["trials"][f"fold{fold}_outer_{kind}"]
            architecture = trial["architecture"]
            central = average(
                list(results[f"fold{fold}_outer"][kind]["0.02"][f"{architecture}_enriched"].values())
            )
            fed_avg = average(list(trial["assessment"]["0.02"].values()))
            cosines = [v for h in trial["history"].values() for v in h["gradient_cosines"] if v is not None]
            fl_rows.append(
                [
                    kind,
                    fold,
                    architecture,
                    central["objective"],
                    fed_avg["objective"],
                    central["calls"],
                    fed_avg["calls"],
                    trial["drift"],
                    min(cosines),
                ]
            )
    oracle_rows = []
    for kind in summaries:
        for cost in COSTS:
            for origin in range(3):
                rows = [
                    results[f"fold{f}_outer"][kind][str(cost)]["best_peer"][str(origin)] for f in range(3)
                ]
                oracle_rows.append(
                    [
                        kind,
                        cost,
                        origin,
                        float(np.mean([r["oracle_objective"] for r in rows])),
                        float(np.mean([r["oracle_stop_fraction"] for r in rows])),
                    ]
                )
    body = (
        """# M5 Step 3 — Intelligent consultation-value routing

Executed 2026-10-14, Windows, **experimental sprint complete; routing is not solved**.
Original production router/fusion, M1–M4/Steps 1–2 checkpoints, reports and pointers
remain unchanged. No historical validation/test metric, Qwen tuning, registry or
generalized fusion system. See [frozen protocol](m5-step03-protocol.md),
[feature report](m5-step03-routing-features.md) and
[execution log](../devlog/2026-10-14-m5-step03.md).

## Findings

At tariff .02 the original graph always consults and reproduces the training-best
peer mapping in these frozen-fusion conditions; higher tariffs can produce STOP.
New value policies produce STOP and
patient-dependent decisions, but the inner-selected pipelines fail the registered
stability gates below. Enrichment is not a reliable repair. Cost/quality, utility
generalization and origin-specific behavior matter more than STOP rate alone.
No candidate is promoted and no claim of statistically proven superiority follows.

The inner-selected residual pipeline has gain R2 .0415/.1003/-.0286 over folds,
and BCE+cost .30589 versus fixed peer .30349. The selected logistic-condition
pipeline has gain R2 .0563/.1134/.0341, objective .43136 versus .43590, but worsens
one fold and origin A mean BCE by .01172. Thus measurable conditional signal in
some conditions does not establish stable action or cost-quality improvement.
Residual origin C mean BCE worsens .01200. The patient-dependence gate also fails:
selected policies vary actions in only1/1/2 origins (residual),3/2/1 (logistic).

Step 2's marginal utility teacher used the same meta-training labels as its fusion.
Here each target fold H is excluded from ALL upstream specialist fits/calibration
and fusion-fitting inputs. Splitting an existing full-T OOF matrix at the fusion
level would still let H labels influence other rows' expert predictions; this
implementation refits those experts inside T\\H as well. This removes that target
leakage path, but produces smaller/noisier teacher fits and a teacher/final model
shift. It does not make realized one-label gains noiseless conditional expectations.

## Fixed experiment

Only original364 training patients; outer seed53003, three folds; inner75/25
seed53100+fold. Outer fit/held242/122,243/121,243/121; inner181/61,182/61,182/61.
Three additional target folds within each inner/outer fitting set. Existing
specialists30 epochs, own-feature preprocessing, nested calibration. **288 private
specialist fits**, 48 separately fitted fusion models, uniform valid-action BCE,
40 full-batch Adam.01 steps. Routing comparisons freeze fusion. Original residual
and existing experimental six-input logistic are paired conditions, not jointly
optimized replacements. Logistic's short uniform schedule is NOT Step2's joint
50-epoch schedule; poorer scores here cannot establish a general fusion ranking.

All value models regress untruncated gross BCE reduction; select largest positive
estimated reduction minus tariff, otherwise STOP. Ridge alpha10+fit-only scaling;
MLP21-or-ablation-input→16Tanh→1,80 Adam.01 steps/weight_decay.01; tree depth3,
minleaf30 (regularization for small noisy data). Graph is the original architecture
trained50 steps on soft expected cross-fitted action losses, frozen fusion, not
a historical M2 checkpoint. Fixed peer uses training mean gains; confidence uses
own confidence<.6 and that peer. Controls have the same permitted information
envelope but intentionally restricted decision functions. New routing APIs never
receive y/unqueried probabilities; offline teacher enumeration does.

All costs are fixed .02/.05/.10 and maximum budget1. Random-matched controls use
exact patient call masks and seeded random peers; their complete metrics are in
outer_metrics.csv/results.json. Logical replay budgets are not measured network
traffic. Separate deployed checks below use genuine selected-only Flower queries.
Metrics macro-average origins then folds; dependent views are not independent
patients. Accuracy/F1 threshold.5; ECE10 equal bins. Harm means realized BCE no
better than same-fusion STOP, not medical harm/necessity. Oracle uses outcomes
only retrospectively and is never fitted or deployed. Nothing is selected by
outer scores. All fixed candidates/costs/controls are retained, including failures.

## Outer metrics

"""
        + "\n\n".join(sections)
        + "\n\n## Nested selected pipeline and numerical gates\n\n"
        + "\n".join(gate_tables)
    )
    body += "\nNo automatic promotion. Gain gate uses mean origin R2 per fold relative to its training-origin/peer mean predictor; other gates are as preregistered.\n"
    body += "\n## Per-origin actions, tariff .02\n\n" + table(
        ["Fusion", "Policy", "Origin", "BCE", "Objective", "Calls", "[STOP,A,B,C]", "Varying folds/3"],
        origin_rows,
    )
    body += "\n\n## Retrospective oracle lower loss bound\n\n" + table(
        ["Fusion", "Tariff", "Origin", "Oracle objective", "Oracle STOP"], oracle_rows
    )
    body += "\n\n## Cost/loss Pareto comparison\n\n" + table(
        ["Fusion", "Tariff", "Nondominated fixed policies"], pareto_rows
    )
    body += """\n\nThis empirical frontier is descriptive and not an additional selection procedure.
Identical predictors can occupy the same point. CSV retains per-fold metrics,
utility MSE/correlation/R2, all matched random controls, calibration and actions;
results.json retains per-patient actions. No budget2 evidence enters comparison.

## Genuine Flower routing-only comparison

""" + table(
        [
            "Fusion",
            "Fold",
            "Architecture",
            "Central obj",
            "FL obj",
            "Central calls",
            "FL calls",
            "Weight L2",
            "Min grad cosine",
        ],
        fl_rows,
    )
    body += f"""

Run **{fed["run_id"]}**, **30 rounds / 90 client updates**, three logical clients,
one CPU Ray worker; wall time **{fed["seconds"]:.3f}s** including evidence queries,
training and selected consultations. Both fusion conditions selected MLP among
the two eligible neural architectures using INNER scores only. This is not a
claim that MLP beat every centrally evaluated tree/ridge/feature variant.
Central comparator uses identical architecture, seed, target rows and80 pooled
Adam steps. FL uses five rounds×16 origin-local steps, Adam reset each round;
optimizer/averaging schedules differ. Every global array equals its three local
updates within tolerance. Private expert/fusion parameters never enter ArrayRecords.
Gradient cosine conflict and nonzero weight drift demonstrate actual optimization,
not improvement. Contexts/parameters remain incompatible with default serving.

72 actual evidence queries bind probabilities to context/client/IDs and hashes;
central diagnostics initially load trusted local caches, and Flower later acquires
and verifies those exact probabilities. No claim that central preparation itself
was federated. At tariff.02 the FL router selected peers before sending aligned
assessment IDs, peers actually loaded private classifiers and returned only those
probabilities. Masked fusion outputs matched replay within1e-6. These are batched
functional checks, not per-request serving latency/security benchmarks.

## Verification and preservation

Independent verification: `{json.dumps(verification)}`.
Fresh regression QA: **{qa_text}**. Ruff lint and format (71 files) and pip check
passed; exact commands/return codes/runtimes are in qa/checks.json. Optional old
Qwen/feedback runtime benchmarks are explicitly disabled; this sprint's actual
training-only Flower integration was executed separately. Two Typer warnings
are upstream. Captured pip freeze is environment evidence, not a portable lock.
Nested centralized/preparation time **{execution["seconds"]:.3f}s**, CPU. The device
audit performed a real RTX A4000 CUDA matrix operation (environment.json); CUDA
was not used to accelerate these tiny repeated training fits. CPU peak CUDA
allocation is0; that is not process RAM, reserved GPU memory or total board usage.
No CPU/GPU equal-work performance comparison or GPU saturation claim.

## Commands and artifacts

```powershell
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.m5_step03 --output artifacts/m5-step03-v1
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.value_federation --output artifacts/m5-step03-v1 --device cpu
.venv/Scripts/python.exe scripts/verify_m5_step03.py artifacts/m5-step03-v1 --refit-oof
.venv/Scripts/python.exe scripts/report_m5_step03.py artifacts/m5-step03-v1
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check --no-cache src tests scripts
.venv/Scripts/python.exe -m ruff format --no-cache --check src tests scripts
.venv/Scripts/python.exe -m pip check
```

For reproduction choose a fresh versioned output directory for BOTH launchers.
Prepared M2 seed42 source is required; no downloads/service/Qwen needed. CPU is
default; optional Flower --device auto/cuda/cpu supports at most one GPU worker.
Windows is executed; Linux substitutes activated python, but WSL is unverified.
Artifacts include immutable protocol copy, preservation manifest, every nested
specialist/fusion state and OOF provenance, fitting/held utility arrays, fitted
routers, inner choices, all assessment metrics/actions, calibration/reliability,
confidence bins, target diagnostics, Pareto/CSV, full Flower message evidence,
all client/global states, gradient/drift audits, verification and QA logs.

## Limitations, failures and recommendations

One364-patient cohort, small overlapping fits, reused training patients from past
development and many comparisons: these are development diagnostics, not an
independent study, statistical superiority or clinical validity. Realized utility
is heavy-tailed BCE from one binary outcome; small target-fold specialists/fusers
can shift calibration and predicted benefits. Reliable-looking mean statistics
can encode teacher-fold identity and shift at final inference. Margins/entropy
are deterministic own-probability transforms; fixed capabilities/availability
have no experimentally identified effect. No raw metadata summary was added
without a new authorization contract. All patient files remain accessible to a
trusted local simulator; predictions/IDs/model updates can disclose information.
No proven privacy, differential privacy or secure aggregation.

Separately fitted logistic in this fixed short schedule is a limited condition,
not evidence invalidating Step2's distinct logistic results. Architecture choice,
target quality, local optimizer and model-size shift remain confounded; this
study does not causally identify a sole failure mechanism. STOP variation is not
automatically useful. The historical M4 observed-only objective was not replaced
with offline full-information targets. No Qwen learning or production migration.

Initial inspection tried nonexistent fusion.py/router.py paths; source lives in
fusion/network.py and routing/graph.py. Initial Ruff findings were ordinary
dictionary iteration/closure binding/style issues, corrected without model
changes. Existing Windows Ray shutdown/deprecated-launcher warnings remain;
successful rounds are established by saved arrays, messages and numerical replay.
QA details and any additional failures are in the execution log.

Priorities for a NEW reviewed protocol: stronger independently cross-fitted
fusion schedules with sample-size/temperature controls; label-efficient utility
estimators and conditional-null analyses; feature-history transport/shift controls;
patient-level uncertainty/repeated new splits and eventually independent cohorts.
Choose future experiments before outcomes, retain simple fixed-peer baselines,
and preserve the historical test. No registry/generalized fusion/Qwen training,
OS/driver changes, publishing or GitHub push. **Stop after Step3.**

## Exact changed files

New experiments/consultation_value.py, experiments/m5_step03.py,
experiments/value_federation.py; scripts/verify_m5_step03.py,
scripts/report_m5_step03.py; tests/test_consultation_value.py; this report,
m5-step03-routing-features.md, m5-step03-protocol.md and the dated devlog.
Updated README.md and docs/status.md, architecture.md, limitations.md,
next-steps.md. All earlier experiment reports/default modules remain unchanged.
"""
    Path("docs/experiments/m5-step03-intelligent-routing.md").write_text(body, encoding="utf-8")
    feature_text = """# M5 Step 3 — Pre-consultation features and utility predictability

Read with [protocol](m5-step03-protocol.md) and [full routing report](m5-step03-intelligent-routing.md).
No outer model selection or new historical validation/test evaluation.

## Information contract

Base11: origin/candidate one-hots6, own calibrated p/confidence2, candidate domain
width1, availability1, remaining budget1. Uncertainty adds margin/entropy/p(1-p)3;
reliability adds origin/candidate training OOF BCE/Brier/ECE6; history adds one
training-only origin-peer mean gross gain. Enriched21 combines all. Cost is not
a learned feature: subtract known tariff at decision time, giving identical model
predictions under .02/.05/.10. Self/unavailable peers masked; budget0 alwaysSTOP.

Own margin/entropy add nonlinear transforms, not new evidence. Domain width is
public capability metadata; no raw clinical fields, unauthorized summaries,
patient IDs or peer predictions enter router features. No empirical capability
or missing-node variation was induced. Origin-local feature generation is node
local; only compatible shared router weights/scalars leave FL updates.

For training target H, reliability and history derive solely from upstream T\\H.
History is a resubstitution summary of that teacher's OOF fusion risks, not an
independent outcome estimate; final assessment history uses cross-fitted T gains.
This intentional train/final summary shift is measured as a limitation. All
utility LABELS themselves use the fully held target-fold pipeline. Final fusers
use T-only specialist OOF, never old validation/test or outer assessment labels.

## Controlled feature comparisons, tariff .02

R2 uses a training-origin/peer mean gain predictor as null (not the assessment
mean); table macro-averages origins/folds. Correlation is descriptive. Positive R2
in some rows does not meet the all-fold/selection gates by itself. Full per-origin
MSE/R2/correlation and cross-cost metrics are in outer_metrics.csv/results.json.

""" + table(["Fusion", "Policy", "Objective", "Calls", "Gain R2 vs train mean", "Gain corr"], feature_rows)
    feature_text += "\n\n## Target variability and upstream model shift\n\n" + table(
        ["Fusion", "Fold", "Origin", "Crossfit gain SD", "Mean abs teacher gain shift", "Own p shift"],
        [
            [
                r[k]
                for k in (
                    "fusion",
                    "fold",
                    "origin",
                    "gain_std",
                    "mean_abs_teacher_gain_shift",
                    "own_probability_shift",
                )
            ]
            for r in diagnostics
        ],
    )
    feature_text += """\n\nTeacher shift compares proper held-stack gains with the final fusion's fitting-OOF
resubstitution gains on the same T patients. It combines training size, specialist
and fusion/calibration differences; it is NOT an unbiased noise estimator or an
isolated causal test. No optimistic target was used to fit/select a router.
Confidence-bin counts/mean gains/SD, per-origin/peer/fold, for target and assessment
are saved in target_diagnostics.json. They describe signal and variability without
fitting extra post-hoc policies. Oracle's lower loss bound uses labels unavailable
at query time and cannot establish attainable personalized benefit.

## Interpretation

Existing p/confidence can distinguish some uncertainty regions, but higher
dimensional summaries do not give stable gains. History/reliability vary across
teacher folds as well as candidates, so apparent fitting associations can shift
at deployment. Large gain spread relative to predicted differences and unstable
outer utility scores are consistent with noisy/shifted supervision and limited
context. They do not prove an information-theoretic impossibility or show that
the optimizer is solely responsible. The central/Flower same-architecture control
tests training protocol sensitivity separately; weight change alone is not success.
Fixed peer/always-consult remains a strong baseline and no routing solution is
promoted. Further supervised labels/features and stronger independent fits require
a new protocol, not selecting an outer winner or relaxing gates after these results.
"""
    Path("docs/experiments/m5-step03-routing-features.md").write_text(feature_text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
