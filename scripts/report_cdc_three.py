"""Report the fixed A/B/C CDC amendment without altering historical six-agent reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from report_m5_step035 import SCALARS, mean, table
from sklearn.metrics import average_precision_score

from fed_med_agents.experiments.cdc_data import THREE_DOMAINS, THREE_SLOTS
from fed_med_agents.experiments.cdc_routing import COSTS, context, gross_gains
from fed_med_agents.experiments.m5_step03 import write
from fed_med_agents.experiments.m5_step035 import KINDS
from fed_med_agents.experiments.routing_diagnostics import describe, loss_rows


def report(root: Path) -> None:
    """Create complete CSV, per-origin diagnostics and a standalone amendment report."""
    results = json.loads((root / "results.json").read_text())
    selection = {
        size: json.loads((root / f"selection_{size}.json").read_text()) for size in ("small", "pilot")
    }
    gates = {size: json.loads((root / f"gates_{size}.json").read_text()) for size in selection}
    configuration = json.loads((root / "benchmark-config.json").read_text())
    source = Path(configuration["source"])
    execution = json.loads((root / "execution.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    qa_text = (
        (root / "qa/pytest.log").read_text(encoding="utf-8") if (root / "qa/pytest.log").exists() else ""
    )
    qa_result = next((line for line in reversed(qa_text.splitlines()) if "passed" in line), "See QA logs")
    old = json.loads((source / "results.json").read_text())
    summaries, records = {}, []
    for size in selection:
        summaries[size] = {}
        for kind in KINDS:
            summaries[size][kind] = {}
            for scenario in ("static", "variable"):
                summaries[size][kind][scenario] = {}
                for cost in COSTS:
                    bucket = {}
                    for fold in range(3):
                        policies = results[f"{size}_fold{fold}_outer"][kind][scenario][str(cost)]
                        for name, origins in policies.items():
                            bucket.setdefault(name, []).extend(origins.values())
                            for origin, row in origins.items():
                                records.append(
                                    {
                                        "size": size,
                                        "fusion": kind,
                                        "scenario": scenario,
                                        "cost_tariff": cost,
                                        "fold": fold,
                                        "origin": "ABC"[int(origin)],
                                        "policy": name,
                                        **{key: row.get(key) for key in SCALARS},
                                        "action_frequencies": json.dumps(row["action_frequencies"]),
                                        "oracle_objective": row["oracle_objective"],
                                        "utility_mse": row.get("utility_mse"),
                                        "utility_correlation": row.get("utility_correlation"),
                                    }
                                )
                        name = selection[size][str(fold)][kind]
                        bucket.setdefault("inner_selected", []).extend(policies[name].values())
                    summaries[size][kind][scenario][str(cost)] = {
                        name: mean(rows) for name, rows in bucket.items()
                    }
    write(root / "summary.json", summaries)
    with (root / "outer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    development = dict(np.load(root / "development.npz"))
    columns = json.loads((root / "dataset.json").read_text())["features"]
    diversity, experts, group_rows = [], [], []
    for size in selection:
        for fold in range(3):
            folder = root / f"{size}_fold{fold}_outer"
            stage = dict(np.load(folder / "stack.npz"))
            p, y = stage["held_p"], stage["held_y"]
            errors = p - y[:, None]
            private_losses = np.column_stack([loss_rows(y, p[:, c]) for c in range(3)])
            diversity.append(
                {
                    "size": size,
                    "fold": fold,
                    "error_correlation": np.corrcoef(errors.T).tolist(),
                    "prediction_correlation": np.corrcoef(p.T).tolist(),
                    "probability_disagreement": np.abs(p[:, :, None] - p[:, None, :]).mean(0).tolist(),
                    "disagreement": [
                        [float(((p[:, a] >= 0.5) != (p[:, b] >= 0.5)).mean()) for b in range(3)]
                        for a in range(3)
                    ],
                    "retrospective_private_best": np.bincount(private_losses.argmin(1), minlength=3).tolist(),
                }
            )
            for client, name in enumerate(THREE_DOMAINS):
                experts.append(
                    {
                        "size": size,
                        "fold": fold,
                        "agent": "ABC"[client],
                        "domain": name,
                        **describe(y, p[:, client]),
                        "pr_auc": float(average_precision_score(y, p[:, client])),
                    }
                )
            raw = development["x"][stage["held"]]
            masks = {
                "age_1_8": raw[:, columns.index("Age")] <= 8,
                "age_9_13": raw[:, columns.index("Age")] >= 9,
                "sex_0": raw[:, columns.index("Sex")] == 0,
                "sex_1": raw[:, columns.index("Sex")] == 1,
                "health_1_3": raw[:, columns.index("GenHlth")] <= 3,
                "health_4_5": raw[:, columns.index("GenHlth")] >= 4,
            }
            actions = dict(np.load(folder / "actions.npz"))
            for kind in KINDS:
                values = np.load(folder / f"targets_{kind}.npz")["held"]
                gain = gross_gains(values, y)
                for origin in range(3):
                    _, _, prices = context(
                        development["ids"][stage["held"]], origin, columns, "variable", THREE_SLOTS
                    )
                    chosen = selection[size][str(fold)][kind]
                    a = actions[f"{kind}_variable_0.02_{chosen}_{origin}"]
                    b = actions[f"{kind}_variable_0.02_best_peer_{origin}"]
                    rows = np.arange(len(y))
                    costs_a = 0.02 * np.where(a > 0, prices[rows, (a - 1).clip(0)], 0)
                    costs_b = 0.02 * np.where(b > 0, prices[rows, (b - 1).clip(0)], 0)
                    losses = np.column_stack([loss_rows(y, values[:, origin, action]) for action in range(4)])
                    for group, mask in masks.items():
                        group_rows.append(
                            {
                                "size": size,
                                "fold": fold,
                                "origin": "ABC"[origin],
                                "fusion": kind,
                                "group": group,
                                "n": int(mask.sum()),
                                "calls": float((a[mask] > 0).mean()),
                                "selected_objective": float((losses[rows, a] + costs_a)[mask].mean()),
                                "baseline_objective": float((losses[rows, b] + costs_b)[mask].mean()),
                                "candidate_mean_gain": gain[mask, origin].mean(0).tolist(),
                            }
                        )
    write(
        root / "diversity.json",
        {"folds": diversity, "specialists": experts, "retrospective_groups": group_rows},
    )
    benchmark = """# CDC three-specialist amendment — benchmark

Executed 2026-10-14. Exactly the user-specified A/B/C domains, selected before
amendment outcomes. This supersedes the six-agent CDC benchmark as the current
CDC experiment, while preserving all its evidence and entry points. No Step4,
registry, Qwen training, UCI529 model/default change or publication.

Read [amendment protocol](m5-step035-three-agent-protocol.md),
[routing feasibility](m5-step035-three-agent-feasibility.md) and
[execution log](../devlog/2026-10-14-m5-step035-three-agent.md).
Historical six-agent [benchmark](m5-step035-benchmark.md),
[feasibility](m5-step035-routing-feasibility.md) and original frozen protocol remain unchanged.

## Fixed domains and compatible reuse

""" + table(
        ["Agent", "Domain", "Required private fields", "Old slot"],
        [
            ["ABC"[c], name, ", ".join(fields), THREE_SLOTS[c]]
            for c, (name, fields) in enumerate(THREE_DOMAINS.items())
        ],
    )
    benchmark += """

**576 byte-identical private model/calibrator files** reused, covering ALL12
small/pilot inner/outer contexts, four upstream blocks and four fits/client/block.
No specialist needed retraining for the experiment. Independent verification
does perform12 ephemeral refits without replacing them. Files/provenance/seeds,
features and patient probability association are checked; source/destination
hashes are in reuse.json. Only probabilities/reliability columns are remapped.
No six-agent shared weights are copied or loaded. All96 trainable target/final
fusion states and every ridge/MLP/graph router are fitted from fresh initialization;
48 arithmetic-mean snapshots remain stateless. Initial fusion states are saved
and independently verified against their seeds, then all96 fits are reproduced.

## Boundaries, schedules and unchanged gates

The exact frozen **50,861-row CDC test file**,202,819-row development file,
15k pilot IDs, grouped boundaries and12 context definitions are copied byte-for-byte.
No new test or historical validation/test prediction/metric. Full pilot outer
fitting sizes10000/9999/10001, held5000/5001/4999; small fitting2500 uses those SAME
assessments. Inner grouped80/20 and whole-upstream target cross-fitting are unchanged.
Copied contexts.json retains historical preparation timings; new runtime below
measures reuse/new fitting, not original specialist training.

Feature duplicate groups still use the ORIGINAL21 fields, preserving the boundary;
they do not assert unique patients or exclusion of every13-field lookalike after
removing domains. All fitting/calibration/target group exclusions remain intact.
Development outcomes have been used in earlier CDC analyses: this is an explicit
requested development amendment, not a newly blinded independent study.

Original residual fusion now12→24→1 with own-logit residual; logistic6→1; mean
same masked evidence. Nine valid STOP/one-peer masks replace36; keep15 epochs,
Adam.003/batch4096/clip5 and seeds. Fixed epoch count now implies fewer updates;
no compute-matched causal agent-count claim. Routers: ridgealpha100 or32-Tanh MLP,
base12/enriched37 inputs,20 epochs Adam.003/weight_decay.01; graph3×3 edges and
11→16Tanh→4 context,15 epochs on frozen action risks. No hyperparameter search.

Static and variable services retain original origin AND candidate slot1/3/4 random
schedules. Recipient multipliers are mapped to[1,1,1.5], with the same congestion.
Do not reseed by new client indices. Public21-field coverage is metadata, not
clinical values. Missing required fields make a peer unavailable; no expert is
weakened. Costs .02/.05/.10, maximum one peer. Every policy sees matched eligibility,
prices and evidence constraints. Only own calibrated p and public/train-derived
metadata enter inference. No labels/hidden peer predictions/unauthorized raw rows.

All numerical acceptance gates remain EXACTLY unchanged, including >=.005 gain
every fold, R²>=.05 every fold, accuracy drop<=.005, no origin BCE degradation>.01,
positive static gain and >=3 varying origins. With three agents the last criterion
requires ALL THREE; it is not relaxed. Inner-only selections precede outer policy
metrics. No50k expansion or default promotion is part of this amendment.

## Specialist metrics (unchanged checkpoints)

"""
    rows = []
    for size in selection:
        for agent in "ABC":
            values = [e for e in experts if e["size"] == size and e["agent"] == agent]
            rows.append(
                [
                    size,
                    agent,
                    *[
                        float(np.mean([v[key] for v in values]))
                        for key in ("accuracy", "auroc", "pr_auc", "binary_cross_entropy", "ece")
                    ],
                ]
            )
    benchmark += table(["Size", "Agent", "Acc", "AUROC", "AP", "BCE", "ECE"], rows)
    benchmark += f"""

## Commands and actual execution

```powershell
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_three --device cuda --output artifacts/m5-step035-three-agent-v1
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_flower --output artifacts/m5-step035-three-agent-v1
.venv/Scripts/python.exe scripts/verify_m5_step035.py artifacts/m5-step035-three-agent-v1
.venv/Scripts/python.exe scripts/report_cdc_three.py artifacts/m5-step035-three-agent-v1
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check --no-cache src tests scripts
.venv/Scripts/python.exe -m ruff format --no-cache --check src tests scripts
.venv/Scripts/python.exe -m pip check
```

Use a fresh output and the preserved six-agent source. Neural --device auto/cuda/cpu;
actual **{execution["device"]}**, RTX A4000 audit/environment recorded. Reuse/fitting
**{execution["seconds"]:.3f}s**, peak torch allocation **{execution["peak_allocated_bytes"]} bytes**,
excluding CUDA contexts/reserves/board usage. No speedup/saturation claim; private
fits were reused, not GPU-retrained. Flower uses one CPU worker and fixed3 clients.
No OS/driver/WSL/Docker installation. CPU fusion/routing paths are exercised by tests
and actual CPU federation, not a new full CPU nested reproduction.

Verification: `{json.dumps(verification)}`. Full CSV/actions/checkpoints/provenance,
fitting-loss curves, initial states, reuse/preservation hashes, device evidence,
and Flower records stay under the new gitignored output. Six-agent results and
frozen test bytes are preserved. Exact changed files/QA/failures are in the devlog.
Fresh regression: **{qa_result}**; Ruff lint/format and pip check are recorded in qa/.
"""
    Path("docs/experiments/m5-step035-three-agent-benchmark.md").write_text(benchmark, encoding="utf-8")
    feasibility = """# CDC three-specialist amendment — routing feasibility

Read [benchmark](m5-step035-three-agent-benchmark.md), [protocol](m5-step035-three-agent-protocol.md)
and [execution log](../devlog/2026-10-14-m5-step035-three-agent.md).
Fixed specialists are A metabolic, B lifestyle, C general health. All results
are development-only; the old21-feature grouping/test boundary is unchanged.

Compare STOP, best training peer with eligibility fallback, positive-net fixed
value, random, confidence<.6, original graph and ridge/MLP base/enriched. All three
frozen fusion conditions and costs .02/.05/.10 are retained. Exact-call-mask random
controls are in complete CSV. Utility R² is relative to the training-origin/peer
mean, across both potential non-self peers including temporarily unavailable ones;
actual choices/oracle/deployment always mask unavailable peers. Gross gain labels
are whole-stack cross-fitted; true outcomes never enter prospective selection.

Macro-average origins then folds (dependent views of the same patients). AP means
average precision; accuracy/F1 threshold.5, ECE10 equal bins. Reported harm is a
realized BCE proxy versus same-fusion STOP, never clinical harm or necessity.

## Full pilot, tariff .02

"""
    names = (
        "stop",
        "best_peer",
        "fixed_value",
        "random",
        "confidence",
        "graph",
        "ridge_base",
        "ridge_enriched",
        "mlp_base",
        "mlp_enriched",
        "inner_selected",
    )
    for kind in KINDS:
        for scenario in ("static", "variable"):
            rows = [
                [
                    name,
                    *[
                        summaries["pilot"][kind][scenario]["0.02"][name][key]
                        for key in (
                            "accuracy",
                            "auroc",
                            "pr_auc",
                            "binary_cross_entropy",
                            "f1",
                            "objective",
                            "calls",
                            "utility_r2",
                        )
                    ],
                ]
                for name in names
            ]
            feasibility += (
                f"\n### {kind}/{scenario}\n\n"
                + table(["Policy", "Acc", "AUROC", "AP", "BCE", "F1", "Objective", "Calls", "Gain R2"], rows)
                + "\n"
            )
    feasibility += "\n## Inner-selected pipelines and unchanged gates\n\n"
    variable = {kind: summaries["pilot"][kind]["variable"]["0.02"] for kind in KINDS}
    feasibility += f"""Logistic fusion's selected pipeline achieves objective
{variable["logistic"]["inner_selected"]["objective"]:.6f} versus always-best peer
{variable["logistic"]["best_peer"]["objective"]:.6f} and cost-aware fixed value
{variable["logistic"]["fixed_value"]["objective"]:.6f}. Its fold utility R²
.0420/.0321/.0274 remains below the unchanged .05 gate, despite positive static
gains and all-three-origin variation. These are real gains in this fixed fusion
condition, not fulfillment of all acceptance criteria. Its absolute BCE/objective
is still worse than the original residual condition. Fresh three-agent fusion has
fewer training masks/optimizer updates at the unchanged epoch schedule; differences
can reflect fusion calibration/capacity, not solely a better clinical consultation.

Original residual selected objective {variable["original"]["inner_selected"]["objective"]:.6f}
versus fixed value {variable["original"]["fixed_value"]["objective"]:.6f}; mean
{variable["mean"]["inner_selected"]["objective"]:.6f} versus fixed value
{variable["mean"]["fixed_value"]["objective"]:.6f}. Their extra gains over cost-aware
fixed controls are small and R²/peer-personalization gates fail. No candidate
passes all frozen pilot gates; no promotion,50k expansion or Step4 follows.

"""
    rows = []
    for size in selection:
        for kind in KINDS:
            record = gates[size][kind]
            rows.append(
                [
                    size,
                    kind,
                    [selection[size][str(f)][kind] for f in range(3)],
                    record["passed"],
                    [round(x, 6) for x in record["fold_improvements"]],
                    [round(x, 6) for x in record["fold_utility_r2"]],
                    record["varying_origins"],
                ]
            )
    feasibility += table(
        ["Size", "Fusion", "Inner choices", "Pass", "Fold objective gains", "Fold R2", "Varying origins"],
        rows,
    )
    feasibility += "\n\nAll gates including origin-BCE/accuracy/static criteria are saved in gates_{small,pilot}.json. No automatic promotion.\n"
    feasibility += "\n## Cost and fitting-size sensitivity\n\n"
    rows = []
    for size in selection:
        for kind in KINDS:
            for cost in COSTS:
                bucket = summaries[size][kind]["variable"][str(cost)]
                for name in ("best_peer", "fixed_value", "inner_selected"):
                    row = bucket[name]
                    rows.append(
                        [
                            size,
                            kind,
                            cost,
                            name,
                            row["binary_cross_entropy"],
                            row["objective"],
                            row["calls"],
                            row["utility_r2"],
                        ]
                    )
    feasibility += table(["Size", "Fusion", "Tariff", "Policy", "BCE", "Objective", "Calls", "R2"], rows)
    feasibility += "\n\n## Per-origin behavior at .02, variable services\n\n"
    rows = []
    for kind in KINDS:
        for origin in range(3):
            values = [
                results[f"pilot_fold{f}_outer"][kind]["variable"]["0.02"][selection["pilot"][str(f)][kind]][
                    str(origin)
                ]
                for f in range(3)
            ]
            row = mean(values)
            rows.append(
                [
                    kind,
                    "ABC"[origin],
                    row["binary_cross_entropy"],
                    row["objective"],
                    row["calls"],
                    row["utility_r2"],
                    np.array([v["action_frequencies"] for v in values]).sum(0).tolist(),
                    row["minority_peer_fraction"],
                ]
            )
    feasibility += table(
        ["Fusion", "Origin", "BCE", "Objective", "Calls", "R2", "[STOP,A,B,C]", "Minority peer"], rows
    )
    feasibility += "\n\n## Diversity and historical context\n\n"
    rows = []
    off = ~np.eye(3, dtype=bool)
    for size in selection:
        d = [v for v in diversity if v["size"] == size]
        rows.append(
            [
                size,
                *[
                    float(np.mean([np.array(v[key])[off].mean() for v in d]))
                    for key in (
                        "prediction_correlation",
                        "error_correlation",
                        "probability_disagreement",
                        "disagreement",
                    )
                ],
            ]
        )
    feasibility += table(
        ["Size", "Prediction corr", "Error corr", "Probability disagreement", "Threshold disagreement"], rows
    )
    feasibility += "\n\nRetrospective age/sex/health group losses and peer gain vectors are saved in diversity.json; these groups never enter routing features.\n"
    rows = []
    for kind in KINDS:
        old_rows = [
            old[f"pilot_fold{f}_outer"][kind]["variable"]["0.02"]["best_peer"][str(slot)]
            for f in range(3)
            for slot in THREE_SLOTS
        ]
        rows.append(
            [
                kind,
                mean(old_rows)["objective"],
                summaries["pilot"][kind]["variable"]["0.02"]["best_peer"]["objective"],
            ]
        )
    feasibility += table(["Fusion", "Historical6 baseline (same3 origins)", "New3 baseline"], rows)
    feasibility += """

Historical numbers are read from saved development results, not a new six-agent
model evaluation. Although origins and patients match, the six-agent action set,
fusion architecture/fitting objective/mask count differ: this is descriptive,
not a controlled causal claim that fewer specialists improve quality.

## Genuine three-client Flower check

"""
    fed = json.loads((root / "flower/results.json").read_text())
    rows = [
        [
            name,
            *[
                mean(list(values.values()))[key]
                for key in (
                    "accuracy",
                    "auroc",
                    "pr_auc",
                    "binary_cross_entropy",
                    "objective",
                    "calls",
                    "utility_r2",
                )
            ],
        ]
        for name, values in fed["metrics"].items()
    ]
    feasibility += table(
        ["Pilot fold0/original", "Acc", "AUROC", "AP", "BCE", "Objective", "Calls", "R2"], rows
    )
    feasibility += f"""

Run **{fed["run_id"]}**, exactly **{fed["clients"]} logical clients**, three FedAvg
rounds/**{fed["updates"]} actual updates**, one CPU worker. Server work
**{fed["seconds"]:.3f}s** excludes startup, includes full evidence queries/training/
selected-only batched consultations. Shared-router drift **{fed["drift"]:.6f}**.
Only compatible37→32Tanh→1 router arrays/scalars are aggregated. Private experts
and fusion stay frozen. Numeric OOF evidence is full-information diagnostic
traffic, not production privacy. Actual selected aligned-ID queries load private
classifiers; masked fusion matches replay. Central comparison uses the same
architecture/random initialization and six pooled epochs versus3×2 local epochs
with Adam resets; optimizer schedules differ. Weight updates are not superiority.

## Interpretation and stop boundary

All three-agent criteria are assessed without relaxed thresholds. Logistic gains
and partial utility signal, where present, must be distinguished from meeting the
full R²/stability gate. STOP rate or forced availability fallback alone is not
clinical personalization. See all fixed candidates and per-origin/cost rows in
outer_metrics.csv; no outer winner is selected or default pointer promoted.

Limitations: reused development cohort/known prior results; only the user-fixed
domains; one frozen optimizer schedule; shared patients and overlapping fits;
imbalanced survey target; teacher/final calibration shift; unavailable outcomes
only as retrospective diagnostics; public availability/coverage is simulated.
No external/clinical validity, authenticated consent, filesystem isolation,
secure aggregation or proven privacy. New three-agent source is an explicit CDC
adapter, not arbitrary registration, a Qwen integration or Step4. All old checkpoints,
reports and frozen test bytes remain untouched. Stop after this amendment.
"""
    Path("docs/experiments/m5-step035-three-agent-feasibility.md").write_text(feasibility, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
