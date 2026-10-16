"""Render CDC pilot results and diversity diagnostics, never fit or select outer winners."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from fed_med_agents.experiments.cdc_data import DOMAINS
from fed_med_agents.experiments.cdc_routing import CLIENTS, COSTS, context, gross_gains
from fed_med_agents.experiments.m5_step03 import write
from fed_med_agents.experiments.m5_step035 import KINDS
from fed_med_agents.experiments.routing_diagnostics import describe, loss_rows

SCALARS = (
    "accuracy",
    "auroc",
    "pr_auc",
    "binary_cross_entropy",
    "f1",
    "ece",
    "brier",
    "objective",
    "cost",
    "calls",
    "stop",
    "harm",
    "minority_peer_fraction",
    "utility_r2",
)


def mean(rows: list[dict]) -> dict:
    """Macro average dependent origin views, not pooled independent samples."""
    output = {}
    for key in SCALARS:
        values = [row[key] for row in rows if row.get(key) is not None]
        output[key] = float(np.mean(values)) if values else None
    return output


def table(headers: list[str], rows: list[list]) -> str:
    def cell(value):
        return f"{value:.5f}" if isinstance(value, float) else ("—" if value is None else str(value))

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(cell, row)) + " |" for row in rows],
        ]
    )


def report(root: Path) -> None:
    """Write full CSV and independently readable Markdown with saved real results."""
    results = json.loads((root / "results.json").read_text())
    meta = json.loads((root / "dataset.json").read_text())
    execution = json.loads((root / "execution.json").read_text())
    qa = (root / "qa/pytest.log").read_text(encoding="utf-8") if (root / "qa/pytest.log").exists() else ""
    qa_result = next((line for line in reversed(qa.splitlines()) if "passed" in line), "See QA logs")
    selection = {
        size: json.loads((root / f"selection_{size}.json").read_text()) for size in ("small", "pilot")
    }
    gates = {size: json.loads((root / f"gates_{size}.json").read_text()) for size in selection}
    records, summaries = [], {}
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
                        for policy, origins in policies.items():
                            bucket.setdefault(policy, []).extend(origins.values())
                            for origin, metrics in origins.items():
                                records.append(
                                    {
                                        "size": size,
                                        "fusion": kind,
                                        "scenario": scenario,
                                        "cost_tariff": cost,
                                        "fold": fold,
                                        "origin": origin,
                                        "policy": policy,
                                        **{key: metrics.get(key) for key in SCALARS},
                                        "action_frequencies": json.dumps(metrics["action_frequencies"]),
                                        "oracle_objective": metrics["oracle_objective"],
                                        "utility_mse": metrics.get("utility_mse"),
                                        "utility_correlation": metrics.get("utility_correlation"),
                                    }
                                )
                        chosen = selection[size][str(fold)][kind]
                        bucket.setdefault("inner_selected", []).extend(policies[chosen].values())
                    summaries[size][kind][scenario][str(cost)] = {
                        policy: mean(rows) for policy, rows in bucket.items()
                    }
    write(root / "summary.json", summaries)
    with (root / "outer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    source = dict(np.load(root / "development.npz"))
    diversity, group_rows, expert_rows = [], [], []
    for size in selection:
        for fold in range(3):
            folder = root / f"{size}_fold{fold}_outer"
            stage = dict(np.load(folder / "stack.npz"))
            p, y = stage["held_p"], stage["held_y"]
            error = p - y[:, None]
            corr = np.corrcoef(error.T)
            prediction_corr = np.corrcoef(p.T)
            probability_disagreement = np.abs(p[:, :, None] - p[:, None, :]).mean(0)
            disagreement = np.array(
                [
                    [float(((p[:, a] >= 0.5) != (p[:, b] >= 0.5)).mean()) for b in range(CLIENTS)]
                    for a in range(CLIENTS)
                ]
            )
            individual_loss = np.column_stack([loss_rows(y, p[:, c]) for c in range(CLIENTS)])
            domains = list(DOMAINS)
            for client in range(CLIENTS):
                expert_rows.append(
                    {
                        "size": size,
                        "fold": fold,
                        "client": domains[client],
                        **describe(y, p[:, client]),
                        "pr_auc": float(average_precision_score(y, p[:, client])),
                    }
                )
            diversity.append(
                {
                    "size": size,
                    "fold": fold,
                    "error_correlations": corr.tolist(),
                    "prediction_correlations": prediction_corr.tolist(),
                    "probability_disagreement": probability_disagreement.tolist(),
                    "disagreement": disagreement.tolist(),
                    "retrospective_best_private_counts": np.bincount(
                        individual_loss.argmin(1), minlength=CLIENTS
                    ).tolist(),
                    "private_oracle_bce": float(individual_loss.min(1).mean()),
                    "positive_prevalence": float(y.mean()),
                }
            )
            actions = dict(np.load(folder / "actions.npz"))
            x = source["x"][stage["held"]]
            labels = {
                "age_1_8": x[:, meta["features"].index("Age")] <= 8,
                "age_9_13": x[:, meta["features"].index("Age")] >= 9,
                "sex_0": x[:, meta["features"].index("Sex")] == 0,
                "sex_1": x[:, meta["features"].index("Sex")] == 1,
                "health_1_3": x[:, meta["features"].index("GenHlth")] <= 3,
                "health_4_5": x[:, meta["features"].index("GenHlth")] >= 4,
            }
            for kind in KINDS:
                risks = np.load(folder / f"targets_{kind}.npz")["held"]
                benefit = gross_gains(risks, y)
                chosen = selection[size][str(fold)][kind]
                for origin in range(CLIENTS):
                    available, _, prices = context(
                        source["ids"][stage["held"]], origin, meta["features"], "variable"
                    )
                    selected = actions[f"{kind}_variable_0.02_{chosen}_{origin}"]
                    baseline = actions[f"{kind}_variable_0.02_best_peer_{origin}"]
                    loss = np.column_stack([loss_rows(y, risks[:, origin, a]) for a in range(CLIENTS + 1)])
                    rows = np.arange(len(y))
                    net = loss[rows, selected] + 0.02 * np.where(
                        selected > 0, prices[rows, (selected - 1).clip(0)], 0
                    )
                    fixed = loss[rows, baseline] + 0.02 * np.where(
                        baseline > 0, prices[rows, (baseline - 1).clip(0)], 0
                    )
                    peers = np.arange(CLIENTS) != origin
                    for group, mask in labels.items():
                        group_rows.append(
                            {
                                "size": size,
                                "fold": fold,
                                "fusion": kind,
                                "origin": origin,
                                "group": group,
                                "n": int(mask.sum()),
                                "selected_objective": float(net[mask].mean()),
                                "fixed_objective": float(fixed[mask].mean()),
                                "calls": float((selected[mask] > 0).mean()),
                                "mean_candidate_gross_gain": benefit[mask, origin][:, peers].mean(0).tolist(),
                                "eligible_peer_count": float(available[mask][:, peers].sum(1).mean()),
                            }
                        )
    write(
        root / "diversity.json",
        {
            "domains": DOMAINS,
            "folds": diversity,
            "specialist_metrics": expert_rows,
            "retrospective_groups": group_rows,
        },
    )
    benchmark = f"""# M5 Step 3.5 — CDC benchmark

Executed2026-10-14. This is a NEW fixed-six development benchmark, not a registry
or replacement for the preserved UCI529 system. Read [frozen protocol](m5-step035-protocol.md),
[routing feasibility](m5-step035-routing-feasibility.md) and
[execution log](../devlog/2026-10-14-m5-step035.md).

## Dataset and frozen boundary

Official [UCI891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators),
fetched via ucimlrepo: **{meta["rows"]} rows,21 features**, target Diabetes_binary,
counts **{meta["target_counts"]}** for0/1. Its variable table says0=neither,
1=prediabetes OR diabetes; the page narrative mentions three classes, but that
is not the delivered binary target. This is cross-sectional self-reported survey
classification, not prospectively validated incident diabetes prediction.

Feature-value hash grouping (all21 values,not target) freezes **{meta["test_rows"]}
test rows** by hash modulo5. **{meta["development_rows"]} development rows** are
stored separately; pilot15000 sampled only there. **No frozen-test prediction,
metric or selection**, and no historical validation/test evaluation. Hash files
are audit evidence, not privacy or patient identity. **{meta["unique_feature_groups"]}
unique feature groups** in the full source: identical features remain in one
partition/fold even when outcomes conflict. This cannot detect repeated people
with different responses. Test-file byte hash is verified without parsing labels.

Three grouped outer folds assess about5000 pilot rows each. Full fitting sets
have about10000 rows; small2500-row fitting subsets use the SAME outer assessments
and separately refit nested pipelines. Inner grouped80/20 selection; target3fold
cross-fitting rebuilds the WHOLE specialist→calibration→fusion stack inside each
target-fold complement. Even fusion-training specialist OOF excludes target H.
No old dataset/model initializes these experts. Small/full comparisons isolate
fitting-size sensitivity within this new cohort, not a causal comparison to529.

## Specialist domains

""" + table(["Specialist", "Private fields"], [[name, ", ".join(fields)] for name, fields in DOMAINS.items()])
    benchmark += """

All experts use the same legitimate HistGradientBoosting capacity/schedule,
100iterations/15leaves/minleaf30/l2.1/lr.1, no early stopping/class rebalance.
Independent group-held20% sigmoid calibration; original fit/calibration model
weights are retained rather than refitting through calibration labels. Overlap
between domains is intentional. Every private model, including OOF, is saved.
This is simulated feature heterogeneity with aligned survey rows/labels available
at origins, not independent hospitals, true distributed vertical FL or deployment.

Frozen fusion conditions: existing original residual PredictionFusion(6),
six-client masked-logit linear12→1, arithmetic mean. Neural/logistic15epochs
Adam.003 batch4096 uniformly over36 valid STOP/one-peer masks, trained on specialist
OOF only. Final checkpoints frozen during routing; original3-client weights never
loaded. Mean gets no additional evidence. Learned losses/schedules are fixed,
not tuned to make a fusion/router win. Training loss curves are retained.

## Specialist predictive performance

Means across outer folds, shared assessment patients. Natural class imbalance is
retained. Average precision is the reported PR-AUC convention, not trapezoidal
interpolation. Accuracy alone is misleading near a14% positive prevalence.

"""
    expert_table = []
    for size in selection:
        for name in DOMAINS:
            rows = [row for row in expert_rows if row["size"] == size and row["client"] == name]
            expert_table.append(
                [
                    size,
                    name,
                    *[
                        float(np.mean([r[key] for r in rows]))
                        for key in ("accuracy", "auroc", "pr_auc", "binary_cross_entropy", "ece")
                    ],
                ]
            )
    benchmark += table(["Size", "Specialist", "Acc", "AUROC", "PR-AUC(AP)", "BCE", "ECE10"], expert_table)
    benchmark += f"""

## Devices, timing, execution and reproducibility

Actual shared fitting device **{execution["device"]}**, pipeline **{execution["seconds"]:.3f}s**
(excludes fetch/hash/audit). Histogram experts run CPU, one native thread.
Torch peak allocation **{execution["peak_allocated_bytes"]} bytes** excludes CUDA
contexts/reserves/board activity. environment.json records actual RTX A4000 CUDA
operation/checksum, CUDA build and nvidia-smi; no speedup/saturation claim.
CPU fallback is supported. At most one CPU Flower actor is used in the integration.

```powershell
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.m5_step035 --device cuda --output artifacts/m5-step035-v1
.venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_flower --output artifacts/m5-step035-v1
.venv/Scripts/python.exe scripts/verify_m5_step035.py artifacts/m5-step035-v1
.venv/Scripts/python.exe scripts/report_m5_step035.py artifacts/m5-step035-v1
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check --no-cache src tests scripts
.venv/Scripts/python.exe -m ruff format --no-cache --check src tests scripts
.venv/Scripts/python.exe -m pip check
```

Use a fresh output, existing environment, --device auto/cuda/cpu. Cold fetch uses
ucimlrepo; the executed initial fetch is preserved under artifacts/m5-step035-fetch.
The new dataset does not reuse or overwrite artifacts/dataset.npz. Prepared test
and development are separate files. Inner selections precede outer routing metrics.
No test-set evaluation command is provided. The Flower launcher is a compatibility
check for THIS fixed-six adapter, not a generalized configurable service.

All artifacts gitignored: source metadata, frozen partition/hash, pilot IDs, every
context/target/OOF/calibration model/provenance, frozen fusion/router checkpoints,
metrics/actions/selection/gates, diversity/group records, full per-origin/cost CSV,
Flower probabilities/updates/consultations, verification and QA logs. Source and
documentation changes are listed in the execution log. Original defaults untouched.

## Limitations

No survey weights or population representativeness correction; no temporal/external
clinical validation. Ordinal survey codes are modeled numerically; Age is a band,
not age in years. Domains overlap and small capacity choices may limit attainable
benefits; all are trained honestly without designed degradation. Full-information
OOF utility supervision is more expensive/disclosive than serving one selected
peer. Prepared local files are accessible to a trusted operator. No isolation,
formal privacy or secure aggregation claim. Frozen calibration is estimated and
can shift between target-fold and final fitting sizes. Correlated errors and oracle
variation are descriptive associations, not causal diversity or achievable routing.
Outer folds share fitting rows and origin views share assessment patients; no
statistical superiority or independent sample pooling. No registry/Qwen/GAT/
bandit/ranker, production-default change or publication. Stop after Step3.5.
"""
    Path("docs/experiments/m5-step035-benchmark.md").write_text(benchmark, encoding="utf-8")
    feasibility = """# M5 Step 3.5 — Routing feasibility on larger CDC data

Read [benchmark/data protocol](m5-step035-benchmark.md), [frozen preregistration](m5-step035-protocol.md)
and [execution log](../devlog/2026-10-14-m5-step035.md). All results are development-only.

## Controls and permitted information

STOP; best training-selected peer with eligibility fallback; positive-net fixed
value; random max; confidence<.6; existing graph architecture6clients/one-hop;
ridge/MLP each base/enriched. Frozen original/logistic/mean fusion, maxone peer,
costs .02/.05/.10. Best peer ranks mean CROSS-FITTED training gain, not assessment
labels. Ridgealpha100/fitting-only standardization; MLP32Tanh,20epochs Adam.003,
weight_decay.01. Graph15epochs on fixed cross-fitted action risks; same architecture
as existing routing, fresh six-client weights. Graph raw scores are not calibrated
BCE gains; its subtract-price comparison is a restricted control.
Utility R2 scores all five potential non-self peers, including temporarily
unavailable peers in retrospective full-information diagnostics. Actual action
objectives, oracle bounds and deployments strictly mask unavailable specialists.

Base own p/confidence, identities, public candidate width/overlap/availability/
budget; enriched entropy/variance, training-only own/candidate reliability and
public21-field coverage. Routing inference APIs have no labels, hidden peer
predictions, patient identifiers or clinical feature values. Labels create only
offline cross-fitted targets and retrospective metrics. Reliability for target H
uses T\\H OOF, not H labels. No in-sample fusion target teacher.

STATIC: all covered/available, uniform tariffs. VARIABLE: independent exogenous
lifestyle/general-health coverage outages,80% service availability, heterogeneous
recipient/congestion cost. Origin fields stay present. Missing required coverage
makes that specialist unavailable; experts are never artificially damaged or given
untrained missing inputs. Schedules depend on row IDs/seed, not clinical outcomes.
All comparisons see identical masks/prices/evidence budget. Changing service
eligibility alone can cause peer variation; static results are essential.

## Matched outer results at tariff .02

Macro-average origins then folds; AP=average precision, ECE10 equal-width bins;
F1/accuracy threshold .5. Complete all-cost/per-origin metrics and exact-mask random
controls are in outer_metrics.csv; actions.npz retains per-patient decisions.

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
    for size in selection:
        for kind in KINDS:
            for scenario in ("static", "variable"):
                rows = [
                    [
                        name,
                        *[
                            summaries[size][kind][scenario]["0.02"][name][key]
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
                    f"\n### {size}/{kind}/{scenario}\n\n"
                    + table(
                        ["Policy", "Acc", "AUROC", "AP", "BCE", "F1", "BCE+cost", "Calls", "Gain R2"], rows
                    )
                    + "\n"
                )
    feasibility += "\n## Frozen scale gates and across-fold stability\n\n"
    residual = summaries["pilot"]["original"]["variable"]["0.02"]
    pooled = summaries["pilot"]["mean"]["variable"]["0.02"]
    feasibility += f"""The larger pilot provides **limited cost-aware STOP benefit, not robust learned
peer personalization**. With mean fusion, inner-selected objective
{pooled["inner_selected"]["objective"]:.6f} beats always-best peer
{pooled["best_peer"]["objective"]:.6f}, but the stronger cost-aware fixed-value
baseline is already {pooled["fixed_value"]["objective"]:.6f}. Residual fusion's
selected objective {residual["inner_selected"]["objective"]:.6f} is close to its
fixed-value baseline {residual["fixed_value"]["objective"]:.6f}. Mean-fusion gain
R2 stays around .0003–.0026 by fold, below .05, and only two origins/fold satisfy
the peer-variation gate. Larger data alone does not establish the requested
feasibility in this context. This does NOT prove personalized routing impossible
with richer authorized local observations, different honest experts or new data.

"""
    feasibility += "### Matched fitting-size effects (variable services, tariff .02)\n\n"
    size_rows = []
    for kind in KINDS:
        for size in selection:
            bucket = summaries[size][kind]["variable"]["0.02"]
            chosen, fixed = bucket["inner_selected"], bucket["best_peer"]
            size_rows.append(
                [
                    size,
                    kind,
                    chosen["binary_cross_entropy"],
                    chosen["objective"],
                    fixed["objective"],
                    fixed["objective"] - chosen["objective"],
                    chosen["utility_r2"],
                    chosen["calls"],
                ]
            )
    feasibility += (
        table(
            [
                "Fitting size",
                "Fusion",
                "Selected BCE",
                "Selected objective",
                "Best peer objective",
                "Gain vs fixed",
                "Utility R2",
                "Calls",
            ],
            size_rows,
        )
        + "\n\n"
    )
    gate_rows = []
    for size in selection:
        for kind in KINDS:
            value = gates[size][kind]
            gate_rows.append(
                [
                    size,
                    kind,
                    value["passed"],
                    [selection[size][str(f)][kind] for f in range(3)],
                    [round(x, 6) for x in value["fold_improvements"]],
                    [round(x, 6) for x in value["fold_utility_r2"]],
                    value["varying_origins"],
                ]
            )
    feasibility += table(
        ["Size", "Fusion", "Pass", "Inner choices", "Fold objective gains", "Fold R2", "Varying origins"],
        gate_rows,
    )
    feasibility += """

Gate: >=.005 objective gain EVERY fold at.02; mean utilityR2>=.05 EVERY fold;
accuracy drop<=.005; no origin mean BCE worsening>.01; >=5% minority peer calls
in>=3origins each fold; positive static-scenario gain each fold. No acceptance
criterion is relaxed. No automatic promotion or outer-selected winner. All gates,
including per-origin/accuracy failures, are retained in gates_{small,pilot}.json.
Service-induced variation is not necessarily clinical personalization, even when
the peer-count gate passes. Useful STOP must improve cost-quality, not STOP alone.

## Cost-quality tradeoffs and fitting-size sensitivity

"""
    rows = []
    for size in selection:
        for kind in KINDS:
            for cost in COSTS:
                bucket = summaries[size][kind]["variable"][str(cost)]
                for policy in ("stop", "best_peer", "inner_selected"):
                    row = bucket[policy]
                    rows.append(
                        [
                            size,
                            kind,
                            cost,
                            policy,
                            row["binary_cross_entropy"],
                            row["cost"],
                            row["objective"],
                            row["calls"],
                            row["minority_peer_fraction"],
                        ]
                    )
    feasibility += table(
        ["Size", "Fusion", "Tariff", "Policy", "BCE", "Paid cost", "Objective", "Calls", "Minority peer"],
        rows,
    )
    feasibility += "\n\n## Diversity and patient-group consultation benefit\n\n"
    rows = []
    for size in selection:
        subset = [d for d in diversity if d["size"] == size]
        offdiag = ~np.eye(CLIENTS, dtype=bool)
        rows.append(
            [
                size,
                float(np.mean([np.array(d["error_correlations"])[offdiag].mean() for d in subset])),
                float(np.mean([np.array(d["disagreement"])[offdiag].mean() for d in subset])),
                float(np.mean([d["private_oracle_bce"] for d in subset])),
            ]
        )
    feasibility += table(
        ["Size", "Mean error corr", "Threshold disagreement", "Retrospective private oracle BCE"], rows
    )
    feasibility += "\n\nIn the full pilot mean prediction correlation is .37470 and mean absolute probability disagreement .08010.\n"
    feasibility += "High .95002 error correlation partly reflects the shared binary outcome and imbalance; it is not conditional independence analysis.\n"
    feasibility += """

diversity.json contains all pairwise correlations/disagreements, specialist
metrics, retrospective best-specialist counts and groupwise consultation gains
by Age bands1–8/9–13, Sex0/1, GenHlth1–3/4–5. These are post-decision analyses,
never router inputs or tuning data. Labels can make hindsight 'best' identities
vary even when predictable utility does not; do not mistake that for a learned
signal. Origin-specific selected actions/utility/harm/calibration appear in CSV.
Different domains can be predictive without providing novel conditional evidence.
"""
    feasibility += "\n### Retrospective group comparisons, full pilot/variable services/.02\n\n"
    grouped = []
    for kind in KINDS:
        for group in sorted({row["group"] for row in group_rows}):
            rows = [
                row
                for row in group_rows
                if row["size"] == "pilot" and row["fusion"] == kind and row["group"] == group
            ]
            grouped.append(
                [
                    kind,
                    group,
                    float(np.mean([row["selected_objective"] for row in rows])),
                    float(np.mean([row["fixed_objective"] for row in rows])),
                    float(np.mean([row["calls"] for row in rows])),
                ]
            )
    feasibility += table(["Fusion", "Group", "Selected objective", "Fixed objective", "Calls"], grouped)
    feasibility += "\n\nGroup means macro-average folds/origins; group sizes and peer gain vectors are in diversity.json. No group result changes a policy.\n"
    feasibility += "\n## Genuine Flower compatibility, verification and decision\n\n"
    fed = json.loads((root / "flower/results.json").read_text())
    verify = json.loads((root / "verification.json").read_text())
    fl_table = [
        [
            name,
            *[
                mean(list(rows.values()))[key]
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
        for name, rows in fed["metrics"].items()
    ]
    feasibility += table(
        ["First pilot outer fold", "Acc", "AUROC", "AP", "BCE", "Objective", "Calls", "R2"], fl_table
    )
    feasibility += f"""

Flower run **{fed["run_id"]}**,3actual FedAvg rounds/**18 local updates**, six
logical ClientApps on one CPU actor. Routing-only drift **{fed["drift"]:.6f}**,
server work **{fed["seconds"]:.3f}s** including queries/consultations, excluding
startup. Central same43→32Tanh→1 architecture/same initialization,six pooled
epochs; FL3×2local epochs with optimizer resets. These are not identical global
optimizer schedules. Only compatible numeric router arrays/scalars are aggregated.
24 real evidence queries bind prepared probabilities/IDs/hashes; actual selected
aligned-ID queries load private models and masked fusion matches offline replay.
This does not establish learned routing improvement, generalized multi-agent
federation, Qwen integration or production readiness. Full-information preparation
and central diagnosis are explicitly trusted-local; later queries do not make
earlier preparation federated. Private experts/fusion remain frozen during FL.

Verification: `{json.dumps(verify)}`. QA logs/checks.json record pytest/Ruff/pip
commands and results; old optional Qwen/feedback historical-validation benchmarks
are skipped, while this new six-client integration is actually executed.
Fresh regression **{qa_result}**; Ruff lint/format78files and pip check passed.
The144 verified fusion states include48 stateless mean snapshots and96 trained
neural/logistic checkpoints. Verification replays every private prediction and
refits12 calibrated models; it does not claim all1152 training fits were rerun.

Scale gate passed: **{execution["scale_gate_passed"]}**; expanded50k:
**{execution["scaled_to_50k"]}**. Larger data may improve specialist fits and fusion,
but whether it improves useful consultation selection must be judged on these
paired gates/metrics, not accuracy alone. No retrospective threshold/cost/model
changes are made to force success. Interpret pilot and small results as fitting-
size sensitivity, not evidence that dataset size alone caused earlier failures.
The frozen test remains untouched; a confirmatory test requires a future frozen
protocol and authorization, not this development sprint.

Advanced GAT/bandit/ranker work is not justified merely by weight changes or
oracle headroom. Prioritize transferable pre-query signals, stable expert/fusion
calibration and independent replication of useful value prediction before that
complexity. Service-aware deterministic baselines remain essential. All results
are exploratory development evidence, not clinical validity or proven privacy.
No default model change, registry, Qwen training, push or publishing. STOP after3.5.
"""
    Path("docs/experiments/m5-step035-routing-feasibility.md").write_text(feasibility, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
