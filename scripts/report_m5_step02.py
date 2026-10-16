"""Render the three Step 2 reports from actual nested and feedback evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from fed_med_agents.experiments.diabetes import logits
from fed_med_agents.experiments.repair_models import CANDIDATES, action_outputs, make_model
from fed_med_agents.experiments.routing_diagnostics import describe, loss_rows
from fed_med_agents.models import Classifier


def table(headers: list, rows: list) -> str:
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.5f}"
        return "undefined" if value is None else str(value).replace("|", "/")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step02-v1"))
    args = parser.parse_args()
    root = args.output.resolve()
    torch.set_num_threads(1)
    data = json.loads((root / "results.json").read_text())
    verification = json.loads((root / "verification.json").read_text())
    if not verification.get("feedback"):
        raise RuntimeError("Complete feedback and independent verification before final reports")
    contexts = data["contexts"]
    metrics = ["accuracy", "auroc", "binary_cross_entropy", "f1", "ece", "brier", "calls", "primary"]

    def rows_for(name, field="assessment"):
        if name in ("stop", "random", "cyclic"):
            return [v for fold in range(3) for v in contexts[f"fold{fold}_outer"]["controls"][name].values()]
        return [v for fold in range(3) for v in contexts[f"fold{fold}_outer"][name][field].values()]

    def average(rows):
        return [float(np.mean([v[m] for v in rows])) for m in metrics]

    names = [*CANDIDATES, "utility_bins", "central_pooled", "central_interleaved", "stop", "random", "cyclic"]
    common_table = table(
        ["New nested variant", "Accuracy", "AUROC", "BCE", "F1", "ECE10", "Brier", "Calls", "BCE+.02 calls"],
        [[name, *average(rows_for(name))] for name in names],
    )
    with (root / "outer_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["fold", "origin", "variant", *metrics, "stop_fraction", "harm", "actions_STOP_A_B_C"]
        )
        for name in names:
            for index, row in enumerate(rows_for(name)):
                writer.writerow(
                    [
                        index // 3,
                        index % 3,
                        name,
                        *[row[m] for m in metrics],
                        row["stop_fraction"],
                        row["harm"],
                        row["action_frequencies"],
                    ]
                )
    scope = f"""Executed 2026-10-14, Windows/Python 3.11/Flower 1.39/PyTorch CUDA.
Only the original **364 training patients** enter any new fitting, calibration,
selection or assessment. Three stratified outer assessment folds; each outer
fitting set has an independent 80/20 inner fit/selection split. Experts are
refit/calibrated within every context, and fusion/router fitting uses target-label
excluding specialist OOF probabilities. Source age was reconstructed by inverting
its old affine transform/rounding and then refit inside each new fitting subset.
The historical 78 validation and 78 test patients are not newly evaluated.

Original Step 1 historical validation reference: accuracy/BCE .93162/.19573;
identical-evidence mean .95014/.26517; random+.same fusion .93305/.19798.
Those used different patients/fits and are **not** paired with the new table.
The paired original-protocol baseline below is `avg10` (five rounds, ten epochs,
Adam .01 reset, cost .02). Metrics macro-average three dependent origins and three
outer folds; AUROC/F1 are means, not pooled patient metrics or significance tests.
Calibration is ten equal-width ECE and Brier, threshold fixed .5. Folds partition
one cohort and fitting sets overlap. No independent clinical validity is claimed.

[Preregistered protocol](m5-step02-protocol.md) fixes candidates, data partitions,
cost tariffs and promotion gates before results. All five terminal rounds are
reported without outer-based round/model selection. Costs .02/.1/.3 are BCE units
per actual selected call. Primary objective is BCE+.02*calls. Availability is
matched, all three domains present, maximum one peer; this does not identify the
effect of variable capability/availability. No default/serving pointer is changed.

## Original protocol versus every candidate/control

{common_table}

Controls STOP/random/cyclic use the same avg10 fusion. Utility bins use the
uniform-pretrained original fusion. All inference routing uses only own probability,
origin identity/confidence and public domains/availability; no label or unqueried
peer probability enters selection. Complete OOF evidence is deliberately used in
full-information fitting; assessment replays masked evidence from Flower-acquired
probabilities. These logical inference call budgets are not selected-only deployed
traffic measurements. No raw training features/private specialist weights are in
shared update messages. This is not a proven privacy guarantee.
"""
    selection_rows = []
    selected_outer = []
    baseline_outer = []
    for fold in range(3):
        name = data["selection"][str(fold)]["candidate"]
        a = list(contexts[f"fold{fold}_outer"][name]["assessment"].values())
        b = list(contexts[f"fold{fold}_outer"]["avg10"]["assessment"].values())
        av, bv = average(a), average(b)
        selection_rows.append([fold, name, av[-1], bv[-1], bv[-1] - av[-1]])
        selected_outer.extend(a)
        baseline_outer.extend(b)
    primary_gate = all(row[-1] >= 0.01 for row in selection_rows)
    accuracy_gate = (
        np.mean([v["accuracy"] for v in selected_outer])
        >= np.mean([v["accuracy"] for v in baseline_outer]) - 0.01
    )
    origin_gate = all(
        np.mean([selected_outer[f * 3 + o]["binary_cross_entropy"] for f in range(3)])
        <= np.mean([baseline_outer[f * 3 + o]["binary_cross_entropy"] for f in range(3)]) + 0.01
        for o in range(3)
    )
    promotion = {
        "each_fold_primary_improvement_at_least_.01": primary_gate,
        "accuracy_gate": bool(accuracy_gate),
        "per_origin_bce_gate": bool(origin_gate),
        "automatic_promotion": False,
        "original_retained": True,
    }
    (root / "promotion.json").write_text(json.dumps(promotion, indent=2))
    selection = f"""## Nested selection and promotion decision

Candidate names were selected only on the INNER selection objective, with declared
order as tie-break. Outer metrics did not change the selected name.

{table(["Outer fold", "Inner-selected pipeline", "Outer objective", "avg10 objective", "Improvement"], selection_rows)}

Selected-pipeline aggregate: {dict(zip(metrics, average(selected_outer)))}.
Predeclared promotion gates: `{promotion}`. **Original retained; no model promoted.**
In particular the unchanged avg10 selection on one fold provides zero improvement,
so the all-fold >=.01 gate fails even though other folds improve. The fixed logistic
candidate is promising, but choosing it after inspecting all outer scores would
override the nested selection protocol. No such override occurred. No significance
or medical-safety inference is attached to this gate.
"""
    environment = json.loads((root / "environment.json").read_text())
    all_reports = [
        report
        for ctx in contexts.values()
        for record in ctx.values()
        for h in record.get("history", {}).values()
        for report in h.get("reports", [])
    ]
    peak = max(r["peak_allocated_bytes"] for r in all_reports)
    wall = sum(record.get("runtime_seconds", 0) for ctx in contexts.values() for record in ctx.values())
    proofs = f"""## Flower, isolation and reproduction evidence

Main Flower run **{data["flower_run_id"]}**: **{verification["aggregations"]}** actual
FedAvg/FedProx aggregations, **{verification["client_updates"]}** client updates,
six contexts and {verification["assessment_replays"]} final per-origin metric
replays independently verified. Three clients per round, same compatible state
architecture within a variant. Different fusion architectures are never mixed.
All new private expert hashes match their pre-FL hashes; all
**{verification["preserved_files"]}** historical artifact/report hashes match.
No old checkpoint/report/policy pointer was overwritten. Candidate arrays, client
audits, optimizer persistence, step counts and payload keys are saved under
`{root}/runs`. `flower_evidence.json` binds probabilities/IDs to actual queries.
`verification.json` checks disjoint nested IDs/OOF provenance, inner-only selection,
all weighted aggregates, freeze semantics and recorded-device metric replay.

CPU specialist preparation; CUDA shared optimization with at most one full-GPU
Ray actor serving three logical clients; CPU feedback simulations. Environment
audit ran an actual CUDA matrix operation. Torch peak memory is not total board
memory. Timings include diagnostic instrumentation; no GPU speed/saturation claim.
Actual device **{environment["device_name"]}**, CUDA checksum
**{environment["tensor_operation_checksum"]}**, maximum local torch allocated memory
**{peak} bytes**. Sum of main strategy runtimes **{wall:.3f} s**, excluding private
preparation, framework startup/evidence acquisition and central controls. Per-variant
and feedback runtimes are recorded separately. CPU/GPU speed was not compared on
an identical workload; no acceleration claim follows.
Windows Ray shutdown diagnostics and deprecated Python simulation launcher remain.

```powershell
python -m fed_med_agents.experiments.m5_step02 --prepare-only --output artifacts/my-m5-step02
python -m fed_med_agents.experiments.m5_step02 --device cuda --output artifacts/my-m5-step02
python -m fed_med_agents.experiments.repair_feedback --output artifacts/my-m5-step02
python scripts/verify_m5_step02.py artifacts/my-m5-step02
python scripts/report_m5_step02.py --output artifacts/my-m5-step02
python -m pytest -q
python -m ruff check --no-cache src tests scripts
python -m ruff format --no-cache --check src tests scripts
python -m pip check
```

Use the venv Python and a fresh versioned path. Original execution prepared the
same frozen contexts with `repair_data.prepare_nested` after separately hashing
prior artifacts. The CLI prepare-only entry records hashes/copies the protocol
for future reproductions. Historical Step 1 is preserved, not regenerated.
"""
    qa = ""
    for name in ("pytest", "ruff", "format", "dependencies"):
        file = root / "qa" / f"{name}.log"
        if file.exists():
            qa += (
                f"\n`qa/{name}.log`:\n\n```text\n"
                + "\n".join(file.read_text(encoding="utf-8").splitlines()[-4:])
                + "\n```\n"
            )
    failures = """## Failures, limitations and unchanged scope

Two initial unit tests failed because a standalone Flower Message lacked the
process TaskIdentity normally supplied by its runtime. A test-only fixture now
sets/restores that identity; no model/acceptance criteria were loosened. Existing
Ruff cache instability is avoided with --no-cache. Actual Flower training completed
without failed client updates; Windows Ray warnings persist.
The first report-generation attempt shadowed the top-level results dictionary
with an NPZ reader and failed while writing summary.json. Renamed the local
reader and regenerated reports successfully; no training results changed.

One cohort, three small nested assessment folds, a single prescribed optimizer
seed per fold and many fixed exploratory comparisons: no significance or external
validity. Logistic changes capacity, identity/confidence features and residual
form together; it does not isolate one cause. Smaller local epochs use less work,
so they are not compute-matched. Pretraining/value regression changes objective,
schedule and supervision. Its teacher targets are meta-level in-sample training
quantities, not independent clinical outcomes. No historical validation/test label
was used to choose configurations. Stronger claims require independent data and
preregistered follow-up. No dynamic registry, variable-client framework, generalized
fusion API, new SLM, Qwen fine-tuning, publishing or GitHub push. Stop after Step 2.
"""
    optimization = []
    for name in ("avg10", "avg1", "avg2", "avg5", "persistent2", "prox2"):
        reports = []
        cosines = []
        gradcos = []
        runtime = []
        for fold in range(3):
            record = contexts[f"fold{fold}_outer"][name]
            runtime.append(record["runtime_seconds"])
            for r, h in record["history"].items():
                if int(r):
                    reports.extend(h["reports"])
                    cosines.extend(v for v in h["local_update_cosines"]["fusion."] if v is not None)
                    gradcos.extend(v for v in h["origin_gradient_cosines"]["fusion."] if v is not None)
        optimization.append(
            [
                name,
                5,
                int(sum(r["steps"] for r in reports) / 3),
                float(np.mean([r["drift"]["router."] for r in reports])),
                float(np.mean([r["drift"]["fusion."] for r in reports])),
                float(np.mean(cosines)),
                float(np.mean(gradcos)),
                float(np.mean(runtime)),
            ]
        )
    optimization_text = """## Optimization interpretation

Reduced local work is not an unconditional repair: one epoch underfits relative
to avg10, two epochs are close, and five epochs slightly reduce mean BCE. FedProx
mu=.01 slightly improves over plain two epochs here; persistent Adam also changes
results only modestly. These are descriptive paired outcomes, not reliable
superiority. Compare per-fold/per-origin JSON rather than selecting the best mean.

FedProx is installed Flower 1.39 FedProx: configure_train injects proximal-mu=.01;
clients add mu/2 times squared distance from received weights. Its server arithmetic
remains weighted FedAvg and is independently checked. Persistent Adam keeps only
each origin's moments across parameter broadcasts, with a fresh optimizer at every
context/variant. Stale local moments are not globally averaged. Unit tests verify
zero initial proximal gradient, later divergence and actual persistent step counts.

Central pooled: 60/round in original Step 1 size, but **40/round** at these fitting
sizes (ceil(n/64)=4, ten epochs). It averages three origin losses per step: 120
origin-loss evaluations. Interleaved uses the same 120 origin-loss evaluations but
120 optimizer steps. Each ten-epoch client takes 40 steps; three clients total120,
then one aggregate. Both central schedules reset each round. Pooled BCE is not
uniformly better than FL on these new folds; the Step 1 pattern is not universal.
Interleaved central has more global optimizer steps and remains confounded by
that difference. Conflicting local objectives are observed, not proof that FedAvg
arithmetic alone is wrong.

Local statistics below average all outer-fold client rounds. Steps are total
origin-loss computations per outer fold across three clients (not FLOP counts).
Cosines are A/B, A/C, B/C averaged across rounds/folds; they are not clinical
correlations. Proximal gradient is zero at the global broadcast for the origin
gradient diagnostic; local updates include it.
"""
    optimization_text += table(
        [
            "Variant",
            "FL rounds",
            "Origin-loss computations/fold",
            "Mean router local L2",
            "Mean fusion local L2",
            "Fusion update cosine",
            "Origin fusion gradient cosine",
            "Mean FL seconds/fold",
        ],
        optimization,
    )
    routing_rows = []
    for name in ("avg10", "cost10", "cost30", "value", "utility_bins", "stop", "random", "cyclic"):
        rows = rows_for(name)
        harm = [r["harm"] for r in rows if r["harm"] is not None]
        routing_rows.append(
            [
                name,
                float(np.mean([r["stop_fraction"] for r in rows])),
                float(np.mean([r["calls"] for r in rows])),
                float(np.mean([r["entropy"] for r in rows])),
                float(np.mean([r["hard_minus_soft_bce"] for r in rows])),
                float(np.mean(harm)) if harm else None,
                *[float(np.mean([r["objectives"][str(c)] for r in rows])) for c in (0.02, 0.1, 0.3)],
            ]
        )
    routing_table = table(
        [
            "Policy",
            "STOP fraction",
            "Calls",
            "Score entropy",
            "Hard-soft BCE",
            "Harm/calls",
            "Objective .02",
            "Objective .1",
            "Objective .3",
        ],
        routing_rows,
    )
    action_rows = [
        [
            fold,
            name,
            origin,
            contexts[f"fold{fold}_outer"][name]["assessment"][str(origin)]["action_frequencies"],
        ]
        for fold in range(3)
        for name in ("avg10", "cost10", "cost30", "value", "utility_bins")
        for origin in range(3)
    ]
    diagnostic = []
    for fold in range(3):
        key = f"fold{fold}_outer"
        folder = root / "contexts" / key / "seed_42"
        p = []
        for o in range(3):
            with np.load(folder / f"client_{o}/local.npz") as d:
                p.append(d["assessment"].copy())
        p = np.asarray(p, dtype=np.float32).T
        with np.load(folder / "cohort.npz") as d:
            y = d["y"][d["val"]]
        model = make_model()
        model.load_state_dict(
            torch.load(root / "runs" / key / "value_pretrain/server_round_3.pt", weights_only=True)
        )
        for o in range(3):
            with torch.no_grad():
                out = torch.sigmoid(action_outputs(model, torch.tensor(p), torch.full((len(y),), o))).numpy()
            peer = [i + 1 for i in range(3) if i != o]
            benefit = loss_rows(y, out[:, 0]) - np.stack([loss_rows(y, out[:, a]) for a in peer]).min(0)
            confidence = 2 * np.abs(p[:, o] - 0.5)
            corr = (
                float(np.corrcoef(confidence, benefit)[0, 1])
                if confidence.std() > 0 and benefit.std() > 0
                else None
            )
            diagnostic.append([fold, o, corr, float((benefit > 0.02).mean())])
    routing_text = """## STOP and usefulness interpretation

At the primary .02 tariff the selected pipelines still always consult; useful
individualized STOP is not established. Raising training cost to .1/.3 increases
STOP but predictive BCE worsens. At their higher assigned evaluation tariffs,
fewer calls can give a better cost-aware objective; compare the whole tradeoff,
not STOP count alone. No inference uses label-aware best actions. Always-consult
is competitive among the tested low-cost policies, not proven Bayes-optimal.

Marginal-value regression predicts fitting-only realized loss reduction from
pre-query context. Inference sets STOP value zero and subtracts .02 from eligible
peer estimates. Uniform-pretrained fusion remains frozen. The simple utility bins
use fixed confidence bins and fitting-only mean gains. Both can STOP, but fail
to improve the primary mean objective over avg10. They do not establish calibrated
probabilities of correctness or useful prospective value estimates.

Entropy/hard-soft quantities use normalized raw scores. For value regression these
are diagnostics, not its MSE fitting objective. For deterministic controls they
describe the underlying avg10 score distribution, not sampling randomness of the
control. Harm is selected-action realized BCE >= same-fusion STOP BCE; it is not
clinical harm. Availability/capability flags are constant here: their predictive
value cannot be identified. Correlations below are retrospective assessment
diagnostics only and never feed deployed choices.
"""
    routing_text += (
        routing_table
        + "\n### Action counts [STOP,A,B,C]\n\n"
        + table(["Fold", "Policy", "Origin", "Counts"], action_rows)
    )
    routing_text += "\n### Confidence and retrospective best-peer gain\n\n" + table(
        ["Fold", "Origin", "Pearson confidence/gain", "Fraction gain > .02"], diagnostic
    )
    fusion_rows = []
    for name in ("avg10", "logistic", "small", "value"):
        rows = rows_for(name, "same_avg10_evidence")
        fusion_rows.append([name, *average(rows)])
    mean_rows = []
    for r in rows_for("avg10"):
        mean_rows.append(
            r
            | r["mean_same_evidence"]
            | {"primary": r["mean_same_evidence"]["binary_cross_entropy"] + 0.02 * r["calls"]}
        )
    fusion_rows.append(["arithmetic_mean", *average(mean_rows)])
    fusion_text = """## Identical-evidence fusion comparison

The following uses avg10's exact action mask for EVERY fusion, preventing extra
peer evidence from explaining differences. Original residual12→24→1, slim12→8→1,
linear6→1 masked logits/presence without origin identity, uniform-pretrained
original residual, and arithmetic mean. Logistic improves BCE/Brier and accuracy
here, but ECE does not improve consistently. Reduced hidden capacity alone is not
a clear repair. Separately pretrained original fusion plus value routing does not
beat the baseline primary objective. Its shorter pretraining and different router
objective confound attribution to staging alone.

Logistic removes confidence, origin identity and residual form together; results
do not establish overreliance on identity or excessive flexibility as the isolated
cause. All fitting inputs are specialist OOF predictions. Threshold remains .5;
BCE and accuracy/ECE optimize different properties, so arithmetic mean is not
unconditionally better. Correlated expert errors and fold/final calibration shift
remain plausible contributors; no validation/test recalibration occurred.
"""
    fusion_text += "\nExperimental logistic/slim states require their explicit variant classes and cannot be loaded into the default original model. They are compatible across the three clients within their own trial, not interchangeable production checkpoints. No serving adapter or default pointer changed.\n"
    fusion_text += table(["Fusion, fixed avg10 mask", *metrics], fusion_rows)
    calibration = []
    correlations = []
    for fold in range(3):
        folder = root / "contexts" / f"fold{fold}_outer" / "seed_42"
        probabilities = []
        with np.load(folder / "cohort.npz") as cohort:
            held_y = cohort["y"][cohort["val"]]
        for origin in range(3):
            private = folder / f"client_{origin}"
            metadata = json.loads((private / "specialist.json").read_text())
            with np.load(private / "local.npz") as d:
                model = Classifier(d["x"].shape[1])
                model.load_state_dict(torch.load(private / "specialist.pt", weights_only=True))
                final = torch.sigmoid(
                    torch.tensor(
                        logits(model, d["x"][d["train"]], torch.device("cpu")) / metadata["temperature"]
                    )
                ).numpy()
                oof, fit_y, held = d["oof"], d["train_y"], d["assessment"]
                a, b, c = describe(fit_y, oof), describe(fit_y, final), describe(held_y, held)
                calibration.append(
                    [
                        fold,
                        origin,
                        float(np.abs(oof - final).mean()),
                        a["binary_cross_entropy"],
                        b["binary_cross_entropy"],
                        c["binary_cross_entropy"],
                        a["ece"],
                        c["ece"],
                    ]
                )
                probabilities.append(held)
        errors = np.asarray(probabilities).T - held_y[:, None]
        correlation = np.corrcoef(errors.T)
        correlations.append(
            [
                fold,
                float(correlation[0, 1]),
                float(correlation[0, 2]),
                float(correlation[1, 2]),
                float(
                    ((np.asarray(probabilities)[1] >= 0.5) != (np.asarray(probabilities)[2] >= 0.5)).mean()
                ),
            ]
        )
    (root / "fusion_diagnostics.json").write_text(
        json.dumps({"calibration": calibration, "error_correlations": correlations}, indent=2)
    )
    fusion_text += "\n### Frozen expert distribution/calibration diagnostics\n\nFinal-on-fit probabilities below are resubstitution diagnostics ONLY, never fusion fitting or model selection inputs. The shift combines fold sample size, specialist refitting and calibration; it does not isolate a cause.\n\n"
    fusion_text += table(
        [
            "Fold",
            "Origin",
            "Mean abs(OOF−final fit)",
            "OOF BCE",
            "Final-fit BCE (optimistic)",
            "Assessment BCE",
            "OOF ECE",
            "Assessment ECE",
        ],
        calibration,
    )
    fusion_text += (
        "\nSigned probability-error correlations are associations, not conditional independence or causal evidence of redundancy.\n\n"
        + table(
            ["Fold", "A/B error corr", "A/C error corr", "B/C error corr", "B/C threshold disagreement"],
            correlations,
        )
    )
    audits = json.loads((root / "feedback_audit.json").read_text())
    feedback_rows = []
    feedback_info = []
    learning_curve = []
    for steps in (2, 8):
        trial = json.loads((root / f"feedback_steps{steps}/metrics.json").read_text())
        for policy in ("frozen", "adaptive"):
            m = trial["stream"][policy]
            feedback_rows.append(
                [
                    steps,
                    policy,
                    m["accuracy"],
                    m["auroc_macro_origin"],
                    m["binary_cross_entropy"],
                    m["f1"],
                    m["consultations_per_patient"],
                    m["unnecessary_consultation_fraction"],
                ]
            )
        audit = audits[f"steps{steps}"]["steps"]
        for h in trial["history"]:
            if "version" in h:
                learning_curve.append(
                    [
                        steps,
                        h["episode"],
                        h["version"],
                        h["approved_after"],
                        h["validation"]["mean_bce"],
                        h["shared_change_l2"],
                        h["stream_metrics"]["adaptive"]["binary_cross_entropy"],
                        h["stream_metrics"]["frozen"]["binary_cross_entropy"],
                    ]
                )
        feedback_info.append(
            [
                steps,
                trial["flower_run_id"],
                trial["approved_version"],
                len(audit),
                min(v["ess"] for v in audit),
                min(v["min_propensity"] for v in audit),
                max(v["clipped_fraction"] for v in audit),
                max(v["preclip_norm"] for v in audit),
                trial["runtime_seconds"],
            ]
        )
    feedback_text = """## Feedback: actual two-step versus eight-step replay

The ORIGINAL M4 Apps/pipeline are unchanged. Both new 120-patient streams use outer
fold0 fitting-only experts and its original-training-only holdout, same cold
initialization, patient order, random draws, delayed true labels, epsilon.2, max
budget1, lr.0003, window128, triggers60/120/123. The only changed training setting
is two versus eight local replay steps. Source training IDs are remapped local
positions with original IDs retained; no original held-out labels are evaluated.
The guard uses a reused outer-training-study assessment holdout: these are guarded
development scores, not independent feedback generalization estimates.

Frozen/adaptive policies share maximum budgets; realized counts are reported, not
assumed equal. Frozen predictions/actions across both trials are independently
checked identical. Every label is a verified training source outcome revealed
after prediction, never an SLM acknowledgement or guessed peer outcome. Only
actually consulted evidence reaches local replay. The offline reference was fit
on all fitting outcomes and is excluded from claims of adaptive improvement.
"""
    feedback_text += table(
        ["Local steps", "Policy", "Accuracy", "Macro-origin AUROC", "BCE", "F1", "Calls", "Harm/calls"],
        feedback_rows,
    )
    feedback_text += "\n### Actual feedback learning curves and versions\n\n" + table(
        [
            "Local steps",
            "Label cutoff",
            "Candidate",
            "Approved pointer",
            "Guard BCE",
            "Update L2",
            "Adaptive stream BCE",
            "Frozen stream BCE",
        ],
        learning_curve,
    )
    feedback_text += "\nThese are six additional genuine Flower rounds and 18 client updates, separate from the 348 main aggregations. Every version/update is retained. Labels from the historical validation/test partitions never enter these curves.\n"
    feedback_text += (
        "\nActual local-step audit (including steps AFTER the first optimizer update):\n\n"
        + table(
            [
                "Steps",
                "Flower run",
                "Approved",
                "Audited local steps",
                "Min ESS",
                "Min propensity",
                "Max clipped fraction",
                "Max preclip norm",
                "Runtime seconds",
            ],
            feedback_info,
        )
    )
    old = audits["historical_local_steps"]["steps"]
    feedback_text += f"\nHistorical M4 replay: {len(old)} actual local steps reconstructed; maximum clipped fraction {max(v['clipped_fraction'] for v in old):.5f}. Local reconstructed checkpoints agree within 2e-5 (CPU/CUDA tolerance). Full per-step ESS, gradients, loss/reward age, action frequencies and model-version ages persist in feedback_audit.json.\n"
    entropy_rows = [
        [
            name,
            min(v["policy_entropy"] for v in item["steps"]),
            max(v["policy_entropy"] for v in item["steps"]),
            min(v["logged_reward_cost_mean"] for v in item["steps"]),
            max(v["logged_reward_cost_mean"] for v in item["steps"]),
        ]
        for name, item in audits.items()
    ]
    feedback_text += (
        "\nThe maximum entropy for three valid actions is ln(3)≈1.09861. Entropy stays high at actual local steps; there is no entropy penalty in the M4 objective. Logged reward cost is recorded BCE plus actual consultation cost (the router surrogate also subtracts its fixed .5 baseline).\n\n"
        + table(
            ["Replay", "Min entropy", "Max entropy", "Min mean logged cost", "Max mean logged cost"],
            entropy_rows,
        )
    )
    feedback_text += """
More replay steps increase movement/work; any small BCE changes must not be
called meaningful adaptation or deployed continuous learning. Sparse observed
actions, high entropy, stale reward versions and small learning rate remain
structural limitations. Matched prefixes do not reveal unconsulted counterfactuals.
No independent delay/entropy/learning-rate/regularization sweep was preregistered;
those causal hypotheses remain unresolved. The original full M4 protocol remains
the historical baseline, not overwritten or silently retuned.
Every stream label is eventually observed after the drain (120/120). Here
feedback sparsity means one action's evidence out of three valid actions, not
manufactured missing diagnoses. Two and eight steps tie on accuracy/AUROC/F1;
the small BCE difference is descriptive and does not establish meaningful learning.
"""
    reports = {
        "m5-step02-optimization-repair.md": "# M5 Step 2 — federated optimization repair\n\n"
        + scope
        + optimization_text
        + selection
        + feedback_text
        + proofs
        + qa
        + failures,
        "m5-step02-routing-repair.md": "# M5 Step 2 — STOP and routing repair\n\n"
        + scope
        + routing_text
        + selection
        + proofs
        + qa
        + failures,
        "m5-step02-fusion-ablation.md": "# M5 Step 2 — fixed fusion ablations\n\n"
        + scope
        + fusion_text
        + selection
        + proofs
        + qa
        + failures,
    }
    destination = Path("docs/experiments")
    for name, text in reports.items():
        (destination / name).write_text(text, encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "promotion": promotion,
                "outer_metrics": {name: dict(zip(metrics, average(rows_for(name)))) for name in names},
                "inner_selection": data["selection"],
            },
            indent=2,
        )
    )
    print(json.dumps(promotion, indent=2))


if __name__ == "__main__":
    main()
