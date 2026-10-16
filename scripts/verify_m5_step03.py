"""Independently replay M5 Step 3 provenance, predictions, utilities and Flower arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from fed_med_agents.experiments.consultation_value import (
    choose,
    fit_network,
    predict,
    summarize,
    value_network,
)
from fed_med_agents.experiments.diabetes import generate_oof, logits
from fed_med_agents.experiments.m5_step03 import (
    digest,
    fit_fusion,
    flatten_gains,
    gains,
    matrix,
    probabilities,
    reliability,
    risks,
    select,
    unpack_predictions,
    write,
)
from fed_med_agents.experiments.repair_data import training_source
from fed_med_agents.experiments.repair_models import make_model
from fed_med_agents.experiments.routing_diagnostics import snapshot
from fed_med_agents.models import Classifier


def compare(actual, expected) -> int:
    """Recursive metric check including all actions, with tight numerical tolerance."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        return sum(compare(actual[k], v) for k, v in expected.items())
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    return 1


def verify(root: Path, refit_oof: bool = False) -> dict:
    """Read saved states and original-training source only; never repair old evidence."""
    torch.set_num_threads(1)
    contexts = json.loads((root / "contexts.json").read_text())
    results = json.loads((root / "results.json").read_text())
    selected = json.loads((root / "selection.json").read_text())
    assert selected == select(results)
    local, y, original = training_source(Path("artifacts/milestone02/seed_42"))
    counts = {
        "private_states": 0,
        "fusion_states_refitted": 0,
        "local_router_updates_replayed": 0,
        "oof_columns_refitted": 0,
        "utility_rows": 0,
        "metric_fields": 0,
        "aggregations": 0,
        "client_updates": 0,
    }
    outer_held = []
    for key, context in contexts.items():
        folder = root / key
        fit, held = set(context["fit"]), set(context["held"])
        assert not fit & held
        if key.endswith("outer"):
            outer_held.extend(held)
            inner = contexts[key.replace("outer", "inner")]
            assert set(inner["fit"]) | set(inner["held"]) == fit
        provenance = json.loads((folder / "provenance.json").read_text())["folds"]
        targets = []
        for fold in provenance:
            parent, target = set(fold["all_upstream_fit"]), set(fold["target"])
            assert not parent & target and parent | target == fit
            targets.extend(target)
        assert sorted(targets) == sorted(fit)
        for block in ("final", "target0", "target1", "target2"):
            stage = folder / block / "seed_42"
            with np.load(stage / "cohort.npz") as cohort:
                np.testing.assert_array_equal(cohort["original_ids"], original)
                train, assessment = cohort["train"], cohort["val"]
                assert set(train).issubset(fit) and not set(train) & set(assessment)
            for client in range(3):
                path = stage / f"client_{client}"
                meta = json.loads((path / "specialist.json").read_text())
                assert digest(path / "specialist.pt") == meta["specialist_sha256"]
                assert set(meta["preprocessing"]["fit_ids"]) == set(train)
                for fold in meta["folds"]:
                    upstream = set(fold["model_fit_ids"]) | set(fold["calibration_ids"])
                    assert not upstream & set(fold["held_ids"])
                    assert upstream | set(fold["held_ids"]) == set(train)
                    assert set(fold["preprocessing"]["fit_ids"]) == set(fold["model_fit_ids"])
                with np.load(path / "local.npz") as data:
                    np.testing.assert_array_equal(data["train_y"], y[train])
                    model = Classifier(data["x"].shape[1])
                    model.load_state_dict(torch.load(path / "specialist.pt", weights_only=True))
                    actual = torch.sigmoid(
                        torch.tensor(
                            logits(model, data["x"][assessment], torch.device("cpu")) / meta["temperature"]
                        )
                    ).numpy()
                    np.testing.assert_allclose(actual, data["assessment"], atol=1e-7, rtol=1e-6)
                    if refit_oof:
                        seed = context["seed"] if block == "final" else context["seed"] + 10 + int(block[-1])
                        regenerated, _, _ = generate_oof(
                            local[client], y, train, seed, client, torch.device("cpu"), client == 0, folds=3
                        )
                        np.testing.assert_allclose(regenerated, data["oof"], atol=1e-7, rtol=1e-6)
                        counts["oof_columns_refitted"] += 1
                counts["private_states"] += 1
        for kind in ("original", "logistic"):
            data = dict(np.load(folder / f"{kind}.npz"))
            np.testing.assert_allclose(gains(data["target_risks"], data["y"]), data["gain"], atol=1e-7)
            for fold_id, fold in enumerate(provenance):
                teacher = make_model(kind)
                teacher.load_state_dict(
                    torch.load(folder / f"target{fold_id}" / f"fusion_{kind}.pt", weights_only=True)
                )
                tp, hp = probabilities(folder / f"target{fold_id}")
                positions, parent = fold["target_positions"], fold["all_upstream_fit"]
                refitted = fit_fusion(tp, y[parent], kind, context["seed"], None)
                torch.testing.assert_close(snapshot(refitted), snapshot(teacher), atol=1e-7, rtol=1e-6)
                counts["fusion_states_refitted"] += 1
                np.testing.assert_allclose(data["target_risks"][positions], risks(teacher, hp), atol=1e-7)
                np.testing.assert_allclose(data["p"][positions], hp, atol=1e-7)
                np.testing.assert_allclose(
                    data["rel"][positions],
                    np.tile(reliability(tp, y[parent]), (len(positions), 1, 1)),
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    data["history"][positions],
                    np.tile(gains(risks(teacher, tp), y[parent]).mean(0), (len(positions), 1, 1)),
                    atol=1e-7,
                )
            fusion = make_model(kind)
            fusion.load_state_dict(torch.load(folder / f"fusion_{kind}.pt", weights_only=True))
            tp, _ = probabilities(folder / "final")
            refitted = fit_fusion(tp, data["y"], kind, context["seed"], None)
            torch.testing.assert_close(snapshot(refitted), snapshot(fusion), atol=1e-7, rtol=1e-6)
            counts["fusion_states_refitted"] += 1
            np.testing.assert_allclose(risks(fusion, data["held_p"]), data["held_risks"], atol=1e-7)
            counts["utility_rows"] += len(data["y"]) * 6
            routing = dict(np.load(folder / f"routing_{kind}.npz"))
            np.testing.assert_array_equal(flatten_gains(data["gain"]), routing["target"])
            for name in (
                n for n in results[key][kind]["0.02"] if n.startswith(("ridge_", "mlp_", "tree_", "linear_"))
            ):
                family, group = name.split("_", 1)
                x = matrix(data["held_p"], data["held_rel"], data["held_history"], group)
                if family in ("mlp", "linear"):
                    model = value_network(x.shape[1], family)
                    model.load_state_dict(torch.load(folder / f"router_{kind}_{name}.pt", weights_only=True))
                else:
                    model = joblib.load(folder / f"router_{kind}_{name}.joblib")
                np.testing.assert_allclose(
                    unpack_predictions(predict(model, x), len(data["held_y"])),
                    routing[name],
                    atol=1e-7,
                    rtol=1e-6,
                )
            for tariff, policies in results[key][kind].items():
                for name, origins in policies.items():
                    for origin_name, saved in origins.items():
                        origin = int(origin_name)
                        actions = np.array(saved["actions"])
                        estimate = routing[name][:, origin] if name in routing else None
                        if name == "fixed_value":
                            estimate = np.tile(data["mean_gain"][origin], (len(actions), 1))
                        if estimate is not None:
                            np.testing.assert_array_equal(actions, choose(estimate, origin, float(tariff)))
                        counts["metric_fields"] += compare(
                            summarize(
                                data["held_y"],
                                data["held_risks"][:, origin],
                                actions,
                                origin,
                                float(tariff),
                                estimate,
                                data["mean_gain"],
                            ),
                            saved,
                        )
    assert sorted(outer_held) == list(range(364))
    federated = json.loads((root / "federated/results.json").read_text())
    for trial, record in federated["trials"].items():
        stage = trial[: len("fold0_outer")]
        kind = trial[len(stage) + 1 :]
        folder = root / "federated" / stage / kind
        for round_id in range(1, 6):
            global_state = torch.load(folder / f"server_round_{round_id}.pt", weights_only=True)
            clients = [
                torch.load(folder / f"client_{i}_round_{round_id}.pt", weights_only=True) for i in range(3)
            ]
            for parameter in global_state:
                torch.testing.assert_close(
                    global_state[parameter], sum(s[parameter] for s in clients) / 3, atol=1e-6, rtol=1e-5
                )
            for origin in range(3):
                audit = json.loads((folder / f"client_{origin}_round_{round_id}.json").read_text())
                assert audit["client"] == origin and audit["steps"] == 16 and audit["drift"] > 0
                assert audit["payload_keys"] == ["arrays", "metrics"]
                assert all("fusion" not in k and "specialist" not in k for k in clients[origin])
                route = dict(np.load(root / stage / f"routing_{kind}.npz"))
                replay_model = value_network(route["x"].shape[1], record["architecture"])
                replay_model.load_state_dict(
                    torch.load(folder / f"server_round_{round_id - 1}.pt", weights_only=True)
                )
                n = len(route["target"]) // 3
                fit_network(
                    replay_model,
                    route["x"][origin * n : (origin + 1) * n],
                    route["target"][origin * n : (origin + 1) * n],
                    16,
                )
                torch.testing.assert_close(snapshot(replay_model), clients[origin], atol=1e-7, rtol=1e-6)
                counts["local_router_updates_replayed"] += 1
            counts["aggregations"] += 1
            counts["client_updates"] += 3
        data = dict(np.load(root / stage / f"{kind}.npz"))
        route = dict(np.load(root / stage / f"routing_{kind}.npz"))
        model = value_network(route["x"].shape[1], record["architecture"])
        model.load_state_dict(global_state)
        estimate = unpack_predictions(predict(model, route["held_x"]), len(data["held_y"]))
        for tariff, origins in record["assessment"].items():
            for origin_name, saved in origins.items():
                origin = int(origin_name)
                actions = choose(estimate[:, origin], origin, float(tariff))
                counts["metric_fields"] += compare(
                    summarize(
                        data["held_y"],
                        data["held_risks"][:, origin],
                        actions,
                        origin,
                        float(tariff),
                        estimate[:, origin],
                        data["mean_gain"],
                    ),
                    saved,
                )
    for query in federated["consultations"]:
        held = np.load(root / query["stage"] / "indices.npz")["held"].tolist()
        p = np.load(root / query["stage"] / f"{query['kind']}.npz")["held_p"]
        rows = [held.index(i) for i in query["ids"]]
        np.testing.assert_allclose(query["p"], p[rows, query["peer"]], atol=1e-7, rtol=1e-6)
    preserved = json.loads((root / "preserved-before.json").read_text())
    changed = [path for path, expected in preserved.items() if digest(Path(path)) != expected]
    assert not changed, changed
    assert digest(root / "protocol.md") == digest(Path("docs/experiments/m5-step03-protocol.md"))
    counts.update(preserved_files=len(preserved), changed=changed, success=True)
    write(root / "verification.json", counts)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--refit-oof", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.refit_oof), indent=2))
