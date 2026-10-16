"""Replay CDC development-only predictions, full stack exclusions, metrics and Flower updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fed_med_agents.devices import select_device
from fed_med_agents.experiments.cdc_data import (
    DOMAINS,
    THREE_DOMAINS,
    THREE_SLOTS,
    expert_predict,
    fit_expert,
)
from fed_med_agents.experiments.cdc_routing import (
    CDCFusion,
    context,
    decide,
    fit_fusion,
    gross_gains,
    metrics,
    outcomes,
    router_network,
    train_router,
)
from fed_med_agents.experiments.m5_step03 import digest, write
from fed_med_agents.experiments.m5_step035 import (
    KINDS,
    VALUES,
    gates,
    infer_values,
    reliability,
    select_inner,
)
from fed_med_agents.experiments.routing_diagnostics import snapshot


def compare(actual: dict, expected: dict) -> int:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if value is None:
            assert actual[key] is None
        else:
            np.testing.assert_allclose(actual[key], value, atol=2e-6, rtol=1e-5, err_msg=key)
    return len(expected)


def verify(root: Path, report_path: Path | None = None) -> dict:
    """Do not load frozen test labels; check its byte hash and excluded source IDs only."""
    torch.set_num_threads(1)
    metadata = json.loads((root / "dataset.json").read_text())
    amended = (root / "benchmark-config.json").exists()
    domains, slots = (THREE_DOMAINS, THREE_SLOTS) if amended else (DOMAINS, None)
    clients = len(domains)
    assert digest(root / "frozen_test.npz") == metadata["frozen_test_sha256"]
    data = dict(np.load(root / "development.npz"))
    x, y, groups = (data[k] for k in ("x", "y", "groups"))
    assert (groups % 5 != 0).all()
    assert not set(data["ids"]) & set(np.load(root / "frozen_test_ids.npy"))
    results = json.loads((root / "results.json").read_text())
    contexts = json.loads((root / "contexts.json").read_text())
    device = select_device(json.loads((root / "execution.json").read_text())["device"])
    counts = {
        "private_models": 0,
        "private_models_refitted": 0,
        "fusion_states": 0,
        "assessment_metric_fields": 0,
        "aggregations": 0,
        "client_updates_replayed": 0,
        "fusion_optimizations_refitted": 0,
    }
    for size in ("small", "pilot"):
        selected = json.loads((root / f"selection_{size}.json").read_text())
        assert selected == select_inner(results, size)
        assert gates(results, selected, size) == json.loads((root / f"gates_{size}.json").read_text())
    for key, configuration in contexts.items():
        folder = root / key
        fit, held = np.array(configuration["fit"]), np.array(configuration["held"])
        assert not set(groups[fit]) & set(groups[held])
        stage = dict(np.load(folder / "stack.npz"))
        np.testing.assert_array_equal(stage["fit"], fit)
        np.testing.assert_array_equal(stage["held_y"], y[held])
        assert len(fit) == len(np.unique(fit))
        if key.endswith("outer"):
            inner_key = key.replace("outer", "inner")
            inner = contexts[inner_key]
            assert set(inner["fit"]) | set(inner["held"]) == set(fit)
            assert contexts[key.replace("small", "pilot")]["held"] == held.tolist()
        target_folds = json.loads((folder / "target_provenance.json").read_text())["folds"]
        target_positions = []
        for block in ("final", "target0", "target1", "target2"):
            block_folder = folder / block
            provenance = json.loads((block_folder / "provenance.json").read_text())
            train = np.array(provenance["train"])
            assessment = np.array(provenance["held"])
            assert not set(groups[train]) & set(groups[assessment])
            assert set(train).issubset(fit)
            saved = dict(np.load(block_folder / "probabilities.npz"))
            for client, domain in enumerate(domains.values()):
                info = provenance["clients"][str(client)]
                assert info["columns"] == list(domain)
                for name, upstream in [(f"oof_{i}", fold) for i, fold in enumerate(info["folds"])] + [
                    ("final", info["final"])
                ]:
                    fitting, calibration = np.array(upstream["model_fit"]), np.array(upstream["calibration"])
                    assert not set(groups[fitting]) & set(groups[calibration])
                    assert set(fitting).issubset(train) and set(calibration).issubset(train)
                    evaluated = assessment if name == "final" else np.array(upstream["held"])
                    assert not (set(groups[fitting]) | set(groups[calibration])) & set(groups[evaluated])
                    model = joblib.load(block_folder / f"client_{client}" / f"{name}.joblib")
                    assert model["columns"] == [metadata["features"].index(c) for c in domain]
                    expected = (
                        saved["held"][:, client]
                        if name == "final"
                        else saved["train"][upstream["held_positions"], client]
                    )
                    np.testing.assert_allclose(expert_predict(model, x[evaluated]), expected, atol=1e-7)
                    counts["private_models"] += 1
                assert digest(block_folder / f"client_{client}/final.joblib") == info["sha256"]
            if block == "final":
                # One actual calibrated refit per context, no saved states overwritten.
                info = provenance["clients"]["0"]["final"]
                model, _ = fit_expert(
                    x,
                    y,
                    groups,
                    train,
                    [metadata["features"].index(c) for c in next(iter(domains.values()))],
                    info["seed"],
                )
                np.testing.assert_allclose(
                    expert_predict(model, x[assessment]), saved["held"][:, 0], atol=1e-7
                )
                counts["private_models_refitted"] += 1
            else:
                fold = target_folds[int(block[-1])]
                assert fold["fit"] == train.tolist() and fold["target"] == assessment.tolist()
                assert set(train) | set(assessment) == set(fit)
                positions = fold["positions"]
                target_positions.extend(positions)
                np.testing.assert_array_equal(stage["p"][positions], saved["held"])
                np.testing.assert_allclose(
                    stage["rel"][positions],
                    np.tile(reliability(saved["train"], y[train]), (len(positions), 1)),
                    atol=1e-7,
                )
            for kind in KINDS:
                fusion = CDCFusion(kind, clients).to(device)
                path = (
                    folder / f"fusion_{kind}.pt" if block == "final" else block_folder / f"fusion_{kind}.pt"
                )
                fusion.load_state_dict(torch.load(path, weights_only=True))
                if amended:
                    torch.manual_seed(configuration["seed"])
                    initial = snapshot(CDCFusion(kind, clients))
                    torch.testing.assert_close(
                        initial,
                        torch.load(path.with_name(f"fusion_initial_{kind}.pt"), weights_only=True),
                        atol=0,
                        rtol=0,
                    )
                    if kind != "mean":
                        refitted, _ = fit_fusion(
                            saved["train"], y[train], kind, configuration["seed"], device
                        )
                        torch.testing.assert_close(snapshot(refitted), snapshot(fusion), atol=1e-6, rtol=1e-5)
                        assert any(not torch.equal(initial[k], v) for k, v in snapshot(fusion).items())
                        counts["fusion_optimizations_refitted"] += 1
                actual = outcomes(fusion, saved["held"], device)
                targets = np.load(folder / f"targets_{kind}.npz")
                expected = targets["held"] if block == "final" else targets["train"][positions]
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                counts["fusion_states"] += 1
        assert sorted(target_positions) == list(range(len(fit)))
        actions = dict(np.load(folder / "actions.npz"))
        for kind in KINDS:
            targets = np.load(folder / f"targets_{kind}.npz")
            mean_gain = gross_gains(targets["train"], stage["y"]).mean(0)
            for scenario in ("static", "variable"):
                for origin in range(clients):
                    available, covered, prices = context(
                        data["ids"][held], origin, metadata["features"], scenario, slots
                    )
                    estimates = {}
                    for name in VALUES:
                        enriched = name.endswith("enriched")
                        if name.startswith("ridge"):
                            model = joblib.load(folder / f"router_{kind}_{name}.joblib")
                        else:
                            model = router_network(2 * clients + 6 + (25 if enriched else 0)).to(device)
                            model.load_state_dict(
                                torch.load(folder / f"router_{kind}_{name}.pt", weights_only=True)
                            )
                        estimates[name] = infer_values(
                            model,
                            stage["held_p"][:, origin],
                            origin,
                            available,
                            covered,
                            stage["held_rel"],
                            enriched,
                            device,
                            domains,
                        )
                    estimates["fixed_value"] = np.tile(mean_gain[origin], (len(held), 1))
                    for tariff, policies in results[key][kind][scenario].items():
                        for name, origin_rows in policies.items():
                            selected = actions[f"{kind}_{scenario}_{tariff}_{name}_{origin}"]
                            if name in estimates:
                                np.testing.assert_array_equal(
                                    selected,
                                    decide(estimates[name], origin, available, prices, float(tariff)),
                                )
                            recalculated = metrics(
                                stage["held_y"],
                                targets["held"][:, origin],
                                selected,
                                origin,
                                available,
                                prices,
                                float(tariff),
                                estimates.get(name),
                                mean_gain,
                            )
                            recalculated.pop("actions")
                            counts["assessment_metric_fields"] += compare(
                                recalculated, origin_rows[str(origin)]
                            )
    # Fixed-six integration: compatible shared arrays, all clients, true local optimizer steps.
    flower = root / "flower"
    import os

    from fed_med_agents.experiments.cdc_flower import STAGE, training

    os.environ["SWARMFL_CDC_ROOT"] = str(root.resolve())
    fed = json.loads((flower / "results.json").read_text())
    for round_id in range(1, 4):
        state = torch.load(flower / f"server_round_{round_id}.pt", weights_only=True)
        prior = torch.load(flower / f"server_round_{round_id - 1}.pt", weights_only=True)
        updates = []
        for client in range(clients):
            xtrain, labels = training(client)
            model = router_network(xtrain.shape[1])
            model.load_state_dict(prior)
            train_router(model, xtrain, labels, 2, 53530 + round_id, torch.device("cpu"))
            actual = torch.load(flower / f"client_{client}_round_{round_id}.pt", weights_only=True)
            torch.testing.assert_close(snapshot(model), actual, atol=1e-7, rtol=1e-6)
            updates.append(actual)
            counts["client_updates_replayed"] += 1
        for parameter, value in state.items():
            torch.testing.assert_close(
                value, sum(c[parameter] for c in updates) / clients, atol=1e-6, rtol=1e-5
            )
        counts["aggregations"] += 1
    held = np.load(root / STAGE / "stack.npz")["held"].tolist()
    expected = np.load(root / STAGE / "stack.npz")["held_p"]
    for message in fed["consultations"]:
        rows = [held.index(i) for i in message["ids"]]
        np.testing.assert_allclose(message["p"], expected[rows, message["peer"]], atol=1e-7)
    before = json.loads((root / "private-before-federation.json").read_text())
    assert all(digest(Path(p)) == h for p, h in before.items())
    preserved = json.loads((root / "preserved-before.json").read_text())
    changed = [p for p, h in preserved.items() if digest(Path(p)) != h]
    assert not changed, changed
    protocol = "m5-step035-three-agent-protocol.md" if amended else "m5-step035-protocol.md"
    assert digest(root / "protocol.md") == digest(Path("docs/experiments") / protocol)
    if amended:
        source = Path(json.loads((root / "benchmark-config.json").read_text())["source"])
        for filename in (
            "contexts.json",
            "development.npz",
            "pilot_ids.npy",
            "frozen_test.npz",
            "frozen_test_ids.npy",
        ):
            assert digest(root / filename) == digest(source / filename)
        reused = json.loads((root / "reuse.json").read_text())["models"]
        for record in reused:
            assert digest(Path(record["source"])) == digest(Path(record["destination"])) == record["sha256"]
        counts["byte_identical_reused_models"] = len(reused)
    counts.update(
        preserved_files=len(preserved),
        changed=changed,
        frozen_test_evaluations=0,
        frozen_test_hash_unchanged=True,
        success=True,
    )
    write(report_path or root / "verification.json", counts)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, help="Separate summary path for read-only historical replay")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.output, args.report), indent=2))
