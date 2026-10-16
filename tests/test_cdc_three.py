"""Three-agent amendment: fixed membership, unchanged boundaries and private-state reuse."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord

from fed_med_agents.experiments.cdc_data import DOMAINS, THREE_DOMAINS, THREE_SLOTS
from fed_med_agents.experiments.cdc_flower import profile
from fed_med_agents.experiments.cdc_routing import (
    CDCFusion,
    context,
    fit_fusion,
    outcomes,
    prequery,
    router_network,
)
from fed_med_agents.experiments.cdc_three import reuse_context
from fed_med_agents.experiments.m5_step035 import KINDS, gates
from fed_med_agents.experiments.routing_diagnostics import snapshot


def test_membership_is_the_user_fixed_order():
    assert THREE_SLOTS == (1, 3, 4)
    assert list(THREE_DOMAINS) == ["metabolic", "lifestyle", "general_health"]
    assert list(THREE_DOMAINS.values()) == [DOMAINS[name] for name in THREE_DOMAINS]


def test_mapped_schedules_are_exact_historical_realizations():
    columns = list(dict.fromkeys(c for fields in DOMAINS.values() for c in fields))
    for origin, source in enumerate(THREE_SLOTS):
        for scenario in ("static", "variable"):
            old = context(np.arange(100), source, columns, scenario)
            new = context(np.arange(100), origin, columns, scenario, THREE_SLOTS)
            np.testing.assert_array_equal(new[0], old[0][:, THREE_SLOTS])
            np.testing.assert_array_equal(new[1], old[1])
            np.testing.assert_array_equal(new[2], old[2][:, THREE_SLOTS])


@pytest.mark.parametrize("kind", ["original", "logistic"])
def test_three_fusion_cold_training_rejects_six_agent_weights(kind):
    torch.set_num_threads(1)
    with pytest.raises(RuntimeError):
        CDCFusion(kind, 3).load_state_dict(snapshot(CDCFusion(kind, 6)))
    p = np.random.default_rng(42).uniform(0.05, 0.5, (50, 3)).astype(np.float32)
    y = np.tile(np.array([0.0, 1.0], dtype=np.float32), 25)
    model, log = fit_fusion(p, y, kind, 53530, torch.device("cpu"))
    assert model.clients == 3 and log["steps"] > 0
    assert outcomes(model, p, torch.device("cpu")).shape == (50, 3, 4)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_prequery_and_shared_arrays_have_three_agent_schema():
    for enriched, width in ((False, 12), (True, 37)):
        x = prequery(
            np.full(5, 0.2), 0, 1, np.ones((5, 3)), np.ones((5, 21)), np.ones((5, 3)), enriched, THREE_DOMAINS
        )
        assert x.shape == (5, width)
        state = ArrayRecord(snapshot(router_network(width))).to_torch_state_dict()
        for _ in range(3):
            router_network(width).load_state_dict(state, strict=True)
        with pytest.raises(RuntimeError):
            router_network(43 if enriched else 18).load_state_dict(state)


def test_reuse_is_byte_copy_and_column_remapping(tmp_path):
    source, target = tmp_path / "six", tmp_path / "three"
    source.mkdir()
    (source / "target_provenance.json").write_text("{}")
    array = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.savez(source / "stack.npz", p=array, rel=array, held_p=array, held_rel=array, fit=np.arange(4))
    for block in ("final", "target0", "target1", "target2"):
        path = source / block
        path.mkdir()
        records = {}
        for slot, (name, fields) in enumerate(DOMAINS.items()):
            records[str(slot)] = {"name": name, "columns": list(fields)}
            folder = path / f"client_{slot}"
            folder.mkdir()
            (folder / "final.joblib").write_bytes(f"private-{slot}".encode())
        (path / "provenance.json").write_text(json.dumps({"clients": records, "train": [1], "held": [2]}))
        np.savez(path / "probabilities.npz", train=array, held=array)
    reused = []
    reuse_context(source, target, [], reused)
    assert len(reused) == 12
    np.testing.assert_array_equal(np.load(target / "stack.npz")["p"], array[:, THREE_SLOTS])
    for block in ("final", "target0", "target1", "target2"):
        assert sorted(p.name for p in (target / block).glob("client_*")) == [
            "client_0",
            "client_1",
            "client_2",
        ]
        for client, slot in enumerate(THREE_SLOTS):
            assert (
                target / block / f"client_{client}/final.joblib"
            ).read_bytes() == f"private-{slot}".encode()


def test_flower_profile_uses_exactly_three_clients_and_rejects_other_membership(tmp_path, monkeypatch):
    monkeypatch.setenv("SWARMFL_CDC_ROOT", str(tmp_path))
    assert len(profile()[0]) == 6  # Historical entry point remains reproducible.
    config = {"clients": 3, "source_slots": THREE_SLOTS, "domains": THREE_DOMAINS}
    (tmp_path / "benchmark-config.json").write_text(json.dumps(config))
    assert profile() == (THREE_DOMAINS, THREE_SLOTS)
    config["clients"] = 4
    (tmp_path / "benchmark-config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        profile()


def test_numeric_gates_are_not_relaxed_for_three_origins():
    results, selection = {}, {}
    for fold in range(3):
        selection[str(fold)] = {kind: "ridge_base" for kind in KINDS}
        results[f"pilot_fold{fold}_outer"] = {}
        for kind in KINDS:
            policies = {
                name: {
                    str(o): {
                        "objective": objective,
                        "accuracy": 0.9,
                        "binary_cross_entropy": 0.25,
                        "utility_r2": 0.1,
                        "minority_peer_fraction": 0.1,
                    }
                    for o in range(3)
                }
                for name, objective in (("best_peer", 0.3), ("ridge_base", 0.29))
            }
            results[f"pilot_fold{fold}_outer"][kind] = {
                "static": {"0.02": policies},
                "variable": {"0.02": policies},
            }
    assert all(row["passed"] for row in gates(results, selection, "pilot").values())
    results["pilot_fold0_outer"]["mean"]["variable"]["0.02"]["ridge_base"]["2"]["minority_peer_fraction"] = (
        0.0
    )
    assert not gates(results, selection, "pilot")["mean"]["gates"]["three_varying_origins"]
