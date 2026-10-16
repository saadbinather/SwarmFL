"""Leakage, pre-query isolation and compatible routing-only update contracts."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord

from fed_med_agents.experiments.consultation_value import (
    FEATURE_SETS,
    choose,
    features,
    fit_network,
    value_network,
)
from fed_med_agents.experiments.m5_step03 import fit_fusion, gains, probabilities, risks, select
from fed_med_agents.experiments.repair_data import prepare_context
from fed_med_agents.experiments.routing_diagnostics import snapshot


@pytest.mark.parametrize("group", FEATURE_SETS)
def test_prequery_feature_isolation(group):
    rng = np.random.default_rng(4)
    p = rng.uniform(0.1, 0.9, (20, 3)).astype(np.float32)
    rel, history = np.zeros((20, 3, 3)), np.ones((20, 3, 3))
    before = features(p[:, 1], 1, 2, rel, history, group)
    p[:, [0, 2]] = np.nan
    after = features(p[:, 1], 1, 2, rel, history, group)
    np.testing.assert_array_equal(before, after)
    assert "y" not in inspect.signature(features).parameters
    assert "peer_probability" not in inspect.signature(features).parameters


def test_cost_budget_and_availability():
    benefit = np.array([[100.0, 0.02, 0.01], [100.0, 0.4, 0.3]])
    np.testing.assert_array_equal(choose(benefit, 0, 0.02), [0, 2])
    np.testing.assert_array_equal(choose(benefit, 0, 0.02, budget=0), [0, 0])
    available = np.array([[True, False, False], [True, False, True]])
    np.testing.assert_array_equal(choose(benefit, 0, 0.02, available), [0, 3])
    with pytest.raises(ValueError):
        choose(benefit, 0, 0.02, budget=2)
    with pytest.raises(ValueError):
        choose(np.full((2, 3), np.nan), 0, 0.02)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_compatible_router_only_updates(kind):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = value_network(21, kind)
    initial = snapshot(model)
    rng = np.random.default_rng(42)
    report = fit_network(
        model, rng.normal(size=(30, 21)).astype(np.float32), rng.normal(size=30).astype(np.float32), steps=3
    )
    state = ArrayRecord(snapshot(model)).to_torch_state_dict()
    assert not any("fusion" in k or "specialist" in k for k in state)
    for _ in range(3):
        value_network(21, kind).load_state_dict(state, strict=True)
    assert report["gradient_norm"] > 0
    assert any(not torch.equal(initial[k], state[k]) for k in state)


def test_full_upstream_excludes_utility_target_labels(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    local = [rng.integers(0, 2, (48, width)).astype(np.float32) for width in (3, 5, 8)]
    local[0][:, 0] = rng.integers(20, 70, 48)
    y = np.tile(np.array([0.0, 1.0], dtype=np.float32), 24)
    fit, held = np.arange(36), np.arange(36, 48)
    changed = y.copy()
    changed[held] = 1 - changed[held]
    outputs = []
    for name, labels in (("a", y), ("b", changed)):
        folder = tmp_path / name
        prepare_context(folder, local, labels, np.arange(48), fit, held, 123)
        p, hp = probabilities(folder)
        model = fit_fusion(p, labels[fit], "original", 123, folder / "fusion.pt")
        outputs.append((p, hp, snapshot(model), risks(model, hp)))
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
    for key in outputs[0][2]:
        assert torch.equal(outputs[0][2][key], outputs[1][2][key])
    assert not np.allclose(gains(outputs[0][3], y[held]), gains(outputs[1][3], changed[held]))


def test_selection_ignores_outer_outcomes():
    names = [f"{family}_{group}" for group in FEATURE_SETS for family in ("ridge", "mlp", "tree")]
    results = {}
    for fold in range(3):
        results[f"fold{fold}_inner"] = {
            kind: {
                "0.02": {
                    name: {str(o): {"objective": float(i)} for o in range(3)} for i, name in enumerate(names)
                }
            }
            for kind in ("original", "logistic")
        }
    expected = select(results)
    results.update({f"fold{fold}_outer": {"fabricated_outcome": -1e20} for fold in range(3)})
    assert select(results) == expected
