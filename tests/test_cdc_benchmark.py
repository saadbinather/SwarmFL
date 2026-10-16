"""CDC benchmark group exclusion, private feature isolation and routing contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord
from threadpoolctl import threadpool_limits

from fed_med_agents.experiments.cdc_data import DOMAINS, expert_predict, fit_expert, group_ids, grouped_split
from fed_med_agents.experiments.cdc_routing import (
    CDCFusion,
    context,
    decide,
    fit_fusion,
    prequery,
    router_network,
    train_router,
)
from fed_med_agents.experiments.m5_step035 import KINDS, VALUES, select_inner
from fed_med_agents.experiments.routing_diagnostics import snapshot


def test_duplicate_feature_groups_ignore_labels():
    x = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)
    groups = group_ids(x)
    assert groups[0] == groups[1] and groups[0] != groups[2]
    np.testing.assert_array_equal(groups % 5 == 0, group_ids(x.copy()) % 5 == 0)


def test_calibration_groups_and_private_feature_isolation():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(240, 21)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32)
    groups = group_ids(x)
    train, held = np.arange(180), np.arange(180, 240)
    fit, cal = grouped_split(train, y, groups, 42)
    assert not set(groups[fit]) & set(groups[cal])
    with threadpool_limits(limits=1):
        model, provenance = fit_expert(x, y, groups, train, [0, 1, 2], 42)
        changed_y = y.copy()
        changed_y[held] = 1 - changed_y[held]
        other, other_provenance = fit_expert(x, changed_y, groups, train, [0, 1, 2], 42)
        assert provenance == other_provenance
        baseline = expert_predict(model, x[held])
        np.testing.assert_array_equal(baseline, expert_predict(other, x[held]))
        modified = x[held].copy()
        modified[:, 3:] = 99999
        np.testing.assert_array_equal(baseline, expert_predict(model, modified))


@pytest.mark.parametrize("kind", KINDS)
def test_fusion_never_uses_absent_predictions(kind):
    model = CDCFusion(kind)
    p = torch.full((4, 6), 0.2)
    o = torch.zeros(4, dtype=torch.long)
    mask = torch.zeros_like(p, dtype=torch.bool)
    mask[:, [0, 3]] = True
    actual = model(p, mask, o)
    p[:, [1, 2, 4, 5]] = torch.nan
    torch.testing.assert_close(model(p, mask, o), actual)


@pytest.mark.parametrize("kind", ["original", "logistic"])
def test_cpu_fusion_fitting_freezes_final_weights(kind):
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    p = rng.uniform(0.02, 0.7, (80, 6)).astype(np.float32)
    y = (rng.random(80) < p[:, 0]).astype(np.float32)
    model, log = fit_fusion(p, y, kind, 42, torch.device("cpu"))
    assert log["steps"] > 0 and np.isfinite(log["epoch_losses"]).all()
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_service_context_stable_and_public():
    columns = list(dict.fromkeys(c for domain in DOMAINS.values() for c in domain))
    ids = np.arange(150)
    available, covered, prices = context(ids, 0, columns, "variable")
    for a, b in zip(context(ids[::-1], 0, columns, "variable"), (available, covered, prices), strict=True):
        np.testing.assert_array_equal(a[::-1], b)
    assert available[:, 0].all()
    assert not available.all() and not covered.all() and len(np.unique(prices)) > 1
    for peer, domain in enumerate(DOMAINS.values()):
        assert covered[available[:, peer]][:, [columns.index(c) for c in domain]].all()


@pytest.mark.parametrize("enriched", [False, True])
def test_prequery_has_no_peer_evidence(enriched):
    own = np.array([0.1, 0.4, 0.8], dtype=np.float32)
    rel = np.full((3, 6), 0.4, dtype=np.float32)
    x = prequery(own, 0, 1, np.ones((3, 6)), np.ones((3, 21)), rel, enriched)
    assert x.shape == (3, 43 if enriched else 18)
    assert np.isfinite(x).all()
    with pytest.raises(TypeError):
        prequery(own, 0, 1, np.ones((3, 6)), np.ones((3, 21)), rel, enriched, y=np.ones(3))


def test_budget_cost_and_no_available_peers():
    value = np.ones((3, 6))
    available = np.zeros_like(value, dtype=bool)
    available[:, 0] = True
    np.testing.assert_array_equal(decide(value, 0, available, value, 0.02), [0, 0, 0])
    available[:, 1] = True
    np.testing.assert_array_equal(decide(value, 0, available, value, 1), [0, 0, 0])
    np.testing.assert_array_equal(decide(value, 0, available, value, 0.02, budget=0), [0, 0, 0])
    with pytest.raises(ValueError):
        decide(value, 0, available, value, 0.02, budget=2)
    with pytest.raises(ValueError):
        decide(value, 0, available, value, float("nan"))


def test_six_compatible_private_free_router_arrays():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = router_network(43)
    before = snapshot(model)
    rng = np.random.default_rng(42)
    log = train_router(
        model,
        rng.normal(size=(50, 43)).astype(np.float32),
        rng.normal(size=50).astype(np.float32),
        2,
        42,
        torch.device("cpu"),
    )
    state = ArrayRecord(snapshot(model)).to_torch_state_dict()
    assert log["gradient_norm"] > 0 and any(not torch.equal(before[k], v) for k, v in state.items())
    assert not any("fusion" in k or "specialist" in k for k in state)
    for _ in range(6):
        router_network(43).load_state_dict(state, strict=True)


def test_nested_selection_ignores_outer_metrics():
    results = {
        f"pilot_fold{fold}_inner": {
            kind: {
                "variable": {
                    "0.02": {
                        name: {str(origin): {"objective": float(k)} for origin in range(6)}
                        for k, name in enumerate(VALUES)
                    }
                }
            }
            for kind in KINDS
        }
        for fold in range(3)
    }
    selected = select_inner(results, "pilot")
    results["pilot_fold0_outer"] = {"outcome": "cannot be selected on"}
    assert select_inner(results, "pilot") == selected
