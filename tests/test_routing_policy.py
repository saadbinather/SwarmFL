"""Unified routing, observed-only updates and legacy M4 version compatibility."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from test_registry import registry, session

from fed_med_agents.experiments.cdc_scaling import prefix, split_ids
from fed_med_agents.feedback.events import PredictionEvent, VerifiedOutcome, digest
from fed_med_agents.feedback.learning import routing_feedback_loss
from fed_med_agents.feedback.versions import PolicyRegistry
from fed_med_agents.routing.policy import RoutingPolicy


@pytest.mark.parametrize("kind", ["fixed", "mlp", "graph"])
def test_common_policy_registry_lifecycle_and_roundtrip(kind, tmp_path):
    torch.manual_seed(3)
    policy = RoutingPolicy(kind, ("A", "B", "C"))
    policy.gains.fill_(0.5)
    reg = registry()
    active = session(reg)
    contexts = {"B": [0.0] * 63, "C": [0.0] * 63}
    decision = policy.route(active, contexts, 0.2, version="v000")
    assert set(decision.eligible) == {"B", "C"}
    reg.update("B", available=False)
    assert "B" not in policy.route(active, contexts, 0.2, version="v000").eligible
    reg.unregister("C")
    assert policy.route(active, contexts, 0.2, version="v000").destination is None
    path = tmp_path / "router.pt"
    policy.save(path)
    restored = RoutingPolicy.load(path)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in policy.state_dict().items())
    versions = PolicyRegistry(tmp_path / "versions")
    versions.save("v000", policy.state_dict(), {"eligible": True, "router_config": policy.config()})
    versions.approve("v000")
    versions.save("v001", policy.state_dict(), {"eligible": False, "router_config": policy.config()})
    with pytest.raises(ValueError, match="Rejected"):
        versions.approve("v001")
    versions.rollback("v000")
    assert versions.current() == "v000"
    assert versions.load().config() == policy.config()
    policy.gains.fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        policy.save(tmp_path / "invalid.pt")
    torch.save({"config": policy.config(), "state": policy.state_dict()}, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="Nonfinite"):
        RoutingPolicy.load(tmp_path / "invalid.pt")


def records():
    result = []
    for i in range(40):
        e = PredictionEvent(
            request_id=str(i),
            patient_id=str(i),
            origin=0,
            episode=1,
            available_domains=[True] * 3,
            action=2,
            consulted=[1],
            propensity=1 / 3,
            action_score=0.1,
            evidence=[0.2, 0.3, None],
            final_risk=0.25,
            origin_only_risk=0.2,
            model_version="frozen",
            policy_version="v000",
            consultation_cost=0.02,
            event_timestamp="2026-10-15T01:00:00+00:00",
        )
        o = VerifiedOutcome(
            request_id=str(i),
            patient_id=str(i),
            prediction_digest=digest(e),
            label=i % 2,
            authority="simulation-training-label",
            received_episode=2,
            verified_timestamp="2026-10-15T02:00:00+00:00",
        )
        result.append((e, o))
    context = {str(i): {"features": [[0.0] * 63 for _ in range(3)], "costs": [0.02] * 3} for i in range(40)}
    return result, context


def test_observed_update_and_insufficient_feedback_retains_exact_state():
    torch.manual_seed(4)
    model = RoutingPolicy("mlp", ("A", "B", "C"))
    replay, contexts = records()
    before = copy.deepcopy(model.state_dict())
    assert not model.update(replay[:1], contexts)["updated"]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    log = model.update(replay, contexts)
    assert log["updated"] and log["gradient_norm"] > 0
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    # There is no unchosen evidence/reward in either API; extra reward fields
    # cannot affect optimization even when present in the local context dictionary.
    other = copy.deepcopy(contexts)
    for row in other.values():
        row["unchosen_reward"] = -100000
    a, _ = routing_feedback_loss(model, replay, contexts, torch.device("cpu"))
    b, _ = routing_feedback_loss(model, replay, other, torch.device("cpu"))
    assert torch.equal(a, b)
    bad = copy.deepcopy(replay)
    bad[0][0].action = 3
    with pytest.raises(ValueError, match="Action|action"):
        routing_feedback_loss(model, bad, contexts, torch.device("cpu"))


@pytest.mark.parametrize("kind", ["fixed", "mlp", "graph"])
def test_supervised_fit_masks_self_and_unavailable(kind):
    torch.manual_seed(8)
    model = RoutingPolicy(kind, ("A", "B", "C"), 5)
    origins = torch.arange(3).repeat_interleave(8)
    features = torch.randn(24, 3, 5)
    gains = torch.randn(24, 3) * 0.1
    own = torch.full((24,), 0.2)
    available = torch.ones(24, 3, dtype=torch.bool)
    costs = torch.full((24, 3), 0.02)
    result = model.fit(features, origins, gains, own, available, costs, seed=8, epochs=1)
    assert result["steps"] == (0 if kind == "fixed" else 1)
    available[:] = False
    scores = model.logits(features, origins, own, available, costs)
    assert (scores.argmax(-1) == 0).all()
    assert torch.isneginf(scores[:, 1:]).all()


def test_grouped_splits_and_nested_prefixes():
    n = 100000
    data = {"y": np.arange(n) % 2, "groups": np.arange(n) // 2}
    pool, guard, assessment = split_ids(data, 55001)
    assert not set(data["groups"][pool]) & set(data["groups"][guard])
    assert not set(data["groups"][assessment]) & set(data["groups"][guard])
    a = prefix(pool, data["groups"], 15001)
    b = prefix(pool, data["groups"], 30000)
    assert np.array_equal(a, b[: len(a)]) and len(a) == 15002


def test_validation_gates_and_actual_eligible_version_rollback(tmp_path):
    from fed_med_agents.feedback.versions import routing_update_gates

    rows = [{"objective": 0.35, "accuracy": 0.86, "binary_cross_entropy": 0.34} for _ in range(3)]
    assert all(routing_update_gates(rows, rows, rows, [32] * 3).values())
    worse = copy.deepcopy(rows)
    worse[0]["binary_cross_entropy"] += 0.02
    assert not routing_update_gates(rows, rows, worse, [32] * 3)["origin_bce"]
    assert not routing_update_gates(rows, rows, rows, [31] * 3)["feedback_sufficient"]
    model = RoutingPolicy("mlp", ("A", "B", "C"))
    versions = PolicyRegistry(tmp_path)
    versions.save("v000", model.state_dict(), {"eligible": True, "router_config": model.config()})
    versions.approve("v000")
    with torch.no_grad():
        next(model.parameters()).add_(0.001)
    versions.save("v001", model.state_dict(), {"eligible": True, "router_config": model.config()})
    versions.approve("v001")
    assert versions.current() == "v001"
    versions.rollback("v000")
    assert versions.current() == "v000"
    assert not torch.equal(next(versions.load("v001").parameters()), next(versions.load().parameters()))


def test_authorized_adapter_context_ignores_hidden_peer_values_and_fields():
    from fed_med_agents.experiments.cdc_data import DOMAINS
    from fed_med_agents.experiments.cdc_scaling import tensors

    columns = list(dict.fromkeys(field for fields in DOMAINS.values() for field in fields))
    rng = np.random.default_rng(2)
    x = rng.uniform(1, 4, (9, 21)).astype(np.float32)
    p = np.full((9, 3), 0.2, np.float32)
    data = {"x": x, "ids": np.arange(9)}
    original = tensors(data, columns, np.arange(9), p, np.zeros_like(p), "variable", torch.device("cpu"))
    p[:, 1:] = 0.9
    permitted = set(DOMAINS["metabolic"])
    for i, name in enumerate(columns):
        if name not in permitted:
            x[:, i] = 99
    changed = tensors(data, columns, np.arange(9), p, np.zeros_like(p), "variable", torch.device("cpu"))
    assert torch.equal(original["features"][:9], changed["features"][:9])
    assert torch.equal(original["own"][:9], changed["own"][:9])
