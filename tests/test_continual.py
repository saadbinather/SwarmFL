"""Continual lifecycle, delayed signed feedback, independent consumption and atomic versions."""

from __future__ import annotations

import copy

import pytest
import torch

from fed_med_agents.feedback.continual import ContinualController
from fed_med_agents.feedback.events import FeedbackLedger, VerifiedOutcome, digest, sign
from fed_med_agents.feedback.learning import fusion_feedback_update
from fed_med_agents.feedback.versions import PolicyPair, PolicyRegistry
from fed_med_agents.fusion.dynamic import DynamicFusion
from fed_med_agents.routing.policy import RoutingPolicy
from tests.test_registry import registry, reply, session, specialist

KEYS = ("ab" * 32, "cd" * 32)


def setup(tmp_path, *, stop=False, mean=False):
    router = RoutingPolicy("fixed", tuple("ABC"))
    router.gains.fill_(0 if stop else 0.2)
    fusion = DynamicFusion(
        "mean" if mean else "set",
        [specialist(n).model_dump(mode="json") for n in "ABC"],
        {"A": 0.3, "B": 0.5, "C": 0.7},
    )
    pair = PolicyPair(
        tmp_path / "pair", PolicyRegistry(tmp_path / "routers"), PolicyRegistry(tmp_path / "fusions")
    )
    pair.routers.save("r000", router.state_dict(), {"eligible": True, "router_config": router.config()})
    pair.fusions.save("f000", fusion.state_dict(), {"eligible": True, "fusion_config": fusion.config()})
    pair.activate("r000", "f000", reason="controlledfixture")
    ledger = FeedbackLedger(tmp_path / "local/ledger", 0, {"123", "124"}, *KEYS)
    ctrl = ContinualController(tmp_path / "local", pair, ledger)
    return ctrl, pair


def outcome(event, **changes):
    values = {
        "request_id": event.request_id,
        "patient_id": event.patient_id,
        "prediction_digest": digest(event),
        "label": 1,
        "authority": "simulation-training-label",
        "received_episode": 2,
        "verified_timestamp": "2099-01-01T00:00:00+00:00",
    }
    return sign(VerifiedOutcome(**(values | changes)), KEYS[1])


def test_prediction_verified_feedback_restart_and_independent_consumption(tmp_path):
    ctrl, pair = setup(tmp_path)
    reg = registry()
    event = ctrl.predict("req1", session(reg), {}, 0.2, reply, episode=1)
    assert event.action == 2 and event.evidence == [0.2, 0.3, None]
    assert event.model_version == "f000" and event.policy_version == "r000"
    assert not ctrl.pending("fusion", 2, minimum=1)[0]
    ctrl.receive(outcome(event))
    assert not ctrl.pending("fusion", 1, minimum=1)[0]
    assert not ctrl.pending("fusion", 2, minimum=2)[0]
    records, contexts, provenance = ctrl.pending("fusion", 2, minimum=1)
    assert len(records) == 1 and len(contexts) == 1
    _, model = pair.load()
    before = copy.deepcopy(model.state_dict())
    log = fusion_feedback_update(model, records, provenance, minimum=1)
    assert log["updated"] and log["multi_evidence"] == 1
    assert any(not torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    ctrl.commit("fusion", ["req1"])
    assert not ctrl.pending("fusion", 2, minimum=1)[0]
    assert len(ctrl.pending("router", 2, minimum=1)[0]) == 1
    restarted = ContinualController(
        tmp_path / "local", pair, FeedbackLedger(tmp_path / "local/ledger", 0, {"123", "124"}, *KEYS)
    )
    assert not restarted.pending("fusion", 2, minimum=1)[0]
    assert len(restarted.pending("router", 2, minimum=1)[0]) == 1
    with pytest.raises(ValueError, match="consumed"):
        restarted.commit("fusion", ["req1"])
    with pytest.raises(ValueError, match="Duplicate"):
        restarted.receive(outcome(event))
    with pytest.raises(ValueError, match="Duplicate"):
        restarted.finish("req1", session(reg), reply)
    with pytest.raises(ValueError, match="Unverified"):
        restarted.commit("router", ["unknown"])
    with pytest.raises(ValueError, match="Duplicate"):
        restarted.prepare("req2", session(reg), {}, 0.2, episode=3)


@pytest.mark.parametrize(
    "change",
    [
        {"received_episode": 1},
        {"authority": "untrusted"},
        {"prediction_digest": "bad"},
        {"patient_id": "124"},
    ],
)
def test_invalid_outcomes_never_train(tmp_path, change):
    ctrl, _ = setup(tmp_path)
    event = ctrl.predict("req1", session(registry()), {}, 0.2, reply, episode=1)
    with pytest.raises(ValueError):
        ctrl.receive(outcome(event, **change))
    assert not ctrl.pending("router", 100, minimum=1)[0]
    tampered = outcome(event)
    tampered["payload"]["label"] = 0
    with pytest.raises(ValueError, match="tampered"):
        ctrl.receive(tampered)


def test_stop_outage_unavailable_inflight_and_missing_feedback(tmp_path):
    ctrl, pair = setup(tmp_path, stop=True)
    reg = registry()
    calls = []

    def transport(*args):
        calls.append(args)
        return reply(*args)

    event = ctrl.predict("stop1", session(reg), {}, 0.0, transport, episode=1)
    assert event.final_risk == 0.0 and event.action == 0 and not calls
    assert not ctrl.pending("fusion", 999, minimum=1)[0]
    # No absent outcome acquires a fabricated label during restart.
    ContinualController(
        tmp_path / "local", pair, FeedbackLedger(tmp_path / "local/ledger", 0, {"123", "124"}, *KEYS)
    )
    other = tmp_path / "other"
    ctrl, _ = setup(other)
    reg = registry()
    ctrl.prepare("inflight", session(reg), {}, 0.2, episode=1)
    reg.update("B", available=False)
    with pytest.raises(ValueError, match="unavailable|Unavailable"):
        ctrl.finish("inflight", session(reg), transport)
    assert not ctrl.ledger.predictions


def test_pair_atomic_rejection_component_independence_rollback_and_contract(tmp_path):
    _ctrl, pair = setup(tmp_path)
    original = pair.current()
    r, f = pair.load()
    # A rejected router cannot activate even alongside an eligible fusion.
    pair.routers.save("r001", r.state_dict(), {"eligible": False, "router_config": r.config()})
    with torch.no_grad():
        next(f.parameters()).add_(0.01)
    pair.fusions.save("f001", f.state_dict(), {"eligible": True, "fusion_config": f.config()})
    with pytest.raises(ValueError, match="Rejected"):
        pair.activate("r001", "f001", reason="bad")
    assert pair.current() == original
    pair.activate("r000", "f001", reason="fusiononly")
    assert pair.current() == {"router": "r000", "fusion": "f001"}
    pair.rollback("r000", "f000")
    assert pair.current() == original
    assert not torch.equal(next(pair.fusions.load("f001").parameters()), next(pair.load()[1].parameters()))
    incompatible = f.config()
    incompatible["declarations"][0]["model_adapter"]["checkpoint_sha256"] = "e" * 64
    other = DynamicFusion(**incompatible)
    pair.fusions.save("f002", other.state_dict(), {"eligible": True, "fusion_config": other.config()})
    with pytest.raises(ValueError, match="contract"):
        pair.activate("r000", "f002", reason="wrongcheckpoint")
    assert pair.current() == original


def test_signed_local_context_tampering_fails_recovery(tmp_path):
    import json

    ctrl, pair = setup(tmp_path)
    ctrl.predict("req1", session(registry()), {}, 0.2, reply, episode=1)
    file = ctrl._file("req1")
    data = json.loads(file.read_text())
    data["payload"]["data"]["own"] = 0.9
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tampered"):
        ContinualController(tmp_path / "local", pair, ctrl.ledger)


def test_fusion_singleton_exact_and_nontrainable_baselines(tmp_path):
    ctrl, pair = setup(tmp_path, stop=True)
    event = ctrl.predict("req1", session(registry()), {}, 0.2, reply, episode=1)
    ctrl.receive(outcome(event))
    records, _, provenance = ctrl.pending("fusion", 2, minimum=1)
    _, fusion = pair.load()
    before = copy.deepcopy(fusion.state_dict())
    log = fusion_feedback_update(fusion, records, provenance, minimum=1)
    assert log["multi_evidence"] == 0
    assert all(torch.equal(v, fusion.state_dict()[k]) for k, v in before.items())
    assert not fusion_feedback_update(fusion, [], {}, minimum=1)["updated"]
    bad = records + records
    with pytest.raises(ValueError, match="Duplicate"):
        fusion_feedback_update(fusion, bad, provenance, minimum=1)
    for kind in ("mean", "weighted"):
        model = DynamicFusion(kind, fusion.config()["declarations"], fusion.reliability)
        assert not fusion_feedback_update(model, records, provenance, minimum=1)["updated"]


def test_controller_training_dispatch_skips_and_does_not_activate_or_consume(tmp_path):
    ctrl, pair = setup(tmp_path)
    original = pair.current()
    before = pair.load()[1].state_dict()
    candidate, log = ctrl.train_local("fusion", 2, minimum=1)
    assert not log["updated"] and all(torch.equal(v, candidate.state_dict()[k]) for k, v in before.items())
    event = ctrl.predict("req1", session(registry()), {}, 0.2, reply, episode=1)
    ctrl.receive(outcome(event))
    candidate, log = ctrl.train_local("fusion", 2, minimum=1)
    assert log["updated"] and pair.current() == original
    assert len(ctrl.pending("fusion", 2, minimum=1)[0]) == 1  # Dispatch never commits prematurely.
    _, log = ctrl.train_local("router", 2, minimum=1)
    assert not log["updated"]  # Fixed fallback remains a safe nontrainable policy.
    broken = copy.deepcopy(before)
    next(iter(broken.values())).fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        ctrl.train_local("fusion", 2, minimum=1, state=broken)
    assert len(ctrl.pending("fusion", 2, minimum=1)[0]) == 1
