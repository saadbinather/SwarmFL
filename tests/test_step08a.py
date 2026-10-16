"""External synthetic contracts and leakage/failure boundaries; no CDC artifacts."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from scripts import validate_step08a as demo


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("external") / "study"
    demo.initialize(root, preserve_history=False)
    return root


@pytest.fixture
def root(prepared, tmp_path):
    import shutil

    target = tmp_path / "study"
    shutil.copytree(prepared, target)
    return target


def outcome(event, **changes):
    values = {
        "request_id": event.request_id,
        "patient_id": event.patient_id,
        "prediction_digest": demo.digest(event),
        "label": 1,
        "authority": "simulation-training-label",
        "received_episode": 2,
        "verified_timestamp": (
            datetime.fromisoformat(event.event_timestamp) + timedelta(seconds=1)
        ).isoformat(),
    }
    return demo.sign(demo.VerifiedOutcome(**(values | changes)), demo.KEYS[1])


def prediction(root):
    patient = demo.partitions(root)["feedback"][1]
    ctrl = demo.controller(root, 0)
    own = demo.ExternalAdapter(root, 0).local("feedback", [patient])[0]
    event = ctrl.predict(
        "request",
        demo.session(demo.registry(root), demo.IDS[0], (patient,)),
        demo.contexts(own),
        own,
        lambda entry, request: demo.ExternalAdapter(
            root, demo.IDS.index(entry.specialist.specialist_id)
        ).answer(request),
        episode=1,
    )
    return ctrl, event


def test_disjoint_populations_and_fitting_only_preprocessing(root):
    cfg = demo.partitions(root)
    assert cfg["frozen_before_fitting"]
    assert cfg["split_counts"] == {"training": 360, "feedback": 120, "validation": 120}
    assert cfg["consultation"]["maximum_calls"] == 1
    assert cfg["feedback_protocol"]["simulated_horizon_days"] == 30
    assert cfg["federation"]["physical_clients"] == 3
    assert len(cfg["metrics"]) == 12 and len(cfg["specialists"]) == 3
    assert all(
        patient.startswith("syn-") and patient[4:].isalnum()
        for role in ("training", "feedback", "validation")
        for patient in cfg[role]
    )
    model = demo.joblib.load(root / "private" / f"{demo.IDS[0]}.joblib")
    training = np.load(root / "training/features.npz")["x"]
    np.testing.assert_array_equal(model[0].mean_, training[:, :2].mean(0))
    cfg["feedback"][0] = cfg["training"][0]
    demo.write(root / "protocol.json", cfg)
    with pytest.raises(AssertionError):
        demo.partitions(root)


def test_external_inference_never_opens_outcomes(root, monkeypatch):
    original = np.load

    def guarded(path, *args, **kwargs):
        assert "outcomes" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded)
    prediction(root)


def test_validation_labels_cannot_enter_feedback(root):
    patient = demo.partitions(root)["validation"][0]
    ctrl = demo.controller(root, 0)
    with pytest.raises(ValueError, match="Unauthorized patient"):
        ctrl.prepare("bad", demo.session(demo.registry(root), demo.IDS[0], (patient,)), {}, 0.5, episode=1)
    with pytest.raises(ValueError, match="training rows"):
        demo.ExternalAdapter(root, 0).local("training", [demo.partitions(root)["training"][0]])


def test_authorized_new_patient_filters_and_rejects_missing_wrong_values(root):
    reg = demo.registry(root)
    patient = demo.partitions(root)["validation"][0]
    requests = []

    def transport(entry, request):
        requests.append(request)
        return demo.ExternalAdapter(root, 1).answer(request)

    values = {f"synthetic_{j}": 0.1 for j in range(6)} | {"secret": 1234}
    demo.session(reg, demo.IDS[0], (patient,)).consult(demo.IDS[1], transport, features=values)
    assert set(requests[0]["features"]) == {"synthetic_2", "synthetic_3"}
    for fields in ({}, values | {"synthetic_2": "bad"}, values | {"synthetic_2": float("nan")}):
        with pytest.raises(ValueError):
            demo.session(reg, demo.IDS[0], (patient,)).consult(demo.IDS[1], transport, features=fields)
    malicious = requests[0] | {"features": values}
    with pytest.raises(ValueError):
        demo.ExternalAdapter(root, 1).answer(malicious)


def test_task_horizon_schema_and_checkpoint_rejected_before_private_load(root, monkeypatch):
    reg = demo.registry(root)
    removed = reg.unregister(demo.IDS[2])
    values = removed.specialist.model_dump()
    values["task"]["label_definition"] = "Synthetic event within 90 days"
    with pytest.raises(ValueError, match="Incompatible task"):
        reg.register(demo.Specialist.model_validate(values), node_id=3)
    wrong_output = removed.specialist.model_dump()
    wrong_output["output_schema"]["positive_label"] = "synthetic-no-event"
    with pytest.raises(ValueError, match="positive label|output schema"):
        reg.register(demo.Specialist.model_validate(wrong_output), node_id=3)
    ctrl, event = prediction(root)
    assert event.policy_version == "r000" and event.model_version == "f000"
    request = {}

    def capture(entry, payload):
        request.update(payload)
        return demo.ExternalAdapter(root, 1).answer(payload)

    patient = event.patient_id
    demo.session(demo.registry(root), demo.IDS[0], (patient,)).consult(demo.IDS[1], capture)
    monkeypatch.setattr(demo.joblib, "load", lambda *_: pytest.fail("Invalid request loaded a private model"))
    for changes in (
        {"checkpoint_sha256": "e" * 64},
        {"task": values["task"]},
        {"authorized_features": []},
        {"labels": [1]},
    ):
        with pytest.raises(ValueError):
            demo.ExternalAdapter(root, 1).answer(request | changes)
    assert not ctrl.pending("fusion", 100, minimum=1)[0]


def test_stop_selected_only_budget_outages_timeout_and_no_fallback(root):
    reg = demo.registry(root)
    patient = demo.partitions(root)["feedback"][0]
    calls = []

    def transport(entry, request):
        calls.append(entry.specialist.specialist_id)
        return demo.ExternalAdapter(root, demo.IDS.index(entry.specialist.specialist_id)).answer(request)

    stop = demo.RoutingPolicy("fixed", demo.IDS, 3)
    req = demo.session(reg, demo.IDS[0], (patient,))
    assert stop.route(req, {}, 0.5, version="stop").destination is None and not calls
    req.consult(demo.IDS[1], transport)
    assert calls == [demo.IDS[1]] and req.cost == 0.02
    with pytest.raises(ValueError):
        req.consult(demo.IDS[2], transport)
    for changes in ({"max_cost": 0.0}, {"permitted": frozenset()}, {"max_calls": 0}):
        with pytest.raises(ValueError):
            demo.session(reg, demo.IDS[0], (patient,), **changes).consult(demo.IDS[1], transport)
    reg.update(demo.IDS[1], available=False)
    assert [e.specialist.specialist_id for e in demo.session(reg, demo.IDS[0], (patient,)).eligible()] == [
        demo.IDS[2]
    ]
    assert calls == [demo.IDS[1]]
    reg.update(demo.IDS[1], available=True)
    timed = demo.session(reg, demo.IDS[0], (patient,))

    def timeout(*_):
        raise TimeoutError("Synthetic deadline")

    with pytest.raises(TimeoutError):
        timed.consult(demo.IDS[1], timeout)
    assert timed.calls == 1 and not timed.eligible() and calls == [demo.IDS[1]]


@pytest.mark.parametrize(
    "changes",
    [
        {"received_episode": 1},
        {"patient_id": "unknown"},
        {"prediction_digest": "wrong-version-digest"},
        {"request_id": "missing"},
        {"authority": "wrong-authority"},
    ],
)
def test_bad_feedback_never_trainable(root, changes):
    ctrl, event = prediction(root)
    with pytest.raises(ValueError):
        ctrl.receive(outcome(event, **changes))
    assert not ctrl.pending("fusion", 100, minimum=1)[0]


def test_tamper_duplicate_replay_wrong_origin_and_missing_feedback(root):
    ctrl, event = prediction(root)
    tampered = outcome(event)
    tampered["payload"]["label"] = 0
    with pytest.raises(ValueError, match="tampered"):
        ctrl.receive(tampered)
    with pytest.raises(ValueError, match="Wrong origin"):
        demo.controller(root, 1).ledger.prediction(demo.sign(event, demo.KEYS[0]))
    assert not ctrl.pending("router", 999, minimum=1)[0]
    ctrl.receive(outcome(event))
    with pytest.raises(ValueError, match="Duplicate"):
        ctrl.receive(outcome(event))
    assert not ctrl.pending("fusion", 1, minimum=1)[0]
    assert not ctrl.pending("fusion", 2, minimum=64)[0]
    changed = event.model_copy(update={"model_version": "f999"})
    assert demo.digest(changed) != demo.digest(event)


def test_controlled_gates_activation_recovery_and_exact_rollback(root):
    ctrl, event = prediction(root)
    ctrl.receive(outcome(event))
    pair = demo.bundle(root)
    r, f = pair.load()
    bad = [{"objective": 1.0, "accuracy": 0.8, "binary_cross_entropy": 1.0}]
    good = [{"objective": 0.5, "accuracy": 0.8, "binary_cross_entropy": 0.5}]
    rejected = demo.routing_update_gates(good, good, bad, [64], minimum_feedback=64)
    assert not all(rejected.values())
    pair.fusions.save(
        "fReject", demo.snapshot(f), {"eligible": all(rejected.values()), "fusion_config": f.config()}
    )
    before = pair.current()
    with pytest.raises(ValueError, match="Rejected"):
        pair.activate("r000", "fReject", reason="forced failing gate fixture")
    assert pair.current() == before
    with torch.no_grad():
        f.head[-1].bias.add_(0.1)
    assert all(demo.routing_update_gates(bad, bad, good, [64], minimum_feedback=64).values())
    pair.fusions.save("fApproved", demo.snapshot(f), {"eligible": True, "fusion_config": f.config()})
    pair.activate("r000", "fApproved", reason="controlled approval fixture, no accuracy claim")
    assert pair.current() == {"router": "r000", "fusion": "fApproved"}
    pair.rollback("r000", "f000")
    assert pair.current() == before
    assert demo.distance(demo.snapshot(pair.load()[0]), demo.snapshot(r)) == 0
    restarted = demo.controller(root, 0)
    assert len(restarted.pending("fusion", 2, minimum=1)[0]) == 1
    restarted.commit("fusion", [event.request_id])
    assert not demo.controller(root, 0).pending("fusion", 2, minimum=1)[0]
    with pytest.raises(ValueError, match="consumed"):
        restarted.commit("fusion", [event.request_id])
    with pytest.raises(ValueError):
        pair.load(router="r000")
    wrong = demo.RoutingPolicy("fixed", ("unknown", "ids", "only"), 3)
    pair.routers.save("rWrong", demo.snapshot(wrong), {"eligible": True, "router_config": wrong.config()})
    with pytest.raises(ValueError, match="identities"):
        pair.activate("rWrong", "f000", reason="inconsistent bundle")
    assert pair.current() == before
    sidecar = ctrl._file(event.request_id)
    data = demo.read(sidecar)
    data["payload"]["data"]["own"] = 0.99
    demo.write(sidecar, data)
    with pytest.raises(ValueError, match="tampered"):
        demo.controller(root, 0)


def test_nonfinite_predictions_updates_and_unseen_checkpoint_ids(root):
    reg = demo.registry(root)
    patient = demo.partitions(root)["feedback"][0]
    req = demo.session(reg, demo.IDS[0], (patient,))
    fusion = demo.bundle(root).load()[1]
    evidence = demo.FusionEvidence(
        demo.IDS[0], patient, float("nan"), reg.get(demo.IDS[0]).specialist.model_adapter.checkpoint_sha256
    )
    with pytest.raises(ValueError, match="probability"):
        fusion.fuse([evidence], req)
    state = copy.deepcopy(demo.snapshot(fusion))
    next(iter(state.values())).fill_(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        demo.controller(root, 0).train_local("fusion", 2, state=state)
    incompatible_state = copy.deepcopy(demo.snapshot(fusion))
    incompatible_state.pop(next(iter(incompatible_state)))
    with pytest.raises(RuntimeError):
        demo.controller(root, 0).train_local("fusion", 2, state=incompatible_state)
    incompatible = demo.RoutingPolicy("fixed", tuple("ABC"))
    with pytest.raises(ValueError, match="origin"):
        incompatible.route(req, {}, 0.5, version="old")
    stale = demo.FusionEvidence(demo.IDS[0], patient, 0.2, "0" * 64)
    with pytest.raises(ValueError, match="checkpoint"):
        fusion.fuse([stale], req)


def test_local_trace_contains_no_feature_or_label_values(root):
    request = {
        "origin": demo.IDS[0],
        "destination": demo.IDS[1],
        "patient_ids": ["synthetic-case"],
        "mode": "new_patient",
        "features": {"synthetic_2": 123.456},
    }
    sanitized = demo.audit_record(request)
    assert "123.456" not in json.dumps(sanitized)
    assert all(patient.startswith("syn-") for patient in demo.partitions(root)["feedback"])


def test_controlled_external_compatibility_and_lifecycle(root):
    rejected = demo.historical_compatibility(root)
    assert set(rejected.values()) <= {"ValueError", "RuntimeError", "RegistryError"}
    fixture = demo.controlled_lifecycle(root)
    assert fixture["rejection"] == "ValueError"
    assert fixture["recovered"] == {"router": "r000", "fusion": "fApproved"}
    assert fixture["rollback"] == {"router": "r000", "fusion": "f000"}
    assert fixture["router_exact"] and fixture["fusion_exact"]


@pytest.mark.skipif(
    os.environ.get("SWARMFL_REAL_STEP08A") != "1", reason="Opt-in new synthetic three-client Flower evidence"
)
def test_real_three_client_flower_and_numerical_replay():
    result = demo.verify(demo.ROOT)
    assert result["local_optimizer_replays"] == 6 and result["FedAvg_replays"] == 2
    assert result["deterministic_split_and_data_replays"] == 600
    assert result["feedback_consumed_once_per_component"] == 648
    assert result["changed"] == []
