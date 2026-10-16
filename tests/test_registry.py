"""Registry lifecycle, compatibility, privacy boundaries and genuine A/B/C integration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fed_med_agents.collaboration.registry import (
    ConsultationSession,
    FeatureSchema,
    ModelAdapter,
    OutputSchema,
    RegistryError,
    Specialist,
    SpecialistRegistry,
    TaskSchema,
)
from fed_med_agents.experiments.cdc_data import THREE_DOMAINS

TASK = TaskSchema(
    name="cdc891-binary-risk",
    labels=("neither", "prediabetes-or-diabetes"),
    label_definition="cross-sectional prediabetes-or-diabetes",
)
OUTPUT = OutputSchema(positive_label=TASK.labels[1])


def specialist(name: str) -> Specialist:
    """Only the user-fixed three CDC identities/domains appear in tests."""
    slot = ("A", "B", "C").index(name)
    fields = tuple(list(THREE_DOMAINS.values())[slot])
    return Specialist(
        specialist_id=name,
        task=TASK,
        output_schema=OUTPUT,
        model_adapter=ModelAdapter(
            adapter_id=f"histogram-{name}",
            model_family="cdc-three",
            checkpoint_sha256=name.lower() * 64,
            feature_names=fields,
            input_dim=len(fields),
            alignment_namespace="cdc-development",
        ),
        feature_schema=tuple(FeatureSchema(name=field, minimum=0, maximum=100) for field in fields),
        capabilities=frozenset({"aligned_prediction", "new_patient_prediction"}),
        consultation_cost=0.02,
    )


def registry() -> SpecialistRegistry:
    result = SpecialistRegistry(TASK, OUTPUT, model_family="cdc-three")
    for node, name in enumerate(("A", "B", "C"), 1):
        result.register(specialist(name), node_id=node)
    return result


def session(reg: SpecialistRegistry, **changes) -> ConsultationSession:
    values = {
        "permitted": frozenset({"B", "C"}),
        "feature_permissions": {
            name: frozenset(specialist(name).model_adapter.feature_names) for name in ("B", "C")
        },
        "available_features": frozenset(field for domain in THREE_DOMAINS.values() for field in domain),
        "alignment_namespace": "cdc-development",
    }
    return ConsultationSession(reg, "A", ("123",), **(values | changes))


def reply(entry, request):
    return {
        key: request[key]
        for key in ("origin", "destination", "patient_ids", "adapter_id", "checkpoint_sha256")
    } | {"probabilities": [0.3] * len(request["patient_ids"])}


def test_dynamic_lifecycle_discovery_and_live_session_lookup():
    reg = SpecialistRegistry(TASK, OUTPUT)
    reg.register(specialist("A"), node_id=1)
    reg.register(specialist("B"), node_id=2)
    assert [r.specialist.specialist_id for r in reg.discover()] == ["A", "B"]
    reg.register(specialist("C"), node_id=3)
    pending = session(reg)
    old = reg.unregister("C")
    with pytest.raises(RegistryError, match="Unknown"):
        pending.consult("C", reply)
    reg.register(old.specialist, node_id=old.node_id)
    assert reg.get("C").generation > old.generation
    assert pending.consult("C", reply)["destination"] == "C"
    assert not pending.eligible()
    assert len(reg.discover(capability="aligned_prediction")) == 3
    assert not reg.discover(capability="unsupported")


def test_duplicate_ids_nodes_and_checkpoint_families_rejected():
    reg = registry()
    with pytest.raises(RegistryError, match="Duplicate specialist"):
        reg.register(specialist("B"), node_id=4)
    removed = reg.unregister("C")
    with pytest.raises(RegistryError, match="Duplicate physical"):
        reg.register(removed.specialist, node_id=2)
    bad = removed.specialist.model_dump()
    bad["model_adapter"]["model_family"] = "uci529-incompatible"
    with pytest.raises(RegistryError, match="checkpoint family"):
        reg.register(Specialist.model_validate(bad), node_id=3)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "different-task"),
        ("label_definition", "future incident disease"),
        ("labels", ("negative", "positive")),
    ],
)
def test_task_and_label_semantics_are_exact(field, value):
    reg = registry()
    reg.unregister("C")
    values = specialist("C").model_dump()
    values["task"][field] = value
    values["output_schema"]["positive_label"] = values["task"]["labels"][1]
    with pytest.raises(RegistryError, match="task, labels or output"):
        reg.register(Specialist.model_validate(values), node_id=3)


def test_output_feature_order_dimension_and_shared_schema_validation():
    values = specialist("C").model_dump()
    values["model_adapter"]["feature_names"] = tuple(reversed(values["model_adapter"]["feature_names"]))
    with pytest.raises(ValueError, match="feature order/dimension"):
        Specialist.model_validate(values)
    values = specialist("C").model_dump()
    values["model_adapter"]["input_dim"] += 1
    with pytest.raises(ValueError, match="feature order/dimension"):
        Specialist.model_validate(values)
    reg = registry()
    reg.unregister("C")
    values = specialist("C").model_dump()
    values["output_schema"]["dtype"] = "float64"
    with pytest.raises(RegistryError, match="output schema"):
        reg.register(Specialist.model_validate(values), node_id=3)
    values = specialist("C").model_dump()
    values["model_adapter"].update(feature_names=("BMI",), input_dim=1)
    values["feature_schema"] = ({"name": "BMI", "dtype": "integer"},)
    with pytest.raises(RegistryError, match="shared feature schema"):
        reg.register(Specialist.model_validate(values), node_id=3)


def test_live_availability_cost_and_call_budgets():
    reg = registry()
    pending = session(reg, max_cost=0.025)
    reg.update("B", available=False)
    with pytest.raises(RegistryError, match="unavailable"):
        pending.consult("B", reply)
    assert pending.calls == 0 and [r.specialist.specialist_id for r in pending.eligible()] == ["C"]
    reg.update("B", available=True, consultation_cost=0.03)
    with pytest.raises(RegistryError, match="cost budget"):
        pending.consult("B", reply)
    reg.update("B", consultation_cost=0.01)
    pending.consult("B", reply)
    assert pending.calls == 1 and pending.cost == 0.01 and not pending.eligible()
    with pytest.raises(RegistryError, match="budget exhausted"):
        pending.consult("C", reply)
    for value in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            reg.update("B", consultation_cost=value)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"permitted": frozenset({"C"})}, "Unauthorized specialist"),
        ({"feature_permissions": {"B": frozenset({"Smoker"})}}, "Unauthorized required"),
        ({"available_features": frozenset({"Smoker"})}, "Missing required"),
        ({"max_calls": 0}, "budget exhausted"),
        ({"alignment_namespace": "uci529-test"}, "alignment namespace"),
    ],
)
def test_rejections_do_not_call_transport_or_consume_budget(change, reason):
    pending = session(registry(), **change)

    def forbidden(*_):
        pytest.fail("Rejected request reached transport")

    with pytest.raises(RegistryError, match=reason):
        pending.consult("B", forbidden)
    assert pending.calls == 0 and pending.cost == 0


def test_self_unbound_origin_availability_and_request_task_rejection():
    reg = registry()
    with pytest.raises(RegistryError, match="Self-consultation"):
        session(reg).consult("A", reply)
    reg.unregister("B")
    reg.register(specialist("B"))
    with pytest.raises(RegistryError, match="physical transport"):
        session(reg).consult("B", reply)
    reg.update("A", available=False)
    with pytest.raises(RegistryError, match="unavailable"):
        session(reg).consult("C", reply)
    reg.update("A", available=True)
    requirements = session(reg).requirements()
    requirements["task"] = TASK.model_copy(update={"label_definition": "other outcome"})
    with pytest.raises(RegistryError, match="request task/labels"):
        reg.validate("A", "C", **requirements)


def test_aligned_payload_has_no_values_and_authorized_new_patient_subset_only():
    pending = session(registry())
    requests = []

    def capture(entry, request):
        requests.append(request)
        return reply(entry, request)

    pending.consult("B", capture)
    assert "features" not in requests[0] and "labels" not in requests[0] and "weights" not in requests[0]
    data = {field: 1 for domain in THREE_DOMAINS.values() for field in domain}
    data["unrelated-secret"] = "do not forward"
    session(registry()).consult("B", capture, features=data)
    assert set(requests[-1]["features"]) == set(THREE_DOMAINS["lifestyle"])
    assert requests[-1]["mode"] == "new_patient"
    for bad in (float("nan"), "Yes", True, -1):
        invalid = data | {"Smoker": bad}
        with pytest.raises(RegistryError):
            session(registry()).consult("B", capture, features=invalid)
    with pytest.raises(RegistryError, match="Missing required feature"):
        session(registry()).consult("B", capture, features={})


@pytest.mark.parametrize(
    "change",
    [
        {"destination": "C"},
        {"origin": "B"},
        {"patient_ids": ["999"]},
        {"adapter_id": "wrong"},
        {"checkpoint_sha256": "0" * 64},
        {"probabilities": []},
        {"probabilities": [float("nan")]},
        {"probabilities": [1.1]},
        {"probabilities": [True]},
        {"features": {"Smoker": 1}},
    ],
)
def test_reply_association_and_output_schema(change):
    pending = session(registry())
    with pytest.raises(RegistryError):
        pending.consult("B", lambda entry, request: reply(entry, request) | change)
    assert pending.calls == 1  # A dispatched attempt consumes its reservation.


def test_transport_failure_and_mutation_cannot_reuse_budget_or_change_association():
    pending = session(registry())

    def broken(*_):
        raise TimeoutError("test transport deadline")

    with pytest.raises(TimeoutError):
        pending.consult("B", broken)
    assert not pending.eligible()

    def mutate(entry, request):
        request["patient_ids"][0] = "999"
        return reply(entry, request)

    with pytest.raises(RegistryError, match="patient_ids"):
        session(registry()).consult("B", mutate)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"destination": "C"}, "Wrong recipient"),
        ({"checkpoint_sha256": "0" * 64}, "identity mismatch"),
        ({"authorized_features": []}, "not authorized"),
        ({"features": {"Smoker": 1}}, "raw feature transfer"),
        ({"origin": "B"}, "Self-consultation"),
    ],
)
def test_private_handler_rejects_misrouting_and_payloads_before_loading_rows(monkeypatch, change, reason):
    from flwr.app import ConfigRecord, Context, Message, RecordDict
    from flwr.supercore.task_identity import TaskIdentity

    from fed_med_agents.experiments import cdc_flower

    for key, value in (("_run_id", 1), ("_node_id", 0), ("_task_id", 1)):
        monkeypatch.setattr(TaskIdentity, key, value)
    monkeypatch.setattr(cdc_flower, "registry_declaration", lambda _: specialist("B"))
    captured = []

    def capture(entry, request):
        captured.append(request)
        return reply(entry, request)

    session(registry()).consult("B", capture)
    request = captured[0] | change
    message = Message(
        RecordDict({"config": ConfigRecord({"request": json.dumps(request)})}),
        dst_node_id=2,
        message_type="query.registry_predict",
    )
    ctx = Context(run_id=1, node_id=2, node_config={"partition-id": 1}, state=RecordDict(), run_config={})
    # No dataset/model fixture is supplied: any attempt to load rows would fail.
    with pytest.raises(RegistryError, match=reason):
        cdc_flower.registry_predict(message, ctx)


@pytest.mark.skipif(
    os.environ.get("SWARMFL_REAL_REGISTRY") != "1", reason="Opt-in real A/B/C Flower lifecycle"
)
def test_real_flower_registry_lifecycle_and_fusion(tmp_path):
    from fed_med_agents.experiments.cdc_flower import STAGE
    from fed_med_agents.experiments.cdc_routing import CDCFusion
    from fed_med_agents.experiments.m5_step03 import digest

    source = Path("artifacts/m5-step035-three-agent-v1").resolve()
    destination = Path(os.environ.get("SWARMFL_REGISTRY_EVIDENCE_DIR", str(tmp_path / "registry"))).resolve()
    frozen_hash = digest(source / "frozen_test.npz")
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "fed_med_agents.experiments.cdc_flower",
            "--registry-demo",
            "--output",
            str(source),
            "--evidence-output",
            str(destination),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((destination / "flower/results.json").read_text())
    assert evidence["physical_clients"] == 3 and evidence["training_updates"] == 0
    lifecycle = evidence["logical_lifecycle"]
    assert lifecycle[0]["registered"] == ["A", "B"] and lifecycle[1]["registered"] == ["A", "B", "C"]
    assert lifecycle[2]["available"] == ["A", "C"]
    assert lifecycle[-2]["registered"] == ["A", "B"] and lifecycle[-1]["registered"] == ["A", "B", "C"]
    assert evidence["rejections"] and all(not row["transport_called"] for row in evidence["rejections"])
    assert all("features" not in row["request"] for row in evidence["traffic"])
    fusion = CDCFusion("original", 3)
    fusion.load_state_dict(torch.load(source / STAGE / "fusion_original.pt", weights_only=True))
    for row in evidence["fusion"]:
        with torch.no_grad():
            risk = float(
                fusion(torch.tensor(row["evidence"]), torch.tensor(row["mask"]), torch.tensor([0])).sigmoid()[
                    0
                ]
            )
        np.testing.assert_allclose(risk, row["risk"], atol=2e-6)
        np.testing.assert_allclose(risk, row["expected_saved_risk"], atol=2e-6)
    assert frozen_hash == digest(source / "frozen_test.npz") and evidence["frozen_test_evaluations"] == 0
