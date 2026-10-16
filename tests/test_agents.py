from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from fed_med_agents.agents.coordinator import CoordinationSession, coordinate
from fed_med_agents.agents.provider import OllamaProvider, ProviderError
from fed_med_agents.agents.runtime import ToolAgent
from fed_med_agents.agents.schemas import (
    Capability,
    ClientAction,
    PatientRequest,
    SpecialistResponse,
    ToolCall,
)
from fed_med_agents.agents.specialist import PrivateClassifier, SpecialistAgent
from fed_med_agents.data import ARTIFACTS, ROOT
from fed_med_agents.fusion import CollaborativeModel
from fed_med_agents.models import Classifier


class MockProvider:
    """Deterministic test provider only; never used in the real benchmark."""

    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.messages = []
        self.schemas = []

    def generate(self, messages, schema):
        self.schemas.append(schema)
        self.messages.append([message.copy() for message in messages])
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output, {"model": "TEST-MOCK", "eval_count": 1}


def capabilities():
    return [
        Capability(specialist=name, index=i, domain=f"domain-{name}", required_features=[f"feature_{name}"])
        for i, name in enumerate("ABC")
    ]


class MockTransport:
    def __init__(self):
        self.nodes = {name: i for i, name in enumerate("ABC")}
        self.received = []
        self.returned = []

    def predict(self, destination, request, intake=False):
        self.received.append((destination, request, intake))
        return SpecialistResponse(
            request_id=request.request_id,
            patient_id=request.patient_id,
            specialist=destination,
            status="ok",
            probability=0.6 if destination == "A" else 0.8,
            confidence=0.2 if destination == "A" else 0.6,
        )

    def return_to_origin(self, result):
        self.returned.append(result)


def request(mode="new_patient", budget=1):
    return PatientRequest(
        request_id="unit-request",
        patient_id="simulated-unit",
        origin="A",
        mode=mode,
        alignment_id=42 if mode == "aligned" else None,
        features={f"feature_{name}": 1.0 for name in "ABC"} if mode == "new_patient" else {},
        authorized_features={name: [f"feature_{name}"] for name in "ABC"} if mode == "new_patient" else {},
        budget=budget,
    )


def action(name, arguments=None):
    return ToolCall.model_validate_json(json.dumps({"name": name, "arguments": arguments or {}}))


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "read_training_file", "arguments": {}},
        {"name": "finalize_prediction", "arguments": {"probability": 0.99}},
        {"name": "request_specialist_prediction", "arguments": {"specialist": "B", "path": "secret"}},
    ],
)
def test_nonexistent_tools_and_invented_scores_are_rejected(payload):
    with pytest.raises(ValidationError):
        ToolCall.model_validate_json(json.dumps(payload))


def test_mode_schema_rejects_aligned_feature_transfer_and_nonfinite_values():
    with pytest.raises(ValidationError):
        PatientRequest(
            request_id="r", patient_id="p", origin="A", mode="aligned", alignment_id=42, features={"age": 42}
        )
    with pytest.raises(ValidationError):
        PatientRequest(
            request_id="r", patient_id="p", origin="A", mode="new_patient", features={"age": float("nan")}
        )


def test_mock_provider_executes_valid_typed_tools_and_returns_to_origin():
    transport = MockTransport()
    model = CollaborativeModel()
    session = CoordinationSession(request(), capabilities(), transport, model, torch.device("cpu"), "slm")
    provider = MockProvider(
        [
            json.dumps({"name": "request_specialist_prediction", "arguments": {"specialist": "B"}}),
            json.dumps(
                {"name": "get_consultation_status", "arguments": {"consultation_id": "unit-request-B"}}
            ),
            json.dumps({"name": "finalize_prediction", "arguments": {}}),
        ]
    )
    coordinate(session, provider)
    assert session.finished.status == "ok" and session.finished.consulted == ["B"]
    destination, subset, intake = transport.received[1]
    assert destination == "B" and not intake
    assert subset.origin == "A" and subset.destination == "B"
    assert subset.features == {"feature_B": 1.0}
    assert subset.authorized_features == {"B": ["feature_B"]}
    assert transport.returned[0].request_id == "unit-request"
    assert not any("TEST-MOCK" in str(item) for item in session.tool_trace)
    p = torch.tensor([[0.6, 0.8, 0.5]])
    mask = torch.tensor([[True, True, False]])
    with torch.no_grad():
        expected = torch.sigmoid(model.fusion(p, 2 * (p - 0.5).abs(), mask, torch.tensor([0]))).item()
    assert session.finished.probability == pytest.approx(expected)


@pytest.mark.parametrize("specialist", ["A", "nonexistent"])
def test_self_or_nonexistent_consultation_is_blocked(specialist):
    session = CoordinationSession(
        request(), capabilities(), MockTransport(), CollaborativeModel(), torch.device("cpu"), "slm"
    )
    session.intake()
    result = session.execute(action("request_specialist_prediction", {"specialist": specialist}))
    assert "error" in result and session.tool_failures == 1 and not session.routing_valid


def test_budget_consent_missing_fields_and_duplicate_requests_are_enforced():
    patient = request()
    patient.authorized_features["C"] = []
    session = CoordinationSession(
        patient, capabilities(), MockTransport(), CollaborativeModel(), torch.device("cpu"), "slm"
    )
    session.intake()
    assert not session.eligibility(session.capabilities["C"])
    with pytest.raises(ValueError, match="lacking authorization"):
        session.consult("C")
    session.consult("B")
    with pytest.raises(ValueError, match="Duplicate|exhausted"):
        session.consult("B")
    patient = request()
    patient.features.pop("feature_B")
    session = CoordinationSession(
        patient, capabilities(), MockTransport(), CollaborativeModel(), torch.device("cpu"), "slm"
    )
    assert not session.eligibility(session.capabilities["B"])


def test_malformed_provider_output_retry_and_role_isolation():
    provider = MockProvider(
        [
            "bad JSON",
            '{"name":"run_local_inference","arguments":{}}',
            '{"name":"return_insufficient_features","arguments":{}}',
        ]
    )
    first = ToolAgent(provider, "A", "role A", ClientAction)
    assert first.choose({"request": "A"}).name == "run_local_inference"
    assert any("error" in event for event in first.trace)
    second = ToolAgent(provider, "B", "role B", ClientAction)
    second.choose({"request": "B"})
    assert second.messages[0]["content"] == "role B"
    assert all('"request": "A"' not in message["content"] for message in second.messages)


def test_provider_timeout_does_not_fabricate_fusion():
    session = CoordinationSession(
        request(), capabilities(), MockTransport(), CollaborativeModel(), torch.device("cpu"), "slm"
    )
    provider = MockProvider([ProviderError("timeout"), ProviderError("timeout")])
    coordinate(session, provider)
    assert session.finished.status == "agent_error" and session.finished.probability is None
    assert not session.finished.routing_valid


@pytest.fixture
def private_adapter(tmp_path):
    path = tmp_path / "client_0"
    path.mkdir()
    metadata = {
        "features": ["age", "gender", "obesity"],
        "temperature": 1.0,
        "preprocessing": {"mean": 40.0, "scale": 10.0},
    }
    (path / "specialist.json").write_text(json.dumps(metadata))
    torch.save(Classifier(3).state_dict(), path / "specialist.pt")
    config = {
        "clients": [{"specialist": "A", "domain": "risk"}],
        "numeric_fields": ["age"],
        "encodings": {"gender": {"Male": 1, "Female": 0}, "symptom": {"Yes": 1, "No": 0}},
    }
    return PrivateClassifier(path, 0, config, "cpu")


def test_new_patient_scaling_and_no_alignment_lookup(private_adapter, monkeypatch):
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("New-patient mode must not load stored rows")
    )
    features = {"age": 50, "gender": "Male", "obesity": "No"}
    np.testing.assert_array_equal(private_adapter.encode(features), [[1.0, 1.0, 0.0]])
    patient = PatientRequest(
        request_id="new",
        patient_id="external-sim",
        origin="A",
        mode="new_patient",
        features=features,
        authorized_features={"A": list(features)},
    )
    score = private_adapter.probability(patient)
    assert 0 <= score <= 1
    assert all(not parameter.requires_grad for parameter in private_adapter._model().parameters())


def test_incomplete_patient_never_invokes_classifier(private_adapter, monkeypatch):
    monkeypatch.setattr(
        private_adapter, "probability", lambda *args: pytest.fail("Missing fields must not be imputed")
    )
    patient = PatientRequest(
        request_id="new",
        patient_id="external-sim",
        origin="A",
        mode="new_patient",
        features={"age": 50},
        authorized_features={"A": ["age"]},
    )
    service = SpecialistAgent(
        private_adapter, MockProvider(['{"name":"return_insufficient_features","arguments":{}}'])
    )
    result, _trace = service.predict(patient)
    assert result.status == "insufficient_features" and result.probability is None
    assert set(result.missing_features) == {"gender", "obesity"}


def test_specialist_rejects_unauthorized_or_extraneous_fields(private_adapter):
    patient = PatientRequest(
        request_id="new",
        patient_id="external-sim",
        origin="A",
        mode="new_patient",
        features={"age": 50, "gender": 1, "obesity": 0, "other": 1},
        authorized_features={"A": ["age", "gender", "obesity", "other"]},
    )
    with pytest.raises(ValueError, match="unauthorized or extraneous"):
        SpecialistAgent(private_adapter, MockProvider([])).predict(patient)


def test_unlabeled_supplied_requests_have_no_invented_evaluation_metrics():
    from fed_med_agents.agents.server_app import evaluate

    results = [
        {
            "patient_id": "new-synthetic",
            "probability": 0.8,
            "routing_valid": True,
            "consulted": ["B"],
            "tool_failures": 0,
            "latency_seconds": 1.0,
        }
    ]
    scores = evaluate(results, {})
    assert all(scores[key] is None for key in ("accuracy", "auroc", "f1", "binary_cross_entropy"))
    assert scores["failed_predictions_counted_wrong_accuracy"] is None
    assert scores["prediction_coverage"] == 1.0


def test_loopback_provider_rejects_external_model_endpoint():
    with pytest.raises(ValueError, match="loopback"):
        OllamaProvider(url="https://external.example")


def test_capability_scoped_schema_excludes_self_unavailable_and_used_peers():
    transport = MockTransport()
    patient = request()
    patient.authorized_features["C"] = []
    session = CoordinationSession(
        patient, capabilities(), transport, CollaborativeModel(), torch.device("cpu"), "slm"
    )
    session.intake()
    provider = MockProvider(
        [
            '{"name":"request_specialist_prediction","arguments":{"specialist":"B"}}',
            '{"name":"finalize_prediction","arguments":{}}',
        ]
    )
    agent = ToolAgent(provider, "server", "test schema", ToolCall)
    first = agent.choose(session.observation())
    assert provider.schemas[0]["$defs"]["PredictArguments"]["properties"]["specialist"]["enum"] == ["B"]
    session.execute(first)
    agent.choose(session.observation())
    assert {"$ref": "#/$defs/PredictCall"} not in provider.schemas[1]["oneOf"]


def test_saved_real_predictions_recompute_from_shared_fusion():
    from fed_med_agents.agents.verify import verify

    path = ARTIFACTS / "milestone03-budget-matched"
    if not (path / "metrics.json").exists():
        pytest.skip("Run the real Qwen benchmark first")
    evidence = verify(path, ARTIFACTS / "milestone02/seed_42/shared_best.pt")
    assert evidence["fusion_predictions_recomputed"] > 0
    assert (
        evidence["fusion_predictions_recomputed"] + evidence["no_prediction_responses"]
        == evidence["requests"]
    )
    assert evidence["authorized_subsets_verified"] and evidence["return_hops_verified"]


@pytest.mark.skipif(
    os.environ.get("SWARMFL_REAL_QWEN") != "1", reason="Opt in to genuine local Qwen/Flower integration"
)
def test_real_qwen_flower_origin_b_consultation():
    output = Path(os.environ.get("SWARMFL_M3_INTEGRATION_DIR", str(ARTIFACTS / "milestone03-integration")))
    command = [
        sys.executable,
        "-m",
        "fed_med_agents.experiments.milestone03",
        "--patients",
        "2",
        "--modes",
        "new_patient",
        "--demo",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    assert process.returncode == 0, process.stdout[-4000:] + process.stderr[-2000:]
    metrics = json.loads((output / "metrics.json").read_text())
    traces = [json.loads(line) for line in (output / "traces.jsonl").read_text().splitlines()]
    slm = [trace for trace in traces if trace["policy"] == "slm"]
    assert slm and all(trace["final"]["status"] == "ok" for trace in slm)
    assert all(trace["final"]["consulted"] == ["B"] for trace in slm)
    for trace in slm:
        assert any(
            event.get("action", {}).get("name") == "request_specialist_prediction"
            for event in trace["coordinator_trace"]
        )
        assert any(event.get("action") == "return_to_origin" for event in trace["communication_trace"])
        assert any(event.get("role") == "client_B" for event in trace["communication_trace"])
        assert trace["final"]["probability"] is not None
    assert metrics["fusion_unchanged"]
    assert metrics["tabular_device"] == "cpu"
    assert any(model["size_vram"] > 0 for model in metrics["provider"]["running_models"]["models"])
