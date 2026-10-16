from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.supercore.task_identity import TaskIdentity

from fed_med_agents.collaboration import ConsultationBroker
from fed_med_agents.collaboration.protocol import answer, validate_reply
from fed_med_agents.devices import select_device
from fed_med_agents.evaluation.collaboration import policy_predictions
from fed_med_agents.evaluation.verify import verify
from fed_med_agents.experiments import diabetes
from fed_med_agents.federated import client_app
from fed_med_agents.fusion import CollaborativeModel
from fed_med_agents.models import Classifier
from fed_med_agents.routing import ContextualRouter


def tensors(device="cpu", clients=3):
    torch.manual_seed(42)
    probabilities = torch.rand(30, clients, device=device).clamp(0.05, 0.95)
    labels = (probabilities[:, 1] > 0.5).float()
    origins = torch.arange(30, device=device) % clients
    return (
        probabilities,
        labels,
        origins,
        torch.ones_like(probabilities),
        torch.ones_like(probabilities, dtype=torch.bool),
    )


@pytest.fixture
def task_identity(monkeypatch):
    monkeypatch.setattr(TaskIdentity, "_run_id", 1)
    monkeypatch.setattr(TaskIdentity, "_node_id", 0)
    monkeypatch.setattr(TaskIdentity, "_task_id", 1)


def test_graph_general_dimensions_self_mask_and_missing_clients():
    router = ContextualRouter(5, 2)
    p, _, origin, domains, available = tensors(clients=5)
    available[:, 2] = False
    selected, scores = router.select(origin, p[:, 0], p[:, 0], domains, available)
    assert router.edge_logits.shape == (5, 5)
    assert torch.equal(router.adjacency().diag(), torch.zeros(5))
    assert not selected[:, 2].any()
    assert not selected.gather(1, origin[:, None]).any()
    assert (selected.sum(-1) <= 2).all()
    selected, scores = router.select(origin, p[:, 0], p[:, 0], domains, torch.zeros_like(available))
    assert not selected.any() and torch.isfinite(scores[:, 0]).all()
    assert "probabilities" not in inspect.signature(router.forward).parameters
    assert set(inspect.signature(router.forward).parameters) == {
        "origin",
        "probability",
        "confidence",
        "domains",
        "available",
    }


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
        ),
    ],
)
def test_joint_training_changes_graph_fusion_and_keeps_private_parameters(device, tmp_path):
    p, y, o, domains, available = tensors(device)
    model = CollaborativeModel().to(device)
    private = Classifier(3).to(device)
    private_before = {k: v.clone() for k, v in private.state_dict().items()}
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(5):
        optimizer.zero_grad()
        loss = model.expected_loss(p, y, o, domains, available)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    for prefix in ["router.", "fusion."]:
        assert any(
            not torch.equal(before[k], v) for k, v in model.state_dict().items() if k.startswith(prefix)
        )
    assert all(torch.equal(private_before[k], v) for k, v in private.state_dict().items())
    evidence = available.clone()
    prediction = model.fusion(p, 2 * (p - 0.5).abs(), evidence, o).detach()
    checkpoint = tmp_path / "shared.pt"
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, checkpoint)
    reloaded = CollaborativeModel().to(device)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    torch.testing.assert_close(prediction, reloaded.fusion(p, 2 * (p - 0.5).abs(), evidence, o))
    clone = CollaborativeModel().to(device)
    clone.load_state_dict(
        ArrayRecord({k: v.cpu() for k, v in model.state_dict().items()}).to_torch_state_dict()
    )
    assert not any(key.startswith("net.") for key in model.state_dict())


def test_unconsulted_predictions_cannot_change_router_or_fusion():
    model = CollaborativeModel()
    p, _, origin, domains, available = tensors()
    origin[:] = 0
    selected, scores = model.router.select(origin, p[:, 0], 2 * (p[:, 0] - 0.5).abs(), domains, available)
    evidence = selected.clone()
    evidence[:, 0] = True
    before = model.fusion(p, 2 * (p - 0.5).abs(), evidence, origin)
    changed = p.clone()
    changed[~evidence] = float("nan")
    after = model.fusion(changed, 2 * (changed - 0.5).abs(), evidence, origin)
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(
        scores, model.router(origin, changed[:, 0], 2 * (changed[:, 0] - 0.5).abs(), domains, available)
    )


def test_oof_label_and_preprocessing_leakage(monkeypatch):
    # Fixed folds isolate whether the prediction generator ever uses its held labels.
    class FixedFolds:
        def __init__(self, **kwargs):
            pass

        def split(self, ids, labels):
            yield np.arange(40, 80), np.arange(40)
            yield np.arange(40), np.arange(40, 80)

    monkeypatch.setattr(diabetes, "StratifiedKFold", FixedFolds)

    def fast_fit(x, y, seed, device):
        return diabetes_original_fit(x, y, seed, device, epochs=2)

    diabetes_original_fit = diabetes.fit_specialist
    monkeypatch.setattr(diabetes, "fit_specialist", fast_fit)
    x = np.column_stack([np.arange(80), np.tile([0, 1], 40)]).astype(np.float32)
    labels = np.tile([0, 1], 40).astype(np.float32)
    first, _, provenance = diabetes.generate_oof(
        x, labels, np.arange(80), 42, 0, torch.device("cpu"), True, 2
    )
    altered = labels.copy()
    altered[:40] = 1 - altered[:40]
    second, _, _ = diabetes.generate_oof(x, altered, np.arange(80), 42, 0, torch.device("cpu"), True, 2)
    np.testing.assert_array_equal(first[:40], second[:40])
    for fold in provenance:
        held, fit, calibration = map(set, (fold["held_ids"], fold["model_fit_ids"], fold["calibration_ids"]))
        assert not held & (fit | calibration)
        assert not fit & calibration
        assert set(fold["preprocessing"]["fit_ids"]) == fit
    changed = x.copy()
    changed[:40, 0] = 100000
    _, scale = diabetes.preprocess(x, np.arange(40, 80), True)
    _, changed_scale = diabetes.preprocess(changed, np.arange(40, 80), True)
    assert scale == changed_scale


def test_protocol_destination_patient_alignment_and_missing_records():
    request = {
        "destination": 1,
        "origin": 0,
        "request_ids": ["r1", "r2"],
        "patient_ids": ["case-0042", "case-9999"],
    }
    response = answer(request, 1, np.array([42]), np.array([0.8]))
    validate_reply(request, response)
    assert response["status"] == ["ok", "missing_record"]
    assert response["probabilities"] == [0.8, 0.5]
    with pytest.raises(ValueError, match="wrong specialist"):
        answer(request, 2, np.array([42]), np.array([0.8]))
    response["patient_ids"] = list(reversed(response["patient_ids"]))
    with pytest.raises(ValueError, match="Misassociated"):
        validate_reply(request, response)


def test_broker_uses_flower_messages_and_unavailable_clients(task_identity, tmp_path, monkeypatch):
    class Transport:
        def send_and_receive(self, messages, timeout):
            request = next(iter(messages))
            assert request.metadata.dst_node_id == 123
            assert request.metadata.message_type == "query.predict"
            response = answer(dict(request.content["request"]), 1, np.array([42]), np.array([0.7]))
            with monkeypatch.context() as context:
                context.setattr(TaskIdentity, "_node_id", 123)
                reply = Message(RecordDict({"response": ConfigRecord(response)}), reply_to=request)
            return [reply]

    broker = ConsultationBroker(Transport(), {1: 123}, tmp_path / "trace.jsonl", 42)
    assert broker.predict(0, 1, [42], "inference", "test")["probabilities"] == [0.7]
    assert broker.predict(0, 2, [42], "inference", "test")["status"] == ["unavailable"]
    trace = [json.loads(line) for line in broker.trace.read_text().splitlines()]
    assert trace[0]["patient_id"] == "case-0042" and trace[0]["destination"] == 1
    assert not any(key in trace[0] for key in ["features", "weights", "label"])


def test_client_rejects_heldout_training_cache_and_only_returns_shared_parameters(
    task_identity, tmp_path, monkeypatch
):
    path = tmp_path / "seed_42" / "client_0"
    path.mkdir(parents=True)
    monkeypatch.setenv("FED_MED_M2_DIR", str(tmp_path))
    local = {
        "train": np.arange(30),
        "train_y": np.tile([0, 1], 15).astype(np.float32),
        "oof": np.full(30, 0.6, dtype=np.float32),
    }
    np.savez(path / "local.npz", **local)
    p = np.full((30, 3), 0.6, dtype=np.float32)
    context = Context(1, 1, {"partition-id": 0}, RecordDict(), {})
    invalid = Message(
        RecordDict(
            {
                "config": ConfigRecord(
                    {"seed": 42, "patient_ids": list(range(1, 31)), "probabilities": p.ravel().tolist()}
                )
            }
        ),
        dst_node_id=1,
        message_type="query.cache",
    )
    with pytest.raises(ValueError, match="only aligned training"):
        client_app.cache(invalid, context)
    np.save(path / "training_predictions.npy", p)
    config = {
        "seed": 42,
        "server-round": 1,
        "device": "cpu",
        "budget": 1,
        "learning_rate": 0.01,
        "amp": False,
        "epochs": 1,
        "cost": 0.02,
    }
    model = CollaborativeModel()
    request = Message(
        RecordDict({"config": ConfigRecord(config), "arrays": ArrayRecord(model.state_dict())}),
        dst_node_id=1,
        message_type="train",
    )
    response = client_app.train(request, context)
    assert set(response.content) == {"arrays", "metrics"}
    assert set(response.content["arrays"].to_torch_state_dict()) == set(model.state_dict())
    assert response.content["metrics"]["router-change"] > 0
    assert response.content["metrics"]["fusion-change"] > 0


def test_explicit_cuda_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA requested"):
        select_device("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_numerical_stability():
    p, y, o, d, a = tensors("cuda")
    model = CollaborativeModel().cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model.expected_loss(p, y, o, d, a)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None
    )


def test_inference_only_requests_selected_peer():
    class StrictBroker:
        def __init__(self):
            self.nodes = {0: 100, 1: 101, 2: 102}
            self.domain_available = {0: True, 1: True, 2: True}
            self.calls = []

        def predict(self, origin, destination, patients, mode, phase):
            self.calls.append(destination)
            assert destination in {0, 1}, "Unconsulted specialist C must never be inspected"
            return {
                "probabilities": [0.6 if destination == 0 else 0.8] * len(patients),
                "status": ["ok"] * len(patients),
            }

    model = CollaborativeModel()
    with torch.no_grad():
        for parameter in model.router.parameters():
            parameter.zero_()
        model.router.edge_logits[0, 1] = 10
    broker = StrictBroker()
    result = policy_predictions(model, broker, [42, 43], 0, "learned", "test", torch.device("cpu"), 42)
    assert broker.calls == [0, 1]
    assert np.isfinite(result["probabilities"]).all()
    assert result["average_consultations"] == 1


def test_missing_domain_and_missing_patient_have_explicit_status(task_identity, monkeypatch, tmp_path):
    monkeypatch.setenv("FED_MED_M2_DIR", str(tmp_path))
    monkeypatch.setenv("FED_MED_DEVICE", "cpu")
    path = tmp_path / "seed_42" / "client_0"
    path.mkdir(parents=True)
    metadata = {"domain_available": False, "temperature": 1.0}
    (path / "specialist.json").write_text(json.dumps(metadata))
    request = {
        "seed": 42,
        "origin": 1,
        "destination": 0,
        "mode": "inference",
        "patient_ids": ["case-9999"],
        "request_ids": ["r1"],
    }
    message = Message(
        RecordDict({"request": ConfigRecord(request)}), dst_node_id=1, message_type="query.predict"
    )
    context = Context(1, 1, {"partition-id": 0}, RecordDict(), {})
    response = client_app.predict(message, context).content["response"]
    assert response["status"] == ["missing_domain"] and response["probabilities"] == [0.5]
    metadata["domain_available"] = True
    (path / "specialist.json").write_text(json.dumps(metadata))
    np.savez(path / "local.npz", x=np.zeros((1, 3), dtype=np.float32), ids=[42])
    torch.save(Classifier(3).state_dict(), path / "specialist.pt")
    response = client_app.predict(message, context).content["response"]
    assert response["status"] == ["missing_record"]
    router = ContextualRouter()
    p, _, origin, domains, available = tensors()
    domains[:, 2] = 0
    selected, _ = router.select(origin, p[:, 0], p[:, 0], domains, available)
    assert not selected[:, 2].any()


def test_saved_flower_participation_and_oof_evidence():
    from fed_med_agents.data import ARTIFACTS

    directories = [ARTIFACTS / "milestone02" / f"seed_{seed}" for seed in [42, 43, 44]]
    if any(not (directory / "metrics.json").exists() for directory in directories):
        pytest.skip("Run M2 experiment first to verify saved genuine Flower evidence")
    for directory in directories:
        result = verify(directory)
        assert result["client_updates"] == 15 and result["oof_provenance_verified"]


def test_zero_consultation_budget_has_fusion_gradient_and_no_routing_gradient():
    p, y, origin, domains, available = tensors()
    model = CollaborativeModel(max_consultations=0)
    loss = model.expected_loss(p, y, origin, domains, available)
    loss.backward()
    assert torch.isfinite(loss)
    selected, _ = model.router.select(origin, p[:, 0], p[:, 0], domains, available)
    assert not selected.any()
    assert all(
        parameter.grad is None or torch.all(parameter.grad == 0) for parameter in model.router.parameters()
    )
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in model.fusion.parameters()
    )
