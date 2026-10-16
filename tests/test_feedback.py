from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
from flwr.app import ArrayRecord
from pydantic import ValidationError

from fed_med_agents.agents.schemas import ToolCall
from fed_med_agents.data import ARTIFACTS
from fed_med_agents.feedback.events import FeedbackLedger, PredictionEvent, VerifiedOutcome, digest, sign
from fed_med_agents.feedback.learning import feedback_loss
from fed_med_agents.feedback.server_app import action_distribution
from fed_med_agents.feedback.versions import CapabilityRegistry, PolicyRegistry
from fed_med_agents.fusion import CollaborativeModel

KEY = "11" * 32
OUTCOME_KEY = "22" * 32


def event(**changes):
    return PredictionEvent.model_validate(
        {
            "request_id": "request-1",
            "patient_id": "case-0001",
            "origin": 0,
            "episode": 1,
            "available_domains": [True, True, True],
            "action": 2,
            "consulted": [1],
            "propensity": 0.3,
            "action_score": 0.1,
            "evidence": [0.6, 0.8, None],
            "final_risk": 0.7,
            "origin_only_risk": 0.6,
            "model_version": "v000",
            "policy_version": "ips-v1",
            "consultation_cost": 0.02,
            "event_timestamp": "2026-10-14T10:00:00+00:00",
        }
        | changes
    )


def outcome(prediction, **changes):
    return VerifiedOutcome.model_validate(
        {
            "request_id": prediction.request_id,
            "patient_id": prediction.patient_id,
            "prediction_digest": digest(prediction),
            "label": 0,
            "authority": "simulation-training-label",
            "received_episode": 4,
            "verified_timestamp": "2026-10-14T10:01:00+00:00",
        }
        | changes
    )


def store(path):
    return FeedbackLedger(path, 0, {"case-0001"}, KEY, OUTCOME_KEY)


def test_predictions_are_not_outcomes_and_delayed_verified_replay(tmp_path):
    ledger = store(tmp_path)
    prediction = event()
    ledger.prediction(sign(prediction, KEY))
    assert ledger.replay(100) == []
    verified = outcome(prediction)
    assert verified.label != int(prediction.final_risk >= 0.5)
    ledger.outcome(sign(verified, OUTCOME_KEY))
    assert ledger.replay(3) == []
    assert ledger.replay(4) == [(prediction, verified)]
    assert store(tmp_path).replay(4) == [(prediction, verified)]
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.outcome(sign(verified, OUTCOME_KEY))
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.prediction(sign(prediction, KEY))


@pytest.mark.parametrize("changes", [{"patient_id": "held-out"}, {"origin": 1}])
def test_unauthorized_prediction_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        store(tmp_path).prediction(sign(event(**changes), KEY))


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "unknown"},
        {"prediction_digest": "wrong"},
        {"patient_id": "wrong"},
        {"authority": "slm-acknowledgement"},
        {"received_episode": 1},
        {"verified_timestamp": "2026-10-14T09:00:00+00:00"},
    ],
)
def test_outcome_association_and_authority(tmp_path, changes):
    ledger = store(tmp_path)
    prediction = event()
    ledger.prediction(sign(prediction, KEY))
    with pytest.raises(ValueError):
        ledger.outcome(sign(outcome(prediction, **changes), OUTCOME_KEY))


def test_signature_and_strict_schema(tmp_path):
    envelope = sign(event(), KEY)
    envelope["payload"]["final_risk"] = 0.9
    with pytest.raises(ValueError, match="tampered"):
        store(tmp_path).prediction(envelope)
    for changes in (
        {"label": 1},
        {"evidence": [0.6, 0.8, 0.9]},
        {"propensity": 0.0},
        {"final_risk": float("nan")},
        {"consulted": [0]},
        {"available_domains": [True, False, True]},
    ):
        with pytest.raises(ValidationError):
            event(**changes)


def test_partial_feedback_gradients_compatibility_and_zero_action():
    model = CollaborativeModel()
    prediction = event()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss, audit = feedback_loss(model, [(prediction, outcome(prediction))], torch.device("cpu"))
    assert torch.isfinite(loss) and audit["max_importance_ratio"] <= 2
    loss.backward()
    assert model.router.edge_logits.grad.abs().sum() > 0
    assert model.fusion.network[0].weight.grad.abs().sum() > 0
    optimizer.step()
    for prefix in ("router.", "fusion."):
        assert any(
            not torch.equal(v, initial[k]) for k, v in model.state_dict().items() if k.startswith(prefix)
        )
    compatible = CollaborativeModel()
    compatible.load_state_dict(ArrayRecord(model.state_dict()).to_torch_state_dict())
    zero = event(action=0, consulted=[], evidence=[0.6, None, None], consultation_cost=0.0)
    zero_loss, _ = feedback_loss(model, [(zero, outcome(zero))], torch.device("cpu"))
    assert torch.isfinite(zero_loss)
    with pytest.raises(ValueError, match="Action/evidence"):
        feedback_loss(model, [(event(action=1), outcome(event(action=1)))], torch.device("cpu"))


def test_routing_masks_and_exploration_support():
    model = CollaborativeModel()
    distribution, scores = action_distribution(model, 0, 0.6, [True, True, False], torch.device("cpu"), 0.2)
    assert distribution[0] > 0 and distribution[2] > 0
    assert distribution[1] == 0 and distribution[3] == 0
    assert torch.isneginf(scores[1]) and torch.isneginf(scores[3])


def test_versions_approval_integrity_rollback_capability_registry(tmp_path):
    registry = PolicyRegistry(tmp_path)
    model = CollaborativeModel()
    registry.save("v000", model.state_dict(), {"eligible": True})
    registry.approve("v000")
    registry.save("v001", model.state_dict(), {"eligible": True})
    registry.approve("v001")
    registry.rollback("v000")
    assert registry.current() == "v000"
    with pytest.raises(ValueError, match="Immutable"):
        registry.save("v000", model.state_dict(), {"eligible": True})
    registry.save("v002", model.state_dict(), {"eligible": False})
    with pytest.raises(ValueError, match="Rejected"):
        registry.approve("v002")
    (tmp_path / "v001.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        registry.load("v001")
    capabilities = CapabilityRegistry()
    capabilities.register(0, 100, True)
    with pytest.raises(ValueError, match="Duplicate"):
        capabilities.register(0, 101, True)


def test_slm_tool_reads_latest_approved_version(tmp_path):
    from test_agents import MockProvider, MockTransport, capabilities, request

    from fed_med_agents.agents.coordinator import CoordinationSession
    from fed_med_agents.agents.runtime import ToolAgent

    # Real typed dispatch with a deterministic provider only for this unit test.
    registry = PolicyRegistry(tmp_path)
    model = CollaborativeModel()
    registry.save("v000", model.state_dict(), {"eligible": True})
    registry.approve("v000")
    registry.save("v001", model.state_dict(), {"eligible": True})
    registry.approve("v001")
    session = CoordinationSession(
        request(), capabilities(), MockTransport(), model, torch.device("cpu"), "slm"
    )
    session.policy_registry = registry
    session.intake()
    provider = MockProvider(['{"name":"get_routing_recommendation","arguments":{}}'])
    call = ToolAgent(provider, "server_coordinator", "test prompt", ToolCall).choose(session.observation())
    result = session.execute(call)
    assert result["output"]["version"] == "v001"


@pytest.mark.skipif(os.environ.get("SWARMFL_REAL_FEEDBACK") != "1", reason="opt-in fresh genuine Flower test")
def test_real_feedback_update_and_later_consultation(tmp_path):
    from pathlib import Path

    output = (
        Path(os.environ["SWARMFL_M4_INTEGRATION_DIR"])
        if os.environ.get("SWARMFL_M4_INTEGRATION_DIR")
        else tmp_path / "real-feedback"
    )
    command = [
        sys.executable,
        "-m",
        "fed_med_agents.experiments.milestone04",
        "--device",
        "cpu",
        "--episodes",
        "24",
        "--triggers",
        "3",
        "27",
        "--output",
        str(output),
    ]
    process = subprocess.run(
        command, check=False, capture_output=True, text=True, encoding="utf-8", timeout=300
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((output / "metrics.json").read_text())
    assert result["approved_version"] == "v001"
    assert result["specialists_unchanged"]
    assert {row["client"] for row in result["participation"]} == {0, 1, 2}
    assert result["history"][0]["skipped"] == "insufficient_verified_feedback"
    assert result["history"][1]["shared_change_l2"] > 0
    assert result["later_coordinator"]["version"] == "v001"
    assert any(result["later_coordinator"]["selected"][0])
    from fed_med_agents.feedback.verify import verify

    assert verify(output)["verified"]


def test_preserved_m4_evidence_if_available():
    path = ARTIFACTS / "milestone04"
    if not (path / "metrics.json").exists():
        pytest.skip("No saved M4 run")
    from fed_med_agents.feedback.verify import verify

    assert verify(path)["verified"]
