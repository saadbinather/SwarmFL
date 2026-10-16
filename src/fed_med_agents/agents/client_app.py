"""Inference-only agent ClientApp; original federated training apps are untouched."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

from .provider import OllamaProvider
from .schemas import FinalPrediction, PatientRequest
from .specialist import PrivateClassifier, SpecialistAgent

app = ClientApp()
_adapters: dict[tuple[str, int, str], PrivateClassifier] = {}


def configuration() -> dict[str, Any]:
    return json.loads(os.environ["FED_MED_M3_CONFIG"])


def adapter(context: Context) -> PrivateClassifier:
    config = configuration()
    index = int(context.node_config["partition-id"])
    device = os.environ.get("FED_MED_DEVICE", "auto")
    key = (config["checkpoint_dir"], index, device)
    if key not in _adapters:
        _adapters[key] = PrivateClassifier(
            Path(config["checkpoint_dir"]) / f"client_{index}", index, config, device
        )
    return _adapters[key]


@app.query("agent_capabilities")
def capabilities(message: Message, context: Context) -> Message:
    capability = adapter(context).capability
    return Message(
        RecordDict({"capability": ConfigRecord({"json": capability.model_dump_json()})}), reply_to=message
    )


@app.query("agent_predict")
def predict(message: Message, context: Context) -> Message:
    request = PatientRequest.model_validate_json(message.content["request"]["json"])
    service = SpecialistAgent(adapter(context), OllamaProvider(**configuration()["provider"]))
    result, trace = service.predict(request)
    return Message(
        RecordDict(
            {"response": ConfigRecord({"json": result.model_dump_json(), "trace": json.dumps(trace)})}
        ),
        reply_to=message,
    )


@app.query("agent_intake")
def intake(message: Message, context: Context) -> Message:
    request = PatientRequest.model_validate_json(message.content["request"]["json"])
    capability = adapter(context).capability
    if request.origin != capability.specialist:
        raise ValueError("Origin intake reached the incorrect client")
    own = request.model_copy(
        update={
            "destination": capability.specialist,
            "features": {
                name: value
                for name, value in request.features.items()
                if name in capability.required_features
            },
        }
    )
    service = SpecialistAgent(adapter(context), OllamaProvider(**configuration()["provider"]))
    result, trace = service.predict(own)
    return Message(
        RecordDict(
            {
                "response": ConfigRecord(
                    {
                        "json": result.model_dump_json(),
                        "trace": json.dumps(trace),
                        "origin": capability.specialist,
                    }
                )
            }
        ),
        reply_to=message,
    )


@app.query("agent_result")
def receive_result(message: Message, context: Context) -> Message:
    result = FinalPrediction.model_validate_json(message.content["result"]["json"])
    if result.origin != adapter(context).capability.specialist:
        raise ValueError("Final prediction delivered to the wrong origin")
    # Explicit Flower return hop; record request-local acknowledgement, not training memory.
    context.state["last-result"] = ConfigRecord(
        {"request_id": result.request_id, "patient_id": result.patient_id}
    )
    return Message(
        RecordDict(
            {
                "ack": ConfigRecord(
                    {
                        "request_id": result.request_id,
                        "patient_id": result.patient_id,
                        "origin": result.origin,
                        "received": True,
                    }
                )
            }
        ),
        reply_to=message,
    )
