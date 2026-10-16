"""Deterministic CPU-only three-specialist Flower quickstart.

The task, records, labels, credentials, and checkpoints are synthetic and have no
clinical meaning. The demo sends a single explicitly authorized feature subset to
one selected specialist per case; private fitting rows and model weights stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation

from .collaboration.registry import (
    ConsultationSession,
    FeatureSchema,
    ModelAdapter,
    OutputSchema,
    RegisteredSpecialist,
    Specialist,
    SpecialistRegistry,
    TaskSchema,
)
from .feedback.events import FeedbackLedger, PredictionEvent, VerifiedOutcome, digest, sign
from .feedback.versions import PolicyRegistry
from .fusion.dynamic import DynamicFusion, FusionEvidence
from .routing.policy import RoutingPolicy

IDS = ("linear-a", "linear-b", "linear-c")
TASK = TaskSchema(
    name="synthetic-event",
    labels=("synthetic-negative", "synthetic-positive"),
    label_definition="Artificial Bernoulli outcome; no clinical meaning",
)
OUTPUT = OutputSchema(positive_label=TASK.labels[1])
client_app = ClientApp()
server_app = ServerApp()


def _root() -> Path:
    value = os.environ.get("SWARMFL_QUICKSTART_OUTPUT")
    if not value:
        raise RuntimeError("SWARMFL_QUICKSTART_OUTPUT is required")
    return Path(value)


def _read(name: str) -> Any:
    return json.loads((_root() / name).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _probability(index: int, features: dict[str, float]) -> float:
    model = _read(f"specialist-{index}.json")
    values = np.asarray([features[name] for name in model["features"]], dtype=np.float64)
    score = float(values @ np.asarray(model["weights"]) + model["bias"])
    return 1.0 / (1.0 + math.exp(-score))


def _declarations() -> list[dict[str, Any]]:
    return _read("declarations.json")


@client_app.query("capabilities")
def capabilities(message: Message, context: Context) -> Message:
    """Return public model metadata, never local weights or fitting records."""
    index = int(context.node_config["partition-id"])
    response = {"index": index, "declaration": json.dumps(_declarations()[index])}
    return Message(RecordDict({"response": ConfigRecord(response)}), reply_to=message)


@client_app.query("predict")
def predict(message: Message, context: Context) -> Message:
    """Validate and answer one new-patient request on the selected Flower node."""
    index = int(context.node_config["partition-id"])
    spec = Specialist.model_validate(_declarations()[index])
    request = json.loads(message.content["config"]["request"])
    expected = {
        "origin",
        "destination",
        "patient_ids",
        "adapter_id",
        "checkpoint_sha256",
        "task",
        "mode",
        "authorized_features",
        "features",
    }
    if set(request) != expected or request["destination"] != spec.specialist_id:
        raise ValueError("Invalid recipient or payload fields")
    if request["mode"] != "new_patient" or len(request["patient_ids"]) != 1:
        raise ValueError("Quickstart requires one new synthetic case")
    if request["adapter_id"] != spec.model_adapter.adapter_id:
        raise ValueError("Adapter mismatch")
    if request["checkpoint_sha256"] != spec.model_adapter.checkpoint_sha256:
        raise ValueError("Checkpoint mismatch")
    if request["authorized_features"] != list(spec.model_adapter.feature_names):
        raise ValueError("Feature authorization mismatch")
    if set(request["features"]) != set(spec.model_adapter.feature_names):
        raise ValueError("Missing or extra feature")
    for schema in spec.feature_schema:
        schema.validate_value(request["features"][schema.name])
    probability = _probability(index, request["features"])
    keys = ("origin", "destination", "patient_ids", "adapter_id", "checkpoint_sha256")
    answer = {key: request[key] for key in keys} | {"probabilities": [probability], "client": index}
    audit = {"client": index, "feature_names": sorted(request["features"]), "value_count": 2}
    with (_root() / "consultations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(audit) + "\n")
    return Message(RecordDict({"response": ConfigRecord({"json": json.dumps(answer)})}), reply_to=message)


def _registry(nodes: dict[str, int]) -> SpecialistRegistry:
    registry = SpecialistRegistry(TASK, OUTPUT)
    for declaration in _declarations():
        spec = Specialist.model_validate(declaration)
        registry.register(spec, node_id=nodes[spec.specialist_id])
    return registry


def _session(registry: SpecialistRegistry, origin: str, patient: str) -> ConsultationSession:
    peers = frozenset(name for name in IDS if name != origin)
    return ConsultationSession(
        registry,
        origin,
        (patient,),
        permitted=peers,
        feature_permissions={
            name: frozenset(registry.get(name).specialist.model_adapter.feature_names) for name in peers
        },
        available_features=frozenset(f"x{i}" for i in range(6)),
        max_calls=1,
        max_cost=0.02,
        alignment_namespace="swarmfl-quickstart-v1",
    )


@server_app.main()
def server(grid: Any, context: Context) -> None:
    """Discover three clients and perform selected-only consultations and fusion."""
    del context

    def query(node: int, topic: str, content: dict[str, Any]) -> dict[str, Any]:
        replies = list(
            grid.send_and_receive(
                [
                    Message(
                        RecordDict({"config": ConfigRecord(content)}),
                        dst_node_id=node,
                        message_type=f"query.{topic}",
                    )
                ],
                timeout=60,
            )
        )
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError(f"Flower {topic} query failed")
        return dict(replies[0].content["response"])

    deadline = time.perf_counter() + 60
    while len(list(grid.get_node_ids())) < 3:
        if time.perf_counter() >= deadline:
            raise RuntimeError("Three Flower clients did not connect")
        time.sleep(0.05)
    nodes: dict[str, int] = {}
    for node in grid.get_node_ids():
        info = query(node, "capabilities", {})
        spec = Specialist.model_validate_json(str(info["declaration"]))
        nodes[spec.specialist_id] = node
    if set(nodes) != set(IDS):
        raise RuntimeError("Expected exactly three specialist identities")
    registry = _registry(nodes)
    fusion = DynamicFusion("mean", _declarations())
    router = RoutingPolicy("fixed", IDS, width=3)
    router.gains[:] = 0.03
    cases = _read("cases.json")
    outputs = []
    for case in cases:
        origin = str(case["origin"])
        session = _session(registry, origin, str(case["patient_id"]))
        contexts = {name: [0.0, 0.0, 0.0] for name in IDS if name != origin}
        own_index = IDS.index(origin)
        own_features = {
            name: case["features"][name] for name in (f"x{2 * own_index}", f"x{2 * own_index + 1}")
        }
        own = _probability(own_index, own_features)
        decision = router.route(session, contexts, own, version="demo-router-v1")
        evidence = [
            FusionEvidence(
                origin,
                str(case["patient_id"]),
                own,
                registry.get(origin).specialist.model_adapter.checkpoint_sha256,
            )
        ]
        if decision.destination is not None:

            def transport(target: RegisteredSpecialist, payload: dict[str, Any]) -> dict[str, Any]:
                return json.loads(
                    query(int(target.node_id), "predict", {"request": json.dumps(payload)})["json"]
                )

            reply = session.consult(decision.destination, transport, features=case["features"])
            evidence.append(FusionEvidence.from_reply(reply, str(case["patient_id"])))
        outputs.append(
            {
                "patient_id": case["patient_id"],
                "origin": origin,
                "selected": decision.destination,
                "prediction": fusion.fuse(evidence, session),
                "calls": session.calls,
            }
        )
    _write(
        _root() / "result.json",
        {
            "flower_clients": 3,
            "flower_node_ids": nodes,
            "selected_only": True,
            "predictions": outputs,
        },
    )


def prepare(output: Path) -> None:
    """Generate synthetic fitting records, local linear adapters, and new cases."""
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(8042)
    x = rng.normal(size=(180, 6))
    logits = x @ np.array([0.8, -0.6, 0.5, 0.7, -0.9, 0.4])
    y = (rng.random(len(x)) < 1 / (1 + np.exp(-logits))).astype(float)
    declarations = []
    for index, identity in enumerate(IDS):
        view = x[:, 2 * index : 2 * index + 2]
        design = np.column_stack([view, np.ones(len(view))])
        target = np.log((y + 0.05) / (1.05 - y))
        fitted = np.linalg.pinv(design) @ target
        model = {
            "features": [f"x{2 * index}", f"x{2 * index + 1}"],
            "weights": fitted[:2].tolist(),
            "bias": float(fitted[2]),
        }
        model_path = output / f"specialist-{index}.json"
        _write(model_path, model)
        declarations.append(
            Specialist(
                specialist_id=identity,
                task=TASK,
                output_schema=OUTPUT,
                model_adapter=ModelAdapter(
                    adapter_id=f"example.numpy-linear:{identity}:v1",
                    model_family="external-numpy-linear",
                    checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    feature_names=tuple(model["features"]),
                    input_dim=2,
                    alignment_namespace="swarmfl-quickstart-v1",
                ),
                feature_schema=tuple(
                    FeatureSchema(name=name, minimum=-10, maximum=10) for name in model["features"]
                ),
                capabilities=frozenset({"aligned_prediction", "new_patient_prediction"}),
                consultation_cost=0.02,
            ).model_dump(mode="json")
        )
    _write(output / "declarations.json", declarations)
    _write(
        output / "cases.json",
        [
            {
                "patient_id": f"synthetic-{index}",
                "origin": IDS[index],
                "features": {f"x{i}": float(value) for i, value in enumerate(x[170 + index])},
            }
            for index in range(3)
        ],
    )
    (output / "consultations.jsonl").write_text("", encoding="utf-8")


def versioning_smoke(output: Path) -> dict[str, str]:
    """Exercise immutable approval and exact pointer rollback locally."""
    store = PolicyRegistry(output / "versions")
    model = RoutingPolicy("fixed", IDS, width=3)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    metadata = {"eligible": True, "router_config": model.config()}
    store.save("v000", state, metadata)
    store.approve("v000")
    model.gains[:] = 0.01
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    store.save("v001", state, metadata)
    store.approve("v001")
    store.rollback("v000")
    return {"approved": "v001", "rollback": store.current()}


def feedback_smoke(output: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Authenticate one delayed synthetic outcome and reject pending/future leakage."""
    prediction_key, outcome_key = "18" * 32, "81" * 32
    row = result["predictions"][0]
    now = datetime.now(UTC)
    event = PredictionEvent(
        request_id="quickstart-feedback-0",
        patient_id=str(row["patient_id"]),
        origin=0,
        episode=1,
        available_domains=[True, True, True],
        action=1,
        consulted=[1],
        propensity=1.0,
        action_score=0.1,
        evidence=[float(row["prediction"]), float(row["prediction"]), None],
        final_risk=float(row["prediction"]),
        origin_only_risk=float(row["prediction"]),
        model_version="demo-fusion-v1",
        policy_version="demo-router-v1",
        consultation_cost=0.02,
        event_timestamp=now.isoformat(),
    )
    outcome = VerifiedOutcome(
        request_id=event.request_id,
        patient_id=event.patient_id,
        prediction_digest=digest(event),
        label=1,
        authority="simulation-training-label",
        received_episode=2,
        verified_timestamp=(now + timedelta(seconds=1)).isoformat(),
    )
    ledger = FeedbackLedger(
        output / "feedback-ledger",
        origin=0,
        allowed_patients={event.patient_id},
        prediction_key=prediction_key,
        outcome_key=outcome_key,
    )
    ledger.prediction(sign(event, prediction_key))
    pending_before = len(ledger.replay(1))
    ledger.outcome(sign(outcome, outcome_key))
    return {
        "authenticated": len(ledger.replay(2)) == 1,
        "pending_before_release": pending_before,
        "verified_after_release": len(ledger.replay(2)),
    }


def run(output: Path, *, feedback: bool = False) -> dict[str, Any]:
    """Run a fresh deterministic demonstration and return its evidence summary."""
    prepare(output)
    os.environ["SWARMFL_QUICKSTART_OUTPUT"] = str(output.resolve())
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
            "init_args": {"num_cpus": 3, "num_gpus": 0, "include_dashboard": False},
        },
    )
    result = _read("result.json")
    result["versioning"] = versioning_smoke(output)
    if feedback:
        result["feedback"] = feedback_smoke(output, result)
    _write(output / "result.json", result)
    return result


def main() -> None:
    """Console entry point for the installed-wheel quickstart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("swarmfl-quickstart-output"))
    parser.add_argument("--feedback", action="store_true", help="exercise signed delayed feedback")
    args = parser.parse_args()
    print(json.dumps(run(args.output, feedback=args.feedback), indent=2))


if __name__ == "__main__":
    main()
