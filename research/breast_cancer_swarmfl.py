"""CPU-friendly real Flower/SwarmFL experiment on sklearn's Wisconsin dataset.

Run with::

    python research/breast_cancer_swarmfl.py --output research/results/breast-cancer

The three client models and scalers are fit only on the stratified 80% training
split. The 20% test split is used only by the Flower evaluation workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.simulation import run_simulation
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fed_med_agents.collaboration.registry import (
    FeatureSchema,
    ModelAdapter,
    OutputSchema,
    Specialist,
    SpecialistRegistry,
    TaskSchema,
    ConsultationSession,
)
from fed_med_agents.fusion.dynamic import DynamicFusion, FusionEvidence
from fed_med_agents.routing.policy import RoutingPolicy

SEED = 42
IDS = ("specialist-0", "specialist-1", "specialist-2")
FEATURES = tuple(f"feature-{i:02d}" for i in range(30))
GROUPS = tuple(tuple(FEATURES[i : i + 10]) for i in range(0, 30, 10))
TASK = TaskSchema(
    name="wisconsin-breast-cancer",
    labels=("malignant", "benign"),
    label_definition="sklearn load_breast_cancer target mapping: 0 malignant, 1 benign",
)
OUTPUT = OutputSchema(positive_label="benign")
CLIENT_APP = ClientApp()
SERVER_APP = ServerApp()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare(output: Path) -> dict[str, Any]:
    """Create client-local model artifacts; no test labels enter this step."""
    output.mkdir(parents=True, exist_ok=True)
    data = load_breast_cancer()
    x, y = data.data.astype(np.float64), data.target.astype(np.int64)
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    np.save(output / "test_x.npy", x[test])
    np.save(output / "test_y.npy", y[test])
    reports = {}
    declarations = []
    for index, (specialist_id, names) in enumerate(zip(IDS, GROUPS, strict=True)):
        columns = np.array([FEATURES.index(name) for name in names])
        scaler = StandardScaler().fit(x[train][:, columns])
        model = LogisticRegression(max_iter=1000, random_state=SEED, solver="liblinear")
        model.fit(scaler.transform(x[train][:, columns]), y[train])
        record = {
            "features": list(names),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        }
        model_path = output / f"{specialist_id}.json"
        _write(model_path, record)
        checkpoint = hashlib.sha256(model_path.read_bytes()).hexdigest()
        declarations.append(
            Specialist(
                specialist_id=specialist_id,
                task=TASK,
                output_schema=OUTPUT,
                model_adapter=ModelAdapter(
                    adapter_id=f"sklearn-logistic:{specialist_id}:v1",
                    model_family="sklearn-logistic-regression",
                    checkpoint_sha256=checkpoint,
                    feature_names=names,
                    input_dim=len(names),
                    alignment_namespace="wisconsin-breast-cancer-v1",
                ),
                feature_schema=tuple(FeatureSchema(name=name) for name in names),
                capabilities=frozenset({"aligned_prediction", "new_patient_prediction"}),
                consultation_cost=0.02,
            ).model_dump(mode="json")
        )
        reports[specialist_id] = {"features": list(names), "train_rows": len(train)}
    _write(output / "declarations.json", declarations)
    _write(output / "split.json", {"seed": SEED, "train_rows": len(train), "test_rows": len(test)})
    return {"x": x, "y": y, "train": train, "test": test, "declarations": declarations, "reports": reports}


def _probability(model: dict[str, Any], features: dict[str, float]) -> float:
    values = np.asarray([features[name] for name in model["features"]], dtype=np.float64)
    z = float(((values - model["mean"]) / model["scale"]) @ model["coef"] + model["intercept"])
    return float(1.0 / (1.0 + np.exp(-z)))


@CLIENT_APP.query("capabilities")
def capabilities(message: Message, context: Context) -> Message:
    root = Path(os.environ["SWARMFL_ARTIFACT_ROOT"])
    declaration = _read(root / "declarations.json")[int(context.node_config["partition-id"])]
    return Message(RecordDict({"response": ConfigRecord({"declaration": json.dumps(declaration)})}), reply_to=message)


@CLIENT_APP.query("predict")
def predict(message: Message, context: Context) -> Message:
    index = int(context.node_config["partition-id"])
    request = json.loads(message.content["config"]["request"])
    root = Path(os.environ["SWARMFL_ARTIFACT_ROOT"])
    declarations = _read(root / "declarations.json")
    spec = declarations[index]
    if request["destination"] != spec["specialist_id"] or request["mode"] != "new_patient":
        raise ValueError("Invalid Flower prediction request")
    model = _read(root / f"{spec['specialist_id']}.json")
    probability = _probability(model, request["features"])
    return Message(
        RecordDict(
            {"response": ConfigRecord({"json": json.dumps({
                "origin": request["origin"], "destination": request["destination"],
                "patient_ids": request["patient_ids"], "adapter_id": request["adapter_id"],
                "checkpoint_sha256": request["checkpoint_sha256"], "probabilities": [probability],
            })})}
        ),
        reply_to=message,
    )


def _session(registry: SpecialistRegistry, origin: str, patient: str) -> ConsultationSession:
    peers = frozenset(name for name in IDS if name != origin)
    return ConsultationSession(
        registry, origin, (patient,), permitted=peers,
        feature_permissions={name: frozenset(registry.get(name).specialist.model_adapter.feature_names) for name in peers},
        available_features=frozenset(FEATURES), max_calls=2, max_cost=0.04,
        alignment_namespace="wisconsin-breast-cancer-v1",
    )


def _metrics(y: np.ndarray, probabilities: list[float], consultations: int, elapsed: float) -> dict[str, float]:
    pred = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auroc": float(roc_auc_score(y, probabilities)),
        "consultation_count": consultations,
        "mean_latency_ms": elapsed / len(y) * 1000,
        "total_latency_s": elapsed,
    }


@SERVER_APP.main()
def server(grid: Any, context: Context) -> None:
    del context
    root = Path(os.environ["SWARMFL_ARTIFACT_ROOT"])
    test_x, test_y = np.load(root / "test_x.npy"), np.load(root / "test_y.npy")
    declarations = _read(root / "declarations.json")
    registry = SpecialistRegistry(TASK, OUTPUT, model_family="sklearn-logistic-regression")
    nodes = {}
    deadline = time.perf_counter() + 60
    while len(list(grid.get_node_ids())) < 3:
        if time.perf_counter() > deadline:
            raise RuntimeError("Flower clients did not connect")
        time.sleep(0.05)
    for node in grid.get_node_ids():
        message = Message(RecordDict({"config": ConfigRecord({})}), dst_node_id=node, message_type="query.capabilities")
        replies = list(grid.send_and_receive([message], timeout=60))
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError("Flower capability discovery failed")
        declaration = Specialist.model_validate_json(str(replies[0].content["response"]["declaration"]))
        nodes[declaration.specialist_id] = node
        registry.register(declaration, node_id=node)
    fusion = DynamicFusion("mean", declarations)
    router = RoutingPolicy("fixed", IDS, width=1)
    router.gains[:] = 0.03

    def call(entry: Any, payload: dict[str, Any]) -> dict[str, Any]:
        message = Message(RecordDict({"config": ConfigRecord({"request": json.dumps(payload)})}), dst_node_id=entry.node_id, message_type="query.predict")
        replies = list(grid.send_and_receive([message], timeout=60))
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError("Flower prediction failed")
        return json.loads(replies[0].content["response"]["json"])

    def run_condition(kind: str) -> dict[str, float]:
        probabilities, consultations, started = [], 0, time.perf_counter()
        rng = random.Random(SEED)
        for row, label in zip(test_x, test_y, strict=True):
            patient = f"test-{len(probabilities)}"
            features = {FEATURES[i]: float(row[i]) for i in range(len(FEATURES))}
            origin = IDS[len(probabilities) % len(IDS)]
            if kind.startswith("single-"):
                origin = IDS[int(kind[-1])]
            session = _session(registry, origin, patient)
            own = call(registry.get(origin), {"origin": origin, "destination": origin, "patient_ids": [patient], "adapter_id": registry.get(origin).specialist.model_adapter.adapter_id, "checkpoint_sha256": registry.get(origin).specialist.model_adapter.checkpoint_sha256, "mode": "new_patient", "authorized_features": list(GROUPS[IDS.index(origin)]), "task": TASK.model_dump(mode="json"), "features": {name: features[name] for name in GROUPS[IDS.index(origin)]}})
            evidence = [FusionEvidence.from_reply(own, patient)]
            if kind.startswith("single-"):
                evidence = [FusionEvidence.from_reply(own, patient)]
            elif kind == "mean":
                for target in IDS:
                    if target == origin:
                        continue
                    reply = session.consult(target, lambda entry, payload: call(entry, payload), features=features)
                    evidence.append(FusionEvidence.from_reply(reply, patient))
                consultations += 2
            else:
                if kind == "random":
                    target = rng.choice([name for name in IDS if name != origin])
                else:
                    peer_contexts = {name: [0.0] for name in IDS if name != origin}
                    decision = router.route(session, peer_contexts, evidence[0].probability, version="fixed-v1")
                    target = decision.destination
                assert target is not None
                reply = session.consult(target, lambda entry, payload: call(entry, payload), features=features)
                evidence.append(FusionEvidence.from_reply(reply, patient))
                consultations += 1
            probabilities.append(fusion.fuse(evidence, session))
        return _metrics(test_y, probabilities, consultations, time.perf_counter() - started)

    results = {name: run_condition(name) for name in ("single-0", "single-1", "single-2", "mean", "random", "swarmfl")}
    _write(root / "results.json", results)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("research/results/breast-cancer"))
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    prepared = prepare(args.output)
    del prepared
    os.environ["SWARMFL_ARTIFACT_ROOT"] = str(args.output.resolve())
    run_simulation(
        server_app=SERVER_APP, client_app=CLIENT_APP, num_supernodes=3,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0}, "init_args": {"num_cpus": 3, "num_gpus": 0, "include_dashboard": False},
            "actor": {"runtime_env": {"env_vars": {"SWARMFL_ARTIFACT_ROOT": str(args.output.resolve())}}}},
    )


if __name__ == "__main__":
    main()
