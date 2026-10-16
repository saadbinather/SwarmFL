"""FedAvg over shared router/fusion; genuine Grid-mediated consultation protocol."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ..collaboration import ConsultationBroker
from ..devices import peak_memory, select_device, synchronize
from ..evaluation.collaboration import calibration_error, evaluate_policy, policy_predictions
from ..fusion import CollaborativeModel
from .client_app import directory, state_cpu

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = json.loads(os.environ.get("FED_MED_M2_CONFIG", "{}")) | dict(context.run_config)
    seed, rounds, budget = int(config["seed"]), int(config["rounds"]), int(config["budget"])
    device = select_device(str(config["device"]))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    path = directory(seed)
    private_hashes = {}
    nodes = {}
    domain_available = {}
    deadline = time.perf_counter() + 60
    while len(list(grid.get_node_ids())) < 3:
        if time.perf_counter() > deadline:
            raise RuntimeError("Three clients did not connect")
        time.sleep(0.1)
    for node in grid.get_node_ids():
        message = Message(
            RecordDict({"config": ConfigRecord({"seed": seed})}),
            dst_node_id=node,
            message_type="query.identity",
            group_id="identity",
        )
        replies = list(grid.send_and_receive([message], timeout=60))
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError("Client identity handshake failed")
        client = int(replies[0].content["identity"]["client"])
        if client in nodes:
            raise RuntimeError("Duplicate specialist identity")
        nodes[client] = node
        domain_available[client] = bool(replies[0].content["identity"]["domain_available"])
        private_hashes[str(client)] = replies[0].content["identity"]["specialist_sha256"]
    if set(nodes) != {0, 1, 2}:
        raise RuntimeError("Invalid client identities")
    trace = path / "consultations.jsonl"
    trace.write_text("", encoding="utf-8")
    broker = ConsultationBroker(grid, nodes, trace, seed)
    broker.domain_available = domain_available
    for origin in nodes:
        plan = broker.exchange(origin, "plan", RecordDict({"config": ConfigRecord({"seed": seed})}))["plan"]
        patients = list(plan["patient_ids"])
        columns = []
        for destination in range(3):
            response = broker.predict(origin, destination, patients, "oof", "training/exploration")
            if any(status != "ok" for status in response["status"]):
                raise RuntimeError("Incomplete OOF exploration outcomes")
            columns.append(response["probabilities"])
        values = np.asarray(columns, dtype=np.float32).T
        broker.exchange(
            origin,
            "cache",
            RecordDict(
                {
                    "config": ConfigRecord(
                        {"seed": seed, "patient_ids": patients, "probabilities": values.ravel().tolist()}
                    )
                }
            ),
        )
    model = CollaborativeModel(3, budget).to(device)
    initial = {key: value.clone() for key, value in state_cpu(model).items()}
    torch.save(initial, path / "shared_initial.pt")
    participation = {str(i): [] for i in nodes}
    history = {}
    best = {"bce": float("inf"), "round": -1}

    def aggregate(records: list[RecordDict], weighted_by: str) -> MetricRecord:
        reports = [record["metrics"] for record in records]
        if sorted(int(report["client"]) for report in reports) != [0, 1, 2]:
            raise RuntimeError("Missing/duplicate client updates")
        if any(
            (budget > 0 and report["router-change"] <= 0) or report["fusion-change"] <= 0
            for report in reports
        ):
            raise RuntimeError("Router or fusion failed to train")
        for report in reports:
            participation[str(int(report["client"]))].append(int(report["round"]))
        total = sum(report[weighted_by] for report in reports)
        return MetricRecord(
            {
                key: sum(report[key] * report[weighted_by] for report in reports) / total
                for key in ("loss", "router-change", "fusion-change", "runtime-seconds", "peak-vram-bytes")
            }
        )

    def validate(round_id: int, arrays: ArrayRecord) -> MetricRecord:
        model.load_state_dict(arrays.to_torch_state_dict())
        with np.load(path / "cohort.npz") as file:
            patients = file["val"].tolist()
            labels = file["y"][file["val"]]
        scores = {}
        for origin in range(3):
            prediction = policy_predictions(
                model, broker, patients, origin, "learned", f"validation/round_{round_id}", device, seed
            )
            scores[str(origin)] = evaluate_policy(prediction, labels)
        bce = float(np.mean([value["binary_cross_entropy"] for value in scores.values()]))
        history[str(round_id)] = scores
        if round_id > 0 and bce < best["bce"]:
            best.update(bce=bce, round=round_id)
            torch.save(state_cpu(model), path / "shared_best.pt")
        print(f"M2 validation seed={seed} round={round_id} mean_origin_bce={bce:.6f}", flush=True)
        return MetricRecord({"binary_cross_entropy": bce})

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=3,
        min_available_nodes=3,
        train_metrics_aggr_fn=aggregate,
    )
    result = strategy.start(
        grid,
        initial_arrays=ArrayRecord(initial),
        num_rounds=rounds,
        train_config=ConfigRecord(dict(config)),
        evaluate_fn=validate,
        timeout=300,
    )
    torch.save(result.arrays.to_torch_state_dict(), path / "shared_final.pt")
    model.load_state_dict(torch.load(path / "shared_best.pt", weights_only=True, map_location=device))
    # Model selection is now frozen. This is the first test prediction/label evaluation.
    with np.load(path / "cohort.npz") as file:
        patients = file["test"].tolist()
        labels = file["y"][file["test"]]
    tables, examples = {}, []
    for policy in ("learned", "none", "fixed_mean", "all", "random", "confidence"):
        table = {}
        for origin in range(3):
            prediction = policy_predictions(model, broker, patients, origin, policy, "test", device, seed)
            table[str(origin)] = evaluate_policy(prediction, labels)
            table[str(origin)]["ece_10_bins"] = calibration_error(labels, prediction["probabilities"])
            np.savez(
                path / f"test_{policy}_origin_{origin}.npz",
                patient_ids=patients,
                probabilities=prediction["probabilities"],
                selected=prediction["selected"],
            )
            if policy == "learned" and origin == 0:
                for row in range(min(8, len(patients))):
                    scores = prediction["scores"][row]
                    examples.append(
                        {
                            "patient_id": f"case-{patients[row]:04d}",
                            "origin": 0,
                            "origin_probability": float(prediction["own_probabilities"][row]),
                            "calibrated_probability_confidence": float(
                                2 * abs(prediction["own_probabilities"][row] - 0.5)
                            ),
                            "actions": model.router.actions.cpu().tolist(),
                            "scores": [float(score) if np.isfinite(score) else None for score in scores],
                            "selected": np.flatnonzero(prediction["selected"][row]).tolist(),
                        }
                    )
        tables[policy] = table
    expected = list(range(1, rounds + 1))
    if any(value != expected for value in participation.values()):
        raise RuntimeError("Client participation verification failed")
    unchanged = all(
        broker.exchange(i, "identity", RecordDict({"config": ConfigRecord({"seed": seed})}))["identity"][
            "specialist_sha256"
        ]
        == private_hashes[str(i)]
        for i in range(3)
    )
    if not unchanged:
        raise RuntimeError("Private specialist checkpoint changed during FL")
    final = result.arrays.to_torch_state_dict()
    changes = {
        prefix: sum(
            float(((final[key] - initial[key]) ** 2).sum()) for key in final if key.startswith(prefix)
        )
        ** 0.5
        for prefix in ("router.", "fusion.")
    }
    synchronize(device)
    output = {
        "seed": seed,
        "flower_run_id": context.run_id,
        "config": dict(config),
        "selected_round": best["round"],
        "selection_mean_validation_bce": best["bce"],
        "validation_history": history,
        "client_training_metrics": {str(k): dict(v) for k, v in result.train_metrics_clientapp.items()},
        "test": tables,
        "adjacency": model.router.adjacency().detach().cpu().tolist(),
        "routing_examples": examples,
        "verification": {
            "participation": participation,
            "parameter_l2_changes": changes,
            "specialists_unchanged": unchanged,
        },
        "runtime_seconds": time.perf_counter() - started,
        "server_peak_allocated_bytes": peak_memory(device),
        "device": str(device),
        "consultation_events": len(broker.events),
    }
    (path / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": seed,
                "selected_round": best["round"],
                "test_origin_A": tables["learned"]["0"],
                "adjacency": output["adjacency"],
                "verification": output["verification"],
            },
            indent=2,
        ),
        flush=True,
    )
