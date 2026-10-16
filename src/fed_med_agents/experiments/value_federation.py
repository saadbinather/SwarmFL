"""Optional genuine Flower routing-only comparison for M5 Step 3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ..devices import peak_memory, select_device
from ..models import Classifier
from .consultation_value import COSTS, choose, fit_network, predict, summarize, value_network
from .diabetes import logits
from .m5_step03 import digest, probabilities, unpack_predictions, write
from .repair_models import make_model
from .routing_diagnostics import cosine, snapshot, vector

client_app, server_app = ClientApp(), ServerApp()


def configuration() -> dict:
    return json.loads(os.environ["SWARMFL_VALUE_CONFIG"])


def data_for(stage: str, kind: str) -> dict:
    root = Path(configuration()["output"])
    if stage not in json.loads((root / "contexts.json").read_text()) or kind not in ("original", "logistic"):
        raise ValueError("Unknown context")
    return dict(np.load(root / stage / f"routing_{kind}.npz"))


@client_app.query("evidence")
def evidence(message: Message, context: Context) -> Message:
    """Return private-model probabilities/IDs, never private parameters or features."""
    cfg = message.content["config"]
    root, client = Path(configuration()["output"]), int(context.node_config["partition-id"])
    stage, block = str(cfg["stage"]), str(cfg["block"])
    if stage not in json.loads((root / "contexts.json").read_text()) or block not in (
        "final",
        "target0",
        "target1",
        "target2",
    ):
        raise ValueError("Unknown evidence source")
    folder = root / stage / block / "seed_42" / f"client_{client}"
    with np.load(folder / "local.npz") as data:
        record = {
            "client": client,
            "fit_ids": data["train"].tolist(),
            "held_ids": data["val"].tolist(),
            "oof": data["oof"].tolist(),
            "assessment": data["assessment"].tolist(),
            "private_hash": digest(folder / "specialist.pt"),
        }
    return Message(RecordDict({"response": ConfigRecord(record)}), reply_to=message)


@client_app.query("consult")
def consult(message: Message, context: Context) -> Message:
    """Actually classify requested aligned assessment IDs, no raw-feature forwarding."""
    cfg = message.content["config"]
    root, client = Path(configuration()["output"]), int(context.node_config["partition-id"])
    stage = str(cfg["stage"])
    if stage not in json.loads((root / "contexts.json").read_text()):
        raise ValueError("Unknown context")
    folder = root / stage / "final/seed_42" / f"client_{client}"
    ids = np.asarray(cfg["ids"], dtype=int)
    metadata = json.loads((folder / "specialist.json").read_text())
    with np.load(folder / "local.npz") as data:
        if not set(ids).issubset(data["val"].tolist()):
            raise ValueError("Consultation outside assessment")
        x = data["x"][ids].copy()
    model = Classifier(x.shape[1])
    model.load_state_dict(torch.load(folder / "specialist.pt", weights_only=True))
    p = torch.sigmoid(torch.tensor(logits(model, x, torch.device("cpu")) / metadata["temperature"])).tolist()
    return Message(
        RecordDict({"response": ConfigRecord({"client": client, "ids": ids.tolist(), "p": p})}),
        reply_to=message,
    )


@client_app.train()
def train(message: Message, context: Context) -> Message:
    """Train only this origin's examples and return compatible router arrays/scalars."""
    cfg = message.content["config"]
    client = int(context.node_config["partition-id"])
    data = data_for(str(cfg["stage"]), str(cfg["kind"]))
    n = len(data["target"]) // 3
    x, y = data["x"][client * n : (client + 1) * n], data["target"][client * n : (client + 1) * n]
    device = select_device(str(cfg["device"]))
    torch.set_num_threads(1)
    model = value_network(x.shape[1], str(cfg["architecture"])).to(device)
    model.load_state_dict(message.content["arrays"].to_torch_state_dict())
    before = vector(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    report = fit_network(model, x, y, steps=16)
    report.update(
        client=client,
        num_examples=n,
        drift=float((vector(model) - before).norm()),
        device=str(device),
        peak_allocated_bytes=peak_memory(device),
        payload_keys=["arrays", "metrics"],
    )
    target = Path(configuration()["output"]) / "federated" / str(cfg["stage"]) / str(cfg["kind"])
    name = f"client_{client}_round_{cfg['server-round']}"
    torch.save(snapshot(model), target / f"{name}.pt")
    write(target / f"{name}.json", report)
    return Message(
        RecordDict(
            {
                "arrays": ArrayRecord(snapshot(model)),
                "metrics": MetricRecord({"client": client, "num-examples": n, "loss": report["loss"]}),
            }
        ),
        reply_to=message,
    )


@server_app.main()
def server(grid: Grid, context: Context) -> None:
    cfg, start = configuration(), time.perf_counter()
    root, device = Path(cfg["output"]), select_device(cfg["device"])
    torch.set_num_threads(1)
    deadline = time.perf_counter() + 90
    while len(list(grid.get_node_ids())) != 3:
        if time.perf_counter() > deadline:
            raise TimeoutError("Three clients required")
        time.sleep(0.1)
    nodes = list(grid.get_node_ids())
    identities, traffic = {}, []

    def query(node, content, topic):
        request = Message(
            RecordDict({"config": ConfigRecord(content)}), dst_node_id=node, message_type=f"query.{topic}"
        )
        replies = list(grid.send_and_receive([request], timeout=120))
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError(f"Failed {topic} reply")
        return dict(replies[0].content["response"])

    # Authentic full-information offline probability traffic, including target-fold teachers.
    for stage in json.loads((root / "contexts.json").read_text()):
        for block in ("final", "target0", "target1", "target2"):
            expected, held = probabilities(root / stage / block)
            for node in nodes:
                r = query(node, {"stage": stage, "block": block}, "evidence")
                client = int(r["client"])
                identities[client] = node
                np.testing.assert_allclose(r["oof"], expected[:, client], atol=1e-7)
                np.testing.assert_allclose(r["assessment"], held[:, client], atol=1e-7)
                traffic.append({"stage": stage, "block": block, **r})
    write(root / "federated/evidence.json", {"queries": traffic, "run_id": context.run_id})
    selected = json.loads((root / "federated-selection.json").read_text())
    all_results, communication = {}, []
    for fold in range(3):
        stage = f"fold{fold}_outer"
        for kind, architecture in selected.items():
            target = root / "federated" / stage / kind
            target.mkdir(parents=True)
            data = data_for(stage, kind)
            torch.manual_seed(53200 + fold)
            model = value_network(data["x"].shape[1], architecture).to(device)
            initial = snapshot(model)
            history = {}

            def save(round_id, arrays, model=model, target=target, data=data, history=history):
                state = arrays.to_torch_state_dict()
                model.load_state_dict(state)
                torch.save(snapshot(model), target / f"server_round_{round_id}.pt")
                gradients = []
                n = len(data["target"]) // 3
                for origin in range(3):
                    model.zero_grad()
                    x = torch.tensor(data["x"][origin * n : (origin + 1) * n], device=device)
                    y = torch.tensor(data["target"][origin * n : (origin + 1) * n], device=device)
                    ((model(x).flatten() - y) ** 2).mean().backward()
                    gradients.append(vector(model, True))
                record = {
                    "gradient_cosines": [
                        cosine(gradients[a], gradients[b]) for a, b in ((0, 1), (0, 2), (1, 2))
                    ]
                }
                if round_id:
                    clients = [
                        torch.load(target / f"client_{i}_round_{round_id}.pt", weights_only=True)
                        for i in range(3)
                    ]
                    for key, value in state.items():
                        torch.testing.assert_close(
                            value.cpu(), sum(c[key] for c in clients) / 3, atol=1e-6, rtol=1e-5
                        )
                    record["aggregation_verified"] = True
                history[str(round_id)] = record
                return MetricRecord({"round": round_id})

            result = FedAvg(
                fraction_train=1, fraction_evaluate=0, min_train_nodes=3, min_available_nodes=3
            ).start(
                grid,
                initial_arrays=ArrayRecord(initial),
                num_rounds=5,
                train_config=ConfigRecord(
                    {"stage": stage, "kind": kind, "architecture": architecture, "device": str(device)}
                ),
                evaluate_fn=save,
                timeout=300,
            )
            model.load_state_dict(result.arrays.to_torch_state_dict())
            estimates = unpack_predictions(predict(model, data["held_x"]), len(data["held_x"]) // 6)
            central = value_network(data["x"].shape[1], architecture)
            central.load_state_dict(initial)
            fit_network(central, data["x"], data["target"], 80)
            torch.save(snapshot(central), target / "central.pt")
            torch.testing.assert_close(
                torch.load(root / stage / f"router_{kind}_{architecture}_enriched.pt", weights_only=True),
                snapshot(central),
                atol=1e-7,
                rtol=1e-6,
            )
            with np.load(root / stage / f"{kind}.npz") as source:
                record = {}
                ids = np.load(root / stage / "indices.npz")["held"]
                fusion = make_model(kind)
                fusion.load_state_dict(torch.load(root / stage / f"fusion_{kind}.pt", weights_only=True))
                for cost in COSTS:
                    record[str(cost)] = {}
                    for origin in range(3):
                        actions = choose(estimates[:, origin], origin, cost)
                        record[str(cost)][str(origin)] = summarize(
                            source["held_y"],
                            source["held_risks"][:, origin],
                            actions,
                            origin,
                            cost,
                            estimates[:, origin],
                            source["mean_gain"],
                        )
                        if cost != 0.02:
                            continue
                        # Deploy masked selected-only evidence through real Flower consultations.
                        p = np.full_like(source["held_p"], np.nan)
                        p[:, origin] = source["held_p"][:, origin]
                        for peer in range(3):
                            rows = np.flatnonzero(actions == peer + 1)
                            if not len(rows):
                                continue
                            r = query(
                                identities[peer], {"stage": stage, "ids": ids[rows].tolist()}, "consult"
                            )
                            np.testing.assert_array_equal(r["ids"], ids[rows])
                            p[rows, peer] = r["p"]
                            communication.append(
                                {
                                    "stage": stage,
                                    "kind": kind,
                                    "origin": origin,
                                    "peer": peer,
                                    "ids": r["ids"],
                                    "p": r["p"],
                                }
                            )
                        with torch.no_grad():
                            tensor, mask = torch.tensor(p), torch.tensor(np.isfinite(p))
                            prediction = (
                                fusion.fusion(
                                    tensor, 2 * (tensor - 0.5).abs(), mask, torch.full((len(p),), origin)
                                )
                                .sigmoid()
                                .numpy()
                            )
                        expected = source["held_risks"][np.arange(len(p)), origin, actions]
                        np.testing.assert_allclose(prediction, expected, atol=1e-6)
            final = snapshot(model)
            drift = sum(float(((v - initial[k]) ** 2).sum()) for k, v in final.items()) ** 0.5
            if drift <= 1e-6:
                raise ValueError("Router did not change")
            all_results[f"{stage}_{kind}"] = {
                "architecture": architecture,
                "assessment": record,
                "history": history,
                "drift": drift,
                "run_id": context.run_id,
            }
            write(target / "results.json", all_results[f"{stage}_{kind}"])
    write(
        root / "federated/results.json",
        {
            "trials": all_results,
            "consultations": communication,
            "rounds": 30,
            "client_updates": 90,
            "run_id": context.run_id,
            "seconds": time.perf_counter() - start,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step03-v1"))
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    cfg = {"output": str(root), "device": str(select_device(args.device))}
    os.environ["SWARMFL_VALUE_CONFIG"] = json.dumps(cfg)
    if args.worker:
        from flwr.simulation import run_simulation

        gpu = cfg["device"] == "cuda"
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=3,
            backend_config={
                "client_resources": {"num_cpus": 1, "num_gpus": int(gpu)},
                "init_args": {
                    "num_cpus": 1,
                    "num_gpus": int(gpu),
                    "include_dashboard": False,
                    "object_store_memory": 256 * 1024 * 1024,
                },
            },
        )
        return
    target = root / "federated"
    target.mkdir()  # Never overwrite previous runs.
    write(target / "config.json", cfg)
    with (target / "flower.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                __spec__.name,
                "--worker",
                "--output",
                str(root),
                "--device",
                cfg["device"],
            ],
            env=os.environ | {"PYTHONUTF8": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not (target / "results.json").exists():
        raise RuntimeError("Flower did not finish")


if __name__ == "__main__":
    main()
