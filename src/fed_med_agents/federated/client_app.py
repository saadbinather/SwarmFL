"""Client-local specialists serve queries; FedAvg trains only shared routing/fusion."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ..collaboration.protocol import answer
from ..devices import peak_memory, select_device, synchronize
from ..fusion import CollaborativeModel
from ..models import Classifier

app = ClientApp()


def directory(seed: int) -> Path:
    return Path(os.environ["FED_MED_M2_DIR"]) / f"seed_{seed}"


def client_index(context: Context) -> int:
    return int(context.node_config["partition-id"])


def local_data(seed: int, client: int) -> dict[str, np.ndarray]:
    with np.load(directory(seed) / f"client_{client}" / "local.npz") as file:
        return {key: file[key] for key in file.files}


def state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


@app.query("identity")
def identity(message: Message, context: Context) -> Message:
    seed = int(message.content["config"]["seed"])
    digest = hashlib.sha256(
        (directory(seed) / f"client_{client_index(context)}" / "specialist.pt").read_bytes()
    ).hexdigest()
    metadata = json.loads(
        (directory(seed) / f"client_{client_index(context)}" / "specialist.json").read_text()
    )
    return Message(
        RecordDict(
            {
                "identity": ConfigRecord(
                    {
                        "client": client_index(context),
                        "available": True,
                        "specialist_sha256": digest,
                        "domain_available": bool(metadata.get("domain_available", True)),
                    }
                )
            }
        ),
        reply_to=message,
    )


@app.query("plan")
def plan(message: Message, context: Context) -> Message:
    seed = int(message.content["config"]["seed"])
    local = local_data(seed, client_index(context))
    return Message(
        RecordDict(
            {"plan": ConfigRecord({"patient_ids": local["train"].tolist(), "origin": client_index(context)})}
        ),
        reply_to=message,
    )


@app.query("predict")
def predict(message: Message, context: Context) -> Message:
    request = dict(message.content["request"])
    seed, client = int(request["seed"]), client_index(context)
    if int(request["destination"]) != client:
        raise ValueError("Wrong destination")
    metadata = json.loads((directory(seed) / f"client_{client}" / "specialist.json").read_text())
    if not metadata.get("domain_available", True):
        response = answer(request, client, np.array([], dtype=int), np.array([], dtype=np.float32))
        response["status"] = ["missing_domain"] * len(request["patient_ids"])
        return Message(RecordDict({"response": ConfigRecord(response)}), reply_to=message)
    local = local_data(seed, client)
    started = time.perf_counter()
    if request["mode"] == "oof":
        ids, probabilities = local["train"], local["oof"]
    elif request["mode"] == "inference":
        device = select_device(os.environ.get("FED_MED_DEVICE", "auto"))
        model = Classifier(local["x"].shape[1]).to(device)
        path = directory(seed) / f"client_{client}"
        model.load_state_dict(torch.load(path / "specialist.pt", weights_only=True, map_location=device))
        temperature = json.loads((path / "specialist.json").read_text())["temperature"]
        # Predict only the requested patient's local row, never the full candidate cohort.
        requested = set(request["patient_ids"])
        ids = np.array([patient for patient in local["ids"] if f"case-{patient:04d}" in requested], dtype=int)
        with torch.no_grad():
            probabilities = (
                torch.sigmoid(model(torch.as_tensor(local["x"][ids], device=device)) / temperature)
                .cpu()
                .numpy()
            )
        synchronize(device)
    else:
        raise ValueError("Unknown prediction mode")
    request["specialist_seconds"] = time.perf_counter() - started
    response = answer(request, client, ids, probabilities)
    return Message(RecordDict({"response": ConfigRecord(response)}), reply_to=message)


@app.query("cache")
def cache(message: Message, context: Context) -> Message:
    config = message.content["config"]
    seed, client = int(config["seed"]), client_index(context)
    local = local_data(seed, client)
    if list(config["patient_ids"]) != local["train"].tolist():
        raise ValueError("Training cache must contain only aligned training patients")
    p = np.asarray(config["probabilities"], dtype=np.float32).reshape(len(local["train"]), 3)
    np.testing.assert_allclose(p[:, client], local["oof"], atol=1e-7)
    np.save(directory(seed) / f"client_{client}" / "training_predictions.npy", p)
    return Message(RecordDict({"ack": ConfigRecord({"client": client, "cases": len(p)})}), reply_to=message)


@app.train()
def train(message: Message, context: Context) -> Message:
    config = message.content["config"]
    seed, client = int(config["seed"]), client_index(context)
    device = select_device(str(config["device"]))
    torch.set_num_threads(1)
    torch.manual_seed(seed + int(config["server-round"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = CollaborativeModel(3, int(config["budget"])).to(device)
    model.load_state_dict(message.content["arrays"].to_torch_state_dict())
    initial = state_cpu(model)
    initial = {key: value.clone() for key, value in initial.items()}
    local = local_data(seed, client)
    p = torch.as_tensor(
        np.load(directory(seed) / f"client_{client}" / "training_predictions.npy"), device=device
    )
    # The local file deliberately stores only training labels; no val/test labels exist here.
    y = torch.as_tensor(local["train_y"], device=device)
    origin = torch.full((len(y),), client, dtype=torch.long, device=device)
    available = torch.ones(len(y), 3, dtype=torch.bool, device=device)
    domains = available.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    generator = torch.Generator().manual_seed(seed + int(config["server-round"]))
    amp = bool(config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for _ in range(int(config["epochs"])):
        for batch in torch.randperm(len(y), generator=generator).split(64):
            ids = batch.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                # Router loss probabilities/BCE remain FP32 for numerical stability.
                loss = model.expected_loss(
                    p[ids].float(), y[ids], origin[ids], domains[ids], available[ids], float(config["cost"])
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
    synchronize(device)
    final = state_cpu(model)
    delta = lambda prefix: (
        sum(float(((final[key] - initial[key]) ** 2).sum()) for key in final if key.startswith(prefix)) ** 0.5
    )
    report = {
        "client": client,
        "round": int(config["server-round"]),
        "num-examples": len(y),
        "router-change": delta("router."),
        "fusion-change": delta("fusion."),
        "loss": float(loss.detach().cpu()),
        "runtime-seconds": time.perf_counter() - started,
        "peak-vram-bytes": peak_memory(device),
        "device": str(device),
    }
    (directory(seed) / f"client_{client}" / f"round_{report['round']}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"M2 shared update {report}", flush=True)
    metrics = {key: value for key, value in report.items() if key != "device"}
    return Message(
        RecordDict({"arrays": ArrayRecord(final), "metrics": MetricRecord(metrics)}), reply_to=message
    )
