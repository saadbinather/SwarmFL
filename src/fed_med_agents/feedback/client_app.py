"""Private local feedback supervision and compatible shared-only Flower updates."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ..devices import peak_memory, select_device, synchronize
from ..evaluate import metrics
from ..federated import client_app as specialist
from ..fusion import CollaborativeModel
from .events import FeedbackLedger, VerifiedOutcome, digest, sign, timestamp
from .learning import feedback_loss

app = ClientApp()
app.query("identity")(specialist.identity)
app.query("plan")(specialist.plan)
app.query("predict")(specialist.predict)


def configuration() -> dict:
    return json.loads(os.environ["FED_MED_M4_CONFIG"])


def ledger(client: int, policy: str = "adaptive") -> FeedbackLedger:
    if policy not in {"adaptive", "frozen", "offline"}:
        raise ValueError("Unknown policy ledger")
    config = configuration()
    local = specialist.local_data(config["seed"], client)
    credentials = json.loads((Path(config["output"]) / "credentials.json").read_text())[str(client)]
    return FeedbackLedger(
        Path(config["output"]) / f"client_{client}" / policy,
        client,
        {f"case-{i:04d}" for i in local["train"]},
        **credentials,
    )


def ack(message: Message, values: dict) -> Message:
    return Message(RecordDict({"ack": ConfigRecord(values)}), reply_to=message)


@app.query("feedback_record")
def record(message: Message, context: Context) -> Message:
    client = specialist.client_index(context)
    payload = message.content["config"]
    store = ledger(client, str(payload["policy"]))
    envelope = json.loads(payload["json"])
    store.prediction(envelope)
    return ack(message, {"client": client, "recorded": True})


@app.query("feedback_reveal")
def reveal(message: Message, context: Context) -> Message:
    """Simulation authority reveals a real training label only after stored decision.

    Server receives an acknowledgement, not this label. A deployment must replace
    this trusted local adapter with an authenticated verified-diagnosis source.
    """
    client = specialist.client_index(context)
    payload = message.content["config"]
    store = ledger(client, str(payload["policy"]))
    event = store.predictions[str(payload["request_id"])]
    local = specialist.local_data(configuration()["seed"], client)
    patient = int(event.patient_id.removeprefix("case-"))
    positions = np.flatnonzero(local["train"] == patient)
    if len(positions) != 1:
        raise ValueError("Outcome source excludes non-training patients")
    outcome = VerifiedOutcome(
        request_id=event.request_id,
        patient_id=event.patient_id,
        prediction_digest=digest(event),
        label=int(local["train_y"][positions[0]]),
        authority=store.authority,
        received_episode=int(payload["episode"]),
        verified_timestamp=timestamp(),
    )
    store.outcome(sign(outcome, store.outcome_key))
    return ack(message, {"client": client, "verified": True, "labeled_count": len(store.outcomes)})


@app.query("feedback_summary")
def summary(message: Message, context: Context) -> Message:
    client = specialist.client_index(context)
    config = message.content["config"]
    rows = ledger(client, str(config["policy"])).replay(int(config["episode"]), window=100000)
    if not rows:
        return ack(message, {"count": 0})
    events, outcomes = zip(*rows)
    y = np.array([item.label for item in outcomes])
    p = np.array([item.final_risk for item in events])
    result = metrics(y, p) if len(np.unique(y)) == 2 else {"auroc": None}
    bce = lambda values: (
        -(y * np.log(np.clip(values, 1e-6, 1 - 1e-6)) + (1 - y) * np.log(np.clip(1 - values, 1e-6, 1 - 1e-6)))
    )
    origin_loss = bce(np.array([item.origin_only_risk for item in events]))
    calls = np.array([len(item.consulted) for item in events])
    result.update(
        count=len(y),
        tp=int(((p >= 0.5) & (y == 1)).sum()),
        fp=int(((p >= 0.5) & (y == 0)).sum()),
        fn=int(((p < 0.5) & (y == 1)).sum()),
        correct=int(((p >= 0.5) == y).sum()),
        bce_sum=float(bce(p).sum()),
        calls=int(calls.sum()),
        unnecessary=int(((calls > 0) & (origin_loss <= bce(p))).sum()),
    )
    return ack(message, {"json": json.dumps(result)})


@app.train()
def train(message: Message, context: Context) -> Message:
    config = message.content["config"]
    client = specialist.client_index(context)
    device = select_device(str(config["device"]))
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    rows = ledger(client).replay(int(config["cutoff"]), int(config["window"]))
    if len(rows) < int(config["min_feedback"]):
        raise ValueError("Insufficient authorized feedback: server must skip update")
    model = CollaborativeModel().to(device)
    model.load_state_dict(message.content["arrays"].to_torch_state_dict())
    before = {key: value.clone() for key, value in specialist.state_cpu(model).items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    for _ in range(int(config["epochs"])):
        optimizer.zero_grad()
        loss, audit = feedback_loss(model, rows, device)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite feedback loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
        optimizer.step()
    synchronize(device)
    state = specialist.state_cpu(model)
    delta = lambda prefix: (
        sum(float(((state[k] - before[k]) ** 2).sum()) for k in state if k.startswith(prefix)) ** 0.5
    )
    report = {
        "client": client,
        "num-examples": len(rows),
        "round": int(config["update_id"]),
        "loss": float(loss.detach()),
        "router-change": delta("router."),
        "fusion-change": delta("fusion."),
        "gradient_norm": norm,
        "runtime_seconds": time.perf_counter() - start,
        "peak_vram_bytes": peak_memory(device),
        **audit,
    }
    target = Path(configuration()["output"]) / f"client_{client}" / f"update_{config['update_id']}.json"
    target.write_text(
        json.dumps(
            report
            | {
                "device": str(device),
                "cutoff": int(config["cutoff"]),
                "request_ids": [event.request_id for event, _ in rows],
            },
            indent=2,
        )
    )
    torch.save(state, target.with_suffix(".pt"))
    print(f"M4 local feedback update {report}", flush=True)
    # No feedback events, labels, private rows or specialist state in training replies.
    return Message(
        RecordDict({"arrays": ArrayRecord(state), "metrics": MetricRecord(report)}), reply_to=message
    )
