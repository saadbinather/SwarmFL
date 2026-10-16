"""Real Flower Grid relay. The broker never opens a specialist checkpoint/feature file."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.serverapp import Grid

from .protocol import validate_reply


class ConsultationBroker:
    def __init__(self, grid: Grid, nodes: dict[int, int], trace: Path, seed: int) -> None:
        self.grid, self.nodes, self.trace, self.seed = grid, nodes, trace, seed
        self.events: list[dict[str, Any]] = []
        self.domain_available = {client: True for client in nodes}

    def exchange(self, client: int, action: str, content: RecordDict) -> RecordDict:
        message = Message(
            content=content,
            message_type=f"query.{action}",
            dst_node_id=self.nodes[client],
            group_id="consultations",
        )
        replies = list(self.grid.send_and_receive([message], timeout=120))
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError(f"Client {client} failed query.{action}")
        if replies[0].metadata.src_node_id != self.nodes[client]:
            raise ValueError("Reply came from incorrect Flower node")
        return replies[0].content

    def predict(
        self, origin: int, destination: int, patients: list[int], mode: str, phase: str
    ) -> dict[str, Any]:
        token = uuid.uuid4().hex
        request = {
            "request_ids": [f"{token}-{i}" for i in range(len(patients))],
            "patient_ids": [f"case-{patient:04d}" for patient in patients],
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "seed": self.seed,
            "phase": phase,
        }
        started = time.perf_counter()
        try:
            if destination not in self.nodes:
                raise RuntimeError("Specialist unavailable")
            response = dict(
                self.exchange(destination, "predict", RecordDict({"request": ConfigRecord(request)}))[
                    "response"
                ]
            )
            validate_reply(request, response)
        except RuntimeError as error:
            response = {key: request[key] for key in ("request_ids", "patient_ids", "origin", "destination")}
            response.update(
                status=["unavailable"] * len(patients),
                probabilities=[0.5] * len(patients),
                confidence=[0.0] * len(patients),
                error=str(error),
            )
        elapsed = time.perf_counter() - started
        payload_bytes = len(json.dumps(request).encode()) + len(json.dumps(response).encode())
        with self.trace.open("a", encoding="utf-8") as file:
            for i, patient in enumerate(patients):
                event = {
                    "request_id": request["request_ids"][i],
                    "patient_id": request["patient_ids"][i],
                    "origin": origin,
                    "destination": destination,
                    "phase": phase,
                    "mode": mode,
                    "status": response["status"][i],
                    "prediction": response["probabilities"][i],
                    "confidence": response["confidence"][i],
                    "batch_seconds": elapsed,
                    "batch_size": len(patients),
                    "batch_payload_bytes": payload_bytes,
                    "consultation": origin != destination,
                }
                self.events.append(event)
                file.write(json.dumps(event) + "\n")
        return response
