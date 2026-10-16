"""Structured patient-aligned consultation payloads; no features or private weights."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def answer(
    request: dict[str, Any], destination: int, ids: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    """Validate destination and associate each returned prediction with its exact request."""
    if int(request["destination"]) != destination:
        raise ValueError("Consultation sent to wrong specialist")
    if len(request["request_ids"]) != len(request["patient_ids"]):
        raise ValueError("Request identifiers are not aligned")
    lookup = {f"case-{int(patient):04d}": float(p) for patient, p in zip(ids, probabilities, strict=True)}
    values = [lookup.get(patient, 0.5) for patient in request["patient_ids"]]
    return {
        "request_ids": request["request_ids"],
        "patient_ids": request["patient_ids"],
        "origin": int(request["origin"]),
        "destination": destination,
        "status": ["ok" if patient in lookup else "missing_record" for patient in request["patient_ids"]],
        "probabilities": values,
        "confidence": [2 * abs(p - 0.5) for p in values],
        "specialist_seconds": float(request.get("specialist_seconds", 0.0)),
        "responded_at": time.time(),
    }


def validate_reply(request: dict[str, Any], reply: dict[str, Any]) -> None:
    for key in ("request_ids", "patient_ids", "origin", "destination"):
        if request[key] != reply[key]:
            raise ValueError(f"Misassociated consultation reply: {key}")
    count = len(request["patient_ids"])
    if any(len(reply[key]) != count for key in ("status", "probabilities", "confidence")):
        raise ValueError("Incomplete consultation reply")
    if any(not 0 <= p <= 1 for p in reply["probabilities"]):
        raise ValueError("Invalid consultation probability")
