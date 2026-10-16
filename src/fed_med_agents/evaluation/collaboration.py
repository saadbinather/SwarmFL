"""Policies route before obtaining candidate evidence through Flower."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ..collaboration import ConsultationBroker
from ..devices import synchronize
from ..evaluate import metrics
from ..fusion import CollaborativeModel


def policy_predictions(
    model: CollaborativeModel,
    broker: ConsultationBroker,
    patients: list[int],
    origin_index: int,
    policy: str,
    phase: str,
    device: torch.device,
    seed: int,
    confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    """Return predictions, masks, scores, and end-to-end batched latency.

    No peer predictions are arguments to routing. Only after selection are
    destination specialists queried. The server is the designated fusion component.
    """
    started = time.perf_counter()
    own = broker.predict(origin_index, origin_index, patients, "inference", f"{phase}/{policy}")
    count, clients = len(patients), model.router.clients
    probability = torch.tensor(own["probabilities"], device=device)
    origin = torch.full((count,), origin_index, device=device, dtype=torch.long)
    available = torch.tensor([[i in broker.nodes for i in range(clients)]] * count, device=device)
    domains = torch.tensor(
        [[broker.domain_available.get(i, False) for i in range(clients)]] * count,
        device=device,
        dtype=torch.float32,
    )
    available &= domains.bool()
    selected = torch.zeros(count, clients, dtype=torch.bool, device=device)
    scores = None
    model.eval()
    with torch.no_grad():
        if policy == "learned":
            selected, scores = model.router.select(
                origin, probability, 2 * (probability - 0.5).abs(), domains, available
            )
        elif policy == "random":
            candidates = [i for i in range(clients) if i != origin_index and i in broker.nodes]
            rng = np.random.default_rng(seed + origin_index)
            for row in range(count):
                if candidates:
                    selected[
                        row,
                        rng.choice(
                            candidates,
                            size=min(model.router.max_consultations, len(candidates)),
                            replace=False,
                        ),
                    ] = True
        elif policy == "confidence":
            candidates = [i for i in range(clients) if i != origin_index and i in broker.nodes]
            # Public fixed cyclic destination; no candidate probability is inspected.
            candidates.sort(key=lambda index: (index - origin_index) % clients)
            if candidates and model.router.max_consultations:
                selected[:, candidates[0]] = 2 * (probability - 0.5).abs() < confidence_threshold
        elif policy in {"all", "fixed_mean"}:
            selected = available.clone()
            selected[:, origin_index] = False
        elif policy != "none":
            raise ValueError(f"Unknown policy {policy}")
        origin_ok = torch.tensor([status == "ok" for status in own["status"]], device=device)
        selected &= origin_ok[:, None]
        values = torch.full((count, clients), 0.5, device=device)
        evidence = torch.zeros(count, clients, dtype=torch.bool, device=device)
        values[:, origin_index] = probability
        evidence[:, origin_index] = torch.tensor([status == "ok" for status in own["status"]], device=device)
        selection_cpu = selected.cpu().numpy()
        for destination in range(clients):
            rows = np.flatnonzero(selection_cpu[:, destination])
            if not len(rows):
                continue
            response = broker.predict(
                origin_index, destination, [patients[row] for row in rows], "inference", f"{phase}/{policy}"
            )
            values[rows, destination] = torch.tensor(response["probabilities"], device=device)
            evidence[rows, destination] = torch.tensor(
                [status == "ok" for status in response["status"]], device=device
            )
        if policy in {"all", "fixed_mean"}:
            prediction = (values * evidence).sum(-1) / evidence.sum(-1).clamp(min=1)
        elif policy == "none":
            prediction = probability
        else:
            prediction = torch.sigmoid(model.fusion(values, 2 * (values - 0.5).abs(), evidence, origin))
        prediction = torch.where(evidence[:, origin_index], prediction, torch.full_like(prediction, 0.5))
    synchronize(device)
    elapsed = time.perf_counter() - started
    consultation_counts = selected.sum(-1).cpu().numpy()
    evidence_cpu = evidence.cpu().numpy()
    statuses = [
        "missing_origin"
        if not evidence_cpu[row, origin_index]
        else "partial_evidence"
        if np.any(selection_cpu[row] & ~evidence_cpu[row])
        else "ok"
        for row in range(count)
    ]
    return {
        "probabilities": prediction.cpu().numpy(),
        "average_consultations": float(consultation_counts.mean()),
        "collaboration_percent": float((consultation_counts > 0).mean() * 100),
        "batch_inference_seconds": elapsed,
        "amortized_latency_ms": elapsed * 1000 / count,
        "selected": selection_cpu,
        "own_probabilities": probability.cpu().numpy(),
        "scores": scores.cpu().numpy() if scores is not None else None,
        "prediction_status": statuses,
        "evidence_mask": evidence_cpu,
    }


def evaluate_policy(result: dict[str, Any], labels: np.ndarray) -> dict[str, float]:
    return metrics(labels, result["probabilities"]) | {
        key: result[key]
        for key in (
            "average_consultations",
            "collaboration_percent",
            "batch_inference_seconds",
            "amortized_latency_ms",
        )
    }


def calibration_error(labels: np.ndarray, predictions: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        mask = (predictions >= lower) & (predictions < lower + 1 / bins)
        if mask.any():
            result += mask.mean() * abs(predictions[mask].mean() - labels[mask].mean())
    return float(result)
