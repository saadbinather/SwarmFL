"""M5 Step 1 numerical diagnostics; no production model or defaults are changed."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import brier_score_loss

from ..devices import peak_memory, synchronize
from ..evaluate import metrics
from ..evaluation.collaboration import calibration_error
from ..fusion import CollaborativeModel


def snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return an independent CPU checkpoint, including immutable action buffers."""
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def vector(model: torch.nn.Module, gradients: bool = False, prefix: str = "") -> torch.Tensor:
    """Flatten parameters or zero-filled gradients in stable named order."""
    return torch.cat(
        [
            (torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten().cpu()
            if gradients
            else p.detach().flatten().cpu()
            for name, p in model.named_parameters()
            if name.startswith(prefix)
        ]
    )


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    """Cosine, undefined for a zero vector rather than fabricated as zero."""
    denominator = left.norm() * right.norm()
    return float(left.dot(right) / denominator) if denominator > 0 else None


def fit(
    state: dict[str, torch.Tensor],
    p: np.ndarray,
    y: np.ndarray,
    origins: list[int],
    seed: int,
    round_id: int,
    variant: str,
    device: torch.device,
    cost: float = 0.02,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """M2 local schedule, or interleaved equal-step centralized reference.

    Frozen components keep their common random initialization. Each origin sees
    identical aligned training rows. Only caller-supplied training labels are used.
    """
    start = time.perf_counter()
    model = CollaborativeModel().to(device)
    model.load_state_dict(state)
    before = {prefix: vector(model, prefix=prefix) for prefix in ("router.", "fusion.")}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            not (variant == "frozen_router" and name.startswith("router."))
            and not (variant == "frozen_fusion" and name.startswith("fusion."))
        )
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.01)
    probabilities = torch.as_tensor(p, device=device, dtype=torch.float32)
    labels = torch.as_tensor(y, device=device, dtype=torch.float32)
    available = torch.ones(len(y), 3, device=device, dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed + round_id)
    norms: dict[str, list[float]] = {"router.": [], "fusion.": []}
    for _ in range(10):
        for batch in torch.randperm(len(y), generator=generator).split(64):
            ids = batch.to(device)
            if variant == "central_pooled":
                optimizer.zero_grad(set_to_none=True)
                loss = sum(
                    model.expected_loss(
                        probabilities[ids],
                        labels[ids],
                        torch.full((len(ids),), origin, device=device, dtype=torch.long),
                        available[ids].float(),
                        available[ids],
                        cost,
                    )
                    for origin in origins
                ) / len(origins)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite centralized diagnostic loss")
                loss.backward()
                for prefix, values in norms.items():
                    values.append(float(vector(model, True, prefix).norm()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                continue
            for origin in origins:
                optimizer.zero_grad(set_to_none=True)
                loss = model.expected_loss(
                    probabilities[ids],
                    labels[ids],
                    torch.full((len(ids),), origin, device=device, dtype=torch.long),
                    available[ids].float(),
                    available[ids],
                    cost,
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite diagnostic training loss")
                loss.backward()
                for prefix, values in norms.items():
                    values.append(float(vector(model, True, prefix).norm()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
    synchronize(device)
    return snapshot(model), {
        "loss": float(loss.detach()),
        "runtime_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": peak_memory(device),
        "steps": len(norms["router."]),
        "drift": {prefix: float((vector(model, prefix=prefix) - before[prefix]).norm()) for prefix in norms},
        "mean_gradient_norm": {prefix: float(np.mean(values)) for prefix, values in norms.items()},
    }


def prediction_table(model: CollaborativeModel, p: torch.Tensor, origin: int) -> tuple[torch.Tensor, ...]:
    """Enumerate masked outcomes for diagnosis; router sees only origin context."""
    o = torch.full((len(p),), origin, dtype=torch.long, device=p.device)
    available = torch.ones_like(p, dtype=torch.bool)
    own = p[:, origin]
    scores = model.router(o, own, 2 * (own - 0.5).abs(), available.float(), available)
    identity = torch.nn.functional.one_hot(o, 3).bool()
    masks = model.router.actions[None].bool() | identity[:, None]
    outputs = torch.stack(
        [
            torch.sigmoid(model.fusion(p, 2 * (p - 0.5).abs(), masks[:, action], o))
            for action in range(len(model.router.actions))
        ],
        dim=1,
    )
    means = (p[:, None] * masks).sum(-1) / masks.sum(-1)
    return scores, outputs, means


def loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Finite per-patient binary log loss."""
    safe = np.clip(p.astype(float), 1e-6, 1 - 1e-6)
    return -(y * np.log(safe) + (1 - y) * np.log1p(-safe))


def describe(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Classification, equal-width calibration and Brier diagnostics."""
    return metrics(y, p) | {
        "ece": calibration_error(y, p),
        "brier": float(brier_score_loss(y, p)),
    }


@torch.no_grad()
def evaluate(model: CollaborativeModel, probabilities: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Paired policy/fusion comparisons on a fixed train or validation cohort.

    All values are acquired previously via Flower for full-information diagnosis.
    Controls never select using peer values or labels. Exact-call controls reuse
    only the learned call/no-call mask; arithmetic means use identical evidence.
    """
    p = torch.as_tensor(probabilities, device=next(model.parameters()).device, dtype=torch.float32)
    result = {}
    for origin in range(3):
        scores, outputs, means = prediction_table(model, p, origin)
        q = scores.softmax(-1).cpu().numpy()
        out, mean = outputs.cpu().numpy(), means.cpu().numpy()
        learned = q.argmax(-1)
        rng = np.random.default_rng(seed + origin)
        peers = np.array([i + 1 for i in range(3) if i != origin])
        random = rng.choice(peers, size=len(y))
        cyclic = np.full(len(y), (origin + 1) % 3 + 1)
        policies = {
            "learned": learned,
            "stop": np.zeros(len(y), dtype=int),
            "random_max": random,
            "cyclic_max": cyclic,
            "random_matched": np.where(learned > 0, random, 0),
            "cyclic_matched": np.where(learned > 0, cyclic, 0),
        }
        rows = np.arange(len(y))
        details = {}
        for policy, actions in policies.items():
            selected_loss = loss_rows(y, out[rows, actions])
            stopped_loss = loss_rows(y, out[:, 0])
            call = actions > 0
            details[policy] = {
                "learned_fusion": describe(y, out[rows, actions]),
                "mean_fusion": describe(y, mean[rows, actions]),
                "calls": float(call.mean()),
                "action_frequencies": np.bincount(actions, minlength=4).tolist(),
                "harm_fraction_of_calls": float((selected_loss[call] >= stopped_loss[call]).mean())
                if call.any()
                else None,
                "mean_call_bce_delta": float((selected_loss[call] - stopped_loss[call]).mean())
                if call.any()
                else None,
            }
        action_losses = np.stack([loss_rows(y, out[:, a]) for a in range(4)], axis=1)
        expected = (q * action_losses).sum(-1)
        valid = np.isfinite(scores.cpu().numpy())
        penalized = np.where(valid, action_losses + 0.02 * (np.arange(4) > 0), np.inf)
        result[str(origin)] = {
            "policies": details,
            "soft_expected_bce": float(expected.mean()),
            "hard_bce_minus_soft": float((action_losses[rows, learned] - expected).mean()),
            "soft_expected_calls": float((1 - q[:, 0]).mean()),
            "mean_entropy": float(-(q * np.log(q.clip(1e-30))).sum(-1).mean()),
            "label_oracle_cost_regret": float((penalized[rows, learned] - penalized.min(-1)).mean()),
            "beneficial_call_fraction_at_cost": {
                str(cost): float(
                    ((action_losses[:, 0, None] - action_losses[:, peers]).max(-1) > cost).mean()
                )
                for cost in (0, 0.02, 0.1, 0.5)
            },
            "raw_origin": describe(y, probabilities[:, origin]),
        }
    return result


def gradient_audit(model: CollaborativeModel, p: np.ndarray, y: np.ndarray) -> dict:
    """Compare simultaneous origin gradients and policy-weighted fusion gradients."""
    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    p_t, y_t = torch.tensor(p, device=device), torch.tensor(y, device=device)
    available = torch.ones_like(p_t, dtype=torch.bool)
    gradients = {prefix: [] for prefix in ("router.", "fusion.")}
    weighting = []
    for origin in range(3):
        o = torch.full((len(y),), origin, dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        model.expected_loss(p_t, y_t, o, available.float(), available).backward()
        for prefix, values in gradients.items():
            values.append(vector(model, True, prefix))
        weighted = vector(model, True, "fusion.")
        model.zero_grad(set_to_none=True)
        _, outcomes, _ = prediction_table(model, p_t, origin)
        valid = [a for a in range(4) if a != origin + 1]
        uniform_loss = sum(
            torch.nn.functional.binary_cross_entropy(outcomes[:, a], y_t) for a in valid
        ) / len(valid)
        uniform_loss.backward()
        weighting.append(cosine(weighted, vector(model, True, "fusion.")))
    return {
        "origin_gradient_cosines": {
            prefix: [cosine(v[0], v[1]), cosine(v[0], v[2]), cosine(v[1], v[2])]
            for prefix, v in gradients.items()
        },
        "fusion_policy_vs_uniform_cosines": weighting,
    }
