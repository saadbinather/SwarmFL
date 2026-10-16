"""Experimental pre-query value models; labels/peer evidence are not inference inputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from torch import nn

from .routing_diagnostics import describe, loss_rows

FEATURE_SETS = ("base", "uncertainty", "reliability", "history", "enriched")
COSTS = (0.02, 0.05, 0.10)


def features(
    own: np.ndarray,
    origin: int,
    peer: int,
    reliability: np.ndarray,
    history: np.ndarray,
    group: str = "enriched",
    candidate_capability: float = 1.0,
) -> np.ndarray:
    """Build node-local features from own probability and fitting-only public statistics.

    No feature values, labels, patient IDs or unqueried probabilities are accepted.
    Capability width is fixed metadata, not a clinical risk summary.
    """
    if origin == peer or group not in FEATURE_SETS:
        raise ValueError("Invalid candidate/feature set")
    n = len(own)
    identity = np.broadcast_to(np.r_[np.eye(3)[origin], np.eye(3)[peer]], (n, 6))
    confidence = 2 * np.abs(own - 0.5)
    base = np.column_stack(
        [identity, own, confidence, np.full(n, candidate_capability), np.ones(n), np.ones(n)]
    )
    p = own.clip(1e-6, 1 - 1e-6)
    additions = {
        "uncertainty": np.column_stack(
            [np.abs(own - 0.5), -p * np.log(p) - (1 - p) * np.log(1 - p), p * (1 - p)]
        ),
        "reliability": np.concatenate([reliability[:, origin], reliability[:, peer]], axis=1),
        "history": history[:, origin, peer, None],
    }
    groups = additions if group == "enriched" else ([group] if group != "base" else [])
    return np.column_stack([base, *(additions[g] for g in groups)]).astype(np.float32)


def choose(
    benefit: np.ndarray, origin: int, cost: float, availability: np.ndarray | None = None, budget: int = 1
) -> np.ndarray:
    """Choose the strictly positive best net gain, else STOP; enforce eligibility."""
    if budget not in (0, 1) or cost < 0 or not np.isfinite(cost):
        raise ValueError("Invalid budget/cost")
    if benefit.ndim != 2 or benefit.shape[1] != 3 or not np.isfinite(benefit).all():
        raise ValueError("Expected finite candidate gains")
    scores = benefit.copy() - cost
    scores[:, origin] = -np.inf
    if availability is not None:
        if availability.shape != scores.shape:
            raise ValueError("Invalid availability")
        scores[~availability.astype(bool)] = -np.inf
    actions = scores.argmax(1) + 1
    actions[(scores.max(1) <= 0) | (budget == 0)] = 0
    return actions


def value_network(width: int, kind: str = "mlp") -> nn.Module:
    """Compatible shared router only, with no specialist/fusion parameters."""
    if kind == "linear":
        return nn.Sequential(nn.Linear(width, 1))
    if kind != "mlp":
        raise ValueError("Unknown routing architecture")
    return nn.Sequential(nn.Linear(width, 16), nn.Tanh(), nn.Linear(16, 1))


def fit_network(model: nn.Module, x: np.ndarray, target: np.ndarray, steps: int = 80) -> dict:
    """Fixed full-batch Adam regression, returning genuine gradient evidence."""
    device = next(model.parameters()).device
    inputs, labels = torch.tensor(x, device=device), torch.tensor(target, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    norms = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((model(inputs).flatten() - labels) ** 2).mean()
        loss.backward()
        norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), 5)))
        optimizer.step()
    return {"steps": steps, "loss": float(loss.detach()), "gradient_norm": float(np.mean(norms))}


def fit_regressor(kind: str, x: np.ndarray, target: np.ndarray, seed: int) -> Any:
    """Fit prescribed small regressors without outcome-dependent hyperparameters."""
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    elif kind == "tree":
        model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=42)
    else:
        torch.manual_seed(seed)
        model = value_network(x.shape[1], kind)
        fit_network(model, x, target)
        return model
    model.fit(x, target)
    return model


def predict(model: Any, x: np.ndarray) -> np.ndarray:
    """Predict gross benefits without outcomes or other specialists' predictions."""
    if isinstance(model, nn.Module):
        with torch.no_grad():
            return model(torch.tensor(x, device=next(model.parameters()).device)).flatten().cpu().numpy()
    return model.predict(x)


def summarize(
    y: np.ndarray,
    risks: np.ndarray,
    actions: np.ndarray,
    origin: int,
    cost: float,
    estimates: np.ndarray | None = None,
    training_mean: np.ndarray | None = None,
) -> dict:
    """Retrospective assessment only; oracle labels never enter action selection."""
    if np.any(actions == origin + 1):
        raise ValueError("Self consultation")
    losses = np.column_stack([loss_rows(y, risks[:, a]) for a in range(4)])
    rows, call = np.arange(len(y)), actions > 0
    loss = losses[rows, actions]
    valid = [0, *[i + 1 for i in range(3) if i != origin]]
    objectives = losses + cost * np.array([0, 1, 1, 1])
    oracle = objectives[:, valid].min(1)
    result = {
        **describe(y, risks[rows, actions]),
        "objective": float(loss.mean() + cost * call.mean()),
        "calls": float(call.mean()),
        "stop_fraction": float((~call).mean()),
        "harm": float((loss[call] >= losses[call, 0]).mean()) if call.any() else None,
        "oracle_objective": float(oracle.mean()),
        "oracle_regret": float((loss + cost * call - oracle).mean()),
        "oracle_stop_fraction": float((objectives[:, valid].argmin(1) == 0).mean()),
        "action_frequencies": np.bincount(actions, minlength=4).tolist(),
        "actions": actions.tolist(),
        "patient_dependent": len(np.unique(actions)) > 1,
        "useful_stop_fraction": float((objectives[~call, 0] <= objectives[~call][:, valid[1:]].min(1)).mean())
        if (~call).any()
        else None,
    }
    if estimates is not None:
        peers = [i for i in range(3) if i != origin]
        truth = losses[:, :1] - losses[:, 1:]
        actual, estimated = truth[:, peers].flatten(), estimates[:, peers].flatten()
        error = float(np.mean((actual - estimated) ** 2))
        null_error = float(
            np.mean((actual - np.tile(training_mean[origin, peers], (len(y), 1)).flatten()) ** 2)
        )
        result["utility"] = {
            "mse": error,
            "training_mean_mse": null_error,
            "r2_vs_training_mean": 1 - error / null_error if null_error > 0 else None,
            "correlation": float(np.corrcoef(actual, estimated)[0, 1]) if estimated.std() > 1e-8 else None,
            "positive_net_accuracy": float(((actual > cost) == (estimated > cost)).mean()),
            "gain_std": float(actual.std()),
        }
    return result
