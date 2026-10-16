"""Fixed three-client experimental models, objectives and pre-query routing."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..fusion import CollaborativeModel
from .routing_diagnostics import describe, loss_rows

CANDIDATES = {
    "avg10": {"epochs": 10},
    "avg1": {"epochs": 1},
    "avg2": {"epochs": 2},
    "avg5": {"epochs": 5},
    "persistent2": {"epochs": 2, "persistent": True},
    "prox2": {"epochs": 2, "mu": 0.01},
    "cost10": {"epochs": 10, "cost": 0.1},
    "cost30": {"epochs": 10, "cost": 0.3},
    "logistic": {"epochs": 10, "kind": "logistic"},
    "small": {"epochs": 10, "kind": "small"},
    "value": {"epochs": 2, "mode": "value"},
}


def select_inner(results: dict) -> dict:
    """Select fixed candidates from inner selection only; outer metrics are ignored."""
    selected = {}
    for fold in range(3):
        inner = results[f"fold{fold}_inner"]
        scores = {
            name: float(np.mean([r["primary"] for r in inner[name]["assessment"].values()]))
            for name in [*CANDIDATES, "utility_bins"]
        }
        selected[str(fold)] = {"candidate": min(scores, key=scores.get), "inner_objectives": scores}
    return selected


class LogisticFusion(nn.Module):
    """Trainable masked logit pooling, no origin identity or confidence features."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 1)

    def forward(
        self, p: torch.Tensor, confidence: torch.Tensor, mask: torch.Tensor, origin: torch.Tensor
    ) -> torch.Tensor:
        safe = torch.where(mask, p, torch.full_like(p, 0.5)).clamp(1e-5, 1 - 1e-5)
        return self.linear(torch.cat([torch.logit(safe), mask.float()], -1)).squeeze(-1)


def make_model(kind: str = "original") -> CollaborativeModel:
    """Original compatible graph with an explicitly separate fixed fusion variant."""
    model = CollaborativeModel()
    if kind == "logistic":
        model.fusion = LogisticFusion()
    elif kind == "small":
        model.fusion.network = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 1))
    elif kind != "original":
        raise ValueError("Unknown experimental fusion")
    return model


def action_outputs(model: CollaborativeModel, p: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
    """Masked outcomes for exhaustive OOF training or retrospective diagnosis only."""
    identity = nn.functional.one_hot(origins, 3).bool()
    return torch.stack(
        [
            model.fusion(p, 2 * (p - 0.5).abs(), identity | action.bool(), origins)
            for action in model.router.actions
        ],
        -1,
    )


def query_scores(
    model: CollaborativeModel,
    origins: torch.Tensor,
    own: torch.Tensor,
    available: torch.Tensor,
    value: bool = False,
    cost: float = 0.02,
) -> torch.Tensor:
    """Only pre-consultation evidence is accepted; self/unavailable peers stay masked."""
    scores = model.router(origins, own, 2 * (own - 0.5).abs(), available.float(), available)
    if value:
        scores = scores - cost
        scores = torch.cat([torch.zeros_like(scores[:, :1]), scores[:, 1:]], -1)
    return scores


def uniform_loss(
    model: CollaborativeModel, p: torch.Tensor, y: torch.Tensor, origins: torch.Tensor
) -> torch.Tensor:
    logits = action_outputs(model, p, origins)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid.scatter_(1, origins[:, None] + 1, False)
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, y[:, None].expand_as(logits), reduction="none"
    )
    return losses[valid].mean()


def value_targets(
    model: CollaborativeModel, p: torch.Tensor, y: torch.Tensor, origins: torch.Tensor
) -> torch.Tensor:
    """Training-only supervised gains; assessment labels never create targets."""
    with torch.no_grad():
        logits = action_outputs(model, p, origins)
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits, y[:, None].expand_as(logits), reduction="none"
        )
        return (losses[:, :1] - losses).clamp(-3, 3)


def objective(
    model: CollaborativeModel,
    p: torch.Tensor,
    y: torch.Tensor,
    origins: torch.Tensor,
    mode: str = "joint",
    cost: float = 0.02,
    targets: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "uniform":
        return uniform_loss(model, p, y, origins)
    available = torch.ones_like(p, dtype=torch.bool)
    if mode == "value":
        if targets is None:
            raise ValueError("Fitting-only value targets required")
        own = p.gather(1, origins[:, None]).flatten()
        scores = query_scores(model, origins, own, available)
        valid = torch.isfinite(scores)
        valid[:, 0] = False
        return ((scores[valid] - targets[valid]) ** 2).mean()
    return model.expected_loss(p, y, origins, available.float(), available, cost)


def fit_bins(model: CollaborativeModel, p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Four predefined confidence bins, trained only on fitting OOF evidence."""
    device = next(model.parameters()).device
    tensor = torch.tensor(p, device=device)
    table = np.zeros((3, 4, 4), dtype=np.float32)
    for origin in range(3):
        o = torch.full((len(y),), origin, device=device, dtype=torch.long)
        targets = value_targets(model, tensor, torch.tensor(y, device=device), o).cpu().numpy()
        bins = np.minimum((2 * np.abs(p[:, origin] - 0.5) * 4).astype(int), 3)
        for bucket in range(4):
            rows = bins == bucket
            table[origin, bucket] = targets[rows].mean(0) if rows.sum() >= 8 else targets.mean(0)
    return table


@torch.no_grad()
def evaluate(
    model: CollaborativeModel,
    p: np.ndarray,
    y: np.ndarray,
    mode: str = "joint",
    fixed_actions: list[list[int]] | None = None,
    bins: np.ndarray | None = None,
) -> dict:
    """All metrics on declared fitting/assessment data, never prospective oracle routing."""
    device = next(model.parameters()).device
    values = torch.tensor(p, device=device)
    result = {}
    for origin in range(3):
        o = torch.full((len(y),), origin, device=device, dtype=torch.long)
        scores = query_scores(
            model, o, values[:, origin], torch.ones_like(values, dtype=torch.bool), mode == "value"
        )
        q = scores.softmax(-1).cpu().numpy()
        actions = scores.argmax(-1).cpu().numpy()
        if mode == "stop":
            actions[:] = 0
        elif mode in {"random", "cyclic"}:
            peers = [i + 1 for i in range(3) if i != origin]
            actions = (
                np.random.default_rng(7700 + origin).choice(peers, len(y))
                if mode == "random"
                else np.full(len(y), (origin + 1) % 3 + 1)
            )
        elif mode == "bins":
            bucket = np.minimum((2 * np.abs(p[:, origin] - 0.5) * 4).astype(int), 3)
            benefits = bins[origin, bucket].copy() - 0.02
            benefits[:, 0] = 0
            benefits[:, origin + 1] = -np.inf
            actions = benefits.argmax(-1)
        if fixed_actions is not None:
            actions = np.asarray(fixed_actions[origin])
        if np.any(actions == origin + 1):
            raise ValueError("Self consultation")
        all_p = torch.sigmoid(action_outputs(model, values, o)).cpu().numpy()
        rows = np.arange(len(y))
        pred = all_p[rows, actions]
        call = actions > 0
        selected_loss, stop_loss = loss_rows(y, pred), loss_rows(y, all_p[:, 0])
        evidence = np.eye(3, dtype=bool)[np.full(len(y), origin)]
        for row, action in enumerate(actions):
            if action:
                evidence[row, action - 1] = True
        mean = (p * evidence).sum(-1) / evidence.sum(-1)
        action_losses = np.stack([loss_rows(y, all_p[:, a]) for a in range(4)], -1)
        result[str(origin)] = {
            **describe(y, pred),
            "mean_same_evidence": describe(y, mean),
            "stop_fraction": float((~call).mean()),
            "calls": float(call.mean()),
            "primary": float(selected_loss.mean() + 0.02 * call.mean()),
            "costs": {str(c): float(c * call.mean()) for c in (0.02, 0.1, 0.3)},
            "objectives": {str(c): float(selected_loss.mean() + c * call.mean()) for c in (0.02, 0.1, 0.3)},
            "harm": float((selected_loss[call] >= stop_loss[call]).mean()) if call.any() else None,
            "hard_minus_soft_bce": float((selected_loss - (q * action_losses).sum(-1)).mean()),
            "entropy": float(-(q * np.log(q.clip(1e-30))).sum(-1).mean()),
            "action_frequencies": np.bincount(actions, minlength=4).tolist(),
            "actions": actions.tolist(),
        }
    return result
