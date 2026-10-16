"""Registry-aware fixed, contextual and existing graph routing policies.

The caller constructs authorized origin-local context. Outcomes and unqueried
peer predictions are never arguments to route(). Supervised gains and observed
feedback use separate fitting entry points.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from ..collaboration.registry import ConsultationSession
from .graph import ContextualRouter

if TYPE_CHECKING:
    from ..feedback.events import PredictionEvent, VerifiedOutcome


@dataclass(frozen=True)
class RoutingDecision:
    """Auditable admission snapshot; consult() rechecks live registry state."""

    destination: str | None
    version: str
    eligible: tuple[str, ...]
    scores: dict[str, float]
    registry_generations: dict[str, int]


class RoutingPolicy(nn.Module):
    """Pluggable routing-only state, independent of classifier/fusion weights.

    Fixed/MLP outputs estimate gross BCE gains. Graph retains its existing action
    architecture and predicts action preferences, not calibrated utility values.
    Column order is an immutable checkpoint identity, never discovery order.
    """

    def __init__(self, kind: str, specialist_ids: tuple[str, ...], width: int = 63) -> None:
        super().__init__()
        if kind not in {"fixed", "mlp", "graph"} or len(set(specialist_ids)) != len(specialist_ids):
            raise ValueError("Invalid policy configuration")
        if len(specialist_ids) < 2 or width < 1:
            raise ValueError("Invalid policy dimensions")
        self.kind, self.specialist_ids, self.width = kind, tuple(specialist_ids), width
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.register_buffer("gains", torch.zeros(len(specialist_ids), len(specialist_ids)))
        self.network = (
            nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 1))
            if kind == "mlp"
            else ContextualRouter(len(specialist_ids), 1)
            if kind == "graph"
            else nn.Identity()
        )

    def config(self) -> dict:
        """Portable identity/architecture metadata consumed by M4 PolicyRegistry."""
        return {"kind": self.kind, "specialist_ids": list(self.specialist_ids), "width": self.width}

    def values(self, features: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
        """Estimate peer gains from validated local/public feature vectors only."""
        if features.shape[-2:] != (len(self.specialist_ids), self.width):
            raise ValueError("Incompatible routing context")
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite routing context")
        if self.kind == "graph":
            raise ValueError("Graph preferences are not utility estimates")
        if self.kind == "fixed":
            return self.gains[origins]
        return self.network((features - self.mean) / self.scale).squeeze(-1)

    def logits(
        self,
        features: torch.Tensor,
        origins: torch.Tensor,
        own: torch.Tensor,
        available: torch.Tensor,
        costs: torch.Tensor,
    ) -> torch.Tensor:
        """Common STOP/peer logits; unavailable/self actions always masked."""
        if costs.shape != available.shape or not torch.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("Invalid consultation costs")
        if self.kind == "graph":
            scores = self.network(origins, own, 2 * (own - 0.5).abs(), available.float(), available)
            scores = scores - torch.cat([torch.zeros_like(own[:, None]), costs / 0.1], -1)
        else:
            value = self.values(features, origins)
            scores = torch.cat([torch.zeros_like(own[:, None]), (value - costs) / 0.1], -1)
        permitted = available.bool().clone()
        permitted.scatter_(1, origins[:, None], False)
        return scores.masked_fill(
            torch.cat([torch.zeros_like(own[:, None], dtype=torch.bool), ~permitted], -1), -torch.inf
        )

    def fit(
        self,
        features: torch.Tensor,
        origins: torch.Tensor,
        gains: torch.Tensor,
        own: torch.Tensor,
        available: torch.Tensor,
        costs: torch.Tensor,
        *,
        seed: int,
        epochs: int | None = None,
    ) -> dict:
        """Full-information OOF supervised training, separate from observed feedback.

        Self targets are ignored. Scaler sees only fitting examples/nonself peers.
        Fixed means are also fitted only on supplied training observations.
        """
        peers = torch.arange(len(self.specialist_ids), device=origins.device)[None] != origins[:, None]
        with torch.no_grad():
            for origin in range(len(self.specialist_ids)):
                self.gains[origin] = gains[origins == origin].mean(0)
            selected = features[peers]
            self.mean.copy_(selected.mean(0))
            self.scale.copy_(selected.std(0, unbiased=False).clamp_min(1e-6))
        if self.kind == "fixed":
            return {"steps": 0, "loss": 0.0, "gradient_norm": 0.0}
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.003, weight_decay=0.01 if self.kind == "mlp" else 0
        )
        generator = torch.Generator().manual_seed(seed)
        epochs = epochs or (20 if self.kind == "mlp" else 15)
        steps, norm = 0, 0.0
        for _ in range(epochs):
            for batch in torch.randperm(len(origins), generator=generator).split(4096):
                b = batch.to(origins.device)
                optimizer.zero_grad()
                if self.kind == "mlp":
                    loss = ((self.values(features[b], origins[b]) - gains[b]) ** 2)[peers[b]].mean()
                else:
                    scores = self.logits(features[b], origins[b], own[b], available[b], costs[b])
                    objective = torch.cat([torch.zeros_like(own[b, None]), costs[b] - gains[b]], -1)
                    loss = (scores.softmax(-1) * objective).sum(-1).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite routing loss")
                loss.backward()
                norm = float(nn.utils.clip_grad_norm_(self.parameters(), 5))
                optimizer.step()
                steps += 1
        return {"steps": steps, "loss": float(loss.detach()), "gradient_norm": norm}

    def route(
        self,
        session: ConsultationSession,
        peer_contexts: Mapping[str, list[float]],
        own_probability: float,
        *,
        version: str,
    ) -> RoutingDecision:
        """Discover live eligible/affordable peers and choose STOP or one recipient."""
        if not 0 <= own_probability <= 1 or session.origin not in self.specialist_ids:
            raise ValueError("Invalid origin evidence")
        unknown = set(peer_contexts) - set(self.specialist_ids)
        if unknown:
            raise ValueError("Unknown context identity")
        entries = session.eligible()
        if any(e.specialist.specialist_id not in self.specialist_ids for e in entries):
            raise ValueError("Registry membership incompatible with frozen checkpoint")
        device = self.mean.device
        features = torch.zeros(1, len(self.specialist_ids), self.width, device=device)
        available = torch.zeros(1, len(self.specialist_ids), dtype=torch.bool, device=device)
        costs = torch.zeros_like(available, dtype=torch.float32)
        for entry in entries:
            name = entry.specialist.specialist_id
            index = self.specialist_ids.index(name)
            if self.kind == "mlp":
                features[0, index] = torch.tensor(peer_contexts[name], device=device)
            available[0, index] = True
            costs[0, index] = entry.specialist.consultation_cost
        with torch.no_grad():
            scores = self.logits(
                features,
                torch.tensor([self.specialist_ids.index(session.origin)], device=device),
                torch.tensor([own_probability], device=device),
                available,
                costs,
            )[0]
        action = int(scores.argmax())
        return RoutingDecision(
            self.specialist_ids[action - 1] if action else None,
            version,
            tuple(e.specialist.specialist_id for e in entries),
            {
                "STOP" if i == 0 else self.specialist_ids[i - 1]: float(v)
                for i, v in enumerate(scores)
                if torch.isfinite(v)
            },
            {e.specialist.specialist_id: e.generation for e in entries},
        )

    def save(self, path: Path) -> None:
        """Save an experimental standalone router; immutable versions use PolicyRegistry."""
        if not all(torch.isfinite(value).all() for value in self.state_dict().values()):
            raise ValueError("Nonfinite routing state")
        torch.save({"config": self.config(), "state": self.state_dict()}, path)

    def update(
        self,
        records: list[tuple[PredictionEvent, VerifiedOutcome]],
        contexts: dict,
        *,
        minimum: int = 32,
        epochs: int = 2,
    ) -> dict:
        """Update only routing weights from verified M4 ledger replay.

        Callers pass replay(), not unverified envelopes. Insufficient records or
        a fixed baseline retain the exact state. Approval is a separate step.
        """
        from ..feedback.learning import routing_feedback_loss

        if minimum < 1 or epochs < 1:
            raise ValueError("Positive feedback minimum/epochs required")
        if len(records) < minimum or self.kind == "fixed":
            return {
                "updated": False,
                "records": len(records),
                "reason": "insufficient feedback or fixed policy",
            }
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        device = self.mean.device
        for _ in range(epochs):
            optimizer.zero_grad()
            loss, log = routing_feedback_loss(self, records, contexts, device)
            loss.backward()
            log["gradient_norm"] = float(nn.utils.clip_grad_norm_(self.parameters(), 5))
            optimizer.step()
        return {"updated": True, **log}

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> RoutingPolicy:
        """Load compatible explicit identity/dimensions, never a production default."""
        saved = torch.load(path, map_location=device, weights_only=True)
        policy = cls(**saved["config"]).to(device)
        policy.load_state_dict(saved["state"], strict=True)
        if not all(torch.isfinite(value).all() for value in policy.state_dict().values()):
            raise ValueError("Nonfinite routing state")
        return policy.eval()
