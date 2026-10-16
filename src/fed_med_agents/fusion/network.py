"""Shared, dimension-compatible routing/fusion with explicit evidence masks."""

from __future__ import annotations

import torch
from torch import nn

from ..routing import ContextualRouter


class PredictionFusion(nn.Module):
    def __init__(self, clients: int = 3) -> None:
        super().__init__()
        self.clients = clients
        self.network = nn.Sequential(nn.Linear(4 * clients, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(
        self,
        probabilities: torch.Tensor,
        confidence: torch.Tensor,
        evidence: torch.Tensor,
        origin: torch.Tensor,
    ) -> torch.Tensor:
        # torch.where prevents hidden/NaN evidence from influencing predictions.
        safe = torch.where(evidence, probabilities, torch.full_like(probabilities, 0.5))
        logits = torch.logit(safe.clamp(1e-5, 1 - 1e-5))
        certainties = torch.where(evidence, confidence, torch.zeros_like(confidence))
        identity = nn.functional.one_hot(origin, self.clients).float()
        inputs = torch.cat([logits, certainties, evidence.float(), identity], dim=-1)
        residual = logits.gather(1, origin[:, None]).squeeze(1)
        return self.network(inputs).squeeze(-1) + residual


class CollaborativeModel(nn.Module):
    """Only this state_dict is aggregated; specialist parameters are separate."""

    def __init__(self, clients: int = 3, max_consultations: int = 1) -> None:
        super().__init__()
        self.router = ContextualRouter(clients, max_consultations)
        self.fusion = PredictionFusion(clients)

    def expected_loss(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
        origin: torch.Tensor,
        domains: torch.Tensor,
        available: torch.Tensor,
        cost: float = 0.02,
    ) -> torch.Tensor:
        """Full-information OOF training objective; inference uses select before consultation."""
        confidence = 2 * (probabilities - 0.5).abs()
        own = probabilities.gather(1, origin[:, None]).squeeze(1)
        scores = self.router(origin, own, 2 * (own - 0.5).abs(), domains, available)
        actions = self.router.actions
        batch, action_count = len(labels), len(actions)
        identity = nn.functional.one_hot(origin, self.router.clients).bool()
        evidence = (actions[None].bool() | identity[:, None]) & available[:, None].bool()
        p = probabilities[:, None].expand(-1, action_count, -1).reshape(batch * action_count, -1)
        c = confidence[:, None].expand(-1, action_count, -1).reshape(batch * action_count, -1)
        o = origin[:, None].expand(-1, action_count).reshape(-1)
        output = self.fusion(p, c, evidence.reshape(batch * action_count, -1), o).reshape(batch, action_count)
        bce = nn.functional.binary_cross_entropy_with_logits(
            output, labels[:, None].expand_as(output), reduction="none"
        )
        return (scores.softmax(-1) * (bce + cost * actions.sum(-1))).sum(-1).mean()
