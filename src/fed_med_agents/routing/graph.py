"""One-hop directed routing without access to candidate predictions or labels."""

from __future__ import annotations

from itertools import combinations

import torch
from torch import nn


class ContextualRouter(nn.Module):
    """Choose bounded consultation subsets using origin evidence and public metadata.

    Availability and domain coverage are metadata, not private feature values.
    The empty action permits no consultation. Self-consultations are forbidden.
    """

    def __init__(self, clients: int = 3, max_consultations: int = 1) -> None:
        super().__init__()
        if clients < 2 or not 0 <= max_consultations < clients:
            raise ValueError("Invalid number of clients or consultation budget")
        self.clients = clients
        self.max_consultations = max_consultations
        self.edge_logits = nn.Parameter(torch.zeros(clients, clients))
        self.context = nn.Sequential(nn.Linear(3 * clients + 2, 16), nn.Tanh(), nn.Linear(16, clients + 1))
        subsets = [subset for k in range(max_consultations + 1) for subset in combinations(range(clients), k)]
        actions = torch.zeros(len(subsets), clients)
        for i, subset in enumerate(subsets):
            actions[i, list(subset)] = 1
        self.register_buffer("actions", actions)

    def adjacency(self) -> torch.Tensor:
        """Normalized learned edge preferences; contextual action scores also matter."""
        diagonal = torch.eye(self.clients, device=self.edge_logits.device, dtype=torch.bool)
        return self.edge_logits.masked_fill(diagonal, -torch.inf).softmax(-1)

    def forward(
        self,
        origin: torch.Tensor,
        probability: torch.Tensor,
        confidence: torch.Tensor,
        domains: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        identity = nn.functional.one_hot(origin, self.clients).float()
        context = self.context(
            torch.cat(
                [identity, probability[:, None], confidence[:, None], domains, available.float()], dim=-1
            )
        )
        edges = context[:, : self.clients] + self.edge_logits[origin]
        scores = edges @ self.actions.T
        scores[:, 0] = context[:, -1]
        forbidden = (self.actions[None] * identity[:, None]).sum(-1) > 0
        permitted = available.bool() & domains.bool()
        forbidden |= (self.actions[None] * (~permitted)[:, None]).sum(-1) > 0
        return scores.masked_fill(forbidden, -torch.inf)

    def select(
        self,
        origin: torch.Tensor,
        probability: torch.Tensor,
        confidence: torch.Tensor,
        domains: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self(origin, probability, confidence, domains, available)
        return self.actions[scores.argmax(-1)].bool(), scores
