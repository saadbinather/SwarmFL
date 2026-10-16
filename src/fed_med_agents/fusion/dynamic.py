"""Identity-bound binary prediction-set fusion; private models stay behind adapters."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import torch
from torch import nn

from ..collaboration.registry import ConsultationSession, Specialist


def binding(spec: Specialist) -> dict:
    """Immutable inference contract, excluding live availability/cost and transport node."""
    return {
        k: v
        for k, v in spec.model_dump(mode="json").items()
        if k not in {"available", "consultation_cost", "capabilities"}
    }


@dataclass(frozen=True)
class FusionEvidence:
    """One actual observed prediction. Trusted callers bind transport receipts to a case."""

    specialist_id: str
    patient_id: str
    probability: float
    checkpoint_sha256: str

    @classmethod
    def from_reply(cls, reply: dict, patient_id: str) -> FusionEvidence:
        """Extract one case from a ConsultationSession-validated aligned reply."""
        index = reply["patient_ids"].index(patient_id)
        return cls(
            reply["destination"], patient_id, reply["probabilities"][index], reply["checkpoint_sha256"]
        )


class DynamicFusion(nn.Module):
    """Mean, OOF-reliability weighted, or lightweight DeepSets binary-risk fusion.

    Network dimensions depend on the metadata schema, never specialist count or
    ordering. Bindings are explicit deployment metadata; an unknown identity is
    rejected until a new contract/version has been validated. Standard tensor
    forward() is a trusted training adapter; fuse() checks registry admission.
    """

    def __init__(
        self, kind: str, declarations: list[dict], reliability: dict[str, float] | None = None
    ) -> None:
        super().__init__()
        if kind not in {"mean", "weighted", "set"} or not declarations:
            raise ValueError("Invalid fusion strategy/declarations")
        specs = [Specialist.model_validate(v) for v in declarations]
        if len({s.specialist_id for s in specs}) != len(specs):
            raise ValueError("Duplicate specialist")
        if any(s.task != specs[0].task or s.output_schema != specs[0].output_schema for s in specs):
            raise ValueError("Incompatible task/output schemas")
        self.kind = kind
        self.declarations = {s.specialist_id: s for s in sorted(specs, key=lambda s: s.specialist_id)}
        self.reliability = {} if reliability is None else dict(reliability)
        if kind != "mean" and set(self.reliability) != set(self.declarations):
            raise ValueError("Fitting-only reliability required for every bound specialist")
        if any(
            isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v < 0
            for v in self.reliability.values()
        ):
            raise ValueError("Invalid reliability BCE")
        if kind == "set":
            self.encoder = nn.Sequential(nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(18, 16), nn.ReLU(), nn.Linear(16, 1))
        else:
            self.register_buffer("anchor", torch.zeros(()))

    def config(self) -> dict:
        """Explicit architecture, task/output and exact adapter/checkpoint contracts."""
        declarations = []
        for spec in self.declarations.values():
            record = spec.model_dump(mode="json")
            record["capabilities"] = sorted(spec.capabilities)
            declarations.append(record)
        return {
            "kind": self.kind,
            "declarations": declarations,
            "reliability": self.reliability,
        }

    def fingerprint(self) -> str:
        """Bind Flower shared arrays to the same deployment/training contract."""
        return hashlib.sha256(
            json.dumps(self.config(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def metadata(self, specialist_id: str) -> list[float]:
        """Fixed-width stable identity and declared/fitting-only metadata, no outcomes."""
        spec = self.declarations[specialist_id]
        digest = hashlib.sha256(specialist_id.encode()).digest()
        identity = [
            int.from_bytes(digest[i : i + 4], "little") / (2**32 - 1) * 2 - 1 for i in range(0, 24, 4)
        ]
        return identity + [
            math.log1p(spec.model_adapter.input_dim) / 5,
            self.reliability.get(specialist_id, 0.0),
        ]

    def tokens(
        self, probabilities: torch.Tensor, identities: list[str], origins: torch.Tensor
    ) -> torch.Tensor:
        """Encode a trusted aligned training batch; IDs determine metadata, not column positions.

        origins contains column indexes in the supplied identity order. Runtime
        evidence validation and authorization belong to fuse(), not this adapter.
        """
        if len(set(identities)) != len(identities) or any(i not in self.declarations for i in identities):
            raise ValueError("Unknown/duplicate specialist identity")
        if probabilities.ndim != 2 or probabilities.shape[1] != len(identities) or not len(identities):
            raise ValueError("Incompatible prediction dimensions")
        if not torch.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Invalid probabilities")
        if origins.shape != (len(probabilities),) or ((origins < 0) | (origins >= len(identities))).any():
            raise ValueError("Invalid origin indexes")
        meta = torch.tensor(
            [self.metadata(i) for i in identities], device=probabilities.device, dtype=probabilities.dtype
        )
        return torch.cat(
            [
                torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))[..., None],
                (2 * (probabilities - 0.5).abs())[..., None],
                nn.functional.one_hot(origins, len(identities)).to(probabilities.dtype)[..., None],
                meta[None].expand(len(probabilities), -1, -1),
            ],
            -1,
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return finite logits for nonempty masked sets; hidden padded tokens ignored.

        A singleton is the original risk, independent of learned parameters.
        Probability endpoints are clamped only for finite internal logit math.
        """
        if (
            tokens.ndim != 3
            or tokens.shape[-1] != 11
            or mask.shape != tokens.shape[:2]
            or mask.dtype != torch.bool
        ):
            raise ValueError("Invalid set/mask schema")
        if not mask.any(-1).all() or not torch.isfinite(tokens[mask]).all():
            raise ValueError("Empty/nonfinite observed evidence")
        safe = torch.where(mask[..., None], tokens, torch.zeros_like(tokens))
        count = mask.sum(-1).to(tokens.dtype)
        p = safe[..., 0].sigmoid()
        if self.kind == "weighted":
            weight = (-safe[..., -1]).masked_fill(~mask, -torch.inf).softmax(-1)
            probability = (p * weight).sum(-1)
        else:
            probability = (p * mask).sum(-1) / count
        base = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        if self.kind != "set":
            return base
        hidden = self.encoder(safe).masked_fill(~mask[..., None], 0).sum(1) / count[:, None]
        delta = self.head(torch.cat([hidden, count.log1p()[:, None], base[:, None]], -1)).squeeze(-1)
        return base + delta * (count > 1)

    def fuse(self, evidence: Iterable[FusionEvidence], session: ConsultationSession) -> float:
        """Fuse only supplied observed evidence after live registry/schema authorization.

        Requires one case and origin evidence. Missing/unconsulted peers are absent,
        never imputed. Reject stale/unavailable evidence instead of silently using it.
        No transport calls, inference, consultation reservation or fees occur here.
        """
        observations = sorted(evidence, key=lambda e: e.specialist_id)
        names = [e.specialist_id for e in observations]
        if not observations or len(set(names)) != len(names):
            raise ValueError("Empty/duplicate evidence")
        if len(session.patient_ids) != 1 or session.origin not in names:
            raise ValueError("Missing origin/single case required")
        for e in observations:
            if e.specialist_id not in self.declarations:
                raise ValueError("Unknown specialist contract")
            entry = session.registry.get(e.specialist_id)
            spec = entry.specialist
            if binding(spec) != binding(self.declarations[e.specialist_id]):
                raise ValueError("Incompatible specialist checkpoint/schema")
            if not spec.available:
                raise ValueError("Unavailable evidence")
            if e.specialist_id != session.origin:
                session.registry.validate(session.origin, e.specialist_id, **session.requirements())
            elif spec.task != session.registry.task or spec.output_schema != session.registry.output_schema:
                raise ValueError("Incompatible origin task/output")
            if (
                e.patient_id != session.patient_ids[0]
                or e.checkpoint_sha256 != spec.model_adapter.checkpoint_sha256
            ):
                raise ValueError("Misaligned case/checkpoint")
            if (
                isinstance(e.probability, bool)
                or not isinstance(e.probability, Real)
                or not math.isfinite(e.probability)
                or not 0 <= e.probability <= 1
            ):
                raise ValueError("Invalid probability")
        if len(observations) == 1:
            return float(observations[0].probability)
        # Mean rules use float64 raw probabilities: arithmetic averaging preserves endpoints.
        if self.kind in {"mean", "weighted"}:
            offset = (
                min(self.reliability[e.specialist_id] for e in observations)
                if self.kind == "weighted"
                else 0.0
            )
            weights = [
                1.0 if self.kind == "mean" else math.exp(offset - self.reliability[e.specialist_id])
                for e in observations
            ]
            total = sum(weights)
            if not total > 0:
                raise ValueError("Degenerate reliability weights")
            return sum(e.probability * w for e, w in zip(observations, weights, strict=True)) / total
        device = next(self.parameters()).device
        p = torch.tensor([[e.probability for e in observations]], device=device)
        tokens = self.tokens(p, names, torch.tensor([names.index(session.origin)], device=device))
        with torch.no_grad():
            value = self(tokens, torch.ones_like(p, dtype=torch.bool)).sigmoid()
        return float(value[0])

    def save(self, path: Path) -> None:
        """Standalone compatible checkpoint; immutable approved versions use PolicyRegistry."""
        if not all(torch.isfinite(v).all() for v in self.state_dict().values()):
            raise ValueError("Nonfinite fusion state")
        torch.save({"fusion_config": self.config(), "state": self.state_dict()}, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> DynamicFusion:
        """Load explicit fusion configuration; historical dense states are never inferred."""
        record = torch.load(path, map_location=device, weights_only=True)
        model = cls(**record["fusion_config"]).to(device)
        model.load_state_dict(record["state"], strict=True)
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("Nonfinite fusion state")
        return model.eval()
