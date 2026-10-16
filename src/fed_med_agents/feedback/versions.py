"""Immutable shared checkpoints and an atomic approved-policy pointer."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from ..fusion import CollaborativeModel
from ..fusion.dynamic import DynamicFusion, binding
from ..routing.policy import RoutingPolicy


def policy_model(metadata: dict) -> CollaborativeModel | RoutingPolicy | DynamicFusion:
    """Preserve legacy M4 architecture; opt in to explicit routing-only versions."""
    if "fusion_config" in metadata:
        if "router_config" in metadata:
            raise ValueError("Ambiguous routing/fusion checkpoint")
        return DynamicFusion(**metadata["fusion_config"])
    config = metadata.get("router_config")
    return CollaborativeModel() if config is None else RoutingPolicy(**config)


def routing_update_gates(
    previous: list[dict],
    baseline: list[dict],
    candidate: list[dict],
    feedback_counts: list[int],
    *,
    minimum_feedback: int = 32,
    accuracy_tolerance: float = 0.005,
    origin_bce_tolerance: float = 0.01,
) -> dict[str, bool]:
    """Validation-only router gates; callers freeze tolerances before evaluation.

    Lists must have the same immutable origin order. These engineering gates do
    not establish scientific superiority or authorize production promotion.
    """
    if not previous or not len(previous) == len(baseline) == len(candidate) == len(feedback_counts):
        raise ValueError("Incompatible validation origins")
    keys = ("objective", "accuracy", "binary_cross_entropy")
    if any(
        not math.isfinite(row[key])
        for rows in (previous, baseline, candidate)
        for row in rows
        for key in keys
    ):
        raise ValueError("Nonfinite validation metrics")

    def mean(rows: list[dict], key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    return {
        "objective_previous": mean(candidate, "objective") <= mean(previous, "objective"),
        "objective_fixed": mean(candidate, "objective") <= mean(baseline, "objective"),
        "accuracy": mean(candidate, "accuracy") >= mean(previous, "accuracy") - accuracy_tolerance,
        "origin_bce": all(
            c["binary_cross_entropy"] <= p["binary_cross_entropy"] + origin_bce_tolerance
            for c, p in zip(candidate, previous, strict=True)
        ),
        "feedback_sufficient": all(n >= minimum_feedback for n in feedback_counts),
    }


class PolicyRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)

    def save(self, version: str, state: dict[str, torch.Tensor], metadata: dict) -> None:
        if not version.replace("-", "").isalnum():
            raise ValueError("Invalid version")
        target = self.path / f"{version}.pt"
        if target.exists():
            raise ValueError("Immutable model version already exists")
        model = policy_model(metadata)
        model.load_state_dict(state, strict=True)
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Nonfinite shared state")
        torch.save({k: v.detach().cpu() for k, v in state.items()}, target)
        metadata = metadata | {"version": version, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
        (self.path / f"{version}.json").write_text(json.dumps(metadata, indent=2))

    def approve(self, version: str) -> None:
        if not version.replace("-", "").isalnum():
            raise ValueError("Invalid version")
        metadata = json.loads((self.path / f"{version}.json").read_text())
        if not metadata["eligible"]:
            raise ValueError("Rejected checkpoint cannot be approved")
        # Load verifies integrity/architecture before advancing the pointer.
        self.load(version)
        temporary = self.path / "approved.tmp"
        temporary.write_text(json.dumps({"version": version}))
        temporary.replace(self.path / "approved.json")
        with (self.path / "transitions.jsonl").open("a") as file:
            file.write(json.dumps({"approved": version}) + "\n")

    def current(self) -> str:
        return json.loads((self.path / "approved.json").read_text())["version"]

    def load(
        self, version: str | None = None, device: torch.device | None = None
    ) -> CollaborativeModel | RoutingPolicy | DynamicFusion:
        device = device or torch.device("cpu")
        version = version or self.current()
        if not version.replace("-", "").isalnum():
            raise ValueError("Invalid version")
        metadata = json.loads((self.path / f"{version}.json").read_text())
        target = self.path / f"{version}.pt"
        if hashlib.sha256(target.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Checkpoint integrity failure")
        model = policy_model(metadata).to(device)
        model.load_state_dict(torch.load(target, weights_only=True, map_location=device))
        return model.eval()

    def rollback(self, version: str) -> None:
        """Explicitly point back to any retained eligible version; never erase evidence."""
        self.approve(version)


class CapabilityRegistry:
    """Disease-neutral unique identity/capability registration from Flower handshake."""

    def __init__(self) -> None:
        self.nodes: dict[int, int] = {}
        self.available: dict[int, bool] = {}

    def register(self, index: int, node: int, available: bool) -> None:
        if index in self.nodes or node in self.nodes.values():
            raise ValueError("Duplicate capability identity")
        self.nodes[index], self.available[index] = node, available


class PolicyPair:
    """Atomic research router/fusion activation over existing immutable registries.

    One authoritative pair pointer prevents torn mixed-policy reads. Component
    approved.json pointers remain independent legacy APIs and are not consulted.
    Single-writer lifecycle only; atomic replace is not a distributed lock.
    """

    def __init__(self, path: Path, routers: PolicyRegistry, fusions: PolicyRegistry) -> None:
        self.path, self.routers, self.fusions = path, routers, fusions
        path.mkdir(parents=True, exist_ok=True)

    def load(
        self, router: str | None = None, fusion: str | None = None
    ) -> tuple[RoutingPolicy, DynamicFusion]:
        """Load one pair snapshot; require compatible stable specialist identities."""
        if (router is None) != (fusion is None):
            raise ValueError("Specify both component versions or neither")
        if router is None or fusion is None:
            current = self.current()
            router, fusion = current["router"], current["fusion"]
        r, f = self.routers.load(router), self.fusions.load(fusion)
        if not isinstance(r, RoutingPolicy) or not isinstance(f, DynamicFusion):
            raise TypeError("Wrong policy component type")
        if set(r.specialist_ids) != set(f.declarations):
            raise ValueError("Incompatible router/fusion identities")
        return r, f

    def current(self) -> dict[str, str]:
        """Read the complete active pair in one filesystem operation."""
        return json.loads((self.path / "active.json").read_text(encoding="utf-8"))

    def activate(self, router: str, fusion: str, *, reason: str) -> None:
        """Validate both eligible immutable components, then replace one pair pointer."""
        proposed_router, proposed_fusion = self.load(router, fusion)
        if (self.path / "active.json").exists():
            current_router, current_fusion = self.load()
            if (
                proposed_router.specialist_ids != current_router.specialist_ids
                or proposed_router.width != current_router.width
                or {n: binding(s) for n, s in proposed_fusion.declarations.items()}
                != {n: binding(s) for n, s in current_fusion.declarations.items()}
            ):
                raise ValueError("Incompatible active deployment contract")
        for registry, version in ((self.routers, router), (self.fusions, fusion)):
            metadata = json.loads((registry.path / f"{version}.json").read_text(encoding="utf-8"))
            if not metadata.get("eligible"):
                raise ValueError("Rejected component cannot activate")
        record = {"router": router, "fusion": fusion}
        # Immutable transition is written first; uncommitted entries are not active.
        import uuid

        transition = self.path / f"transition-{uuid.uuid4().hex}.json"
        transition.write_text(json.dumps(record | {"reason": reason}), encoding="utf-8")
        temporary = self.path / "active.tmp"
        temporary.write_text(json.dumps(record), encoding="utf-8")
        temporary.replace(self.path / "active.json")

    def rollback(self, router: str, fusion: str) -> None:
        """Restore an earlier retained compatible eligible pair without deleting states."""
        self.activate(router, fusion, reason="rollback")
