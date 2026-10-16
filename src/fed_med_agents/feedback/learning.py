"""Selected-action contextual-bandit gradient; never enumerate peer outcomes."""

from __future__ import annotations

import torch

from ..fusion import CollaborativeModel
from ..fusion.dynamic import DynamicFusion
from .events import PredictionEvent, VerifiedOutcome


def feedback_loss(
    model: CollaborativeModel,
    records: list[tuple[PredictionEvent, VerifiedOutcome]],
    device: torch.device,
    clip: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped IPS cost policy gradient plus importance-weighted fusion BCE.

    Behavior propensities include exploration. Fixed baseline 0.5 is independent
    of action/label. Clipping introduces bias; this is bounded local replay,
    not an unbiased off-policy clinical estimator. Reward uses the logged risk.
    """
    p = torch.tensor(
        [[0.5 if v is None else v for v in event.evidence] for event, _ in records], device=device
    )
    mask = torch.tensor([[v is not None for v in event.evidence] for event, _ in records], device=device)
    origins = torch.tensor([event.origin for event, _ in records], device=device)
    labels = torch.tensor([outcome.label for _, outcome in records], device=device, dtype=torch.float32)
    domains = torch.tensor([event.available_domains for event, _ in records], device=device)
    action = torch.tensor([event.action for event, _ in records], device=device)
    for event, _ in records:
        expected = model.router.actions[event.action].nonzero().flatten().tolist()
        if expected != sorted(event.consulted):
            raise ValueError("Action/evidence mismatch")
    own = p.gather(1, origins[:, None]).squeeze(1)
    scores = model.router(origins, own, 2 * (own - 0.5).abs(), domains.float(), domains)
    selected = scores.softmax(-1).gather(1, action[:, None]).squeeze(1).clamp_min(1e-8)
    behavior = torch.tensor([event.propensity for event, _ in records], device=device)
    ratio = (selected / behavior).clamp(max=clip).detach()
    logits = model.fusion(p, 2 * (p - 0.5).abs(), mask, origins)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    logged = torch.tensor([event.final_risk for event, _ in records], device=device).clamp(1e-6, 1 - 1e-6)
    cost = torch.tensor([event.consultation_cost for event, _ in records], device=device)
    reward_loss = torch.nn.functional.binary_cross_entropy(logged, labels, reduction="none") + cost
    policy = (ratio * (reward_loss - 0.5).detach() * selected.log()).mean()
    total = (ratio * bce).mean() + 0.1 * policy
    return total, {
        "fusion_bce": float(bce.mean().detach()),
        "policy_loss": float(policy.detach()),
        "max_importance_ratio": float(ratio.max()),
        "clipped_fraction": float(((selected / behavior) > clip).float().mean().detach()),
    }


def routing_feedback_loss(
    model: torch.nn.Module,
    records: list[tuple[PredictionEvent, VerifiedOutcome]],
    contexts: dict[str, dict],
    device: torch.device,
    clip: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Router-only selected-action IPS policy loss using authenticated M4 replay.

    Contexts are origin-local prospective snapshots indexed by request ID, never
    counterfactual rewards. Labels and selected predictions stay local. This
    clipped surrogate is biased, not a clinical off-policy estimator.
    """
    if not records:
        raise ValueError("Insufficient verified feedback")
    rows = [contexts[event.request_id] for event, _ in records]
    origins = torch.tensor([e.origin for e, _ in records], device=device)
    features = torch.tensor([row["features"] for row in rows], device=device)
    available = torch.tensor([e.available_domains for e, _ in records], device=device)
    costs = torch.tensor([row["costs"] for row in rows], device=device)
    own = torch.tensor([e.evidence[e.origin] for e, _ in records], device=device)
    actions = torch.tensor([e.action for e, _ in records], device=device)
    for event, _ in records:
        if event.consulted != ([] if event.action == 0 else [event.action - 1]):
            raise ValueError("Routing action/evidence mismatch")
    scores = model.logits(features, origins, own, available, costs)
    selected = scores.softmax(-1).gather(1, actions[:, None]).flatten().clamp_min(1e-8)
    behavior = torch.tensor([e.propensity for e, _ in records], device=device)
    labels = torch.tensor([o.label for _, o in records], device=device, dtype=torch.float32)
    observed = torch.tensor([e.final_risk for e, _ in records], device=device).clamp(1e-6, 1 - 1e-6)
    paid = torch.tensor([e.consultation_cost for e, _ in records], device=device)
    loss = torch.nn.functional.binary_cross_entropy(observed, labels, reduction="none") + paid
    ratio = (selected / behavior).detach().clamp(max=clip)
    objective = (ratio * (loss - 0.5).detach() * selected.log()).mean()
    if not torch.isfinite(objective):
        raise ValueError("Invalid feedback/action support")
    return objective, {
        "loss": float(objective.detach()),
        "records": len(records),
        "max_importance_ratio": float(ratio.max()),
        "clipped_fraction": float(((selected / behavior) > clip).float().mean().detach()),
    }


def fusion_feedback_update(
    model: DynamicFusion,
    records: list[tuple[PredictionEvent, VerifiedOutcome]],
    provenance: dict[str, dict],
    *,
    minimum: int = 32,
    epochs: int = 2,
) -> dict:
    """Selected observed-set BCE update, with no counterfactual or reliability refit.

    Called with authenticated controller replay. Exact singleton passthrough has
    zero fusion-parameter gradient. Duplicate request/patient-origin rows rejected.
    """
    if minimum < 1 or epochs < 1:
        raise ValueError("Positive minimum/epochs required")
    if len(records) < minimum or model.kind != "set":
        return {"updated": False, "records": len(records)}
    if len({(e.origin, e.patient_id) for e, _ in records}) != len(records):
        raise ValueError("Duplicate feedback patient")
    device = next(model.parameters()).device
    # Different record cardinalities are padded only in trusted training tensors.
    width = max(sum(v is not None for v in e.evidence) for e, _ in records)
    tokens = torch.zeros(len(records), width, 11, device=device)
    mask = torch.zeros(len(records), width, dtype=torch.bool, device=device)
    labels = torch.tensor([o.label for _, o in records], device=device, dtype=torch.float32)
    for row, (event, _) in enumerate(records):
        saved = provenance[event.request_id]
        ids = saved["identities"]
        observed = [i for i, v in enumerate(event.evidence) if v is not None]
        if saved["fusion_contract"] != model.fingerprint():
            raise ValueError("Incompatible feedback fusion contract")
        names = [ids[i] for i in observed]
        for name in names:
            if saved["checkpoints"][name] != model.declarations[name].model_adapter.checkpoint_sha256:
                raise ValueError("Stale evidence checkpoint")
        p = torch.tensor([[event.evidence[i] for i in observed]], device=device)
        x = model.tokens(p, names, torch.tensor([observed.index(event.origin)], device=device))
        tokens[row, : len(names)] = x[0]
        mask[row, : len(names)] = True
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    log = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(tokens, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite feedback fusion loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        log.append({"bce": float(loss.detach()), "gradient_norm": float(norm)})
    return {
        "updated": True,
        "records": len(records),
        "curve": log,
        "multi_evidence": int((mask.sum(-1) > 1).sum()),
    }
