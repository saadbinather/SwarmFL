"""CDC numerical fusion/routing for explicit historical-six and amended-three benchmarks."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..fusion.network import PredictionFusion
from .cdc_data import DOMAINS
from .routing_diagnostics import describe, loss_rows

CLIENTS = 6
COSTS = (0.02, 0.05, 0.10)


class CDCFusion(nn.Module):
    """Count-compatible benchmark versions of the existing residual, logit and mean rules."""

    def __init__(self, kind: str, clients: int = CLIENTS) -> None:
        super().__init__()
        self.kind = kind
        self.clients = clients
        if kind == "original":
            self.model = PredictionFusion(clients)
        elif kind == "logistic":
            self.model = nn.Linear(2 * clients, 1)
        elif kind != "mean":
            raise ValueError("Unknown frozen fusion")

    def forward(self, p: torch.Tensor, mask: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        safe = torch.where(mask, p, torch.full_like(p, 0.5)).clamp(1e-6, 1 - 1e-6)
        if self.kind == "original":
            return self.model(safe, 2 * (safe - 0.5).abs(), mask, origin)
        if self.kind == "logistic":
            return self.model(torch.cat([torch.logit(safe), mask.float()], -1)).flatten()
        mean = torch.where(mask, safe, torch.zeros_like(safe)).sum(-1) / mask.sum(-1)
        return torch.logit(mean.clamp(1e-6, 1 - 1e-6))


def fit_fusion(
    p: np.ndarray, y: np.ndarray, kind: str, seed: int, device: torch.device
) -> tuple[CDCFusion, dict]:
    """Uniform STOP/one-peer fitting on strictly OOF specialist predictions."""
    torch.manual_seed(seed)
    clients = p.shape[1]
    model = CDCFusion(kind, clients).to(device)
    if kind == "mean":
        return model, {"epochs": 0, "steps": 0}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    pairs = [(o, a) for o in range(clients) for a in range(clients + 1) if a != o + 1]
    pair = torch.tensor(pairs, device=device)
    probs, labels = torch.tensor(p, device=device), torch.tensor(y, device=device)
    generator = torch.Generator().manual_seed(seed)
    losses, steps = [], 0
    for _ in range(15):
        epoch_losses = []
        for batch in torch.randperm(len(p) * len(pairs), generator=generator).split(4096):
            b = batch.to(device)
            row, action_pair = b % len(p), pair[b // len(p)]
            origin, action = action_pair[:, 0], action_pair[:, 1]
            mask = nn.functional.one_hot(origin, clients).bool()
            mask |= nn.functional.one_hot((action - 1).clamp_min(0), clients).bool() & (action > 0)[:, None]
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(probs[row], mask, origin), labels[row]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            steps += 1
        losses.append(float(np.mean(epoch_losses)))
    model.requires_grad_(False)
    return model, {"epochs": 15, "steps": steps, "epoch_losses": losses}


@torch.no_grad()
def outcomes(model: CDCFusion, p: np.ndarray, device: torch.device) -> np.ndarray:
    """Offline exhaustive outcomes; inference must request the chosen peer first."""
    values = torch.tensor(p, device=device)
    clients = p.shape[1]
    if model.clients != clients:
        raise ValueError("Fusion/evidence client count mismatch")
    result = np.empty((len(p), clients, clients + 1), dtype=np.float32)
    for origin in range(clients):
        o = torch.full((len(p),), origin, device=device)
        for action in range(clients + 1):
            mask = nn.functional.one_hot(o, clients).bool()
            if action:
                mask[:, action - 1] = True
            result[:, origin, action] = model(values, mask, o).sigmoid().cpu().numpy()
    return result


def gross_gains(risks: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Retrospective training labels, never a routing input at query time."""
    losses = -y[:, None, None] * np.log(risks.clip(1e-6, 1 - 1e-6)) - (1 - y[:, None, None]) * np.log1p(
        -risks.clip(1e-6, 1 - 1e-6)
    )
    return (losses[:, :, :1] - losses[:, :, 1:]).astype(np.float32)


def _uniform(ids: np.ndarray, salt: int) -> np.ndarray:
    value = ids.astype(np.uint64) ^ np.uint64(salt)
    with np.errstate(over="ignore"):
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return ((value ^ (value >> np.uint64(31))) >> np.uint64(11)).astype(np.float64) / (2**53)


def context(
    ids: np.ndarray,
    origin: int,
    columns: list[str],
    scenario: str,
    slots: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exogenous service/coverage schedules depend only on IDs, never clinical data."""
    if slots is not None:
        available, covered, prices = context(ids, slots[origin], columns, scenario)
        return available[:, slots], covered, prices[:, slots]
    available = np.ones((len(ids), CLIENTS), dtype=bool)
    covered = np.ones((len(ids), len(columns)), dtype=bool)
    prices = np.ones((len(ids), CLIENTS), dtype=np.float32)
    if scenario == "variable":
        for k, domain in enumerate((DOMAINS["lifestyle"], DOMAINS["general_health"])):
            outage = _uniform(ids, 53570 + origin * 100 + k) < 0.1
            covered[np.ix_(outage, [columns.index(c) for c in domain])] = False
        covered[:, [columns.index(c) for c in list(DOMAINS.values())[origin]]] = True
        for peer, domain in enumerate(DOMAINS.values()):
            available[:, peer] = (_uniform(ids, 53600 + origin * 100 + peer) < 0.8) & covered[
                :, [columns.index(c) for c in domain]
            ].all(1)
        prices = np.array([0.75, 1, 1.25, 1, 1.5, 0.75], dtype=np.float32)[None, :] * np.where(
            _uniform(ids, 53900 + origin)[:, None] < 0.5, 0.8, 1.2
        ).astype(np.float32)
    elif scenario != "static":
        raise ValueError("Unknown service scenario")
    available[:, origin] = True
    return available, covered, prices


def prequery(
    own: np.ndarray,
    origin: int,
    peer: int,
    available: np.ndarray,
    covered: np.ndarray,
    reliability: np.ndarray,
    enriched: bool,
    domain_config: dict = DOMAINS,
) -> np.ndarray:
    """Build legitimate numeric context: no labels, IDs, peer p or raw feature values."""
    if peer == origin:
        raise ValueError("Self candidate")
    n = len(own)
    domains = list(domain_config.values())
    clients = len(domains)
    identity = np.broadcast_to(np.r_[np.eye(clients)[origin], np.eye(clients)[peer]], (n, 2 * clients))
    overlap = len(set(domains[origin]) & set(domains[peer])) / len(domains[peer])
    x = np.column_stack(
        [
            identity,
            own,
            2 * np.abs(own - 0.5),
            np.full(n, len(domains[peer]) / 5),
            np.full(n, overlap),
            available[:, peer],
            np.ones(n),
        ]
    )
    if enriched:
        p = own.clip(1e-6, 1 - 1e-6)
        x = np.column_stack(
            [
                x,
                -p * np.log(p) - (1 - p) * np.log(1 - p),
                p * (1 - p),
                reliability[:, origin],
                reliability[:, peer],
                covered,
            ]
        )
    return x.astype(np.float32)


def design(
    p: np.ndarray,
    ids: np.ndarray,
    columns: list[str],
    rel: np.ndarray,
    enriched: bool,
    domain_config: dict = DOMAINS,
    slots: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Origin/candidate/patient ordering; same variable scheduling at train and assess."""
    arrays = []
    for origin in range(p.shape[1]):
        available, covered, _ = context(ids, origin, columns, "variable", slots)
        for peer in range(p.shape[1]):
            if peer != origin:
                arrays.append(
                    prequery(p[:, origin], origin, peer, available, covered, rel, enriched, domain_config)
                )
    return np.concatenate(arrays)


def target_vector(gain: np.ndarray) -> np.ndarray:
    clients = gain.shape[1]
    return np.concatenate([gain[:, o, c] for o in range(clients) for c in range(clients) if o != c])


def unflatten(values: np.ndarray, n: int) -> np.ndarray:
    output = np.zeros((n, CLIENTS, CLIENTS), dtype=np.float32)
    for k, (o, c) in enumerate((o, c) for o in range(CLIENTS) for c in range(CLIENTS) if o != c):
        output[:, o, c] = values[k * n : (k + 1) * n]
    return output


def router_network(width: int) -> nn.Module:
    """One compatible shared numeric MLP, entirely separate from private experts/fusion."""
    return nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 1))


def train_router(
    model: nn.Module, x: np.ndarray, target: np.ndarray, epochs: int, seed: int, device: torch.device
) -> dict:
    """Fixed mini-batch Adam; also used by actual Flower local handlers."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.01)
    inputs, labels = torch.tensor(x, device=device), torch.tensor(target, device=device)
    generator = torch.Generator().manual_seed(seed)
    steps, norms = 0, []
    for _ in range(epochs):
        for batch in torch.randperm(len(target), generator=generator).split(4096):
            b = batch.to(device)
            optimizer.zero_grad()
            loss = ((model(inputs[b]).flatten() - labels[b]) ** 2).mean()
            loss.backward()
            norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), 5)))
            optimizer.step()
            steps += 1
    return {"steps": steps, "gradient_norm": float(np.mean(norms)), "loss": float(loss.detach())}


def ridge(x: np.ndarray, y: np.ndarray) -> Pipeline:
    """Regularized linear reference with training-only scaling."""
    return make_pipeline(StandardScaler(), Ridge(alpha=100)).fit(x, y)


def decide(
    gain: np.ndarray, origin: int, available: np.ndarray, prices: np.ndarray, cost: float, budget: int = 1
) -> np.ndarray:
    """Largest positive net gain or STOP, without labels or unqueried predictions."""
    if (
        budget not in (0, 1)
        or not np.isfinite(gain).all()
        or not np.isfinite(cost)
        or not np.isfinite(prices).all()
        or (prices < 0).any()
        or cost < 0
    ):
        raise ValueError("Invalid budget/value/cost")
    score = gain - prices * cost
    score[:, origin] = -np.inf
    score[~available] = -np.inf
    action = score.argmax(1) + 1
    action[(score.max(1) <= 0) | (budget == 0)] = 0
    return action


def metrics(
    y: np.ndarray,
    risk: np.ndarray,
    actions: np.ndarray,
    origin: int,
    available: np.ndarray,
    prices: np.ndarray,
    cost: float,
    estimate: np.ndarray | None = None,
    mean_gain: np.ndarray | None = None,
) -> dict:
    """Post-decision metrics; retrospective oracle and harm never select deployed actions."""
    rows = np.arange(len(y))
    clients = risk.shape[1] - 1
    call = actions > 0
    assert not np.any(actions == origin + 1)
    assert available[rows[call], actions[call] - 1].all()
    selected = risk[rows, actions]
    losses = np.column_stack([loss_rows(y, risk[:, a]) for a in range(clients + 1)])
    paid = np.where(call, prices[rows, (actions - 1).clip(0)] * cost, 0)
    action_objective = losses.copy()
    action_objective[:, 1:] += prices * cost
    action_objective[:, origin + 1] = np.inf
    action_objective[:, 1:][~available] = np.inf
    counts = np.bincount(actions, minlength=clients + 1)
    peer_counts = counts[1:]
    result = {
        **describe(y, selected),
        "pr_auc": float(average_precision_score(y, selected)),
        "objective": float(losses[rows, actions].mean() + paid.mean()),
        "cost": float(paid.mean()),
        "calls": float(call.mean()),
        "stop": float((~call).mean()),
        "harm": float((losses[rows[call], actions[call]] >= losses[call, 0]).mean()) if call.any() else None,
        "action_frequencies": counts.tolist(),
        "actions": actions.tolist(),
        "minority_peer_fraction": float(1 - peer_counts.max() / peer_counts.sum()) if call.any() else 0.0,
        "oracle_objective": float(action_objective.min(1).mean()),
        "oracle_peer_frequencies": np.bincount(action_objective.argmin(1), minlength=clients + 1).tolist(),
    }
    if estimate is not None:
        truth = losses[:, :1] - losses[:, 1:]
        peers = np.arange(clients) != origin
        observed, predicted = truth[:, peers].flatten(), estimate[:, peers].flatten()
        mse = float(np.mean((observed - predicted) ** 2))
        null = float(np.mean((truth[:, peers] - mean_gain[origin, peers]) ** 2))
        result["utility_r2"] = 1 - mse / null
        result["utility_mse"] = mse
        result["utility_correlation"] = (
            float(np.corrcoef(observed, predicted)[0, 1]) if predicted.std() > 1e-8 else None
        )
    return result
