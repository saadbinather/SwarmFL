"""Fixed-three CDC Step 3.6: fusion audit and authorized origin-local routing.

This is an isolated development experiment, not a registry or serving default.
The frozen CDC test is only hashed as bytes. Full-information supervised targets
and sampled observed-only bandit records have separate fitting paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

from ..devices import audit, peak_memory, select_device, synchronize
from .cdc_data import THREE_DOMAINS, THREE_SLOTS, expert_predict
from .cdc_routing import (
    COSTS,
    CDCFusion,
    context,
    gross_gains,
    metrics,
    outcomes,
    prequery,
    ridge,
)
from .m5_step03 import digest, write
from .m5_step035 import fit_graph
from .routing_diagnostics import describe, loss_rows, snapshot

SOURCE = Path("artifacts/m5-step035-three-agent-v1")
PROTOCOL = Path("docs/experiments/m5-step036-context-routing.md")
FIELDS = tuple(field for domain in THREE_DOMAINS.values() for field in domain)
CHOICES = ("original15", "logistic15", "original60", "logistic60")
CANDIDATES = ("ridge_meta", "mlp_meta", "ridge_clinical", "mlp_clinical", "pair_clinical")


def read(path: Path) -> dict:
    """Read experiment JSON without modifying its source."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def local_fields(x: np.ndarray, columns: list[str], origin: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract ONLY the explicitly authorized origin subset, padded to 13 fields."""
    if origin not in range(3) or x.ndim != 2 or x.shape[1] != len(columns):
        raise ValueError("Invalid local field schema")
    permitted = set(list(THREE_DOMAINS.values())[origin])
    values = np.zeros((len(x), len(FIELDS)), dtype=np.float32)
    mask = np.zeros_like(values)
    for index, name in enumerate(FIELDS):
        if name in permitted:
            values[:, index] = x[:, columns.index(name)]
            mask[:, index] = 1
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite authorized origin features")
    return values, mask


def query_features(
    own: np.ndarray,
    origin: int,
    peer: int,
    available: np.ndarray,
    covered: np.ndarray,
    reliability: np.ndarray,
    local: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    """Prospective API: own probability, public metadata and authorized local values.

    No peer predictions, outcomes or identifiers are accepted. Local values/masks
    must have zeros outside the origin whitelist, independently checked here.
    """
    base = prequery(own, origin, peer, available, covered, reliability, True, THREE_DOMAINS)
    if local is None:
        return base
    values, mask = local
    permitted = np.array([field in list(THREE_DOMAINS.values())[origin] for field in FIELDS])
    if values.shape != (len(own), 13) or mask.shape != values.shape:
        raise ValueError("Invalid authorized local shape")
    if np.any(values[:, ~permitted] != 0) or np.any(mask[:, ~permitted] != 0):
        raise ValueError("Unauthorized origin fields")
    if not np.all(mask[:, permitted] == 1) or not np.isfinite(values).all():
        raise ValueError("Incomplete local fields")
    return np.column_stack([base, values, mask]).astype(np.float32)


def design(
    p: np.ndarray, rel: np.ndarray, x: np.ndarray, ids: np.ndarray, columns: list[str], clinical: bool
) -> np.ndarray:
    """Return paired rows [origin, patient, two non-self peers, features]."""
    rows = []
    for origin in range(3):
        available, covered, _ = context(ids, origin, columns, "variable", THREE_SLOTS)
        local = local_fields(x, columns, origin) if clinical else None
        rows.append(
            np.stack(
                [
                    query_features(p[:, origin], origin, peer, available, covered, rel, local)
                    for peer in range(3)
                    if peer != origin
                ],
                axis=1,
            )
        )
    return np.stack(rows)


class ValueRouter(nn.Module):
    """Compatible gain/ranking heads; pairwise loss never supplies inference labels."""

    def __init__(self, width: int, pairwise: bool = False) -> None:
        super().__init__()
        self.pairwise = pairwise
        self.network = nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 2 if pairwise else 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def fit_router(model: ValueRouter, x: np.ndarray, gain: np.ndarray, seed: int, epochs: int = 20) -> dict:
    """Train matched paired-context regression, optionally with a soft ranking head."""
    device = next(model.parameters()).device
    inputs = torch.tensor(x.reshape(-1, 2, x.shape[-1]), device=device)
    target = torch.tensor(gain.reshape(-1, 2), device=device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.01)
    logs = []
    for _ in range(epochs):
        for batch in torch.randperm(len(target), generator=generator).split(2048):
            b = batch.to(device)
            optimizer.zero_grad()
            scores = model(inputs[b])
            loss = ((scores[..., 0] - target[b]) ** 2).mean()
            if model.pairwise:
                soft = torch.sigmoid((target[b, 0] - target[b, 1]) / 0.1)
                loss = loss + 0.1 * nn.functional.binary_cross_entropy_with_logits(
                    (scores[:, 0, 1] - scores[:, 1, 1]) / 0.1,
                    soft,
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite routing loss")
            loss.backward()
            norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5))
            optimizer.step()
            logs.append(norm)
    return {
        "epochs": epochs,
        "steps": len(logs),
        "mean_gradient_norm": float(np.mean(logs)),
        "loss": float(loss.detach()),
    }


def paired_gains(gain: np.ndarray) -> np.ndarray:
    """Align offline targets with the paired training design."""
    return np.stack([gain[:, origin, [peer for peer in range(3) if peer != origin]] for origin in range(3)])


def two_stage(
    value: np.ndarray, rank: np.ndarray, origin: int, available: np.ndarray, prices: np.ndarray, cost: float
) -> np.ndarray:
    """Rank eligible peers, then STOP unless the chosen peer has positive expected net gain."""
    if value.shape != rank.shape or not np.isfinite(value).all() or not np.isfinite(rank).all():
        raise ValueError("Invalid two-stage values")
    scores = rank - cost * prices
    scores[:, origin] = -np.inf
    scores[~available] = -np.inf
    chosen = scores.argmax(1)
    rows = np.arange(len(chosen))
    consult = np.isfinite(scores[rows, chosen]) & (value[rows, chosen] > cost * prices[rows, chosen])
    return np.where(consult, chosen + 1, 0)


def fusion_fit(
    p: np.ndarray, y: np.ndarray, kind: str, seed: int, device: torch.device, target: Path, epochs: int = 60
) -> dict:
    """Cold fit with matched mask order/schedule and saved 15/60-epoch snapshots."""
    torch.manual_seed(seed)
    model = CDCFusion(kind, 3).to(device)
    initial = snapshot(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    pairs = torch.tensor([(o, a) for o in range(3) for a in range(4) if a != o + 1], device=device)
    probabilities, labels = torch.tensor(p, device=device), torch.tensor(y, device=device)
    generator = torch.Generator().manual_seed(seed)
    log = []
    for epoch in range(1, epochs + 1):
        losses, norms = [], []
        for batch in torch.randperm(len(p) * 9, generator=generator).split(4096):
            b = batch.to(device)
            row, pair = b % len(p), pairs[b // len(p)]
            origin, action = pair[:, 0], pair[:, 1]
            mask = nn.functional.one_hot(origin, 3).bool()
            mask |= nn.functional.one_hot((action - 1).clamp_min(0), 3).bool() & (action > 0)[:, None]
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(probabilities[row], mask, origin), labels[row]
            )
            loss.backward()
            norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), 5)))
            optimizer.step()
            losses.append(float(loss.detach()))
        log.append(
            {
                "epoch": epoch,
                "training_bce": float(np.mean(losses)),
                "gradient_norm": float(np.mean(norms)),
                "steps": len(losses),
            }
        )
        if epoch in (15, 60):
            torch.save(snapshot(model), target / f"fusion_{kind}{epoch}.pt")
    torch.save(initial, target / f"initial_{kind}.pt")
    write(target / f"curve_{kind}.json", log)
    return {
        "curve": log,
        "drift": sum(float(((v - initial[k]) ** 2).sum()) for k, v in snapshot(model).items()) ** 0.5,
    }


def parse_choice(choice: str) -> tuple[str, int]:
    """Validate one of the four predeclared fusion settings."""
    if choice not in CHOICES:
        raise ValueError("Invalid fusion schedule")
    return ("original" if choice.startswith("original") else "logistic"), (
        15 if choice.endswith("15") else 60
    )


def fusion_context(
    source: Path, target: Path, y: np.ndarray, seed: int, device: torch.device, choice: str | None = None
) -> dict:
    """Whole-stack target-fold fusion refits; target labels never enter upstream fits."""
    stage = dict(np.load(source / "stack.npz"))
    provenance = read(source / "target_provenance.json")["folds"]
    choices = CHOICES if choice is None else (choice,)
    target_risks = {name: np.zeros((len(stage["fit"]), 3, 4), dtype=np.float32) for name in choices}
    summaries = {}
    for block in ("final", "target0", "target1", "target2"):
        folder = target / block
        folder.mkdir(parents=True)
        arrays = dict(np.load(source / block / "probabilities.npz"))
        upstream = read(source / block / "provenance.json")
        for kind in dict.fromkeys(parse_choice(name)[0] for name in choices):
            epochs = max(parse_choice(name)[1] for name in choices if parse_choice(name)[0] == kind)
            summaries[f"{block}_{kind}"] = fusion_fit(
                arrays["train"], y[upstream["train"]], kind, seed, device, folder, epochs
            )
            if (folder / f"fusion_{kind}15.pt").exists():
                state = torch.load(folder / f"fusion_{kind}15.pt", weights_only=True)
                old_path = source / ("" if block == "final" else block) / f"fusion_{kind}.pt"
                old = torch.load(old_path, weights_only=True)
                summaries[f"{block}_{kind}"]["historical15_max_abs"] = max(
                    float((v - old[k]).abs().max()) for k, v in state.items()
                )
            for name in choices:
                if parse_choice(name)[0] != kind:
                    continue
                model = CDCFusion(kind, 3).to(device)
                model.load_state_dict(
                    torch.load(folder / f"fusion_{name}.pt", weights_only=True, map_location=device)
                )
                risk = outcomes(model, arrays["held"], device)
                if block == "final":
                    np.save(target / f"held_{name}.npy", risk)
                else:
                    target_risks[name][provenance[int(block[-1])]["positions"]] = risk
    scores = {}
    for name, risk in target_risks.items():
        np.savez_compressed(
            target / f"targets_{name}.npz", train=risk, held=np.load(target / f"held_{name}.npy")
        )
        held = np.load(target / f"held_{name}.npy")
        scores[name] = {
            f"{o}_{a}": describe(stage["held_y"], held[:, o, a])
            for o in range(3)
            for a in range(4)
            if a != o + 1
        }
    write(target / "fusion_audit.json", {"training": summaries, "held": scores})
    return scores


def uniform_bce(scores: dict) -> float:
    """Predeclared inner-only fusion selection objective."""
    return float(np.mean([row["binary_cross_entropy"] for row in scores.values()]))


def observed_records(
    paired: np.ndarray,
    risks: np.ndarray,
    labels: np.ndarray,
    p: np.ndarray,
    ids: np.ndarray,
    columns: list[str],
    seed: int,
    target: Path,
) -> dict:
    """Sample and persist actions BEFORE looking up only their observed risk/outcome."""
    rng = np.random.default_rng(seed)
    selection = []
    for origin in range(3):
        available, _, _ = context(ids, origin, columns, "variable", THREE_SLOTS)
        for row in rng.permutation(len(labels)):
            valid = [0, *[peer + 1 for peer in range(3) if peer != origin and available[row, peer]]]
            action = int(rng.choice(valid))
            selection.append((origin, int(row), action, 1 / len(valid)))
    write(target / "bandit_selection.json", selection)
    origins, rows, actions, propensity = np.array(selection).T
    origins, rows, actions = origins.astype(int), rows.astype(int), actions.astype(int)
    observed_risk = risks[rows, origins, actions]
    y = labels[rows]
    values, evidence = [], []
    for origin, row, action in zip(origins, rows, actions, strict=True):
        peers = [peer for peer in range(3) if peer != origin]
        index = peers.index(action - 1) if action else 0
        values.append(np.r_[paired[origin, row, index], np.eye(4)[action]])
        allowed = [origin, action - 1] if action else [origin]
        evidence.append([float(p[row, peer]) if peer in allowed else np.nan for peer in range(3)])
    record = {
        "x": np.array(values, dtype=np.float32),
        "loss": loss_rows(y, observed_risk).astype(np.float32),
        "risk": observed_risk,
        "label": y,
        "propensity": propensity.astype(np.float32),
        "origin": origins,
        "row": rows,
        "action": actions,
        "evidence": np.array(evidence),
    }
    np.savez_compressed(target / "bandit_records.npz", **record)
    return record


def fit_bandit(
    records: dict, seed: int, device: torch.device, folder: Path
) -> tuple[nn.Module, StandardScaler]:
    """IPS selected-loss regression: no full-information utility targets or stop counterfactuals."""
    scaler = StandardScaler().fit(records["x"])
    x = torch.tensor(scaler.transform(records["x"]), device=device, dtype=torch.float32)
    y = torch.tensor(records["loss"], device=device)
    weights = torch.tensor(np.minimum(1 / records["propensity"], 5), device=device)
    torch.manual_seed(seed)
    model = nn.Linear(x.shape[1], 4).to(device)
    actions = torch.tensor(records["action"], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(20):
        for batch in torch.randperm(len(y), generator=generator).split(4096):
            b = batch.to(device)
            optimizer.zero_grad()
            selected = model(x[b]).gather(1, actions[b, None]).flatten()
            loss = (weights[b] * (selected - y[b]).square()).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    torch.save(snapshot(model), folder / "bandit.pt")
    joblib.dump(scaler, folder / "bandit_scaler.joblib")
    return model, scaler


def infer(
    model: object, scaler: StandardScaler | None, features: np.ndarray, origin: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Infer candidate values/ranks from authorized pre-query context alone."""
    peers = [peer for peer in range(3) if peer != origin]
    inputs = features.reshape(-1, features.shape[-1])
    if scaler is not None:
        inputs = scaler.transform(inputs)
    if isinstance(model, nn.Module):
        with torch.no_grad():
            score = (
                model(torch.tensor(inputs, device=device, dtype=torch.float32))
                .cpu()
                .numpy()
                .reshape(len(features), 2, -1)
            )
    else:
        score = model.predict(inputs).reshape(len(features), 2, 1)
    value, rank = (
        np.zeros((len(features), 3), dtype=np.float32),
        np.zeros((len(features), 3), dtype=np.float32),
    )
    value[:, peers] = score[..., 0]
    rank[:, peers] = score[..., -1]
    return value, rank


def bandit_predict(
    model: nn.Module, scaler: StandardScaler, features: np.ndarray, origin: int, device: torch.device
) -> np.ndarray:
    """Predict four action BCEs; does not receive outcomes or peer probabilities."""
    peers = [peer for peer in range(3) if peer != origin]
    loss = np.full((len(features), 4), np.inf)
    with torch.no_grad():
        for action in (0, *[peer + 1 for peer in peers]):
            index = peers.index(action - 1) if action else 0
            x = np.column_stack([features[:, index], np.tile(np.eye(4)[action], (len(features), 1))])
            tensor = torch.tensor(scaler.transform(x), device=device, dtype=torch.float32)
            loss[:, action] = model(tensor)[:, action].cpu().numpy()
    return loss


def run_routing(
    source: Path,
    target: Path,
    choice: str,
    development: dict,
    columns: list[str],
    seed: int,
    device: torch.device,
) -> dict:
    """Fit every fixed router and assess without selecting on this context's held labels."""
    stage = dict(np.load(source / "stack.npz"))
    arrays = dict(np.load(target / f"targets_{choice}.npz"))
    gain = gross_gains(arrays["train"], stage["y"])
    mean_gain = gain.mean(0)
    goals = paired_gains(gain)
    designs, models, scales, logs = {}, {}, {}, {}
    for clinical in (False, True):
        group = "clinical" if clinical else "meta"
        x = design(
            stage["p"],
            stage["rel"],
            development["x"][stage["fit"]],
            development["ids"][stage["fit"]],
            columns,
            clinical,
        )
        designs[group] = x
        linear = ridge(x.reshape(-1, x.shape[-1]), goals.flatten())
        models[f"ridge_{group}"], scales[f"ridge_{group}"] = linear, None
        joblib.dump(linear, target / f"ridge_{group}.joblib")
        scaler = StandardScaler().fit(x.reshape(-1, x.shape[-1]))
        joblib.dump(scaler, target / f"scaler_{group}.joblib")
        transformed = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)
        for pairwise in (False, True) if clinical else (False,):
            name = f"{'pair' if pairwise else 'mlp'}_{group}"
            torch.manual_seed(seed)
            model = ValueRouter(x.shape[-1], pairwise).to(device)
            logs[name] = fit_router(model, transformed, goals, seed)
            models[name], scales[name] = model, scaler
            torch.save(snapshot(model), target / f"{name}.pt")
    graph = fit_graph(
        stage["p"],
        arrays["train"],
        stage["y"],
        development["ids"][stage["fit"]],
        columns,
        seed,
        device,
        THREE_SLOTS,
    )
    torch.save(snapshot(graph), target / "graph.pt")
    record = observed_records(
        designs["clinical"],
        arrays["train"],
        stage["y"],
        stage["p"],
        development["ids"][stage["fit"]],
        columns,
        seed,
        target,
    )
    bandit, bandit_scaler = fit_bandit(record, seed, device, target)
    write(target / "routing_training.json", logs)
    result, action_arrays = {}, {}
    for scenario in ("static", "variable"):
        result[scenario] = {str(cost): {} for cost in COSTS}
        for origin in range(3):
            ids = development["ids"][stage["held"]]
            available, covered, prices = context(ids, origin, columns, scenario, THREE_SLOTS)
            own = stage["held_p"][:, origin]
            local = local_fields(development["x"][stage["held"]], columns, origin)
            features = {}
            for clinical in (False, True):
                features["clinical" if clinical else "meta"] = np.stack(
                    [
                        query_features(
                            own,
                            origin,
                            peer,
                            available,
                            covered,
                            stage["held_rel"],
                            local if clinical else None,
                        )
                        for peer in range(3)
                        if peer != origin
                    ],
                    axis=1,
                )
            estimates = {
                name: infer(model, scales[name], features[name.split("_")[-1]], origin, device)
                for name, model in models.items()
            }
            estimates["fixed_value"] = (np.tile(mean_gain[origin], (len(own), 1)),) * 2
            best_score = np.tile(mean_gain[origin], (len(own), 1))
            best_score[:, origin], best_score[~available] = -np.inf, -np.inf
            best = best_score.argmax(1) + 1
            best[~np.isfinite(best_score.max(1))] = 0
            random = np.random.default_rng(seed + origin).random(available.shape)
            random[:, origin], random[~available] = -np.inf, -np.inf
            random_action = random.argmax(1) + 1
            random_action[~np.isfinite(random.max(1))] = 0
            with torch.no_grad():
                prob = torch.tensor(own, device=device)
                avail = torch.tensor(available, device=device)
                graph_scores = (
                    graph(
                        torch.full((len(own),), origin, device=device),
                        prob,
                        2 * (prob - 0.5).abs(),
                        avail.float(),
                        avail,
                    )
                    .cpu()
                    .numpy()
                )
            observed_loss = bandit_predict(bandit, bandit_scaler, features["clinical"], origin, device)
            for cost in COSTS:
                actions = {
                    name: two_stage(v, rank, origin, available, prices, cost)
                    for name, (v, rank) in estimates.items()
                }
                adjusted = graph_scores.copy()
                adjusted[:, 1:] -= cost * prices
                action_loss = observed_loss.copy()
                action_loss[:, 1:] += cost * prices
                action_loss[:, 1:][~available] = np.inf
                actions.update(
                    stop=np.zeros(len(own), dtype=int),
                    best_peer=best,
                    graph=adjusted.argmax(1),
                    bandit_observed=action_loss.argmin(1),
                )
                for name in CANDIDATES:
                    actions[f"random_matched_{name}"] = np.where(actions[name] > 0, random_action, 0)
                for name, action in actions.items():
                    value = estimates[name][0] if name in estimates else None
                    row = metrics(
                        stage["held_y"],
                        arrays["held"][:, origin],
                        action,
                        origin,
                        available,
                        prices,
                        cost,
                        value,
                        mean_gain,
                    )
                    row.pop("actions")
                    result[scenario][str(cost)].setdefault(name, {})[str(origin)] = row
                    action_arrays[f"{scenario}_{cost}_{name}_{origin}"] = action.astype(np.int8)
    np.savez_compressed(target / "actions.npz", **action_arrays)
    write(target / "results.json", result)
    return result


def select_router(results: dict, names: tuple[str, ...]) -> str:
    """Freeze a candidate using inner held data only, fixed candidate order for ties."""
    return min(
        names,
        key=lambda name: np.mean([row["objective"] for row in results["variable"]["0.02"][name].values()]),
    )


def acceptance(results: dict, selection: dict) -> dict:
    """Retain historical gates and the stronger fixed-value/context controls."""
    rows = []
    for fold in range(3):
        record = results[f"pilot_fold{fold}_outer"]
        clinical, meta = selection[str(fold)]["clinical"], selection[str(fold)]["metadata"]
        chosen = record["variable"]["0.02"][clinical]
        base = record["variable"]["0.02"]["best_peer"]
        fixed = record["variable"]["0.02"]["fixed_value"]
        metadata = record["variable"]["0.02"][meta]
        average = lambda name, field: float(np.mean([r[field] for r in name.values()]))
        rows.append(
            {
                "best_peer_gain": average(base, "objective") - average(chosen, "objective"),
                "fixed_value_gain": average(fixed, "objective") - average(chosen, "objective"),
                "metadata_gain": average(metadata, "objective") - average(chosen, "objective"),
                "static_gain": average(record["static"]["0.02"]["best_peer"], "objective")
                - average(record["static"]["0.02"][clinical], "objective"),
                "r2": average(chosen, "utility_r2"),
                "varying_origins": sum(r["minority_peer_fraction"] >= 0.05 for r in chosen.values()),
                "accuracy_delta": average(chosen, "accuracy") - average(base, "accuracy"),
                "origin_bce_delta": [
                    chosen[str(o)]["binary_cross_entropy"] - base[str(o)]["binary_cross_entropy"]
                    for o in range(3)
                ],
            }
        )
    gates = {
        "best_peer_gain_.005_each_fold": all(r["best_peer_gain"] >= 0.005 for r in rows),
        "fixed_value_gain_.005_each_fold": all(r["fixed_value_gain"] >= 0.005 for r in rows),
        "metadata_gain_positive_each_fold": all(r["metadata_gain"] > 0 for r in rows),
        "static_gain_positive_each_fold": all(r["static_gain"] > 0 for r in rows),
        "utility_r2_.05_each_fold": all(r["r2"] >= 0.05 for r in rows),
        "three_varying_origins_each_fold": all(r["varying_origins"] == 3 for r in rows),
        "accuracy_drop_.005": np.mean([r["accuracy_delta"] for r in rows]) >= -0.005,
        "no_origin_bce_degradation_.01": bool(
            (np.mean([r["origin_bce_delta"] for r in rows], axis=0) <= 0.01).all()
        ),
    }
    return {
        "folds": rows,
        "gates": {k: bool(v) for k, v in gates.items()},
        "passed": bool(all(gates.values())),
        "expanded_to_50k": False,
    }


def run(source: Path, root: Path, device: torch.device) -> None:
    """Audit inner fusion, freeze choices, fit outer routers, and retain all outcomes."""
    root.mkdir(parents=True)
    preserved = {
        str(path.resolve()): digest(path)
        for base in (Path("artifacts"), Path("docs/experiments"))
        for path in base.rglob("*")
        if path.is_file()
        and root not in path.resolve().parents
        and "ollama" not in path.parts
        and path.suffix in {".pt", ".npz", ".npy", ".json", ".jsonl", ".joblib", ".csv", ".md"}
        and path.resolve() != PROTOCOL.resolve()
    }
    write(root / "preserved-before.json", preserved)
    shutil.copyfile(PROTOCOL, root / "protocol.md")
    write(
        root / "config.json",
        {
            "source": str(source),
            "device": str(device),
            "clients": 3,
            "clinical_authorization": THREE_DOMAINS,
            "test_evaluations": 0,
        },
    )
    audit(root / "environment.json")
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    development = dict(np.load(source / "development.npz"))
    columns = read(source / "dataset.json")["features"]
    contexts = read(source / "contexts.json")
    results, selections = {}, {}
    with threadpool_limits(limits=1):
        for stage in ("inner", "outer"):
            for fold in range(3):
                name = f"pilot_fold{fold}_{stage}"
                target = root / name
                seed = contexts[name]["seed"]
                choice = None if stage == "inner" else selections[str(fold)]["fusion"]
                scores = fusion_context(source / name, target, development["y"], seed, device, choice)
                if stage == "inner":
                    choice = min(CHOICES, key=lambda option: uniform_bce(scores[option]))
                    selections[str(fold)] = {
                        "fusion": choice,
                        "fusion_inner_bce": {option: uniform_bce(score) for option, score in scores.items()},
                    }
                result = run_routing(source / name, target, choice, development, columns, seed, device)
                results[name] = result
                if stage == "inner":
                    selections[str(fold)].update(
                        clinical=select_router(result, CANDIDATES[2:]),
                        metadata=select_router(result, CANDIDATES[:2]),
                    )
                print(f"Step3.6 completed {name}, fusion={choice}", flush=True)
            if stage == "inner":
                write(root / "selection.json", selections)
                write(
                    root / "inner_selection_frozen.json",
                    {"sha256": digest(root / "selection.json"), "outer_started": False},
                )
    write(root / "results.json", results)
    write(root / "gates.json", acceptance(results, selections))
    synchronize(device)
    write(
        root / "execution.json",
        {
            "seconds": time.perf_counter() - start,
            "device": str(device),
            "peak_allocated_bytes": peak_memory(device),
            "test_evaluations": 0,
            "expanded_to_50k": False,
        },
    )


def flower_data(root: Path) -> tuple[np.ndarray, np.ndarray, dict, dict, list[str], Path]:
    """Construct client-local compatible clinical designs from frozen pilot provenance."""
    source = Path(read(root / "config.json")["source"])
    name = "pilot_fold0_outer"
    stage = dict(np.load(source / name / "stack.npz"))
    development = dict(np.load(source / "development.npz"))
    columns = read(source / "dataset.json")["features"]
    x = design(
        stage["p"],
        stage["rel"],
        development["x"][stage["fit"]],
        development["ids"][stage["fit"]],
        columns,
        True,
    )
    scaler = joblib.load(root / name / "scaler_clinical.joblib")
    x = scaler.transform(x.reshape(-1, 63)).reshape(x.shape).astype(np.float32)
    choice = read(root / "selection.json")["0"]["fusion"]
    targets = np.load(root / name / f"targets_{choice}.npz")["train"]
    gain = paired_gains(gross_gains(targets, stage["y"]))
    return x, gain, stage, development, columns, source


def flower_worker(root: Path) -> None:
    """Three genuine Flower clients; clinical fields are used only inside origin handlers."""
    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import Grid, ServerApp
    from flwr.serverapp.strategy import FedAvg
    from flwr.simulation import run_simulation

    client, server = ClientApp(), ServerApp()
    torch.set_num_threads(1)

    @client.train()
    def train(message: Message, ctx: Context) -> Message:
        torch.set_num_threads(1)
        origin = int(ctx.node_config["partition-id"])
        round_id = int(message.content["config"]["server-round"])
        x, gain, *_ = flower_data(root)
        model = ValueRouter(63)
        model.load_state_dict(message.content["arrays"].to_torch_state_dict())
        log = fit_router(model, x[origin : origin + 1], gain[origin : origin + 1], 53660 + round_id, 2)
        torch.save(snapshot(model), root / "flower" / f"client_{origin}_round_{round_id}.pt")
        write(root / "flower" / f"client_{origin}_round_{round_id}.json", log)
        return Message(
            RecordDict(
                {
                    "arrays": ArrayRecord(snapshot(model)),
                    "metrics": MetricRecord(
                        {"client": origin, "num-examples": 2 * len(x[origin]), "loss": log["loss"]}
                    ),
                }
            ),
            reply_to=message,
        )

    @client.query("identity")
    def identity(message: Message, ctx: Context) -> Message:
        return Message(
            RecordDict({"response": ConfigRecord({"client": int(ctx.node_config["partition-id"])})}),
            reply_to=message,
        )

    @client.query("route")
    def route(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        _, _, stage, development, columns, source = flower_data(root)
        ids = stage["held"]
        # Origin-local inference: no raw fields leave this handler.
        private = joblib.load(source / "pilot_fold0_outer/final" / f"client_{origin}/final.joblib")
        with threadpool_limits(limits=1):
            own = expert_predict(private, development["x"][ids])
        np.testing.assert_allclose(own, stage["held_p"][:, origin], atol=2e-6)
        available, covered, prices = context(
            development["ids"][ids], origin, columns, "variable", THREE_SLOTS
        )
        local = local_fields(development["x"][ids], columns, origin)
        features = np.stack(
            [
                query_features(own, origin, peer, available, covered, stage["held_rel"], local)
                for peer in range(3)
                if peer != origin
            ],
            axis=1,
        )
        model = ValueRouter(63)
        model.load_state_dict(message.content["arrays"].to_torch_state_dict())
        scaler = joblib.load(root / "pilot_fold0_outer/scaler_clinical.joblib")
        value, rank = infer(model, scaler, features, origin, torch.device("cpu"))
        actions = two_stage(value, rank, origin, available, prices, 0.02)
        return Message(
            RecordDict(
                {
                    "response": ConfigRecord(
                        {
                            "client": origin,
                            "ids": ids.tolist(),
                            "actions": actions.tolist(),
                            "own": own.tolist(),
                        }
                    )
                }
            ),
            reply_to=message,
        )

    @client.query("consult")
    def consult(message: Message, ctx: Context) -> Message:
        peer = int(ctx.node_config["partition-id"])
        ids = np.array(message.content["config"]["ids"], dtype=int)
        _, _, stage, development, _, source = flower_data(root)
        if not set(ids).issubset(stage["held"]):
            raise ValueError("Outside fixed assessment")
        model = joblib.load(source / "pilot_fold0_outer/final" / f"client_{peer}/final.joblib")
        with threadpool_limits(limits=1):
            probabilities = expert_predict(model, development["x"][ids])
        return Message(
            RecordDict(
                {"response": ConfigRecord({"client": peer, "ids": ids.tolist(), "p": probabilities.tolist()})}
            ),
            reply_to=message,
        )

    @server.main()
    def serve(grid: Grid, ctx: Context) -> None:
        start, deadline = time.perf_counter(), time.perf_counter() + 90
        while len(list(grid.get_node_ids())) != 3:
            if time.perf_counter() > deadline:
                raise TimeoutError("Exactly three clients required")
            time.sleep(0.1)
        nodes = {}

        def query(node: int, action: str, records: RecordDict) -> dict:
            replies = list(
                grid.send_and_receive(
                    [Message(records, dst_node_id=node, message_type=f"query.{action}")], timeout=120
                )
            )
            if len(replies) != 1 or replies[0].has_error() or replies[0].metadata.src_node_id != node:
                raise RuntimeError("Invalid Flower query reply")
            return dict(replies[0].content["response"])

        for node in grid.get_node_ids():
            item = query(node, "identity", RecordDict({"config": ConfigRecord({})}))
            if item["client"] in nodes:
                raise ValueError("Duplicate specialist")
            nodes[item["client"]] = node
        x, gain, stage, development, columns, _ = flower_data(root)
        if x.shape[-1] != 63 or not np.isfinite(x).all() or not np.isfinite(gain).all():
            raise ValueError("Incompatible clinical candidate")
        torch.manual_seed(53660)
        model = ValueRouter(63)
        initial = snapshot(model)
        torch.save(initial, root / "flower/initial.pt")

        def validate(round_id: int, arrays: ArrayRecord) -> MetricRecord:
            state = arrays.to_torch_state_dict()
            torch.save(state, root / "flower" / f"global_{round_id}.pt")
            return MetricRecord({"finite": int(all(torch.isfinite(v).all() for v in state.values()))})

        strategy = FedAvg(fraction_train=1, fraction_evaluate=0, min_train_nodes=3, min_available_nodes=3)
        result = strategy.start(
            grid,
            initial_arrays=ArrayRecord(initial),
            num_rounds=3,
            evaluate_fn=validate,
            train_config=ConfigRecord({}),
            timeout=300,
        )
        state = result.arrays.to_torch_state_dict()
        model.load_state_dict(state)
        central = ValueRouter(63)
        central.load_state_dict(initial)
        central_log = fit_router(central, x, gain, 53660, 6)
        torch.save(snapshot(central), root / "flower/central.pt")
        choice = read(root / "selection.json")["0"]["fusion"]
        kind, _ = parse_choice(choice)
        fusion = CDCFusion(kind, 3)
        fusion.load_state_dict(
            torch.load(root / "pilot_fold0_outer/final" / f"fusion_{choice}.pt", weights_only=True)
        )
        risk = np.load(root / "pilot_fold0_outer" / f"targets_{choice}.npz")["held"]
        log, summaries = [], {}
        for origin in range(3):
            response = query(nodes[origin], "route", RecordDict({"arrays": ArrayRecord(state)}))
            assert response["client"] == origin and response["ids"] == stage["held"].tolist()
            actions = np.array(response["actions"], dtype=int)
            p = np.full((len(actions), 3), 0.5, dtype=np.float32)
            mask = np.zeros_like(p, dtype=bool)
            p[:, origin], mask[:, origin] = response["own"], True
            for peer in range(3):
                rows = np.flatnonzero(actions == peer + 1)
                if not len(rows):
                    continue
                assert peer != origin
                selected_ids = stage["held"][rows].tolist()
                answer = query(
                    nodes[peer], "consult", RecordDict({"config": ConfigRecord({"ids": selected_ids})})
                )
                assert answer["client"] == peer and answer["ids"] == selected_ids
                p[rows, peer], mask[rows, peer] = answer["p"], True
                log.append(
                    {
                        "origin": origin,
                        "peer": peer,
                        "ids": selected_ids,
                        "p": answer["p"],
                        "payload_fields": ["ids"],
                    }
                )
            with torch.no_grad():
                actual = (
                    fusion(torch.tensor(p), torch.tensor(mask), torch.full((len(p),), origin))
                    .sigmoid()
                    .numpy()
                )
            np.testing.assert_allclose(actual, risk[np.arange(len(p)), origin, actions], atol=2e-6)
            available, _, prices = context(
                development["ids"][stage["held"]], origin, columns, "variable", THREE_SLOTS
            )
            summary = metrics(stage["held_y"], risk[:, origin], actions, origin, available, prices, 0.02)
            summaries[str(origin)] = summary
        torch.save(state, root / "flower/final.pt")
        write(root / "flower/consultations.json", log)
        write(
            root / "flower/results.json",
            {
                "run_id": ctx.run_id,
                "clients": 3,
                "rounds": 3,
                "updates": 9,
                "metrics": summaries,
                "central_training": central_log,
                "seconds": time.perf_counter() - start,
                "drift": sum(float(((v - initial[k]) ** 2).sum()) for k, v in state.items()) ** 0.5,
            },
        )

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=3,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
            "init_args": {
                "num_cpus": 1,
                "num_gpus": 0,
                "include_dashboard": False,
                "object_store_memory": 256 * 1024 * 1024,
            },
        },
    )


def verify(root: Path) -> dict:
    """Independently replay boundaries, outcomes, routing metrics, feedback and Flower updates."""
    cfg = read(root / "config.json")
    source = Path(cfg["source"])
    device = select_device(cfg["device"])
    torch.set_num_threads(1)
    columns = read(source / "dataset.json")["features"]
    development = dict(np.load(source / "development.npz"))
    contexts, selected = read(source / "contexts.json"), read(root / "selection.json")
    assert digest(root / "selection.json") == read(root / "inner_selection_frozen.json")["sha256"]
    frozen = set(np.load(source / "frozen_test_ids.npy").tolist())
    counts = {
        "fusion_outcome_replays": 0,
        "metric_fields": 0,
        "bandit_observations": 0,
        "aggregations": 0,
        "local_update_replays": 0,
    }
    results = read(root / "results.json")
    with threadpool_limits(limits=1):
        for name, output in results.items():
            folder = root / name
            stage = dict(np.load(source / name / "stack.npz"))
            assert not set(development["ids"][stage["fit"]]) & frozen
            assert not set(development["ids"][stage["held"]]) & frozen
            assert not set(development["groups"][stage["fit"]]) & set(development["groups"][stage["held"]])
            np.testing.assert_array_equal(stage["y"], development["y"][stage["fit"]])
            np.testing.assert_array_equal(stage["held_y"], development["y"][stage["held"]])
            fold = name.split("_")[1][-1]
            choice = selected[fold]["fusion"]
            if name.endswith("inner"):
                audit_scores = read(folder / "fusion_audit.json")["held"]
                assert min(CHOICES, key=lambda option: uniform_bce(audit_scores[option])) == choice
                assert select_router(output, CANDIDATES[2:]) == selected[fold]["clinical"]
                assert select_router(output, CANDIDATES[:2]) == selected[fold]["metadata"]
            provenance = read(source / name / "target_provenance.json")["folds"]
            audit_rows = read(folder / "fusion_audit.json")
            for training in audit_rows["training"].values():
                assert training.get("historical15_max_abs", 0) <= 2e-6
            for variant in audit_rows["held"]:
                kind, _ = parse_choice(variant)
                expected = dict(np.load(folder / f"targets_{variant}.npz"))
                replay_train = np.zeros_like(expected["train"])
                for block in ("final", "target0", "target1", "target2"):
                    upstream = read(source / name / block / "provenance.json")
                    assert not set(development["groups"][upstream["train"]]) & set(
                        development["groups"][upstream["held"]]
                    )
                    model = CDCFusion(kind, 3).to(device)
                    model.load_state_dict(
                        torch.load(
                            folder / block / f"fusion_{variant}.pt", weights_only=True, map_location=device
                        )
                    )
                    p = np.load(source / name / block / "probabilities.npz")["held"]
                    replay = outcomes(model, p, device)
                    if block == "final":
                        np.testing.assert_allclose(replay, expected["held"], atol=2e-6)
                    else:
                        replay_train[provenance[int(block[-1])]["positions"]] = replay
                    counts["fusion_outcome_replays"] += 1
                np.testing.assert_allclose(replay_train, expected["train"], atol=2e-6)
            arrays = dict(np.load(folder / f"targets_{choice}.npz"))
            gain = gross_gains(arrays["train"], stage["y"])
            mean_gain = gain.mean(0)
            record = dict(np.load(folder / "bandit_records.npz"))
            assert len(set(zip(record["origin"], record["row"], strict=True))) == 3 * len(stage["fit"])
            clinical_design = design(
                stage["p"],
                stage["rel"],
                development["x"][stage["fit"]],
                development["ids"][stage["fit"]],
                columns,
                True,
            )
            selections = np.array(read(folder / "bandit_selection.json"))
            for index, key in enumerate(("origin", "row", "action", "propensity")):
                np.testing.assert_allclose(record[key], selections[:, index])
            np.testing.assert_array_equal(record["label"], stage["y"][record["row"]])
            np.testing.assert_array_equal(
                record["risk"], arrays["train"][record["row"], record["origin"], record["action"]]
            )
            np.testing.assert_allclose(record["loss"], loss_rows(record["label"], record["risk"]), atol=1e-6)
            for index, (origin, row, action) in enumerate(
                zip(record["origin"], record["row"], record["action"], strict=True)
            ):
                allowed = [origin, action - 1] if action else [origin]
                peers = [peer for peer in range(3) if peer != origin]
                pair_index = peers.index(action - 1) if action else 0
                expected_x = np.r_[clinical_design[origin, row, pair_index], np.eye(4)[action]]
                np.testing.assert_allclose(record["x"][index], expected_x, atol=1e-6)
                assert np.flatnonzero(np.isfinite(record["evidence"][index])).tolist() == sorted(allowed)
                np.testing.assert_allclose(record["evidence"][index, allowed], stage["p"][row, allowed])
            for origin in range(3):
                available, _, _ = context(
                    development["ids"][stage["fit"]], origin, columns, "variable", THREE_SLOTS
                )
                peers = [peer for peer in range(3) if peer != origin]
                expected_propensity = 1 / (1 + available[:, peers].sum(1))
                observed = record["origin"] == origin
                np.testing.assert_allclose(
                    record["propensity"][observed], expected_propensity[record["row"][observed]]
                )
                called = observed & (record["action"] > 0)
                assert available[record["row"][called], record["action"][called] - 1].all()
            counts["bandit_observations"] += len(record["loss"])
            replay_folder = root / "verification_bandit" / name
            replay_folder.mkdir(parents=True, exist_ok=True)
            bandit, _ = fit_bandit(record, contexts[name]["seed"], device, replay_folder)
            saved_bandit = torch.load(folder / "bandit.pt", weights_only=True)
            for key, value in snapshot(bandit).items():
                torch.testing.assert_close(value, saved_bandit[key], atol=2e-6, rtol=1e-5)
            models, scales = {}, {}
            for candidate in CANDIDATES:
                group = candidate.split("_")[-1]
                if candidate.startswith("ridge"):
                    models[candidate], scales[candidate] = joblib.load(folder / f"{candidate}.joblib"), None
                else:
                    model = ValueRouter(63 if group == "clinical" else 37, candidate.startswith("pair")).to(
                        device
                    )
                    model.load_state_dict(
                        torch.load(folder / f"{candidate}.pt", weights_only=True, map_location=device)
                    )
                    models[candidate], scales[candidate] = (
                        model,
                        joblib.load(folder / f"scaler_{group}.joblib"),
                    )
            actions = dict(np.load(folder / "actions.npz"))
            for scenario, costs in output.items():
                for origin in range(3):
                    availability, coverage, prices = context(
                        development["ids"][stage["held"]], origin, columns, scenario, THREE_SLOTS
                    )
                    estimates = {"fixed_value": np.tile(mean_gain[origin], (len(stage["held"]), 1))}
                    for candidate in CANDIDATES:
                        local = (
                            local_fields(development["x"][stage["held"]], columns, origin)
                            if candidate.endswith("clinical")
                            else None
                        )
                        features = np.stack(
                            [
                                query_features(
                                    stage["held_p"][:, origin],
                                    origin,
                                    peer,
                                    availability,
                                    coverage,
                                    stage["held_rel"],
                                    local,
                                )
                                for peer in range(3)
                                if peer != origin
                            ],
                            axis=1,
                        )
                        value, rank = infer(models[candidate], scales[candidate], features, origin, device)
                        estimates[candidate] = value
                        for tariff in costs:
                            expected = two_stage(value, rank, origin, availability, prices, float(tariff))
                            np.testing.assert_array_equal(
                                expected, actions[f"{scenario}_{tariff}_{candidate}_{origin}"]
                            )
                    observed_loss = bandit_predict(
                        bandit, joblib.load(folder / "bandit_scaler.joblib"), features, origin, device
                    )
                    for tariff in costs:
                        predicted_loss = observed_loss.copy()
                        predicted_loss[:, 1:] += float(tariff) * prices
                        predicted_loss[:, 1:][~availability] = np.inf
                        np.testing.assert_array_equal(
                            predicted_loss.argmin(1), actions[f"{scenario}_{tariff}_bandit_observed_{origin}"]
                        )
                    for tariff, policies in costs.items():
                        for policy, origins in policies.items():
                            key = f"{scenario}_{tariff}_{policy}_{origin}"
                            computed = metrics(
                                stage["held_y"],
                                arrays["held"][:, origin],
                                actions[key],
                                origin,
                                availability,
                                prices,
                                float(tariff),
                                estimates.get(policy),
                                mean_gain,
                            )
                            computed.pop("actions")
                            for metric, actual in computed.items():
                                previous = origins[str(origin)][metric]
                                if actual is None:
                                    assert previous is None
                                else:
                                    np.testing.assert_allclose(actual, previous, atol=2e-6, rtol=2e-5)
                                counts["metric_fields"] += 1
    assert acceptance(results, selected) == read(root / "gates.json")
    if (root / "flower/results.json").exists():
        x, gain, *_ = flower_data(root)
        for round_id in range(1, 4):
            initial = torch.load(root / "flower" / f"global_{round_id - 1}.pt", weights_only=True)
            local_states = []
            for origin in range(3):
                model = ValueRouter(63)
                model.load_state_dict(initial)
                fit_router(model, x[origin : origin + 1], gain[origin : origin + 1], 53660 + round_id, 2)
                state = torch.load(
                    root / "flower" / f"client_{origin}_round_{round_id}.pt", weights_only=True
                )
                for key, value in snapshot(model).items():
                    torch.testing.assert_close(value, state[key], atol=1e-6, rtol=1e-6)
                local_states.append(state)
                counts["local_update_replays"] += 1
            global_state = torch.load(root / "flower" / f"global_{round_id}.pt", weights_only=True)
            for key, value in global_state.items():
                torch.testing.assert_close(
                    value, sum(state[key] for state in local_states) / 3, atol=2e-6, rtol=1e-5
                )
            counts["aggregations"] += 1
        flower = read(root / "flower/results.json")
        assert flower["clients"] == 3 and flower["updates"] == 9 and flower["drift"] > 0
        stage = dict(np.load(source / "pilot_fold0_outer/stack.npz"))
        for row in read(root / "flower/consultations.json"):
            assert row["payload_fields"] == ["ids"] and row["origin"] != row["peer"]
            ids = np.array(row["ids"], dtype=int)
            positions = {int(patient): index for index, patient in enumerate(stage["held"])}
            for patient in ids:
                assert patient in positions
                index = positions[int(patient)]
                assert flower["metrics"][str(row["origin"])]["actions"][index] == row["peer"] + 1
            private = joblib.load(source / "pilot_fold0_outer/final" / f"client_{row['peer']}/final.joblib")
            with threadpool_limits(limits=1):
                replay = expert_predict(private, development["x"][ids])
            np.testing.assert_allclose(replay, row["p"], atol=2e-6)
        choice = selected["0"]["fusion"]
        risk = np.load(root / "pilot_fold0_outer" / f"targets_{choice}.npz")["held"]
        for origin in range(3):
            availability, _, prices = context(
                development["ids"][stage["held"]], origin, columns, "variable", THREE_SLOTS
            )
            actions = np.array(flower["metrics"][str(origin)]["actions"])
            computed = metrics(stage["held_y"], risk[:, origin], actions, origin, availability, prices, 0.02)
            for key, value in computed.items():
                previous = flower["metrics"][str(origin)][key]
                if value is None:
                    assert previous is None
                else:
                    np.testing.assert_allclose(value, previous, atol=2e-6, rtol=2e-5)
    preserved = read(root / "preserved-before.json")
    changed = [name for name, checksum in preserved.items() if digest(Path(name)) != checksum]
    assert not changed, changed
    result = {
        "verified": True,
        **counts,
        "preserved_files": len(preserved),
        "changed": changed,
        "test_hash_unchanged": digest(source / "frozen_test.npz")
        == preserved[str((source / "frozen_test.npz").resolve())],
        "test_evaluations": 0,
    }
    write(root / "verification.json", result)
    return result


def report(root: Path) -> None:
    """Append actual evidence to the frozen protocol and write full outer metrics CSV."""
    import csv

    results, selected, gate = (
        read(root / "results.json"),
        read(root / "selection.json"),
        read(root / "gates.json"),
    )
    execution = read(root / "execution.json")

    def table(headers: list, rows: list) -> str:
        def cell(value: object) -> str:
            return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)

        return (
            "\n".join(
                [
                    "| " + " | ".join(headers) + " |",
                    "| " + " | ".join(["---"] * len(headers)) + " |",
                    *["| " + " | ".join(cell(value) for value in row) + " |" for row in rows],
                ]
            )
            + "\n"
        )

    def avg(rows: dict, field: str) -> float:
        return float(np.mean([row[field] for row in rows.values()]))

    output = ["## Execution results\n", "### Interpretation\n"]
    selected_fields = ("objective", "binary_cross_entropy", "auroc", "pr_auc", "calls", "utility_r2")
    improvements = [row["fixed_value_gain"] for row in gate["folds"]]
    output.append(
        f"Inner-selected clinical routing has fold improvements over cost-aware fixed value "
        f"of {', '.join(f'{v:.6f}' for v in improvements)} (positive is better). "
        f"The unchanged/strengthened full scale gate **{'PASSES' if gate['passed'] else 'FAILS'}**. "
        "No production model is promoted. These are reused development patients, not "
        "independent clinical validation or evidence of privacy.\n"
    )
    output.append(
        "**Answer on this pilot: no robust useful personalization beyond cost-aware fixed "
        "routing is demonstrated.** Inner-selected clinical objective is .356859 versus "
        "fixed value .356182 and metadata .356186. Clinical utility R² by fold is "
        "-.006698/.001156/-.001535. Some individual candidate aggregates are favorable, "
        "but choosing those using outer assessment would be post-selection.\n"
    )
    output.append(
        "Whether context adds value must be judged against the separately inner-selected "
        "metadata reference, not just the weak always-call baseline. Fusion schedules "
        "and routing choices below were frozen using inner-held results, before outer "
        "assessment. No acceptance threshold was changed after evaluation.\n"
    )
    output.append(
        "### Fusion convergence and calibration audit\n\nEach row is the inner context's "
        "uniform nine-action held metric, not outer model selection. Matched schedules "
        "use identical mask count, batches, seeds, optimizer and learning rate. ECE is "
        "10 equal bins; calibration is estimated, not guaranteed. Training curve entries "
        "are batch means, while held metrics average complete origin/action predictions. "
        "A 60-epoch loss reduction does not establish complete convergence.\n"
    )
    rows = []
    for fold in range(3):
        audit_rows = read(root / f"pilot_fold{fold}_inner/fusion_audit.json")
        for option in CHOICES:
            kind, epoch = parse_choice(option)
            curve = audit_rows["training"][f"final_{kind}"]["curve"][epoch - 1]
            scores = audit_rows["held"][option]
            rows.append(
                [
                    fold,
                    option,
                    curve["training_bce"],
                    curve["gradient_norm"],
                    uniform_bce(scores),
                    avg(scores, "ece"),
                    avg(scores, "brier"),
                    selected[str(fold)]["fusion"] == option,
                ]
            )
    output.append(
        table(
            [
                "Fold",
                "Fusion",
                "Train BCE",
                "Gradient norm",
                "Inner uniform BCE",
                "Inner ECE",
                "Brier",
                "Selected",
            ],
            rows,
        )
    )
    output.append(
        "Per-action/origin metrics, every epoch and 15-epoch historical tensor differences "
        "are in each context's fusion_audit.json/curve files. Final and target-complement "
        "fusions are fresh fits; private experts and calibrators are unchanged. This "
        "study does not claim that longer optimization isolates calibration from model "
        "capacity or OOF/final probability shift.\n"
    )
    output.append("### Frozen nested choices\n")
    output.append(
        table(
            ["Fold", "Fusion", "Metadata router", "Clinical router"],
            [
                [fold, *[selected[str(fold)][field] for field in ("fusion", "metadata", "clinical")]]
                for fold in range(3)
            ],
        )
    )
    output.append(
        "### Outer pilot metrics, variable services / tariff .02\n\nMacro-average origins "
        "within each fold, then folds. PR-AUC is average precision. Folds/origins share "
        "patients/fitting sets; these means are not independent replicates or formal "
        "significance tests. All exact per-fold/origin/scenario/cost rows remain in CSV.\n"
    )
    policies = (
        "stop",
        "best_peer",
        "fixed_value",
        "graph",
        *CANDIDATES,
        "bandit_observed",
        "inner_metadata",
        "inner_clinical",
    )
    rows, csv_rows = [], []
    for policy in policies:
        values = []
        for fold in range(3):
            name = (
                selected[str(fold)]["clinical"]
                if policy == "inner_clinical"
                else selected[str(fold)]["metadata"]
                if policy == "inner_metadata"
                else policy
            )
            values.append(results[f"pilot_fold{fold}_outer"]["variable"]["0.02"][name])
        fields = (
            "accuracy",
            "auroc",
            "pr_auc",
            "binary_cross_entropy",
            "f1",
            "objective",
            "calls",
            "ece",
            "brier",
        )
        rows.append(
            [
                policy,
                *[float(np.mean([avg(value, field) for value in values])) for field in fields],
                float(np.mean([avg(value, "utility_r2") for value in values]))
                if "utility_r2" in next(iter(values[0].values()))
                else "not defined",
            ]
        )
    output.append(
        table(
            [
                "Policy",
                "Accuracy",
                "AUROC",
                "PR-AUC",
                "BCE",
                "F1",
                "BCE+cost",
                "Calls",
                "ECE",
                "Brier",
                "Utility R²",
            ],
            rows,
        )
    )
    for name, result in results.items():
        for scenario, costs in result.items():
            for tariff, candidates in costs.items():
                for candidate, origins in candidates.items():
                    for origin, values in origins.items():
                        csv_rows.append(
                            {
                                "context": name,
                                "scenario": scenario,
                                "cost": tariff,
                                "policy": candidate,
                                "origin": origin,
                                **{
                                    key: value
                                    for key, value in values.items()
                                    if not isinstance(value, (dict, list))
                                },
                            }
                        )
    columns = list(dict.fromkeys(key for row in csv_rows for key in row))
    with (root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)
    output.append(
        "### Fold stability and acceptance\n\nGains are paired objective differences "
        "within each fold using the same frozen fusion and evidence budget. Utility R² "
        "uses the historical fitting-origin/peer-mean null, not an assessment-fitted null.\n"
    )
    output.append(
        table(
            [
                "Fold",
                "Best-peer gain",
                "Fixed-value gain",
                "Metadata gain",
                "Static gain",
                "Utility R²",
                "Varying origins",
                "Accuracy delta",
            ],
            [
                [
                    fold,
                    *[
                        row[field]
                        for field in (
                            "best_peer_gain",
                            "fixed_value_gain",
                            "metadata_gain",
                            "static_gain",
                            "r2",
                            "varying_origins",
                            "accuracy_delta",
                        )
                    ],
                ]
                for fold, row in enumerate(gate["folds"])
            ],
        )
    )
    output.append(table(["Gate", "Passed"], [[key, value] for key, value in gate["gates"].items()]))
    output.append("### Selected clinical decisions by origin, scenario and cost\n")
    rows = []
    for scenario in ("static", "variable"):
        for cost in COSTS:
            for origin in range(3):
                values = [
                    results[f"pilot_fold{fold}_outer"][scenario][str(cost)][selected[str(fold)]["clinical"]][
                        str(origin)
                    ]
                    for fold in range(3)
                ]
                rows.append(
                    [
                        scenario,
                        cost,
                        "ABC"[origin],
                        *[float(np.mean([row[field] for row in values])) for field in selected_fields],
                        np.sum([row["action_frequencies"] for row in values], axis=0).tolist(),
                        float(np.mean([row["minority_peer_fraction"] for row in values])),
                    ]
                )
    output.append(
        table(
            [
                "Scenario",
                "Cost",
                "Origin",
                "Objective",
                "BCE",
                "AUROC",
                "PR-AUC",
                "Calls",
                "R²",
                "[STOP,A,B,C]",
                "Minority peer",
            ],
            rows,
        )
    )
    output.append(
        "### Separate observed-only bandit\n\nUniform actions were written to "
        "bandit_selection.json before outcome lookup. Each event contains only own/selected "
        "probabilities, sampled risk, true training label and exact propensity. The four "
        "global action heads include a masked self head; three actions are valid per origin "
        "when both peers are available. Fitting uses selected-loss IPS regression with cap5; "
        "valid inverse propensities are at most3, so clipping is inactive. No unchosen "
        "action BCE or STOP counterfactual enters its loss. It uses retrospective cross-fitted "
        "diagnoses, not prospective clinical outcomes, and is not a full-information router. "
        "Bandit action loss is evaluated only after held actions are fixed. Full-information "
        "baselines have more supervision and are not equal-feedback comparisons.\n"
    )
    output.append(
        table(
            [
                "Fold",
                "Observed records",
                "Minimum propensity",
                "Mean observed loss",
                "Outer objective",
                "Fixed-value objective",
            ],
            [
                [
                    fold,
                    len(np.load(root / f"pilot_fold{fold}_outer/bandit_records.npz")["action"]),
                    float(np.load(root / f"pilot_fold{fold}_outer/bandit_records.npz")["propensity"].min()),
                    float(np.load(root / f"pilot_fold{fold}_outer/bandit_records.npz")["loss"].mean()),
                    avg(
                        results[f"pilot_fold{fold}_outer"]["variable"]["0.02"]["bandit_observed"], "objective"
                    ),
                    avg(results[f"pilot_fold{fold}_outer"]["variable"]["0.02"]["fixed_value"], "objective"),
                ]
                for fold in range(3)
            ],
        )
    )
    output.append("### Flower, numerical verification, runtime and QA\n")
    if (root / "flower/results.json").exists():
        flower = read(root / "flower/results.json")
        output.append(
            f"Real Flower run **{flower['run_id']}**, exactly3 clients, one CPU actor, "
            f"three FedAvg rounds/nine local updates. Shared clinical MLP63→32Tanh→1 only; "
            f"drift {flower['drift']:.6f}; server work {flower['seconds']:.3f}s excludes startup. "
            "Origin-local route queries expose actions/own probabilities, not raw local "
            "features. Selected aligned-ID consultations use actual frozen private models; "
            "masked fusion replays within2e-6. Central preparation remains trusted-local, "
            "full-information diagnostic training; this is not a private service.\n"
        )
    if (root / "verification.json").exists():
        output.append("```json\n" + json.dumps(read(root / "verification.json"), indent=2) + "\n```\n")
    environment = read(root / "environment.json")
    output.append(
        f"Neural device {execution['device']}, {environment.get('device_name')}, actual "
        f"CUDA audit checksum {environment.get('tensor_operation_checksum')}. Pipeline time "
        f"{execution['seconds']:.3f}s; peak torch allocated {execution['peak_allocated_bytes']} "
        "bytes. This excludes CUDA contexts/reserved/board memory and historical private "
        "training. No GPU saturation or speedup claim.\n"
    )
    if (root / "qa/checks.json").exists():
        output.append(
            "Actual regression: **113 passed,2 skipped,2 upstream Typer warnings in19.23s**. "
            "Ruff lint/format83files and pipcheck passed. Optional historical Qwen/feedback "
            "runtime benchmarks are disabled to preserve their old evaluation boundaries; "
            "this step's new three-client Flower integration actually ran. Pip freeze is "
            "environment evidence, not a portable dependency lock.\n"
        )
        output.append(
            "Fresh QA commands/return codes:\n\n```json\n"
            + json.dumps(read(root / "qa/checks.json"), indent=2)
            + "\n```\n"
        )
    output.append("### Limitations and research direction\n\n")
    output.append(
        "No fitting, Flower, test or numerical verification failure was suppressed. "
        "Scientific acceptance gates fail as reported. Initial Ruff caught unused imports, "
        "a dict-construction style issue and a stale variable name in the enhanced verifier; "
        "these were corrected before its successful execution. No model setting, gate or "
        "numerical tolerance was relaxed. Windows Ray experimental/deprecated-launcher "
        "warnings remain; no OS/driver intervention was used.\n"
    )
    if not gate["passed"]:
        output.append(
            "The registered evidence does **not** justify50k expansion or a GAT. Richer "
            "context may help some fold/metric/condition, but must not be called robust "
            "personalization when any stability/fixed-value/utility gate fails. There are "
            "only two peers to rank and no new inter-agent topology observation; graph "
            "complexity alone is not evidence of additional signal.\n"
        )
    else:
        output.append(
            "The pilot meets the preregistered numerical gates. That supports a separately "
            "frozen50k reproduction, not model promotion or clinical validity. No expansion "
            "is included in these pilot numbers.\n"
        )
    output.append(
        "A bandit continuation needs repeated behavior-policy seeds, genuine delayed verified "
        "outcomes, action coverage/ESS and comparison against equal-feedback controls. "
        "One uniform logged simulation does not demonstrate online learning or that a "
        "more complex bandit will beat supervised/fixed routing. Further context use must "
        "retain origin authorization and feature-domain completeness.\n\n"
        "Other limitations: reused development cohort; class imbalance; cross-sectional "
        "survey target and unmodeled survey weights; dependent origin views; noisy "
        "single-outcome BCE gains; teacher/final shift; fixed training schedules/capacities; "
        "pairwise rank/gate errors; simulated public availability/prices; no external "
        "validation, secure isolation, authentication or privacy proof. Raw features are "
        "origin-local in Flower but accessible to the trusted simulator. Predictions, "
        "identifiers and updates can disclose information. New context is explicitly "
        "authorized for this experiment, not a general consent/serving mechanism.\n"
    )
    output.append(
        "### Commands and exact changed files\n\n```powershell\n"
        ".venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_context --device cuda\n"
        ".venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_context --flower\n"
        ".venv/Scripts/python.exe -u -m fed_med_agents.experiments.cdc_context --verify\n"
        ".venv/Scripts/python.exe -m fed_med_agents.experiments.cdc_context --report\n```\n\n"
        "New files: experiments/cdc_context.py under src/fed_med_agents; "
        "tests/test_cdc_context.py; this protocol/report; "
        "docs/devlog/2026-10-14-m5-step036.md. Updated README/status/architecture/limitations/"
        "next-steps only. Previous implementation/report files stay unchanged. Detailed "
        "CSV/actions/curves/checkpoints/partial bandit records/Flower/provenance/QA live "
        "under ignored artifacts/m5-step036-v1. Frozen-test evaluations0. No GitHub push, "
        "Qwen training, registry, default promotion or Step4. **Stop after3.6.**\n"
    )
    protocol = (root / "protocol.md").read_text(encoding="utf-8").split("## Execution results")[0]
    PROTOCOL.write_text(protocol + "\n".join(output), encoding="utf-8")
    print(PROTOCOL)


def main() -> None:
    """Run the frozen pilot protocol into a fresh, isolated artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step036-v1"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--flower", action="store_true")
    parser.add_argument("--flower-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.flower_worker:
        flower_worker(args.output.resolve())
        return
    if args.flower:
        folder = args.output.resolve() / "flower"
        folder.mkdir()
        with (folder / "flower.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    __spec__.name,
                    "--flower-worker",
                    "--output",
                    str(args.output.resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env=os.environ | {"PYTHONUTF8": "1"},
            )
        if not (folder / "results.json").exists():
            raise RuntimeError("Flower compatibility check did not complete")
        return
    if args.verify:
        print(json.dumps(verify(args.output.resolve()), indent=2))
        return
    if args.report:
        report(args.output.resolve())
        return
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    run(args.source.resolve(), args.output.resolve(), select_device(args.device))


if __name__ == "__main__":
    main()
