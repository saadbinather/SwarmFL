"""Separate group-safe CDC pilot; all model assessment stays inside development data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from threadpoolctl import threadpool_limits

from ..devices import audit, peak_memory, select_device, synchronize
from ..routing import ContextualRouter
from .cdc_data import DOMAINS, acquire, grouped_split, prepare_stack
from .cdc_routing import (
    CLIENTS,
    COSTS,
    context,
    decide,
    design,
    fit_fusion,
    gross_gains,
    metrics,
    outcomes,
    prequery,
    ridge,
    router_network,
    target_vector,
    train_router,
)
from .m5_step03 import digest, write
from .routing_diagnostics import loss_rows, snapshot

KINDS = ("original", "logistic", "mean")
VALUES = ("ridge_base", "ridge_enriched", "mlp_base", "mlp_enriched")


def reliability(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([loss_rows(y, p[:, c]).mean() for c in range(p.shape[1])], dtype=np.float32)


def prepare(
    root: Path,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    columns: list[str],
    fit: np.ndarray,
    held: np.ndarray,
    seed: int,
    device: torch.device,
) -> None:
    """Refit whole upstream stacks within each utility target complement, never held labels."""
    p, hp = prepare_stack(root / "final", x, y, groups, columns, fit, held, seed)
    tp, rel = np.zeros_like(p), np.zeros_like(p)
    target_outcomes = {kind: np.empty((len(fit), CLIENTS, CLIENTS + 1), dtype=np.float32) for kind in KINDS}
    provenance, fitting_logs = [], {}
    for fold, (a, b) in enumerate(
        StratifiedGroupKFold(3, shuffle=True, random_state=seed + 30).split(fit, y[fit], groups[fit])
    ):
        parent, target = fit[a], fit[b]
        folder = root / f"target{fold}"
        pfit, pheld = prepare_stack(folder, x, y, groups, columns, parent, target, seed + 100 + fold)
        tp[b] = pheld
        rel[b] = reliability(pfit, y[parent])
        for kind in KINDS:
            fusion, log = fit_fusion(pfit, y[parent], kind, seed, device)
            target_outcomes[kind][b] = outcomes(fusion, pheld, device)
            torch.save(snapshot(fusion), folder / f"fusion_{kind}.pt")
            fitting_logs[f"target{fold}_{kind}"] = log
        provenance.append({"fit": parent.tolist(), "target": target.tolist(), "positions": b.tolist()})
    for kind in KINDS:
        fusion, log = fit_fusion(p, y[fit], kind, seed, device)
        torch.save(snapshot(fusion), root / f"fusion_{kind}.pt")
        fitting_logs[f"final_{kind}"] = log
        np.savez_compressed(
            root / f"targets_{kind}.npz", train=target_outcomes[kind], held=outcomes(fusion, hp, device)
        )
    np.savez_compressed(
        root / "stack.npz",
        p=tp,
        rel=rel,
        y=y[fit],
        fit=fit,
        held=held,
        held_p=hp,
        held_y=y[held],
        held_rel=np.tile(reliability(p, y[fit]), (len(held), 1)),
    )
    write(root / "target_provenance.json", {"folds": provenance, "seed": seed})
    write(root / "fusion_training.json", fitting_logs)


def fit_graph(
    p: np.ndarray,
    risks: np.ndarray,
    labels: np.ndarray,
    ids: np.ndarray,
    columns: list[str],
    seed: int,
    device: torch.device,
    slots: tuple[int, ...] | None = None,
) -> ContextualRouter:
    """Original compatible graph architecture with frozen cross-fitted action outcomes."""
    torch.manual_seed(seed)
    clients = p.shape[1]
    model = ContextualRouter(clients, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    n = len(p)
    own = torch.tensor(p.T.flatten(), device=device)
    origins = torch.arange(clients, device=device).repeat_interleave(n)
    availability, prices = [], []
    for o in range(clients):
        available, _, price = context(ids, o, columns, "variable", slots)
        availability.append(available)
        prices.append(price)
    available = torch.tensor(np.concatenate(availability), device=device)
    costs = torch.cat(
        [
            torch.zeros((n * clients, 1), device=device),
            torch.tensor(np.concatenate(prices), device=device) * 0.02,
        ],
        -1,
    )
    loss_array = np.stack(
        [
            np.column_stack([loss_rows(labels, risks[:, o, a]) for a in range(clients + 1)])
            for o in range(clients)
        ]
    )
    losses = torch.tensor(loss_array.reshape(-1, clients + 1), device=device)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(15):
        for batch in torch.randperm(n * clients, generator=generator).split(4096):
            b = batch.to(device)
            optimizer.zero_grad()
            scores = model(origins[b], own[b], 2 * (own[b] - 0.5).abs(), available[b].float(), available[b])
            loss = (scores.softmax(-1) * (losses[b] + costs[b])).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    return model


def infer_values(
    model: Any,
    own: np.ndarray,
    origin: int,
    available: np.ndarray,
    covered: np.ndarray,
    rel: np.ndarray,
    enriched: bool,
    device: torch.device,
    domain_config: dict = DOMAINS,
) -> np.ndarray:
    """Prospective API deliberately has no outcomes or unqueried peer probabilities."""
    values = np.zeros((len(own), len(domain_config)), dtype=np.float32)
    for peer in range(len(domain_config)):
        if peer == origin:
            continue
        inputs = prequery(own, origin, peer, available, covered, rel, enriched, domain_config)
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                values[:, peer] = model(torch.tensor(inputs, device=device)).flatten().cpu().numpy()
        else:
            values[:, peer] = model.predict(inputs)
    return values


def run_context(
    root: Path,
    ids: np.ndarray,
    columns: list[str],
    seed: int,
    device: torch.device,
    domain_config: dict = DOMAINS,
    slots: tuple[int, ...] | None = None,
) -> dict:
    """All fixed candidates, three frozen fusion conditions and matched eligibility/cost."""
    data = dict(np.load(root / "stack.npz"))
    clients = len(domain_config)
    assert data["p"].shape[1] == clients
    result, action_arrays = {}, {}
    for kind in KINDS:
        targets = dict(np.load(root / f"targets_{kind}.npz"))
        gain = gross_gains(targets["train"], data["y"])
        mean_gain = gain.mean(0)
        models = {}
        logs = {}
        for enriched in (False, True):
            group = "enriched" if enriched else "base"
            x = design(data["p"], ids[data["fit"]], columns, data["rel"], enriched, domain_config, slots)
            target = target_vector(gain)
            linear = ridge(x, target)
            joblib.dump(linear, root / f"router_{kind}_ridge_{group}.joblib")
            models[f"ridge_{group}"] = linear
            torch.manual_seed(seed)
            network = router_network(x.shape[1]).to(device)
            logs[f"mlp_{group}"] = train_router(network, x, target, 20, seed, device)
            torch.save(snapshot(network), root / f"router_{kind}_mlp_{group}.pt")
            models[f"mlp_{group}"] = network
        graph = fit_graph(
            data["p"], targets["train"], data["y"], ids[data["fit"]], columns, seed, device, slots
        )
        torch.save(snapshot(graph), root / f"router_{kind}_graph.pt")
        result[kind] = {}
        for scenario in ("static", "variable"):
            result[kind][scenario] = {}
            for cost in COSTS:
                result[kind][scenario][str(cost)] = {}
            for origin in range(clients):
                available, covered, prices = context(ids[data["held"]], origin, columns, scenario, slots)
                own = data["held_p"][:, origin]
                estimates = {
                    name: infer_values(
                        model,
                        own,
                        origin,
                        available,
                        covered,
                        data["held_rel"],
                        name.endswith("enriched"),
                        device,
                        domain_config,
                    )
                    for name, model in models.items()
                }
                estimates["fixed_value"] = np.tile(mean_gain[origin], (len(own), 1))
                scores = np.tile(mean_gain[origin], (len(own), 1))
                scores[:, origin] = -np.inf
                scores[~available] = -np.inf
                best = scores.argmax(1) + 1
                best[~np.isfinite(scores.max(1))] = 0
                random = np.random.default_rng(seed + origin).random(available.shape)
                random[:, origin] = -np.inf
                random[~available] = -np.inf
                random_action = random.argmax(1) + 1
                random_action[~np.isfinite(random.max(1))] = 0
                with torch.no_grad():
                    tensor = torch.tensor(own, device=device)
                    availability = torch.tensor(available, device=device)
                    graph_scores = (
                        graph(
                            torch.full((len(own),), origin, device=device),
                            tensor,
                            2 * (tensor - 0.5).abs(),
                            availability.float(),
                            availability,
                        )
                        .cpu()
                        .numpy()
                    )
                for cost in COSTS:
                    actions = {
                        name: decide(values, origin, available, prices, cost)
                        for name, values in estimates.items()
                    }
                    actions.update(
                        stop=np.zeros(len(own), dtype=int),
                        best_peer=best,
                        random=random_action,
                        confidence=np.where(2 * np.abs(own - 0.5) < 0.6, best, 0),
                    )
                    adjusted = graph_scores.copy()
                    adjusted[:, 1:] -= cost * prices
                    actions["graph"] = adjusted.argmax(1)
                    for name in VALUES:
                        actions[f"random_matched_{name}"] = np.where(actions[name] > 0, random_action, 0)
                    for name, selected in actions.items():
                        row = metrics(
                            data["held_y"],
                            targets["held"][:, origin],
                            selected,
                            origin,
                            available,
                            prices,
                            cost,
                            estimates.get(name),
                            mean_gain,
                        )
                        action_arrays[f"{kind}_{scenario}_{cost}_{name}_{origin}"] = selected.astype(np.int8)
                        row.pop("actions")
                        result[kind][scenario][str(cost)].setdefault(name, {})[str(origin)] = row
        write(root / f"router_training_{kind}.json", logs)
    np.savez_compressed(root / "actions.npz", **action_arrays)
    write(root / "results.json", result)
    return result


def select_inner(results: dict, size: str) -> dict:
    """Freeze candidate names from inner-selection scores only, separately per fusion."""
    selected = {}
    for fold in range(3):
        selected[str(fold)] = {}
        for kind in KINDS:
            rows = results[f"{size}_fold{fold}_inner"][kind]["variable"]["0.02"]
            scores = {name: float(np.mean([r["objective"] for r in rows[name].values()])) for name in VALUES}
            selected[str(fold)][kind] = min(scores, key=scores.get)
    return selected


def gates(results: dict, selection: dict, size: str) -> dict:
    """Conservative preregistered scale gates; never replace a production checkpoint."""
    gate = {}
    for kind in KINDS:
        improvements, static, r2, variation, accuracy, changes = [], [], [], [], [], []
        for fold in range(3):
            chosen = selection[str(fold)][kind]
            data = results[f"{size}_fold{fold}_outer"][kind]
            rows = list(data["variable"]["0.02"][chosen].values())
            base = list(data["variable"]["0.02"]["best_peer"].values())
            improvements.append(
                float(np.mean([b["objective"] - a["objective"] for a, b in zip(rows, base, strict=True)]))
            )
            static.append(
                float(
                    np.mean(
                        [
                            data["static"]["0.02"]["best_peer"][str(o)]["objective"]
                            - data["static"]["0.02"][chosen][str(o)]["objective"]
                            for o in range(len(rows))
                        ]
                    )
                )
            )
            r2.append(float(np.mean([r["utility_r2"] for r in rows])))
            variation.append(sum(r["minority_peer_fraction"] >= 0.05 for r in rows))
            accuracy.extend(a["accuracy"] - b["accuracy"] for a, b in zip(rows, base, strict=True))
            changes.append(
                [
                    a["binary_cross_entropy"] - b["binary_cross_entropy"]
                    for a, b in zip(rows, base, strict=True)
                ]
            )
        passed = {
            "all_fold_improvement_.005": all(v >= 0.005 for v in improvements),
            "all_fold_static_gain": all(v > 0 for v in static),
            "all_fold_r2_.05": all(v >= 0.05 for v in r2),
            "three_varying_origins": all(v >= 3 for v in variation),
            "accuracy_drop_.005": np.mean(accuracy) >= -0.005,
            "no_origin_bce_degradation_.01": bool((np.mean(changes, axis=0) <= 0.01).all()),
        }
        gate[kind] = {
            "gates": {k: bool(v) for k, v in passed.items()},
            "passed": bool(all(passed.values())),
            "fold_improvements": improvements,
            "static_improvements": static,
            "fold_utility_r2": r2,
            "varying_origins": variation,
        }
    return gate


def study(root: Path, pilot: np.ndarray, data: dict, columns: list[str], device: torch.device) -> None:
    """Run matched small/full nested studies; save selections before outer routing assessment."""
    x, y, groups, ids = (data[k] for k in ("x", "y", "groups", "ids"))
    contexts, results = {}, {}
    split = StratifiedGroupKFold(3, shuffle=True, random_state=53501)
    outer = list(split.split(pilot, y[pilot], groups[pilot]))
    for size in ("small", "pilot"):
        for fold, (a, b) in enumerate(outer):
            fitting, held = pilot[a], pilot[b]
            if size == "small":
                fitting, _ = train_test_split(
                    fitting, train_size=2500, stratify=y[fitting], random_state=53520 + fold
                )
            inner, selection = grouped_split(fitting, y, groups, 53510 + fold)
            for stage, fit, assessment in (("inner", inner, selection), ("outer", fitting, held)):
                key, seed = f"{size}_fold{fold}_{stage}", 53530 + fold
                start = time.perf_counter()
                prepare(root / key, x, y, groups, columns, fit, assessment, seed, device)
                contexts[key] = {
                    "fit": fit.tolist(),
                    "held": assessment.tolist(),
                    "seed": seed,
                    "prepare_seconds": time.perf_counter() - start,
                }
                write(root / "contexts.json", contexts)
                print(
                    f"Prepared {key}: {len(fit)}/{len(assessment)} in {time.perf_counter() - start:.1f}s",
                    flush=True,
                )
        for key in contexts:
            if key.startswith(size) and key.endswith("inner"):
                results[key] = run_context(root / key, ids, columns, contexts[key]["seed"], device)
                print(f"Completed selection {key}", flush=True)
        selected = select_inner(results, size)
        write(root / f"selection_{size}.json", selected)
        for key in contexts:
            if key.startswith(size) and key.endswith("outer"):
                results[key] = run_context(root / key, ids, columns, contexts[key]["seed"], device)
                print(f"Completed assessment {key}", flush=True)
        write(root / "results.json", results)
        write(root / f"gates_{size}.json", gates(results, selected, size))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step035-v1"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error("Fresh output required; do not overwrite evidence")
    root.mkdir(parents=True)
    preserved = {}
    for path in Path("artifacts").rglob("*"):
        if (
            path.is_file()
            and root not in path.resolve().parents
            and "ollama" not in path.parts
            and path.suffix in {".pt", ".npz", ".npy", ".json", ".jsonl", ".joblib", ".log", ".csv"}
        ):
            preserved[str(path)] = digest(path)
    for path in Path("docs/experiments").glob("*.md"):
        if "m5-step035" not in path.name:
            preserved[str(path)] = digest(path)
    write(root / "preserved-before.json", preserved)
    (root / "protocol.md").write_bytes(Path("docs/experiments/m5-step035-protocol.md").read_bytes())
    audit(root / "environment.json")
    summary = acquire(root)
    device = select_device(args.device)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        data = dict(np.load(root / "development.npz"))
        study(root, np.load(root / "pilot_ids.npy"), data, summary["features"], device)
    synchronize(device)
    scale = json.loads((root / "gates_pilot.json").read_text())
    passed = any(v["passed"] for v in scale.values())
    write(
        root / "execution.json",
        {
            "seconds": time.perf_counter() - start,
            "device": str(device),
            "peak_allocated_bytes": peak_memory(device),
            "scale_gate_passed": passed,
            "scaled_to_50k": False,
            "test_evaluations": 0,
        },
    )
    if passed:
        raise RuntimeError(
            "Pilot passed scale gate: execute preregistered fresh 50k reproduction before completion"
        )
    print("Pilot gate failed: STOP scaling; historical/default models unchanged", flush=True)


if __name__ == "__main__":
    main()
