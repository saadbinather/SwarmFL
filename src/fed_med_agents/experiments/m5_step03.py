"""Training-partition-only nested cross-fitted consultation-value experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..data import GROUPS
from ..devices import audit
from ..fusion import CollaborativeModel
from ..routing import ContextualRouter
from .consultation_value import COSTS, FEATURE_SETS, choose, features, fit_regressor, predict, summarize
from .repair_data import prepare_context, training_source
from .repair_models import action_outputs, make_model, uniform_loss
from .routing_diagnostics import describe, loss_rows, snapshot


def write(path: Path, value: dict) -> None:
    """Write readable strict JSON, failing on nonfinite evidence."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    """SHA256 of an evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Local trusted diagnostic loader; subsequent Flower queries audit transport."""
    train, held = [], []
    for client in range(3):
        with np.load(path / "seed_42" / f"client_{client}" / "local.npz") as data:
            train.append(data["oof"].copy())
            held.append(data["assessment"].copy())
    return np.array(train).T, np.array(held).T


def fit_fusion(p: np.ndarray, y: np.ndarray, kind: str, seed: int, path: Path | None) -> CollaborativeModel:
    """Fit fusion uniformly on fitting-only specialist OOF, then freeze it."""
    torch.manual_seed(seed)
    model = make_model(kind)
    for parameter in model.router.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.fusion.parameters(), lr=0.01)
    inputs, labels = torch.tensor(np.tile(p, (3, 1))), torch.tensor(np.tile(y, 3))
    origins = torch.arange(3).repeat_interleave(len(y))
    for _ in range(40):
        optimizer.zero_grad()
        loss = uniform_loss(model, inputs, labels, origins)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.fusion.parameters(), 5)
        optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if path is not None:
        torch.save(snapshot(model), path)
    return model


@torch.no_grad()
def risks(model: CollaborativeModel, p: np.ndarray) -> np.ndarray:
    """Enumerate offline action outcomes; not a prospective routing input."""
    return np.stack(
        [
            action_outputs(model, torch.tensor(p), torch.full((len(p),), o)).sigmoid().numpy()
            for o in range(3)
        ],
        axis=1,
    )


def reliability(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Train-only calibration/reliability summaries, three statistics per client."""
    return np.array(
        [[describe(y, p[:, c])[key] for key in ("binary_cross_entropy", "brier", "ece")] for c in range(3)],
        dtype=np.float32,
    )


def gains(outcomes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gross BCE gain; consultation tariff is subtracted only by the decision rule."""
    losses = np.stack(
        [np.column_stack([loss_rows(y, outcomes[:, o, a]) for a in range(4)]) for o in range(3)], axis=1
    )
    return (losses[:, :, :1] - losses[:, :, 1:]).astype(np.float32)


def matrix(p: np.ndarray, rel: np.ndarray, history: np.ndarray, group: str) -> np.ndarray:
    """Stable order origin then candidate then patient for all six candidate pairs."""
    widths = [len(columns) for columns in GROUPS.values()]
    return np.concatenate(
        [
            features(p[:, o], o, c, rel, history, group, widths[c] / max(widths))
            for o in range(3)
            for c in range(3)
            if o != c
        ]
    )


def flatten_gains(target: np.ndarray) -> np.ndarray:
    return np.concatenate([target[:, o, c] for o in range(3) for c in range(3) if o != c])


def unpack_predictions(values: np.ndarray, n: int) -> np.ndarray:
    result = np.zeros((n, 3, 3), dtype=np.float32)
    for k, (o, c) in enumerate((o, c) for o in range(3) for c in range(3) if o != c):
        result[:, o, c] = values[k * n : (k + 1) * n]
    return result


def prepare_stage(
    root: Path,
    local: list,
    y: np.ndarray,
    original: np.ndarray,
    fitting: np.ndarray,
    held: np.ndarray,
    seed: int,
) -> None:
    """Cross-fit the entire upstream stack, excluding each target fold everywhere."""
    prepare_context(root / "final", local, y, original, fitting, held, seed)
    p, assessment = probabilities(root / "final")
    np.savez(root / "indices.npz", fit=fitting, held=held, original_ids=original)
    target_p = np.zeros_like(p)
    target_rel = np.zeros((len(fitting), 3, 3), dtype=np.float32)
    all_targets = {
        kind: np.zeros((len(fitting), 3, 4), dtype=np.float32) for kind in ("original", "logistic")
    }
    histories = {kind: np.zeros((len(fitting), 3, 3), dtype=np.float32) for kind in all_targets}
    provenance = []
    splitter = StratifiedKFold(3, shuffle=True, random_state=seed + 100)
    for fold, (train_pos, target_pos) in enumerate(splitter.split(fitting, y[fitting])):
        parent, target = fitting[train_pos], fitting[target_pos]
        folder = root / f"target{fold}"
        prepare_context(folder, local, y, original, parent, target, seed + 10 + fold)
        tp, hp = probabilities(folder)
        target_p[target_pos] = hp
        target_rel[target_pos] = reliability(tp, y[parent])
        for kind, target_outcomes in all_targets.items():
            model = fit_fusion(tp, y[parent], kind, seed, folder / f"fusion_{kind}.pt")
            target_outcomes[target_pos] = risks(model, hp)
            histories[kind][target_pos] = gains(risks(model, tp), y[parent]).mean(0)
        provenance.append(
            {
                "target": target.tolist(),
                "all_upstream_fit": parent.tolist(),
                "target_positions": target_pos.tolist(),
            }
        )
    for kind, target_outcomes in all_targets.items():
        model = fit_fusion(p, y[fitting], kind, seed, root / f"fusion_{kind}.pt")
        # Training-only mean targets are summaries, never assessment-selected weights.
        target_gain = gains(target_outcomes, y[fitting])
        history = target_gain.mean(0)
        np.savez(
            root / f"{kind}.npz",
            p=target_p,
            rel=target_rel,
            history=histories[kind],
            gain=target_gain,
            target_risks=target_outcomes,
            y=y[fitting],
            held_p=assessment,
            held_y=y[held],
            held_risks=risks(model, assessment),
            held_rel=np.tile(reliability(p, y[fitting]), (len(held), 1, 1)),
            held_history=np.tile(history, (len(held), 1, 1)),
            mean_gain=history,
        )
    write(root / "provenance.json", {"folds": provenance, "seed": seed})


def graph_fit(p: np.ndarray, outcomes: np.ndarray, y: np.ndarray, seed: int, cost: float) -> ContextualRouter:
    """Original graph architecture; frozen cross-fitted action losses, no fusion gradient."""
    torch.manual_seed(seed)
    router = make_model().router
    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)
    pp = torch.tensor(p)
    target = torch.tensor(
        np.stack(
            [np.column_stack([loss_rows(y, outcomes[:, o, a]) for a in range(4)]) for o in range(3)], axis=1
        )
    )
    for _ in range(50):
        optimizer.zero_grad()
        loss = 0
        for o in range(3):
            own = pp[:, o]
            scores = router(
                torch.full((len(y),), o),
                own,
                2 * (own - 0.5).abs(),
                torch.ones_like(pp),
                torch.ones_like(pp, dtype=torch.bool),
            )
            loss = (
                loss
                + (scores.softmax(-1) * (target[:, o] + cost * torch.tensor([0, 1, 1, 1]))).sum(1).mean() / 3
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 5)
        optimizer.step()
    return router


def run_stage(root: Path, seed: int) -> dict:
    """Fit fixed candidates, then assess them without exposing outcomes to routers."""
    result = {}
    for kind in ("original", "logistic"):
        data = dict(np.load(root / f"{kind}.npz"))
        p, rel, history, target = (data[k] for k in ("p", "rel", "history", "gain"))
        held, outcomes, y = (data[k] for k in ("held_p", "held_risks", "held_y"))
        means = target.mean(0)
        benefit = {}
        for group in FEATURE_SETS:
            train_x = matrix(p, rel, history, group)
            held_x = matrix(held, data["held_rel"], data["held_history"], group)
            for family in ("ridge", "mlp", "tree"):
                name = f"{family}_{group}"
                model = fit_regressor(family, train_x, flatten_gains(target), seed)
                benefit[name] = unpack_predictions(predict(model, held_x), len(held))
                if family == "mlp":
                    torch.save(snapshot(model), root / f"router_{kind}_{name}.pt")
                else:
                    joblib.dump(model, root / f"router_{kind}_{name}.joblib")
        # Compatible torch linear is a distinct central diagnostic, not ridge relabeled.
        x = matrix(p, rel, history, "enriched")
        hx = matrix(held, data["held_rel"], data["held_history"], "enriched")
        linear = fit_regressor("linear", x, flatten_gains(target), seed)
        benefit["linear_enriched"] = unpack_predictions(predict(linear, hx), len(held))
        torch.save(snapshot(linear), root / f"router_{kind}_linear_enriched.pt")
        np.savez(root / f"routing_{kind}.npz", x=x, target=flatten_gains(target), held_x=hx, **benefit)
        result[kind] = {}
        for cost in COSTS:
            graph = graph_fit(p, data["target_risks"], data["y"], seed, cost)
            torch.save(snapshot(graph), root / f"router_{kind}_graph_{cost}.pt")
            rows = {}
            for origin in range(3):
                estimates = {name: values[:, origin] for name, values in benefit.items()}
                estimates["fixed_value"] = np.tile(means[origin], (len(held), 1))
                actions = {name: choose(value, origin, cost) for name, value in estimates.items()}
                best = means[origin].copy()
                best[origin] = -np.inf
                peer = int(best.argmax()) + 1
                actions["best_peer"] = np.full(len(held), peer)
                actions["confidence"] = np.where(2 * np.abs(held[:, origin] - 0.5) < 0.6, peer, 0)
                actions["stop"] = np.zeros(len(held), dtype=int)
                peers = [c + 1 for c in range(3) if c != origin]
                random = np.random.default_rng(seed + origin).choice(peers, len(held))
                actions["random_max"] = random
                with torch.no_grad():
                    own = torch.tensor(held[:, origin])
                    scores = graph(
                        torch.full((len(held),), origin),
                        own,
                        2 * (own - 0.5).abs(),
                        torch.ones((len(held), 3)),
                        torch.ones((len(held), 3), dtype=torch.bool),
                    )
                    actions["graph"] = scores.argmax(-1).numpy()
                for name in [*benefit, "graph"]:
                    actions[f"random_matched_{name}"] = np.where(actions[name] > 0, random, 0)
                for name, selected in actions.items():
                    rows.setdefault(name, {})[str(origin)] = summarize(
                        y, outcomes[:, origin], selected, origin, cost, estimates.get(name), means
                    )
            result[kind][str(cost)] = rows
    write(root / "results.json", result)
    return result


def select(results: dict) -> dict:
    """Choose ONLY value policy names on inner scores, never outer assessment."""
    names = [f"{family}_{group}" for group in FEATURE_SETS for family in ("ridge", "mlp", "tree")]
    selected = {}
    for fold in range(3):
        selected[str(fold)] = {}
        for kind in ("original", "logistic"):
            rows = results[f"fold{fold}_inner"][kind]["0.02"]
            scores = {name: float(np.mean([r["objective"] for r in rows[name].values()])) for name in names}
            selected[str(fold)][kind] = {"name": min(scores, key=scores.get), "scores": scores}
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step03-v1"))
    parser.add_argument("--source", type=Path, default=Path("artifacts/milestone02/seed_42"))
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error("Use a fresh directory; earlier evidence must be preserved")
    root.mkdir(parents=True)
    manifest = {}
    for path in Path("artifacts").rglob("*"):
        if (
            path.is_file()
            and root not in path.resolve().parents
            and "ollama" not in path.parts
            and path.suffix in {".pt", ".npz", ".npy", ".json", ".jsonl", ".log", ".csv", ".txt"}
        ):
            manifest[str(path)] = digest(path)
    for path in Path("docs/experiments").glob("*.md"):
        if "m5-step03" not in path.name:
            manifest[str(path)] = digest(path)
    write(root / "preserved-before.json", manifest)
    shutil.copyfile("docs/experiments/m5-step03-protocol.md", root / "protocol.md")
    audit(root / "environment.json")
    torch.set_num_threads(1)
    start = time.perf_counter()
    local, y, original = training_source(args.source)
    contexts = {}
    outer = StratifiedKFold(3, shuffle=True, random_state=53003)
    for fold, (fit, assessment) in enumerate(outer.split(original, y)):
        inner, selection = train_test_split(fit, test_size=0.25, stratify=y[fit], random_state=53100 + fold)
        for stage, fitting, held in (("inner", inner, selection), ("outer", fit, assessment)):
            key, seed = f"fold{fold}_{stage}", 53200 + fold
            prepare_stage(root / key, local, y, original, fitting, held, seed)
            contexts[key] = {"fit": fitting.tolist(), "held": held.tolist(), "seed": seed}
            print(f"Prepared {key}: {len(fitting)}/{len(held)}", flush=True)
    write(root / "contexts.json", contexts)
    results = {}
    # Fix selections before any outer policy metric is computed.
    for key in contexts:
        if key.endswith("inner"):
            results[key] = run_stage(root / key, contexts[key]["seed"])
            print(f"Inner candidates complete: {key}", flush=True)
    selections = select(results)
    write(root / "selection.json", selections)
    neural_scores = {}
    for kind in ("original", "logistic"):
        neural_scores[kind] = {
            name: float(
                np.mean(
                    [
                        r["objective"]
                        for fold in range(3)
                        for r in results[f"fold{fold}_inner"][kind]["0.02"][f"{name}_enriched"].values()
                    ]
                )
            )
            for name in ("linear", "mlp")
        }
    write(
        root / "federated-selection.json",
        {kind: min(scores, key=scores.get) for kind, scores in neural_scores.items()},
    )
    for key in contexts:
        if key.endswith("outer"):
            results[key] = run_stage(root / key, contexts[key]["seed"])
            print(f"Outer candidates complete: {key}", flush=True)
    write(root / "results.json", results)
    write(
        root / "execution.json",
        {
            "seconds": time.perf_counter() - start,
            "device": "cpu",
            "specialist_fits": 288,
            "old_labels_used": False,
        },
    )
    print(f"Nested routing completed in {time.perf_counter() - start:.3f}s", flush=True)


if __name__ == "__main__":
    main()
