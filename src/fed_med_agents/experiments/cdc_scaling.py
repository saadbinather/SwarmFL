"""Step 5 development-only scaling and router-only continual learning evidence.

Frozen protocol/results live in ignored artifacts. Historical checkpoints and
frozen test labels are never loaded or overwritten by this runner.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from ..devices import audit, select_device, synchronize
from ..routing.policy import RoutingPolicy
from .cdc_context import fusion_fit, local_fields, query_features
from .cdc_data import THREE_DOMAINS, THREE_SLOTS, prepare_stack
from .cdc_routing import CDCFusion, context, gross_gains, metrics, outcomes
from .m5_step03 import digest, write
from .m5_step035 import reliability
from .routing_diagnostics import snapshot

SOURCE = Path("artifacts/m5-step035-three-agent-v1")
DEFAULT = Path("artifacts/m5-step05-v1")


def initialize(output: Path) -> None:
    """Freeze a fresh artifact-only protocol and historical preservation manifest.

    No fitting occurs. Existing outputs are never overwritten. The registered
    exploratory settings are part of this module for reproducible fresh runs.
    """
    from datetime import datetime

    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "scope": "Exploratory development-only routing sample efficiency; exactly A/B/C; no production promotion",
        "source": "artifacts/m5-step035-three-agent-v1",
        "splits": {
            "seeds": [55001, 55002, 55003],
            "method": "GroupShuffleSplit: approximately 20000 assessment then approximately 20000 validation; remaining approximately 160000 common upstream training pool; prefixes group-disjoint from held sets",
            "sizes": [15000, 50000, 100000, 160000],
        },
        "upstream": "Per replicate common training pool; 3 whole-stack target folds; each target excluded from expert fitting/calibration AND fusion; internal 3-fold specialist OOF; final experts/fusion fit training pool only. Shared upstream evidence across router sizes: routing supervision budget, NOT total pipeline data budget. Fresh hist/calibrator fits necessary for new group boundaries; historical checkpoints unchanged.",
        "fusion": "Existing residual original, 60 epochs Adam .003 batch4096 clip5 uniform9 masks fixed a priori from prior inner evidence; no assessment selection",
        "routers": {
            "fixed": "origin-peer mean OOF gross BCE gain",
            "mlp": "existing32Tanh one-output gain regressor; authorized63 context; fitting-only standardization;20epochs Adam.003 weight_decay.01 batch4096 clip5",
            "graph": "existing ContextualRouter3x3/11-16-4;15epochs Adam.003 soft frozen-action objective; no clinical values accepted by existing architecture",
        },
        "evaluation": {
            "scenarios": ["static", "variable"],
            "tariffs": [0.02, 0.05, 0.1],
            "budget": 1,
            "metrics": [
                "BCE+cost",
                "BCE",
                "AUROC",
                "PR-AUC",
                "accuracy",
                "F1",
                "calibration",
                "calls",
                "cost",
                "STOP",
                "peer diversity",
                "utility R2 when meaningful",
                "per-origin",
                "fold consistency",
                "runtime",
                "CUDA peak allocation",
            ],
            "selection": "All predeclared candidates reported; no outer winner or test use. Validation only for feedback acceptance. Assessment evaluated once per saved policy/config.",
        },
        "scientific_gate": {
            "primary": "variable/.02 clinical MLP versus matched fixed",
            "objective_gain_each_fold": 0.005,
            "utility_r2_each_fold": 0.05,
            "minimum_varying_origins_each_fold": 3,
            "minority_peer_fraction": 0.1,
            "max_accuracy_drop": 0.005,
            "max_origin_bce_increase": 0.01,
            "static_gain_each_fold": 0.0,
            "larger_data_claim": "100k and160k must both improve15k objective and utility R2 in all folds; descriptive curves otherwise",
        },
        "feedback": {
            "demonstration": "First replicate160k MLP; disjoint training-only logged batch from common training pool, no assessment feedback; final specialists and fusion frozen; true labels simulation authority only; actions sampled before selected-risk lookup",
            "behavior": "epsilon .3 mixture approved policy softmax and uniform eligible STOP/peers; record exact propensities",
            "minimum_per_origin": 32,
            "rounds": 3,
            "local_epochs": 2,
            "loss": "router-only clipped IPS logged BCE+cost policy gradient; no unchosen rewards; local contexts and labels stay in client filesystem; compatible routing arrays only",
            "acceptance": "validation variable/.02 objective <= previous AND fixed; accuracy drop<=.005; each origin BCE increase<=.01; finite compatible state; sufficient verified feedback; no production promotion",
            "rollback": "retain initial approved research version; if candidate fails keep it; explicitly rollback initial after any approval; never mark rejected candidate eligible",
        },
        "preservation": "Hash historical artifact/report files and frozen test bytes only; never load test NPZ; compare source development original IDs against saved frozen test ID list",
        "limitations": "Trusted single-machine simulation; overlapping replicates not independent clinical cohorts; no clinical/privacy guarantee; no delayed deployed outcomes; no physical membership changes",
    }
    protocol["frozen_at"] = datetime.now(UTC).isoformat()
    write(output / "protocol.json", protocol)
    manifest = {
        str(f): digest(f)
        for base in (Path("artifacts"), Path("docs/experiments"), Path("docs/devlog"))
        for f in base.rglob("*")
        if f.is_file() and output.resolve() not in f.resolve().parents and "__pycache__" not in str(f)
    }
    write(output / "historical-hashes.json", manifest)
    write(
        output / "execution.json",
        {
            "started": protocol["frozen_at"],
            "status": "initialized",
            "protocol_sha256": digest(output / "protocol.json"),
            "historical_files": len(manifest),
        },
    )


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def split_ids(data: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-disjoint fixed assessment/validation; prefixes never cross boundaries."""
    ids = np.arange(len(data["y"]))
    a, b = next(
        GroupShuffleSplit(1, test_size=20000 / len(ids), random_state=seed).split(ids, groups=data["groups"])
    )
    fit, guard = next(
        GroupShuffleSplit(1, test_size=20000 / len(a), random_state=seed + 1).split(
            a, groups=data["groups"][a]
        )
    )
    # Randomize groups once for nested sample-size prefixes, not using outcomes.
    pool = a[fit]
    rng = np.random.default_rng(seed + 2)
    unique = np.unique(data["groups"][pool])
    rng.shuffle(unique)
    order = {int(g): i for i, g in enumerate(unique)}
    pool = pool[np.argsort([order[int(g)] for g in data["groups"][pool]], kind="stable")]
    return pool, a[guard], b


def prefix(pool: np.ndarray, groups: np.ndarray, size: int) -> np.ndarray:
    end = min(size, len(pool))
    while end < len(pool) and groups[pool[end]] == groups[pool[end - 1]]:
        end += 1
    return pool[:end]


def prepare(
    folder: Path,
    data: dict,
    columns: list[str],
    pool: np.ndarray,
    guard: np.ndarray,
    assessment: np.ndarray,
    seed: int,
    device: torch.device,
) -> None:
    """Cache whole-stack cross-fitted expert/calibration/fusion targets once per split."""
    if (folder / "prepared.json").exists():
        recorded = read(folder / "prepared.json")
        for name, checksum in recorded["checksums"].items():
            assert digest(folder / name) == checksum
        return
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "stack.npz").exists():
        raise ValueError("Incomplete preparation requires explicit fresh output; refusing silent reuse")
    held = np.concatenate([guard, assessment])
    p = np.zeros((len(pool), 3), np.float32)
    rel = np.zeros_like(p)
    risks = np.zeros((len(pool), 3, 4), np.float32)
    target_records = []
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed + 30)
    start = time.perf_counter()
    for fold, (a, b) in enumerate(splitter.split(pool, data["y"][pool], data["groups"][pool])):
        target = folder / f"target{fold}"
        tp, hp = prepare_stack(
            target,
            data["x"],
            data["y"],
            data["groups"],
            columns,
            pool[a],
            pool[b],
            seed + 100 * fold,
            THREE_DOMAINS,
        )
        fusion_fit(tp, data["y"][pool[a]], "original", seed, device, target, 60)
        fusion = CDCFusion("original", 3).to(device)
        fusion.load_state_dict(
            torch.load(target / "fusion_original60.pt", weights_only=True, map_location=device)
        )
        p[b], rel[b], risks[b] = hp, reliability(tp, data["y"][pool[a]]), outcomes(fusion, hp, device)
        target_records.append(
            {"train": pool[a].tolist(), "target": pool[b].tolist(), "positions": b.tolist()}
        )
        print(json.dumps({"stage": str(target), "seconds": time.perf_counter() - start}), flush=True)
    final = folder / "final"
    fp, hp = prepare_stack(
        final, data["x"], data["y"], data["groups"], columns, pool, held, seed + 500, THREE_DOMAINS
    )
    fusion_fit(fp, data["y"][pool], "original", seed, device, final, 60)
    fusion = CDCFusion("original", 3).to(device)
    fusion.load_state_dict(torch.load(final / "fusion_original60.pt", weights_only=True, map_location=device))
    held_risks = outcomes(fusion, hp, device)
    np.savez(
        folder / "stack.npz",
        fit=pool,
        guard=guard,
        assessment=assessment,
        held=held,
        p=p,
        rel=rel,
        risks=risks,
        held_p=hp,
        held_rel=np.broadcast_to(reliability(fp, data["y"][pool]), hp.shape).copy(),
        held_risks=held_risks,
    )
    write(folder / "target_provenance.json", {"folds": target_records})
    protected = {str(f.relative_to(folder)): digest(f) for f in folder.rglob("*") if f.is_file()}
    write(folder / "prepared.json", {"seconds": time.perf_counter() - start, "checksums": protected})


def tensors(
    data: dict,
    columns: list[str],
    positions: np.ndarray,
    p: np.ndarray,
    rel: np.ndarray,
    scenario: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build prospective authorized context: only own p and own clinical fields."""
    features, own, availability, prices, origins = [], [], [], [], []
    for origin in range(3):
        available, covered, price = context(data["ids"][positions], origin, columns, scenario, THREE_SLOTS)
        feature = np.zeros((len(p), 3, 63), np.float32)
        local = local_fields(data["x"][positions], columns, origin)
        for peer in range(3):
            if peer != origin:
                feature[:, peer] = query_features(p[:, origin], origin, peer, available, covered, rel, local)
        features.append(feature)
        own.append(p[:, origin])
        availability.append(available)
        prices.append(price)
        origins.append(np.full(len(p), origin))
    return {
        name: torch.tensor(np.concatenate(values), device=device)
        for name, values in [
            ("features", features),
            ("own", own),
            ("available", availability),
            ("prices", prices),
            ("origins", origins),
        ]
    }


def evaluate(
    policy: RoutingPolicy, inputs: dict, y: np.ndarray, risks: np.ndarray, mean_gain: np.ndarray, cost: float
) -> list[dict]:
    with torch.no_grad():
        scores = policy.logits(
            inputs["features"], inputs["origins"], inputs["own"], inputs["available"], inputs["prices"] * cost
        )
        actions = scores.argmax(-1).cpu().numpy().reshape(3, -1)
        values = (
            None
            if policy.kind == "graph"
            else policy.values(inputs["features"], inputs["origins"]).cpu().numpy().reshape(3, -1, 3)
        )
    results = []
    for origin in range(3):
        sl = slice(origin * len(y), (origin + 1) * len(y))
        result = metrics(
            y,
            risks[:, origin],
            actions[origin],
            origin,
            inputs["available"][sl].cpu().numpy(),
            inputs["prices"][sl].cpu().numpy(),
            cost,
            None if values is None else values[origin],
            mean_gain,
        )
        results.append(result)
    return results


def run(output: Path, device_name: str) -> None:
    protocol = read(output / "protocol.json")
    device = select_device(device_name)
    audit(output / "environment.json")
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    data = dict(np.load(SOURCE / "development.npz"))
    columns = read(SOURCE / "dataset.json")["features"]
    assert not set(data["ids"]) & set(np.load(SOURCE / "frozen_test_ids.npy"))
    all_results = []
    with threadpool_limits(limits=4):
        for fold, seed in enumerate(protocol["splits"]["seeds"]):
            folder = output / f"fold{fold}"
            pool, guard, assessment = split_ids(data, seed)
            prepare(folder, data, columns, pool, guard, assessment, seed, device)
            stack = dict(np.load(folder / "stack.npz"))
            # Fit all size policies before any assessment evaluation.
            policies = []
            for size in protocol["splits"]["sizes"]:
                fit = prefix(pool, data["groups"], size)
                count = len(fit)
                inputs = tensors(
                    data, columns, fit, stack["p"][:count], stack["rel"][:count], "variable", device
                )
                gains = gross_gains(stack["risks"][:count], data["y"][fit])
                target = torch.tensor(gains.transpose(1, 0, 2).reshape(-1, 3), device=device)
                dest = folder / f"size{size}"
                dest.mkdir(exist_ok=True)
                for kind in ("fixed", "mlp", "graph"):
                    path = dest / f"{kind}.pt"
                    if path.exists():
                        policy = RoutingPolicy.load(path, str(device))
                        log = read(dest / f"{kind}.json")
                    else:
                        torch.manual_seed(seed + size)
                        policy = RoutingPolicy(kind, ("A", "B", "C")).to(device)
                        initial = snapshot(policy)
                        t = time.perf_counter()
                        log = policy.fit(
                            inputs["features"],
                            inputs["origins"],
                            target,
                            inputs["own"],
                            inputs["available"],
                            inputs["prices"] * 0.02,
                            seed=seed + size,
                        )
                        synchronize(device)
                        log.update(
                            seconds=time.perf_counter() - t,
                            rows=count,
                            drift=sum(
                                float(((v - initial[k]) ** 2).sum()) for k, v in snapshot(policy).items()
                            )
                            ** 0.5,
                            cuda_peak_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                        )
                        policy.save(path)
                        write(dest / f"{kind}.json", log)
                    policies.append((size, kind, policy, gains.mean(0)))
                print(json.dumps({"fold": fold, "size": size, "rows": count, "stage": "trained"}), flush=True)
                del inputs, target
            for split in ("guard", "assessment"):
                positions = stack[split]
                begin = 0 if split == "guard" else len(guard)
                sl = slice(begin, begin + len(positions))
                for scenario in ("static", "variable"):
                    inputs = tensors(
                        data, columns, positions, stack["held_p"][sl], stack["held_rel"][sl], scenario, device
                    )
                    for size, kind, policy, mean_gain in policies:
                        for cost in (0.02, 0.05, 0.10):
                            results = evaluate(
                                policy, inputs, data["y"][positions], stack["held_risks"][sl], mean_gain, cost
                            )
                            for origin, result in enumerate(results):
                                all_results.append(
                                    dict(
                                        fold=fold,
                                        size=size,
                                        kind=kind,
                                        split=split,
                                        scenario=scenario,
                                        cost_tariff=cost,
                                        origin=origin,
                                        **result,
                                    )
                                )
                    del inputs
            write(folder / "results.json", [r for r in all_results if r["fold"] == fold])
    write(output / "results.json", all_results)
    write(
        output / "training.json",
        {
            "seconds": time.perf_counter() - start,
            "device": str(device),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
            "frozen_test_evaluations": 0,
        },
    )
    print(json.dumps(read(output / "training.json")), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--flower", action="store_true")
    parser.add_argument("--flower-worker", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        initialize(args.output)
    elif args.verify:
        verify(args.output, args.device)
    elif args.summary:
        summarize(args.output)
    elif args.flower_worker:
        flower_worker(args.output)
    elif args.flower:
        launch_flower(args.output)
    else:
        run(args.output, args.device)


def feedback_ids(output: Path) -> np.ndarray:
    """Training-only batch sampled without labels or policy outcomes."""
    ids = np.array(read(output / "fold0/target0/provenance.json")["held"])
    return np.random.default_rng(55599).choice(ids, 96, replace=False)


def flower_worker(output: Path) -> None:
    """Genuine three-node routing-only M4 feedback round trip; no raw rows/labels in messages."""
    import copy
    from datetime import datetime, timedelta

    import joblib
    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp
    from flwr.serverapp.strategy import FedAvg
    from flwr.simulation import run_simulation

    from ..collaboration.registry import ConsultationSession, Specialist, SpecialistRegistry
    from ..feedback.events import FeedbackLedger, PredictionEvent, VerifiedOutcome, sign
    from ..feedback.events import digest as event_digest
    from ..feedback.versions import PolicyRegistry, routing_update_gates
    from . import cdc_flower as adapter

    client, server = ClientApp(), ServerApp()
    torch.set_num_threads(1)
    folder = output / "flower"
    folder.mkdir(exist_ok=True)
    checkpoint = output / "fold0/size160000/mlp.pt"
    stage = dict(np.load(output / "fold0/stack.npz"))
    source = dict(np.load(SOURCE / "development.npz"))
    columns = read(SOURCE / "dataset.json")["features"]
    held = feedback_ids(output)
    assert set(held).issubset(stage["fit"]) and not set(held) & set(stage["held"])
    keys = ("ab" * 32, "cd" * 32)  # Explicit trusted simulator credentials, not deployed secrets.
    client.query("capabilities")(adapter.capabilities)
    client.query("registry_predict")(adapter.registry_predict)

    def load_registry(declarations: list) -> SpecialistRegistry:
        registry = SpecialistRegistry(adapter.CDC_TASK, adapter.CDC_OUTPUT)
        for declaration in declarations:
            registry.register(Specialist.model_validate(declaration["spec"]), node_id=declaration["node"])
        return registry

    def local_inputs(origin: int, ids: np.ndarray, registry: SpecialistRegistry) -> tuple[np.ndarray, dict]:
        private = joblib.load(output / "fold0/target0" / f"client_{origin}/final.joblib")
        from .cdc_data import expert_predict

        with threadpool_limits(limits=1):
            own = expert_predict(private, source["x"][ids])
        rel = reliability(
            np.load(output / "fold0/target0/probabilities.npz")["train"],
            source["y"][np.array(read(output / "fold0/target0/provenance.json")["train"])],
        )
        available = np.tile(
            [registry.get(name).specialist.available for name in ("A", "B", "C")], (len(ids), 1)
        )
        covered = np.ones((len(ids), len(columns)), bool)
        features = np.zeros((len(ids), 3, 63), np.float32)
        for peer in range(3):
            if peer != origin:
                features[:, peer] = query_features(
                    own,
                    origin,
                    peer,
                    available,
                    covered,
                    np.broadcast_to(rel, (len(ids), 3)),
                    local_fields(source["x"][ids], columns, origin),
                )
        costs = np.tile(
            [registry.get(name).specialist.consultation_cost for name in ("A", "B", "C")], (len(ids), 1)
        )
        return own, {"features": features, "available": available, "costs": costs}

    @client.query("routing_request")
    def route_query(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        config = dict(message.content["config"])
        if set(config) != {"declarations", "explore"}:
            raise ValueError("Unexpected context/label payload")
        registry = load_registry(json.loads(config["declarations"]))
        policy = RoutingPolicy.load(checkpoint)
        own, inputs = local_inputs(origin, held, registry)
        actions, propensities, audits = [], [], []
        rng = np.random.default_rng(55500 + origin)
        for row, patient in enumerate(held):
            request = ConsultationSession(
                registry,
                "ABC"[origin],
                (str(int(patient)),),
                permitted=frozenset("ABC"),
                feature_permissions={
                    e.specialist.specialist_id: frozenset(e.specialist.model_adapter.feature_names)
                    for e in registry.discover()
                },
                available_features=frozenset(field for domain in THREE_DOMAINS.values() for field in domain),
                max_calls=1,
                max_cost=0.1,
                alignment_namespace="cdc891-development-index",
            )
            decision = policy.route(
                request,
                {name: inputs["features"][row, i].tolist() for i, name in enumerate("ABC") if i != origin},
                float(own[row]),
                version="v000",
            )
            # Decisions/propensities fixed before any selected peer prediction or outcome lookup.
            scores = torch.full((4,), -torch.inf)
            scores[0] = decision.scores["STOP"]
            for name in decision.eligible:
                scores["ABC".index(name) + 1] = decision.scores[name]
            behavior = scores.softmax(-1).numpy().astype(np.float64)
            if config["explore"]:
                eligible = np.isfinite(scores.numpy())
                behavior = 0.7 * behavior + 0.3 * eligible / eligible.sum()
                behavior /= behavior.sum()
                action = int(rng.choice(4, p=behavior))
            else:
                action = int(scores.argmax())
            actions.append(action)
            propensities.append(float(behavior[action]))
            audits.append({"destination": decision.destination, "eligible": list(decision.eligible)})
        response = {
            "origin": origin,
            "ids": held.tolist(),
            "own": own.tolist(),
            "actions": actions,
            "propensities": propensities,
            "audits": audits,
        }
        write(folder / f"origin_{origin}_decisions_{int(config['explore'])}.json", response)
        return Message(
            RecordDict({"response": ConfigRecord({"json": json.dumps(response)})}), reply_to=message
        )

    @client.query("log_observed")
    def log_observed(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        record = json.loads(message.content["config"]["record"])
        if set(record) != {"origin", "ids", "actions", "propensities", "own", "selected", "declarations"}:
            raise ValueError("Unexpected logging payload")
        assert record["origin"] == origin and record["ids"] == held.tolist()
        registry = load_registry(record["declarations"])
        own, inputs = local_inputs(origin, held, registry)
        np.testing.assert_allclose(own, record["own"], atol=1e-7)
        ledger = FeedbackLedger(folder / f"origin_{origin}/ledger", origin, set(map(str, held)), *keys)
        contexts = {}
        fusion = CDCFusion("original", 3)
        fusion.load_state_dict(torch.load(output / "fold0/target0/fusion_original60.pt", weights_only=True))
        probabilities = np.full((len(held), 3), 0.5, np.float32)
        probabilities[:, origin] = own
        mask = np.zeros_like(probabilities, bool)
        mask[:, origin] = True
        for row, action in enumerate(record["actions"]):
            if action:
                assert action - 1 != origin and record["selected"][row] is not None
                probabilities[row, action - 1] = record["selected"][row]
                mask[row, action - 1] = True
            else:
                assert record["selected"][row] is None
        with torch.no_grad():
            risk = (
                fusion(torch.tensor(probabilities), torch.tensor(mask), torch.full((len(held),), origin))
                .sigmoid()
                .numpy()
            )
        now = datetime.now(UTC)
        for row, patient in enumerate(held):
            action = record["actions"][row]
            evidence = [float(probabilities[row, j]) if mask[row, j] else None for j in range(3)]
            event = PredictionEvent(
                request_id=f"{origin}-{int(patient)}",
                patient_id=str(int(patient)),
                origin=origin,
                episode=1,
                available_domains=inputs["available"][row].tolist(),
                action=action,
                consulted=[] if not action else [action - 1],
                propensity=record["propensities"][row],
                action_score=0.0,
                evidence=evidence,
                final_risk=float(risk[row]),
                origin_only_risk=float(own[row]),
                model_version="frozen-target0-residual60",
                policy_version="v000",
                consultation_cost=float(inputs["costs"][row, action - 1]) if action else 0.0,
                event_timestamp=now.isoformat(),
            )
            ledger.prediction(sign(event, keys[0]))
            # Training-only diagnosis authority; delayed signed outcome, never sent to server.
            outcome = VerifiedOutcome(
                request_id=event.request_id,
                patient_id=event.patient_id,
                prediction_digest=event_digest(event),
                label=int(source["y"][patient]),
                authority="simulation-training-label",
                received_episode=2,
                verified_timestamp=(now + timedelta(seconds=1)).isoformat(),
            )
            ledger.outcome(sign(outcome, keys[1]))
            contexts[event.request_id] = {
                "features": inputs["features"][row].tolist(),
                "costs": inputs["costs"][row].tolist(),
            }
        write(folder / f"origin_{origin}/contexts.json", contexts)
        assert len(ledger.replay(1)) == 0 and len(ledger.replay(2)) == len(held)
        return Message(
            RecordDict({"response": ConfigRecord({"records": len(held), "origin": origin})}), reply_to=message
        )

    @client.train()
    def train_feedback(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        round_id = int(message.content["config"]["server-round"])
        assert set(message.content) == {"arrays", "config"}
        model = RoutingPolicy.load(checkpoint)
        model.load_state_dict(message.content["arrays"].to_torch_state_dict(), strict=True)
        before = snapshot(model)
        ledger = FeedbackLedger(folder / f"origin_{origin}/ledger", origin, set(map(str, held)), *keys)
        log = model.update(ledger.replay(2), read(folder / f"origin_{origin}/contexts.json"))
        log["drift"] = sum(float(((v - before[k]) ** 2).sum()) for k, v in snapshot(model).items()) ** 0.5
        torch.save(snapshot(model), folder / f"local_{round_id}_{origin}.pt")
        write(folder / f"local_{round_id}_{origin}.json", log)
        return Message(
            RecordDict(
                {
                    "arrays": ArrayRecord(snapshot(model)),
                    "metrics": MetricRecord({"num-examples": len(held), "loss": log["loss"]}),
                }
            ),
            reply_to=message,
        )

    @server.main()
    def serve(grid, ctx: Context) -> None:
        started = time.perf_counter()
        deadline = started + 120
        while len(list(grid.get_node_ids())) != 3:
            if time.perf_counter() > deadline:
                raise TimeoutError("Exactly3 physical clients required")
            time.sleep(0.1)
        traffic = []

        def query(node, topic, config):
            request = Message(
                RecordDict({"config": ConfigRecord(config)}), dst_node_id=node, message_type=f"query.{topic}"
            )
            replies = list(grid.send_and_receive([request], timeout=180))
            if len(replies) != 1 or replies[0].has_error():
                raise RuntimeError(f"Failed {topic}: {replies}")
            assert replies[0].metadata.src_node_id == node
            traffic.append(
                {
                    "topic": topic,
                    "node": node,
                    "request_keys": list(config),
                    "response_keys": list(replies[0].content["response"]),
                }
            )
            return dict(replies[0].content["response"])

        registry = SpecialistRegistry(adapter.CDC_TASK, adapter.CDC_OUTPUT)
        for node in grid.get_node_ids():
            response = query(node, "capabilities", {})
            registry.register(Specialist.model_validate_json(response["declaration"]), node_id=node)
        assert {e.specialist.specialist_id for e in registry.discover()} == set("ABC")

        def declarations():
            return [
                {"spec": e.specialist.model_dump(mode="json"), "node": e.node_id} for e in registry.discover()
            ]

        # Same live registry snapshot gates all policies. No model retraining for outages.
        registry.update("B", available=False)
        disabled = json.loads(
            query(
                registry.get("A").node_id,
                "routing_request",
                {"declarations": json.dumps(declarations()), "explore": False},
            )["json"]
        )
        assert 2 not in disabled["actions"] and all("B" not in a["eligible"] for a in disabled["audits"])
        registry.update("B", available=True)
        observations = []
        for origin, name in enumerate("ABC"):
            response = json.loads(
                query(
                    registry.get(name).node_id,
                    "routing_request",
                    {"declarations": json.dumps(declarations()), "explore": True},
                )["json"]
            )
            selected = [None] * len(held)
            for peer, peer_name in enumerate("ABC"):
                rows = np.flatnonzero(np.array(response["actions"]) == peer + 1)
                if not len(rows):
                    continue
                session = ConsultationSession(
                    registry,
                    name,
                    tuple(str(int(held[r])) for r in rows),
                    permitted=frozenset("ABC"),
                    feature_permissions={
                        e.specialist.specialist_id: frozenset(e.specialist.model_adapter.feature_names)
                        for e in registry.discover()
                    },
                    available_features=frozenset(
                        field for domain in THREE_DOMAINS.values() for field in domain
                    ),
                    max_calls=1,
                    max_cost=0.1,
                    alignment_namespace="cdc891-development-index",
                )

                def transport(entry, payload):
                    return json.loads(
                        query(entry.node_id, "registry_predict", {"request": json.dumps(payload)})["json"]
                    )

                prediction = session.consult(peer_name, transport)["probabilities"]
                for r, p in zip(rows, prediction, strict=True):
                    selected[int(r)] = p
            log_record = {k: response[k] for k in ("origin", "ids", "actions", "propensities", "own")}
            log_record.update(selected=selected, declarations=declarations())
            observations.append(
                query(registry.get(name).node_id, "log_observed", {"record": json.dumps(log_record)})
            )
        model = RoutingPolicy.load(checkpoint)
        versions = PolicyRegistry(folder / "versions")
        initial = snapshot(model)
        versions.save(
            "v000",
            initial,
            {
                "eligible": True,
                "router_config": model.config(),
                "scope": "research bootstrap, not production promotion",
            },
        )
        versions.approve("v000")

        def save(round_id, arrays):
            torch.save(arrays.to_torch_state_dict(), folder / f"global_{round_id}.pt")

        result = FedAvg(
            fraction_train=1, fraction_evaluate=0, min_train_nodes=3, min_available_nodes=3
        ).start(
            grid,
            initial_arrays=ArrayRecord(initial),
            num_rounds=3,
            train_config=ConfigRecord({"feedback-cutoff": 2}),
            evaluate_fn=save,
            timeout=300,
        )
        updated = result.arrays.to_torch_state_dict()
        model.load_state_dict(updated)
        assert any(not torch.equal(initial[k], updated[k]) for k in initial if initial[k].is_floating_point())
        # Insufficient-feedback path is exact retention, before separate gate-based candidate approval.
        retained = copy.deepcopy(model.state_dict())
        insufficient = model.update([], {})
        assert not insufficient["updated"] and all(
            torch.equal(v, model.state_dict()[k]) for k, v in retained.items()
        )
        positions = stage["guard"]
        sl = slice(0, len(positions))
        inputs = tensors(
            source,
            columns,
            positions,
            stage["held_p"][sl],
            stage["held_rel"][sl],
            "variable",
            torch.device("cpu"),
        )
        baseline = RoutingPolicy.load(output / "fold0/size160000/fixed.pt")
        previous = RoutingPolicy.load(checkpoint)
        means = baseline.gains.numpy()
        checks = {
            name: evaluate(policy, inputs, source["y"][positions], stage["held_risks"][sl], means, 0.02)
            for name, policy in [("previous", previous), ("fixed", baseline), ("updated", model)]
        }
        gates = routing_update_gates(
            checks["previous"], checks["fixed"], checks["updated"], [r["records"] for r in observations]
        )
        eligible = all(gates.values())
        versions.save(
            "v001",
            updated,
            {"eligible": eligible, "router_config": model.config(), "gates": gates, "scope": "research only"},
        )
        if eligible:
            versions.approve("v001")
        else:
            try:
                versions.approve("v001")
            except ValueError:
                pass
            else:
                raise AssertionError("Rejected candidate approved")
        before_rollback = versions.current()
        versions.rollback("v000")
        assert versions.current() == "v000"
        assert all(torch.equal(v, versions.load().state_dict()[k]) for k, v in initial.items())
        write(folder / "validation.json", checks)
        write(
            folder / "results.json",
            {
                "run_id": ctx.run_id,
                "physical_clients": 3,
                "rounds": 3,
                "local_updates": 9,
                "observations": observations,
                "traffic": traffic,
                "gates": gates,
                "candidate_approved_research_only": eligible,
                "before_rollback": before_rollback,
                "after_rollback": versions.current(),
                "insufficient_feedback": insufficient,
                "seconds": time.perf_counter() - started,
                "drift": sum(float(((updated[k] - initial[k]) ** 2).sum()) for k in initial) ** 0.5,
                "frozen_test_evaluations": 0,
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


def launch_flower(output: Path) -> None:
    """Run isolated genuine Flower process with explicit fresh study checkpoint bindings."""
    import shutil
    import subprocess
    import sys

    folder = output / "flower"
    folder.mkdir(exist_ok=False)
    for name in ("development.npz", "dataset.json"):
        shutil.copyfile(SOURCE / name, output / name)
    write(
        output / "benchmark-config.json",
        {"clients": 3, "source_slots": list(THREE_SLOTS), "domains": THREE_DOMAINS},
    )
    env = os.environ | {
        "SWARMFL_CDC_ROOT": str(output.resolve()),
        "SWARMFL_CDC_STAGE": "fold0",
        "SWARMFL_CDC_PRIVATE_BLOCK": "target0",
        "PYTHONUTF8": "1",
    }
    with (folder / "flower.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-u", "-m", __spec__.name, "--output", str(output), "--flower-worker"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert (folder / "results.json").exists()


def verify(output: Path, device_name: str) -> None:
    """Read-only provenance, private prediction, fusion, metric and FedAvg replay.

    Fits used for local-update replay are ephemeral verification, never new
    experimental selection. Frozen test NPZ is hashed as bytes only.
    """
    import joblib

    from ..feedback.events import FeedbackLedger
    from .cdc_data import expert_predict

    device = select_device(device_name)
    torch.set_num_threads(1)
    data = dict(np.load(SOURCE / "development.npz"))
    columns = read(SOURCE / "dataset.json")["features"]
    protocol = read(output / "protocol.json")
    assert digest(output / "protocol.json") == read(output / "execution.json")["protocol_sha256"]
    assert not set(data["ids"]) & set(np.load(SOURCE / "frozen_test_ids.npy"))
    counts = {
        "private_prediction_replays": 0,
        "fusion_outcomes": 0,
        "metric_fields": 0,
        "aggregates": 0,
        "local_update_replays": 0,
    }

    def compare(saved, actual):
        if isinstance(saved, dict):
            assert set(saved) == set(actual)
            for key in saved:
                compare(saved[key], actual[key])
        elif isinstance(saved, list):
            assert len(saved) == len(actual)
            if saved and isinstance(saved[0], (float, int)):
                np.testing.assert_allclose(actual, saved, atol=2e-6, rtol=2e-6)
            else:
                for a, b in zip(saved, actual, strict=True):
                    compare(a, b)
        elif isinstance(saved, (float, int)):
            assert np.isclose(saved, actual, atol=2e-6, rtol=2e-6), (saved, actual)
            counts["metric_fields"] += 1
        else:
            assert saved == actual

    with threadpool_limits(limits=1):
        for fold, seed in enumerate(protocol["splits"]["seeds"]):
            folder = output / f"fold{fold}"
            stack = dict(np.load(folder / "stack.npz"))
            pool, guard, assessment = split_ids(data, seed)
            for key, ids in [("fit", pool), ("guard", guard), ("assessment", assessment)]:
                np.testing.assert_array_equal(ids, stack[key])
            assert not set(data["groups"][pool]) & set(data["groups"][stack["held"]])
            assert not set(data["groups"][guard]) & set(data["groups"][assessment])
            for relative, checksum in read(folder / "prepared.json")["checksums"].items():
                assert digest(folder / relative) == checksum
            target_rows = read(folder / "target_provenance.json")["folds"]
            for block in ["target0", "target1", "target2", "final"]:
                root = folder / block
                provenance = read(root / "provenance.json")
                arrays = dict(np.load(root / "probabilities.npz"))
                train = np.array(provenance["train"])
                held = np.array(provenance["held"])
                assert not set(data["groups"][train]) & set(data["groups"][held])
                assert set(train).issubset(pool)
                for origin, (_, fields) in enumerate(THREE_DOMAINS.items()):
                    metadata = provenance["clients"][str(origin)]
                    assert metadata["columns"] == list(fields)
                    for name, record in [
                        (f"oof_{f}", record) for f, record in enumerate(metadata["folds"])
                    ] + [("final", metadata["final"])]:
                        fitted = np.array(record["model_fit"])
                        cal = np.array(record["calibration"])
                        assert not set(data["groups"][fitted]) & set(data["groups"][cal])
                        assert set(fitted) | set(cal) <= set(train)
                        private = joblib.load(root / f"client_{origin}" / f"{name}.joblib")
                        assert private["columns"] == [columns.index(field) for field in fields]
                        predict_ids = held if name == "final" else np.array(record["held"])
                        assert not set(data["groups"][predict_ids]) & set(data["groups"][fitted])
                        assert not set(data["groups"][predict_ids]) & set(data["groups"][cal])
                        expected = (
                            arrays["held"][:, origin]
                            if name == "final"
                            else arrays["train"][record["held_positions"], origin]
                        )
                        np.testing.assert_allclose(
                            expert_predict(private, data["x"][predict_ids]), expected, atol=1e-7
                        )
                        counts["private_prediction_replays"] += 1
                model = CDCFusion("original", 3).to(device)
                model.load_state_dict(
                    torch.load(root / "fusion_original60.pt", weights_only=True, map_location=device)
                )
                actual = outcomes(model, arrays["held"], device)
                if block == "final":
                    np.testing.assert_allclose(actual, stack["held_risks"], atol=2e-6)
                    np.testing.assert_array_equal(arrays["held"], stack["held_p"])
                else:
                    index = int(block[-1])
                    positions = target_rows[index]["positions"]
                    np.testing.assert_allclose(actual, stack["risks"][positions], atol=2e-6)
                    np.testing.assert_array_equal(arrays["held"], stack["p"][positions])
                    np.testing.assert_allclose(
                        stack["rel"][positions],
                        np.broadcast_to(reliability(arrays["train"], data["y"][train]), (len(positions), 3)),
                        atol=1e-7,
                    )
                counts["fusion_outcomes"] += 1
            records = read(folder / "results.json")
            for split in ("guard", "assessment"):
                positions = stack[split]
                begin = 0 if split == "guard" else len(guard)
                sl = slice(begin, begin + len(positions))
                for scenario in ("static", "variable"):
                    inputs = tensors(
                        data, columns, positions, stack["held_p"][sl], stack["held_rel"][sl], scenario, device
                    )
                    for size in protocol["splits"]["sizes"]:
                        count = len(prefix(pool, data["groups"], size))
                        means = gross_gains(stack["risks"][:count], data["y"][pool[:count]]).mean(0)
                        for kind in ("fixed", "mlp", "graph"):
                            policy = RoutingPolicy.load(folder / f"size{size}/{kind}.pt", str(device))
                            for cost in (0.02, 0.05, 0.10):
                                actual = evaluate(
                                    policy, inputs, data["y"][positions], stack["held_risks"][sl], means, cost
                                )
                                saved = [
                                    r
                                    for r in records
                                    if r["split"] == split
                                    and r["scenario"] == scenario
                                    and r["size"] == size
                                    and r["kind"] == kind
                                    and r["cost_tariff"] == cost
                                ]
                                assert len(saved) == 3
                                for origin, result in enumerate(actual):
                                    compare(
                                        {
                                            k: v
                                            for k, v in saved[origin].items()
                                            if k
                                            not in {
                                                "fold",
                                                "size",
                                                "kind",
                                                "split",
                                                "scenario",
                                                "cost_tariff",
                                                "origin",
                                            }
                                        },
                                        result,
                                    )
                    del inputs
    flower = output / "flower"
    if (flower / "results.json").exists():
        from ..feedback.versions import PolicyRegistry

        versions = PolicyRegistry(flower / "versions")
        assert versions.current() == "v000"
        stack = dict(np.load(output / "fold0/stack.npz"))
        lookup = {int(patient): row for row, patient in enumerate(stack["fit"])}
        for origin in range(3):
            ids = feedback_ids(output)
            ledger = FeedbackLedger(
                flower / f"origin_{origin}/ledger", origin, set(map(str, ids)), "ab" * 32, "cd" * 32
            )
            contexts = read(flower / f"origin_{origin}/contexts.json")
            policy = RoutingPolicy.load(output / "fold0/size160000/mlp.pt")
            rng = np.random.default_rng(55500 + origin)
            assert len(ledger.replay(2)) == 96
            for event, outcome in ledger.replay(2):
                row = contexts[event.request_id]
                x = torch.tensor([row["features"]])
                available = torch.tensor([event.available_domains])
                with torch.no_grad():
                    logits = policy.logits(
                        x,
                        torch.tensor([origin]),
                        torch.tensor([event.evidence[origin]]),
                        available,
                        torch.tensor([row["costs"]]),
                    )[0]
                prob = logits.softmax(-1).numpy().astype(np.float64)
                allowed = np.isfinite(logits.numpy())
                prob = 0.7 * prob + 0.3 * allowed / allowed.sum()
                prob /= prob.sum()
                assert int(rng.choice(4, p=prob)) == event.action
                assert np.isclose(prob[event.action], event.propensity, atol=1e-8)
                position = lookup[int(event.patient_id)]
                assert outcome.label == int(data["y"][int(event.patient_id)])
                assert np.isclose(event.final_risk, stack["risks"][position, origin, event.action], atol=2e-6)
                for peer in range(3):
                    if peer == origin or peer in event.consulted:
                        assert np.isclose(event.evidence[peer], stack["p"][position, peer], atol=1e-7)
                    else:
                        assert event.evidence[peer] is None
        from ..feedback.versions import routing_update_gates

        validation = read(flower / "validation.json")
        gates = routing_update_gates(
            validation["previous"], validation["fixed"], validation["updated"], [96, 96, 96]
        )
        assert gates == read(flower / "results.json")["gates"]
        for round_id in range(1, 4):
            before = torch.load(flower / f"global_{round_id - 1}.pt", weights_only=True)
            local = []
            for origin in range(3):
                held = feedback_ids(output).tolist()
                ledger = FeedbackLedger(
                    flower / f"origin_{origin}/ledger", origin, set(map(str, held)), "ab" * 32, "cd" * 32
                )
                assert not ledger.replay(1)
                model = RoutingPolicy.load(output / "fold0/size160000/mlp.pt")
                model.load_state_dict(before)
                model.update(ledger.replay(2), read(flower / f"origin_{origin}/contexts.json"))
                expected = torch.load(flower / f"local_{round_id}_{origin}.pt", weights_only=True)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=1e-7)
                local.append(expected)
                counts["local_update_replays"] += 1
            after = torch.load(flower / f"global_{round_id}.pt", weights_only=True)
            for key in after:
                torch.testing.assert_close(after[key], sum(s[key] for s in local) / 3, rtol=2e-6, atol=2e-7)
            counts["aggregates"] += 1
        assert read(flower / "results.json")["physical_clients"] == 3
    historical = read(output / "historical-hashes.json")
    changed = [
        name
        for name, checksum in historical.items()
        if not Path(name).exists() or digest(Path(name)) != checksum
    ]
    assert not changed, changed
    write(
        output / "verification.json",
        dict(
            counts,
            historical_files=len(historical),
            changed=changed,
            frozen_test_evaluations=0,
            frozen_test_sha256=digest(SOURCE / "frozen_test.npz"),
            protocol_sha256=digest(output / "protocol.json"),
        ),
    )
    print(json.dumps(read(output / "verification.json")), flush=True)


def summarize(output: Path) -> None:
    """Report every registered size/candidate and frozen scientific gates, no winner selection."""
    import csv

    records = read(output / "results.json")
    keys = [
        "objective",
        "binary_cross_entropy",
        "accuracy",
        "auroc",
        "pr_auc",
        "calls",
        "cost",
        "stop",
        "minority_peer_fraction",
        "utility_r2",
        "ece",
        "brier",
    ]
    rows = []
    for size in (15000, 50000, 100000, 160000):
        for kind in ("fixed", "mlp", "graph"):
            selected = [
                r
                for r in records
                if r["size"] == size
                and r["kind"] == kind
                and r["split"] == "assessment"
                and r["scenario"] == "variable"
                and r["cost_tariff"] == 0.02
            ]
            row = {"size": size, "kind": kind}
            for key in keys:
                values = [r[key] for r in selected if key in r and r[key] is not None]
                row[key] = float(np.mean(values)) if values else None
            row["fold_objectives"] = [
                float(np.mean([r["objective"] for r in selected if r["fold"] == fold])) for fold in range(3)
            ]
            rows.append(row)
    gates = []
    for size in (15000, 50000, 100000, 160000):
        folds = []
        for fold in range(3):

            def group(kind, scenario="variable", size=size, fold=fold):
                return [
                    r
                    for r in records
                    if r["size"] == size
                    and r["kind"] == kind
                    and r["fold"] == fold
                    and r["split"] == "assessment"
                    and r["scenario"] == scenario
                    and r["cost_tariff"] == 0.02
                ]

            m, f = group("mlp"), group("fixed")
            mean = lambda rs, key: float(np.mean([r[key] for r in rs]))
            gain = mean(f, "objective") - mean(m, "objective")
            static = mean(group("fixed", "static"), "objective") - mean(group("mlp", "static"), "objective")
            checks = {
                "meaningful_gain": gain >= 0.005,
                "utility_r2": mean(m, "utility_r2") >= 0.05,
                "all_origins_vary": all(r["minority_peer_fraction"] >= 0.1 for r in m),
                "accuracy": mean(m, "accuracy") >= mean(f, "accuracy") - 0.005,
                "origin_bce": all(
                    a["binary_cross_entropy"] <= b["binary_cross_entropy"] + 0.01
                    for a, b in zip(m, f, strict=True)
                ),
                "static": static >= 0,
            }
            folds.append(
                {
                    "fold": fold,
                    "gain": gain,
                    "utility_r2": mean(m, "utility_r2"),
                    "static_gain": static,
                    "gates": checks,
                    "pass": all(checks.values()),
                }
            )
        gates.append({"size": size, "folds": folds, "pass": all(f["pass"] for f in folds)})
    scientific = all(g["pass"] for g in gates if g["size"] in (100000, 160000))
    larger = []
    for size in (100000, 160000):
        small = next(r for r in rows if r["size"] == 15000 and r["kind"] == "mlp")
        large = next(r for r in rows if r["size"] == size and r["kind"] == "mlp")
        objective = all(
            a < b for a, b in zip(large["fold_objectives"], small["fold_objectives"], strict=True)
        )
        r2 = []
        for fold in range(3):
            val = lambda s, fold=fold: np.mean(
                [
                    r["utility_r2"]
                    for r in records
                    if r["size"] == s
                    and r["kind"] == "mlp"
                    and r["split"] == "assessment"
                    and r["scenario"] == "variable"
                    and r["cost_tariff"] == 0.02
                    and r["fold"] == fold
                ]
            )
            r2.append(bool(val(size) > val(15000)))
        larger.append(
            {"size": size, "objective_improves_all_folds": objective, "r2_improves_all_folds": all(r2)}
        )
    write(
        output / "summary.json",
        {
            "assessment_variable_002": rows,
            "scientific_gates": gates,
            "meaningful_routing_gate": scientific,
            "larger_data_checks": larger,
            "historical_scores_directly_comparable": False,
        },
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = ["fold", "size", "kind", "split", "scenario", "cost_tariff", "origin"] + keys
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
