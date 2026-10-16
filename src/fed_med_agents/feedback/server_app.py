"""Sequential decisions, delayed local outcomes, genuine FedAvg and approval."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ..collaboration import ConsultationBroker
from ..devices import select_device
from ..evaluate import metrics
from ..evaluation.collaboration import policy_predictions
from ..federated.client_app import state_cpu
from ..fusion import CollaborativeModel
from .events import PredictionEvent, sign, timestamp
from .versions import CapabilityRegistry, PolicyRegistry

app = ServerApp()


def action_distribution(
    model: CollaborativeModel,
    origin: int,
    own: float,
    domains: list[bool],
    device: torch.device,
    epsilon: float,
) -> tuple[np.ndarray, torch.Tensor]:
    """Routing context contains no peer probability or outcome."""
    with torch.no_grad():
        p = torch.tensor([own], device=device)
        availability = torch.tensor([domains], device=device)
        scores = model.router(
            torch.tensor([origin], device=device), p, 2 * (p - 0.5).abs(), availability.float(), availability
        )[0]
        valid = torch.isfinite(scores).float()
        behavior = (1 - epsilon) * scores.softmax(-1) + epsilon * valid / valid.sum()
    return behavior.cpu().numpy().astype(float), scores


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = json.loads(os.environ["FED_MED_M4_CONFIG"])
    output, source = Path(config["output"]), Path(config["source"])
    device = select_device(config["device"])
    torch.set_num_threads(1)
    start = time.perf_counter()
    registry = PolicyRegistry(output / "versions")
    baseline = registry.load("v000", device)
    offline = registry.load("offline", device)
    capabilities = CapabilityRegistry()
    hashes = {}
    deadline = time.perf_counter() + 60
    while len(list(grid.get_node_ids())) < 3:
        if time.perf_counter() > deadline:
            raise RuntimeError("Missing clients")
        time.sleep(0.1)
    for node in grid.get_node_ids():
        message = Message(
            RecordDict({"config": ConfigRecord({"seed": config["seed"]})}),
            dst_node_id=node,
            message_type="query.identity",
        )
        replies = list(grid.send_and_receive([message], timeout=60))
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError("Identity failure")
        identity = replies[0].content["identity"]
        index = int(identity["client"])
        capabilities.register(index, node, bool(identity["domain_available"]))
        hashes[index] = identity["specialist_sha256"]
    if set(capabilities.nodes) != {0, 1, 2}:
        raise ValueError("Three compatible clients required")
    broker = ConsultationBroker(grid, capabilities.nodes, output / "consultations.jsonl", config["seed"])
    broker.domain_available = capabilities.available
    plan = broker.exchange(0, "plan", RecordDict({"config": ConfigRecord({"seed": config["seed"]})}))["plan"]
    patients = np.array(plan["patient_ids"])
    rng = np.random.default_rng(config["seed"])
    patients = rng.permutation(patients)[: config["episodes"]].tolist()
    # Only validation labels enter this server evaluator. Training labels stay local.
    with np.load(source / "cohort.npz") as cohort:
        validation_ids = cohort["val"].tolist()
        validation_y = cohort["y"][cohort["val"]]
    if set(patients) & set(validation_ids) or len(set(patients)) != len(patients):
        raise ValueError("Replay contamination")
    credentials = json.loads((output / "credentials.json").read_text())
    domains = [capabilities.available[i] for i in range(3)]
    pending, history, decisions, participation = [], [], [], []
    release_audit = []
    update_id = 0

    def exchange(origin, action, payload):
        return broker.exchange(origin, action, RecordDict({"config": ConfigRecord(payload)}))["ack"]

    def validate(model):
        scores, routes = {}, []
        for origin in range(3):
            result = policy_predictions(
                model, broker, validation_ids, origin, "learned", "m4/validation", device, config["seed"]
            )
            scores[str(origin)] = metrics(validation_y, result["probabilities"]) | {
                "consultations_per_patient": result["average_consultations"]
            }
            routes.append(result["selected"])
        return {
            "mean_bce": float(np.mean([v["binary_cross_entropy"] for v in scores.values()])),
            "origins": scores,
        }, np.array(routes)

    initial_validation, reference_routes = validate(baseline)
    best_bce = initial_validation["mean_bce"]

    def summaries(episode):
        result = {}
        for policy in ("adaptive", "frozen", "offline"):
            reports = []
            for origin in range(3):
                reply = exchange(origin, "feedback_summary", {"policy": policy, "episode": episode})
                if "json" in reply:
                    reports.append(json.loads(reply["json"]))
            if not reports:
                continue
            count = sum(r["count"] for r in reports)
            calls = sum(r["calls"] for r in reports)
            tp, fp, fn = (sum(r[key] for r in reports) for key in ("tp", "fp", "fn"))
            aucs = [r["auroc"] for r in reports if r["auroc"] is not None]
            result[policy] = {
                "count": count,
                "accuracy": sum(r["correct"] for r in reports) / count,
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                "auroc_macro_origin": float(np.mean(aucs)) if aucs else None,
                "binary_cross_entropy": sum(r["bce_sum"] for r in reports) / count,
                "consultations_per_patient": calls / count,
                "unnecessary_consultation_fraction": sum(r["unnecessary"] for r in reports) / max(1, calls),
                "origins": reports,
            }
        return result

    def release(episode, flush=False):
        for item in pending[:]:
            if not flush and item["due"] > episode:
                continue
            exchange(
                item["origin"],
                "feedback_reveal",
                {"policy": item["policy"], "request_id": item["request_id"], "episode": episode},
            )
            release_audit.append(item | {"released": episode})
            pending.remove(item)

    def update(episode):
        nonlocal update_id, best_bce
        counts = []
        for origin in range(3):
            reply = exchange(origin, "feedback_summary", {"policy": "adaptive", "episode": episode})
            counts.append(json.loads(reply["json"])["count"] if "json" in reply else 0)
        if min(counts) < config["min_feedback"]:
            history.append(
                {"episode": episode, "skipped": "insufficient_verified_feedback", "counts": counts}
            )
            return
        update_id += 1
        previous = registry.current()
        before = registry.load(device=device)
        records_seen = []

        def aggregate(records, weighted_by):
            reports = [dict(record["metrics"]) for record in records]
            if sorted(int(r["client"]) for r in reports) != [0, 1, 2]:
                raise RuntimeError("Missing/duplicate client feedback update")
            if any(r["router-change"] <= 0 or r["fusion-change"] <= 0 for r in reports):
                raise RuntimeError("Shared router/fusion did not learn")
            records_seen.extend(reports)
            participation.extend({"update": update_id, "client": int(r["client"])} for r in reports)
            total = sum(r[weighted_by] for r in reports)
            return MetricRecord(
                {
                    key: sum(r[key] * r[weighted_by] for r in reports) / total
                    for key in ("loss", "router-change", "fusion-change")
                }
            )

        strategy = FedAvg(
            fraction_train=1.0,
            fraction_evaluate=0.0,
            min_train_nodes=3,
            min_available_nodes=3,
            train_metrics_aggr_fn=aggregate,
        )
        train_config = ConfigRecord(
            {
                "device": str(device),
                "cutoff": episode,
                "window": config["window"],
                "min_feedback": config["min_feedback"],
                "epochs": config["epochs"],
                "learning_rate": config["learning_rate"],
                "update_id": update_id,
            }
        )
        result = strategy.start(
            grid,
            initial_arrays=ArrayRecord(state_cpu(before)),
            num_rounds=1,
            train_config=train_config,
            timeout=300,
        )
        state = result.arrays.to_torch_state_dict()
        candidate = registry.load(device=device)
        candidate.load_state_dict(state)
        validation, routes = validate(candidate)
        delta = (
            sum(float(((state[k].cpu() - v.cpu()) ** 2).sum()) for k, v in before.state_dict().items()) ** 0.5
        )
        eligible = bool(
            np.isfinite(validation["mean_bce"])
            and delta <= config["max_update_norm"]
            and validation["mean_bce"] <= best_bce + config["approval_tolerance"]
        )
        version = f"v{update_id:03d}"
        audit = {
            "version": version,
            "parent": previous,
            "episode": episode,
            "eligible": eligible,
            "validation": validation,
            "shared_change_l2": delta,
            "clients": records_seen,
            "routing_change_fraction": float(np.any(routes != reference_routes, axis=-1).mean()),
        }
        registry.save(version, state, audit)
        if eligible:
            registry.approve(version)
            best_bce = min(best_bce, validation["mean_bce"])
        audit["approved_after"] = registry.current()
        audit["stream_metrics"] = summaries(episode)
        history.append(audit)
        print(
            f"M4 update {version} approved={eligible} delta={delta:.8f} validation_bce={validation['mean_bce']:.6f}",
            flush=True,
        )
        (output / "learning_curve.json").write_text(json.dumps(history, indent=2))

    for episode, patient in enumerate(patients, 1):
        origin = (episode - 1) % 3
        # Own evidence is observed first. Labels remain behind the client authority.
        own = broker.predict(origin, origin, [patient], "oof", "m4/stream/origin")
        if own["status"] != ["ok"]:
            raise RuntimeError("Missing OOF origin")
        own_probability = float(own["probabilities"][0])
        active_version = registry.current()
        for policy in ("adaptive", "frozen", "offline"):
            model = (
                registry.load(device=device)
                if policy == "adaptive"
                else (offline if policy == "offline" else baseline)
            )
            distribution, scores = action_distribution(
                model, origin, own_probability, domains, device, config["epsilon"]
            )
            # Paired stochastic controls use the same uniform draw; never another policy's peer evidence.
            uniform = np.random.default_rng(config["seed"] + episode).random()
            action = (
                int(np.searchsorted(np.cumsum(distribution), uniform))
                if policy != "offline"
                else int(scores.argmax())
            )
            action = min(action, len(distribution) - 1)
            selected = model.router.actions[action].nonzero().flatten().tolist()
            values = [None] * 3
            values[origin] = own_probability
            for destination in selected:
                response = broker.predict(origin, destination, [patient], "oof", f"m4/stream/{policy}")
                if response["status"] != ["ok"]:
                    raise RuntimeError("Failed selected feedback consultation")
                values[destination] = float(response["probabilities"][0])
            p = torch.tensor([[0.5 if v is None else v for v in values]], device=device)
            mask = torch.tensor([[v is not None for v in values]], device=device)
            o = torch.tensor([origin], device=device)
            own_mask = torch.nn.functional.one_hot(o, 3).bool()
            with torch.no_grad():
                risk = float(torch.sigmoid(model.fusion(p, 2 * (p - 0.5).abs(), mask, o))[0])
                own_risk = float(torch.sigmoid(model.fusion(p, 2 * (p - 0.5).abs(), own_mask, o))[0])
            event = PredictionEvent(
                request_id=f"{policy}-{episode:04d}",
                patient_id=f"case-{patient:04d}",
                origin=origin,
                episode=episode,
                available_domains=domains,
                action=action,
                consulted=selected,
                propensity=float(distribution[action]) if policy != "offline" else 1.0,
                action_score=float(scores[action]),
                evidence=values,
                final_risk=risk,
                origin_only_risk=own_risk,
                model_version=active_version
                if policy == "adaptive"
                else ("offline" if policy == "offline" else "v000"),
                policy_version="clipped-ips-v1" if policy == "adaptive" else policy,
                consultation_cost=config["cost"] * len(selected),
                event_timestamp=timestamp(),
            )
            exchange(
                origin,
                "feedback_record",
                {
                    "policy": policy,
                    "json": json.dumps(sign(event, credentials[str(origin)]["prediction_key"])),
                },
            )
            decisions.append(event.model_dump())
            pending.append(
                {
                    "request_id": event.request_id,
                    "origin": origin,
                    "policy": policy,
                    "episode": episode,
                    "due": episode + config["delay"],
                }
            )
        release(episode)
        if episode in config["triggers"]:
            update(episode)
        if episode % 30 == 0:
            print(f"M4 stream {episode}/{len(patients)} approved={registry.current()}", flush=True)
    final_episode = len(patients) + config["delay"]
    release(final_episode, flush=True)
    # Final explicitly configured trigger follows label drain; never retroactively alters decisions.
    if final_episode in config["triggers"]:
        update(final_episode)
    final = registry.load(device=device)
    final_validation, final_routes = validate(final)
    candidates = np.argwhere(final_routes.any(-1))
    later_origin, later_row = candidates[0].tolist() if len(candidates) else (0, 0)
    # A subsequent coordinator request loads the approved checkpoint afresh and actually consults via Flower.
    later = policy_predictions(
        registry.load(device=device),
        broker,
        [validation_ids[later_row]],
        later_origin,
        "learned",
        f"m4/later-coordinator/{registry.current()}",
        device,
        config["seed"],
    )
    unchanged = all(
        broker.exchange(i, "identity", RecordDict({"config": ConfigRecord({"seed": config["seed"]})}))[
            "identity"
        ]["specialist_sha256"]
        == hashes[i]
        for i in range(3)
    )
    if not unchanged:
        raise RuntimeError("Private specialist changed")
    logical_bytes = sum(e["batch_payload_bytes"] / e["batch_size"] for e in broker.events)
    output_metrics = {
        "flower_run_id": context.run_id,
        "config": config,
        "source_split": "training",
        "unique_patients": len(patients),
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "stream": summaries(final_episode),
        "history": history,
        "participation": participation,
        "approved_version": registry.current(),
        "specialists_unchanged": unchanged,
        "private_hashes": hashes,
        "consultation_logical_bytes": logical_bytes,
        "shared_update_array_bytes": sum(v.numel() * v.element_size() for v in final.state_dict().values())
        * 6
        * update_id,
        "later_coordinator": {
            "origin": later_origin,
            "patient_id": validation_ids[later_row],
            "version": registry.current(),
            "selected": later["selected"].tolist(),
            "probabilities": later["probabilities"].tolist(),
        },
        "runtime_seconds": time.perf_counter() - start,
    }
    (output / "decisions.json").write_text(json.dumps(decisions, indent=2))
    (output / "releases.json").write_text(json.dumps(release_audit, indent=2))
    (output / "metrics.json").write_text(json.dumps(output_metrics, indent=2))
    print(
        f"M4 completed version={registry.current()} updates={update_id} metrics={output_metrics['stream']}",
        flush=True,
    )
