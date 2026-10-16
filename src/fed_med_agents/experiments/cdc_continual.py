"""Step7 retrospective three-client continual routing/fusion integration experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from ..collaboration.registry import ConsultationSession, Specialist, SpecialistRegistry
from ..devices import audit, select_device, synchronize
from ..feedback.continual import ContinualController
from ..feedback.events import FeedbackLedger, VerifiedOutcome, sign
from ..feedback.events import digest as event_digest
from ..feedback.learning import fusion_feedback_update
from ..feedback.versions import PolicyPair, PolicyRegistry, routing_update_gates
from ..fusion.dynamic import DynamicFusion
from ..routing.policy import RoutingPolicy
from .cdc_context import local_fields, query_features
from .cdc_data import THREE_DOMAINS, expert_predict
from .cdc_scaling import read
from .m5_step03 import digest, write
from .routing_diagnostics import describe, snapshot

ROOT = Path("artifacts/m5-step07-v1")
SOURCE = Path("artifacts/m5-step05-v1")
PRIVATE = SOURCE / "fold0/final"
DATA = Path("artifacts/m5-step035-three-agent-v1")
IDS = list("ABC")
KEYS = ("ab" * 32, "cd" * 32)


def policy_pair(root: Path) -> PolicyPair:
    return PolicyPair(root / "pair", PolicyRegistry(root / "routers"), PolicyRegistry(root / "fusions"))


def group_batches(ids: np.ndarray, groups: np.ndarray, size: int, seed: int) -> list[list[int]]:
    """Label-independent whole-group batches, ordered synthetic time only."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups[ids])
    rng.shuffle(unique)
    rank = {int(g): i for i, g in enumerate(unique)}
    ordered = ids[np.argsort([rank[int(g)] for g in groups[ids]], kind="stable")]
    result = []
    begin = 0
    for _ in range(3):
        end = begin + size
        while end < len(ordered) and groups[ordered[end]] == groups[ordered[end - 1]]:
            end += 1
        result.append(ordered[begin:end].tolist())
        begin = end
    return result


def initialize(root: Path, device: str) -> None:
    """Freeze exact local split IDs and source bindings; never fit private models."""
    if (root / "config.json").exists():
        raise ValueError("Use fresh output, never overwrite a run")
    audit(root / "environment.json")
    select_device(device)
    stage = dict(np.load(SOURCE / "fold0/stack.npz"))
    data = dict(np.load(DATA / "development.npz"))
    protocol = read(root / "protocol.json")
    assert digest(root / "protocol.json") == read(root / "execution.json")["protocol_sha256"]
    feedback = group_batches(stage["guard"], data["groups"], 192, 57001)
    validation = group_batches(stage["assessment"], data["groups"], 1024, 57002)
    seen = set()
    for batch in feedback + validation:
        groups = set(data["groups"][batch])
        assert not seen & groups
        seen |= groups
        assert not groups & set(data["groups"][stage["fit"]])
        assert not set(data["ids"][batch]) & set(np.load(DATA / "frozen_test_ids.npy"))
    router = RoutingPolicy.load(Path(protocol["sources"]["router"]))
    fusion = DynamicFusion.load(Path(protocol["sources"]["fusion"]))
    pair = policy_pair(root)
    pair.routers.save(
        "r000",
        snapshot(router),
        {"eligible": True, "router_config": router.config(), "scope": "researchbootstrap"},
    )
    pair.fusions.save(
        "f000",
        snapshot(fusion),
        {"eligible": True, "fusion_config": fusion.config(), "scope": "researchbootstrap"},
    )
    pair.activate("r000", "f000", reason="research bootstrap")
    write(
        root / "config.json",
        {
            "feedback": feedback,
            "validation": validation,
            "device": str(select_device(device)),
            "declarations": fusion.config()["declarations"],
            "initial_pair": pair.current(),
        },
    )


def registry(declarations: list[dict], nodes: dict[str, int] | None = None) -> SpecialistRegistry:
    specs = [Specialist.model_validate(v) for v in declarations]
    result = SpecialistRegistry(specs[0].task, specs[0].output_schema)
    for i, spec in enumerate(specs):
        result.register(spec, node_id=(nodes or {}).get(spec.specialist_id, i + 1))
    return result


def session(reg: SpecialistRegistry, origin: str, patient: str) -> ConsultationSession:
    return ConsultationSession(
        reg,
        origin,
        (patient,),
        permitted=frozenset(IDS),
        feature_permissions={
            e.specialist.specialist_id: frozenset(e.specialist.model_adapter.feature_names)
            for e in reg.discover()
        },
        available_features=frozenset(f for fields in THREE_DOMAINS.values() for f in fields),
        max_calls=1,
        max_cost=0.1,
        alignment_namespace="cdc891-development-index",
    )


def controller(root: Path, origin: int) -> ContinualController:
    cfg = read(root / "config.json")
    allowed = {str(p) for batch in cfg["feedback"] for p in batch}
    local = root / f"origin{origin}"
    ledger = FeedbackLedger(local / "ledger", origin, allowed, *KEYS)
    return ContinualController(local, policy_pair(root), ledger)


def evaluate(
    router: RoutingPolicy,
    fusion: DynamicFusion,
    p: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    *,
    outage: bool = False,
    forced: np.ndarray | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Matched population/evidence evaluation; full cache never enters feedback fitting."""
    router = router.cpu()
    fusion = fusion.cpu()
    all_actions = []
    rows = []
    for origin in range(3):
        available = np.ones((len(p), 3), bool)
        if outage and origin != 1:
            available[:, 1] = False
        feature = np.zeros((len(p), 3, 63), np.float32)
        rel = np.broadcast_to(np.array([fusion.reliability.get(n, 0.0) for n in IDS], np.float32), p.shape)
        for peer in range(3):
            if peer != origin:
                feature[:, peer] = query_features(
                    p[:, origin],
                    origin,
                    peer,
                    available,
                    np.ones((len(p), 21), bool),
                    rel,
                    local_fields(x, columns, origin),
                )
        with torch.no_grad():
            scores = router.logits(
                torch.tensor(feature),
                torch.full((len(p),), origin),
                torch.tensor(p[:, origin]),
                torch.tensor(available),
                torch.full((len(p), 3), 0.02),
            )
            actions = scores.argmax(-1).numpy() if forced is None else forced[origin]
            mask = torch.zeros((len(p), 3), dtype=torch.bool)
            mask[:, origin] = True
            selected = actions > 0
            mask[np.flatnonzero(selected), actions[selected] - 1] = True
            assert not np.any(actions == origin + 1)
            assert available[np.flatnonzero(selected), actions[selected] - 1].all()
            tokens = fusion.tokens(torch.tensor(p), IDS, torch.full((len(p),), origin))
            risk = fusion(tokens, mask).sigmoid().numpy()
            risk[~selected] = p[~selected, origin]
        cost = selected * 0.02
        rows.append(
            {
                "origin": origin,
                **describe(y, risk),
                "pr_auc": float(average_precision_score(y, risk)),
                "calls": float(selected.mean()),
                "stop": float((~selected).mean()),
                "cost": float(cost.mean()),
                "objective": float(
                    -np.mean(
                        y * np.log(risk.clip(1e-6, 1 - 1e-6)) + (1 - y) * np.log1p(-risk.clip(1e-6, 1 - 1e-6))
                    )
                    + cost.mean()
                ),
                "actions": np.bincount(actions, minlength=4).tolist(),
            }
        )
        all_actions.append(actions)
    return rows, np.stack(all_actions)


def flower_worker(root: Path) -> None:
    from datetime import datetime, timedelta

    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp
    from flwr.serverapp.strategy import FedAvg
    from flwr.simulation import run_simulation

    from . import cdc_flower as adapter

    torch.set_num_threads(1)
    cfg = read(root / "config.json")
    device = select_device(cfg["device"])
    data = dict(np.load(DATA / "development.npz"))
    columns = read(DATA / "dataset.json")["features"]
    stage = dict(np.load(SOURCE / "fold0/stack.npz"))
    held = np.load(PRIVATE / "probabilities.npz")["held"]
    lookup = {int(v): i for i, v in enumerate(stage["held"])}
    client, server = ClientApp(), ServerApp()
    client.query("capabilities")(adapter.capabilities)
    client.query("registry_predict")(adapter.registry_predict)

    @client.query("prepare_continual")
    def prepare(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        cycle = int(message.content["config"]["cycle"])
        declarations = json.loads(message.content["config"]["declarations"])
        reg = registry(declarations)
        ctrl = controller(root, origin)
        ids = np.array(cfg["feedback"][cycle - 1])
        model = joblib.load(PRIVATE / f"client_{origin}/final.joblib")
        with threadpool_limits(limits=1):
            own = expert_predict(model, data["x"][ids])
        _, fusion = ctrl.pair.load()
        rel = np.broadcast_to(np.array([fusion.reliability[n] for n in IDS], np.float32), (len(ids), 3))
        available = np.tile([reg.get(n).specialist.available for n in IDS], (len(ids), 1))
        contexts = {}
        for peer in range(3):
            if peer != origin:
                contexts[IDS[peer]] = query_features(
                    own,
                    origin,
                    peer,
                    available,
                    np.ones((len(ids), 21), bool),
                    rel,
                    local_fields(data["x"][ids], columns, origin),
                )
        plans = []
        for row, patient in enumerate(ids):
            plan = ctrl.prepare(
                f"c{cycle}-o{origin}-p{int(patient)}",
                session(reg, IDS[origin], str(int(patient))),
                {n: v[row].tolist() for n, v in contexts.items()},
                float(own[row]),
                episode=2 * cycle - 1,
            )
            plans.append(plan)
        write(root / f"cycle{cycle}" / f"plans{origin}.json", plans)
        return Message(RecordDict({"response": ConfigRecord({"plans": json.dumps(plans)})}), reply_to=message)

    @client.query("finish_continual")
    def finish(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        cycle = int(message.content["config"]["cycle"])
        ctrl = controller(root, origin)
        reg = registry(json.loads(message.content["config"]["declarations"]))
        replies = json.loads(message.content["config"]["selected_replies"])
        plans = read(root / f"cycle{cycle}" / f"plans{origin}.json")
        events = []
        for plan in plans:

            def transport(entry, payload):
                name = entry.specialist.specialist_id
                source = replies[name]
                patient = payload["patient_ids"][0]
                index = source["patient_ids"].index(patient)
                return {
                    k: payload[k]
                    for k in ("origin", "destination", "patient_ids", "adapter_id", "checkpoint_sha256")
                } | {"probabilities": [source["probabilities"][index]]}

            event = ctrl.finish(plan["request_id"], session(reg, IDS[origin], plan["patient_id"]), transport)
            events.append(event.model_dump())
        write(root / f"cycle{cycle}" / f"events{origin}.json", events)
        assert not ctrl.pending("router", 2 * cycle - 1, minimum=1)[0]
        # No labels in this prediction reply.
        return Message(
            RecordDict({"response": ConfigRecord({"events": json.dumps(events)})}), reply_to=message
        )

    @client.query("release_outcomes")
    def release(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        cycle = int(message.content["config"]["cycle"])
        ctrl = controller(root, origin)
        plans = read(root / f"cycle{cycle}" / f"plans{origin}.json")
        released = 0
        for row, plan in enumerate(plans):
            if row % 10 == 0:
                continue
            event = ctrl.ledger.predictions[plan["request_id"]]
            verified = VerifiedOutcome(
                request_id=event.request_id,
                patient_id=event.patient_id,
                prediction_digest=event_digest(event),
                label=int(data["y"][int(event.patient_id)]),
                authority="simulation-training-label",
                received_episode=2 * cycle,
                verified_timestamp=(
                    datetime.fromisoformat(event.event_timestamp) + timedelta(seconds=1)
                ).isoformat(),
            )
            ctrl.receive(sign(verified, KEYS[1]))
            released += 1
        counts = {component: len(ctrl.pending(component, 2 * cycle)[0]) for component in ("router", "fusion")}
        write(
            root / f"cycle{cycle}" / f"feedback{origin}.json",
            {"released": released, "pending": len(plans) - released, "fresh": counts},
        )
        return Message(
            RecordDict(
                {
                    "response": ConfigRecord(
                        {"released": released, "router": counts["router"], "fusion": counts["fusion"]}
                    )
                }
            ),
            reply_to=message,
        )

    @client.train()
    def train(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        config = message.content["config"]
        cycle = int(config["cycle"])
        component = config["component"]
        ctrl = controller(root, origin)
        records, _contexts, _provenance = ctrl.pending(component, 2 * cycle)
        if not records:
            raise ValueError("Server must skip insufficient feedback before scheduling training")
        router, fusion = ctrl.pair.load()
        model = router if component == "router" else fusion
        contract = json.dumps(model.config(), sort_keys=True)
        if contract != config["contract"]:
            raise ValueError("Incompatible shared component")
        assert set(message.content) == {"arrays", "config"}
        model = model.to(device)
        model.load_state_dict(message.content["arrays"].to_torch_state_dict(), strict=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        before = snapshot(model)
        start = time.perf_counter()
        model, log = ctrl.train_local(
            component, 2 * cycle, device=str(device), state=message.content["arrays"].to_torch_state_dict()
        )
        synchronize(device)
        state = snapshot(model)
        log.update(
            origin=origin,
            device=str(device),
            seconds=time.perf_counter() - start,
            drift=sum(float(((v - before[k]) ** 2).sum()) for k, v in state.items()) ** 0.5,
            cuda_peak_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
            request_ids=[e.request_id for e, _ in records],
        )
        dest = root / f"cycle{cycle}" / component
        torch.save(state, dest / f"local{origin}.pt")
        write(dest / f"local{origin}.json", log)
        loss = log["loss"] if component == "router" else log["curve"][-1]["bce"]
        return Message(
            RecordDict(
                {
                    "arrays": ArrayRecord(state),
                    "metrics": MetricRecord({"num-examples": len(records), "loss": loss}),
                }
            ),
            reply_to=message,
        )

    @client.query("commit_feedback")
    def commit(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        cycle = int(message.content["config"]["cycle"])
        component = message.content["config"]["component"]
        ctrl = controller(root, origin)
        log = read(root / f"cycle{cycle}" / component / f"local{origin}.json")
        ctrl.commit(component, log["request_ids"])
        assert not ctrl.pending(component, 2 * cycle, minimum=1)[0]
        # Simulated restart reauthenticates all local records/consumption.
        controller(root, origin)
        return Message(
            RecordDict({"response": ConfigRecord({"committed": len(log["request_ids"])})}), reply_to=message
        )

    @server.main()
    def serve(grid, ctx: Context) -> None:
        start = time.perf_counter()
        deadline = start + 120
        while len(list(grid.get_node_ids())) != 3:
            if time.perf_counter() > deadline:
                raise TimeoutError("Three physical clients required")
            time.sleep(0.1)
        traffic = []
        nodes = {}
        specs = {}

        def query(node, topic, payload):
            replies = list(
                grid.send_and_receive(
                    [
                        Message(
                            RecordDict({"config": ConfigRecord(payload)}),
                            dst_node_id=node,
                            message_type=f"query.{topic}",
                        )
                    ],
                    timeout=180,
                )
            )
            if len(replies) != 1 or replies[0].has_error() or replies[0].metadata.src_node_id != node:
                raise RuntimeError(f"Failed {topic}: {replies}")
            traffic.append({"topic": topic, "node": node, "keys": list(payload)})
            return dict(replies[0].content["response"])

        for node in grid.get_node_ids():
            response = query(node, "capabilities", {})
            spec = Specialist.model_validate_json(response["declaration"])
            nodes[spec.specialist_id] = node
            specs[spec.specialist_id] = spec
        assert set(nodes) == set(IDS)
        pair = policy_pair(root)
        cycle_rows = []
        for cycle in (1, 2, 3):
            folder = root / f"cycle{cycle}"
            folder.mkdir()
            previous = pair.current()
            router, fusion = pair.load()
            frozen_router = RoutingPolicy.load(Path(read(root / "protocol.json")["sources"]["fallback"]))
            registries = {}
            for origin, name in enumerate(IDS):
                reg = registry([specs[n].model_dump(mode="json") for n in IDS], nodes)
                if cycle == 2 and origin != 1:
                    reg.update("B", available=False)
                declarations = json.dumps([e.specialist.model_dump(mode="json") for e in reg.discover()])
                registries[origin] = (reg, declarations)
                plans = json.loads(
                    query(nodes[name], "prepare_continual", {"cycle": cycle, "declarations": declarations})[
                        "plans"
                    ]
                )
                responses = {}
                for peer in IDS:
                    patients = tuple(p["patient_id"] for p in plans if p["selected"] == peer)
                    if not patients:
                        continue
                    # Only selected patient/recipient groups trigger actual Flower private inference.
                    request = ConsultationSession(
                        reg,
                        name,
                        patients,
                        permitted=frozenset(IDS),
                        feature_permissions={
                            n: frozenset(s.model_adapter.feature_names) for n, s in specs.items()
                        },
                        available_features=frozenset(f for fields in THREE_DOMAINS.values() for f in fields),
                        max_calls=1,
                        max_cost=0.1,
                        alignment_namespace="cdc891-development-index",
                    )

                    def transport(entry, payload):
                        return json.loads(
                            query(entry.node_id, "registry_predict", {"request": json.dumps(payload)})["json"]
                        )

                    responses[peer] = request.consult(peer, transport)
                query(
                    nodes[name],
                    "finish_continual",
                    {"cycle": cycle, "declarations": declarations, "selected_replies": json.dumps(responses)},
                )
            feedback = []
            for name in IDS:
                feedback.append(query(nodes[name], "release_outcomes", {"cycle": cycle}))
            candidates = {}
            drifts = {}
            skips = {}
            for component, model in [("router", router), ("fusion", fusion)]:
                if any(r[component] < 32 for r in feedback):
                    skips[component] = "insufficient verified fresh records"
                    continue
                dest = folder / component
                dest.mkdir()

                def save(round_id, arrays, dest=dest):
                    torch.save(arrays.to_torch_state_dict(), dest / f"global{round_id}.pt")

                result = FedAvg(
                    fraction_train=1, fraction_evaluate=0, min_train_nodes=3, min_available_nodes=3
                ).start(
                    grid,
                    initial_arrays=ArrayRecord(snapshot(model)),
                    num_rounds=1,
                    train_config=ConfigRecord(
                        {
                            "cycle": cycle,
                            "component": component,
                            "contract": json.dumps(model.config(), sort_keys=True),
                        }
                    ),
                    evaluate_fn=save,
                    timeout=300,
                )
                candidate = (
                    RoutingPolicy(**router.config())
                    if component == "router"
                    else DynamicFusion(**fusion.config())
                )
                candidate.load_state_dict(result.arrays.to_torch_state_dict())
                candidates[component] = candidate
                drifts[component] = (
                    sum(
                        float(((v - model.state_dict()[k]) ** 2).sum())
                        for k, v in candidate.state_dict().items()
                    )
                    ** 0.5
                )
                for name in IDS:
                    query(nodes[name], "commit_feedback", {"cycle": cycle, "component": component})
            ids = np.array(cfg["validation"][cycle - 1])
            p = np.array([held[lookup[int(i)]] for i in ids])
            x = data["x"][ids]
            y = data["y"][ids]
            old_scores, old_actions = evaluate(router, fusion, p, x, y, columns, outage=cycle == 2)
            fixed_scores, _ = evaluate(frozen_router, fusion, p, x, y, columns, outage=cycle == 2)
            mean = DynamicFusion("mean", fusion.config()["declarations"], fusion.reliability)
            mean_scores, _ = evaluate(router, mean, p, x, y, columns, outage=cycle == 2, forced=old_actions)
            reports = {"previous": old_scores, "fixed": fixed_scores, "mean_matched": mean_scores}
            gates = {}
            accepted = {}
            versions = {}
            if "router" in candidates:
                reports["router"], _ = evaluate(
                    candidates["router"], fusion, p, x, y, columns, outage=cycle == 2
                )
                gates["router"] = routing_update_gates(
                    old_scores, fixed_scores, reports["router"], [f["router"] for f in feedback]
                )
                accepted["router"] = all(gates["router"].values())
                versions["router"] = f"r{cycle:03}"
                pair.routers.save(
                    versions["router"],
                    snapshot(candidates["router"]),
                    {
                        "eligible": accepted["router"],
                        "router_config": router.config(),
                        "gates": gates["router"],
                    },
                )
            if "fusion" in candidates:
                reports["fusion"], _ = evaluate(
                    router, candidates["fusion"], p, x, y, columns, outage=cycle == 2, forced=old_actions
                )
                gate = routing_update_gates(
                    old_scores, mean_scores, reports["fusion"], [f["fusion"] for f in feedback]
                )
                # Matched evidence => consultation cost identical; objective is BCE+constant.
                with torch.no_grad():
                    token = candidates["fusion"].tokens(
                        torch.tensor(p[:32]), IDS, torch.zeros(min(32, len(p)), dtype=torch.long)
                    )
                    mask = torch.ones(token.shape[:2], dtype=torch.bool)
                    gate["permutation"] = bool(
                        torch.allclose(
                            candidates["fusion"](token, mask),
                            candidates["fusion"](token.flip(1), mask.flip(1)),
                            atol=2e-6,
                            rtol=0,
                        )
                    )
                gates["fusion"] = gate
                accepted["fusion"] = all(gate.values())
                versions["fusion"] = f"f{cycle:03}"
                pair.fusions.save(
                    versions["fusion"],
                    snapshot(candidates["fusion"]),
                    {"eligible": accepted["fusion"], "fusion_config": fusion.config(), "gates": gate},
                )
            next_r = candidates["router"] if accepted.get("router") else router
            next_f = candidates["fusion"] if accepted.get("fusion") else fusion
            joint, _ = evaluate(next_r, next_f, p, x, y, columns, outage=cycle == 2)
            reports["proposed_pair"] = joint
            pair_gate = routing_update_gates(
                old_scores, fixed_scores, joint, [min(f["router"], f["fusion"]) for f in feedback]
            )
            if all(pair_gate.values()) and any(accepted.values()):
                pair.activate(
                    versions["router"] if accepted.get("router") else previous["router"],
                    versions["fusion"] if accepted.get("fusion") else previous["fusion"],
                    reason=f"validated cycle{cycle}",
                )
            active = pair.current()
            active_r, active_f = pair.load()
            reports["active"], _ = evaluate(active_r, active_f, p, x, y, columns, outage=cycle == 2)
            result = {
                "cycle": cycle,
                "previous_pair": previous,
                "active_pair": active,
                "feedback": feedback,
                "drifts": drifts,
                "component_gates": gates,
                "component_eligible": accepted,
                "pair_gate": pair_gate,
                "skipped": skips,
                "validation": reports,
            }
            write(folder / "results.json", result)
            cycle_rows.append(result)
        before = pair.current()
        pair.rollback("r000", "f000")
        assert pair.current() == cfg["initial_pair"]
        # Exact earlier behavior on same held population after reload/rollback.
        rr, ff = pair.load()
        original_r = RoutingPolicy.load(Path(read(root / "protocol.json")["sources"]["router"]))
        original_f = DynamicFusion.load(Path(read(root / "protocol.json")["sources"]["fusion"]))
        for a, b in [(rr, original_r), (ff, original_f)]:
            for k, v in a.state_dict().items():
                assert torch.equal(v, b.state_dict()[k])
        write(
            root / "results.json",
            {
                "run_id": ctx.run_id,
                "physical_clients": 3,
                "cycles": cycle_rows,
                "traffic": traffic,
                "seconds": time.perf_counter() - start,
                "before_rollback": before,
                "after_rollback": pair.current(),
                "frozen_test_evaluations": 0,
                "private_fits": 0,
            },
        )

    gpu = 1 if device.type == "cuda" else 0
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=3,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": gpu},
            "init_args": {
                "num_cpus": 1,
                "num_gpus": gpu,
                "include_dashboard": False,
                "object_store_memory": 256 * 1024 * 1024,
            },
        },
    )


def launch(root: Path, device: str) -> None:
    import subprocess
    import sys

    initialize(root, device)
    env = os.environ | {
        "SWARMFL_CDC_ROOT": str(SOURCE.resolve()),
        "SWARMFL_CDC_STAGE": "fold0",
        "SWARMFL_CDC_PRIVATE_BLOCK": "final",
        "PYTHONUTF8": "1",
    }
    with (root / "flower.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-u", "-m", __spec__.name, "--worker", "--output", str(root)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.output)
    elif args.summary:
        summarize(args.output)
    elif args.worker:
        flower_worker(args.output)
    else:
        launch(args.output, args.device)


def verify(root: Path) -> None:
    """Independently replay signed observations, local optimizers, averages and gates."""
    from ..feedback.events import PredictionEvent
    from ..fusion.dynamic import FusionEvidence

    torch.set_num_threads(1)
    cfg = read(root / "config.json")
    device = select_device(cfg["device"])
    protocol = read(root / "protocol.json")
    assert digest(root / "protocol.json") == read(root / "execution.json")["protocol_sha256"]
    data = dict(np.load(DATA / "development.npz"))
    columns = read(DATA / "dataset.json")["features"]
    stage = dict(np.load(SOURCE / "fold0/stack.npz"))
    probabilities = np.load(PRIVATE / "probabilities.npz")["held"]
    lookup = {int(v): i for i, v in enumerate(stage["held"])}
    frozen_ids = set(np.load(DATA / "frozen_test_ids.npy"))
    allgroups = set()
    allpatients = set()
    for batch in cfg["feedback"] + cfg["validation"]:
        groups = set(data["groups"][batch])
        assert not groups & allgroups
        allgroups |= groups
        assert not groups & set(data["groups"][stage["fit"]])
        assert not set(data["ids"][batch]) & frozen_ids
        assert not set(batch) & allpatients
        allpatients |= set(batch)
    counts = {
        "observations": 0,
        "verified": 0,
        "missing_outcomes": 0,
        "private_prediction_replays": 0,
        "local_optimizer_replays": 0,
        "FedAvg_replays": 0,
        "metric_fields": 0,
    }
    pair = policy_pair(root)
    results = read(root / "results.json")
    initial = cfg["initial_pair"]

    def compare(a, b):
        assert set(a) == set(b)
        for key, v in a.items():
            if isinstance(v, (float, int)):
                assert np.isclose(v, b[key], atol=2e-6, rtol=2e-6), (key, v, b[key])
                counts["metric_fields"] += 1
            else:
                assert v == b[key]

    with threadpool_limits(limits=1):
        for record in results["cycles"]:
            cycle = record["cycle"]
            previous = record["previous_pair"]
            r, f = pair.load(previous["router"], previous["fusion"])
            if cycle == 1:
                assert previous == initial
            else:
                assert previous == results["cycles"][cycle - 2]["active_pair"]
            for origin in range(3):
                ctrl = controller(root, origin)
                reg = registry(cfg["declarations"])
                if cycle == 2 and origin != 1:
                    reg.update("B", available=False)
                ids = np.array(cfg["feedback"][cycle - 1])
                private = joblib.load(PRIVATE / f"client_{origin}/final.joblib")
                own = expert_predict(private, data["x"][ids])
                counts["private_prediction_replays"] += 1
                events = [
                    PredictionEvent.model_validate(e)
                    for e in read(root / f"cycle{cycle}" / f"events{origin}.json")
                ]
                for row, event in enumerate(events):
                    patient = str(int(ids[row]))
                    request = session(reg, IDS[origin], patient)
                    saved = ctrl._read(event.request_id)
                    assert saved["event_digest"] == event_digest(event) and event.patient_id == patient
                    assert (
                        event.model_version == previous["fusion"]
                        and event.policy_version == previous["router"]
                    )
                    assert event.propensity == 1.0 and event.episode == 2 * cycle - 1
                    # Reconstruct prospective contexts from origin-only fields/public service metadata.
                    available = np.tile([reg.get(n).specialist.available for n in IDS], (1, 1))
                    rel = np.array([[f.reliability[n] for n in IDS]], np.float32)
                    contexts = {}
                    for peer in range(3):
                        if peer != origin:
                            contexts[IDS[peer]] = query_features(
                                own[row : row + 1],
                                origin,
                                peer,
                                available,
                                np.ones((1, 21), bool),
                                rel,
                                local_fields(data["x"][ids[row : row + 1]], columns, origin),
                            )[0].tolist()
                    decision = r.route(request, contexts, float(own[row]), version=previous["router"])
                    assert decision.destination == saved["selected"]
                    assert event.action == (
                        0 if decision.destination is None else IDS.index(decision.destination) + 1
                    )
                    observed = []
                    for peer, value in enumerate(event.evidence):
                        if peer not in {origin, *event.consulted}:
                            assert value is None
                        else:
                            expected = probabilities[lookup[int(patient)], peer]
                            assert np.isclose(value, expected, atol=1e-7)
                            observed.append(
                                FusionEvidence(IDS[peer], patient, value, saved["checkpoints"][IDS[peer]])
                            )
                    assert np.isclose(event.final_risk, f.fuse(observed, request), atol=2e-6)
                    assert event.consultation_cost == (0.02 if event.action else 0.0)
                    counts["observations"] += 1
                    outcome = ctrl.ledger.outcomes.get(event.request_id)
                    if row % 10 == 0:
                        assert outcome is None
                        counts["missing_outcomes"] += 1
                    else:
                        assert outcome is not None and outcome.received_episode == 2 * cycle
                        assert outcome.label == int(data["y"][int(patient)])
                        counts["verified"] += 1
                for component in ("router", "fusion"):
                    dest = root / f"cycle{cycle}" / component
                    if not dest.exists():
                        continue
                    log = read(dest / f"local{origin}.json")
                    request_ids = log["request_ids"]
                    assert len(set(request_ids)) == len(request_ids)
                    records = [(ctrl.ledger.predictions[n], ctrl.ledger.outcomes[n]) for n in request_ids]
                    assert all(
                        e.episode == 2 * cycle - 1 and o.received_episode <= 2 * cycle for e, o in records
                    )
                    assert len(records) == record["feedback"][origin][component]
                    provenance = {e.request_id: ctrl._read(e.request_id) for e, _ in records}
                    contexts = {n: v["context"] for n, v in provenance.items()}
                    before = torch.load(dest / "global0.pt", weights_only=True)
                    model = (
                        RoutingPolicy(**r.config()) if component == "router" else DynamicFusion(**f.config())
                    )
                    model = model.to(device)
                    model.load_state_dict(before)
                    if component == "router":
                        model.update(records, contexts)
                    else:
                        fusion_feedback_update(model, records, provenance)
                    after = torch.load(dest / f"local{origin}.pt", weights_only=True)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value.cpu(), after[key], rtol=0, atol=1e-7)
                    counts["local_optimizer_replays"] += 1
                assert (
                    not ctrl.pending("router", 6, minimum=1)[0]
                    and not ctrl.pending("fusion", 6, minimum=1)[0]
                )
            for component in ("router", "fusion"):
                dest = root / f"cycle{cycle}" / component
                if not dest.exists():
                    continue
                local = [torch.load(dest / f"local{o}.pt", weights_only=True) for o in range(3)]
                weights = [len(read(dest / f"local{o}.json")["request_ids"]) for o in range(3)]
                aggregate = torch.load(dest / "global1.pt", weights_only=True)
                for key, value in aggregate.items():
                    torch.testing.assert_close(
                        value,
                        sum(state[key] * n for state, n in zip(local, weights, strict=True)) / sum(weights),
                        rtol=2e-6,
                        atol=2e-7,
                    )
                counts["FedAvg_replays"] += 1
            ids = np.array(cfg["validation"][cycle - 1])
            p = np.array([probabilities[lookup[int(i)]] for i in ids])
            x = data["x"][ids]
            y = data["y"][ids]
            rr = pair.routers.load(f"r{cycle:03}")
            ff = pair.fusions.load(f"f{cycle:03}")
            fixed = RoutingPolicy.load(Path(protocol["sources"]["fallback"]))
            old, actions = evaluate(r, f, p, x, y, columns, outage=cycle == 2)
            fixed_scores, _ = evaluate(fixed, f, p, x, y, columns, outage=cycle == 2)
            mean = DynamicFusion("mean", f.config()["declarations"], f.reliability)
            means, _ = evaluate(r, mean, p, x, y, columns, outage=cycle == 2, forced=actions)
            router_scores, _ = evaluate(rr, f, p, x, y, columns, outage=cycle == 2)
            fusion_scores, _ = evaluate(r, ff, p, x, y, columns, outage=cycle == 2, forced=actions)
            for name, actual in [
                ("previous", old),
                ("fixed", fixed_scores),
                ("mean_matched", means),
                ("router", router_scores),
                ("fusion", fusion_scores),
            ]:
                for a, b in zip(record["validation"][name], actual, strict=True):
                    compare(a, b)
            router_gates = routing_update_gates(
                old, fixed_scores, router_scores, [v["router"] for v in record["feedback"]]
            )
            fusion_gates = routing_update_gates(
                old, means, fusion_scores, [v["fusion"] for v in record["feedback"]]
            )
            assert router_gates == record["component_gates"]["router"]
            for key, val in fusion_gates.items():
                assert val == record["component_gates"]["fusion"][key]
            for component, gates in record["component_gates"].items():
                assert record["component_eligible"][component] == all(gates.values())
                registry_store = pair.routers if component == "router" else pair.fusions
                metadata = read(
                    registry_store.path / f"{'r' if component == 'router' else 'f'}{cycle:03}.json"
                )
                assert metadata["eligible"] == all(gates.values())
            proposed_r = rr if record["component_eligible"]["router"] else r
            proposed_f = ff if record["component_eligible"]["fusion"] else f
            proposed, _ = evaluate(proposed_r, proposed_f, p, x, y, columns, outage=cycle == 2)
            joint = routing_update_gates(
                old, fixed_scores, proposed, [min(v["router"], v["fusion"]) for v in record["feedback"]]
            )
            assert joint == record["pair_gate"]
            for a, b in zip(record["validation"]["proposed_pair"], proposed, strict=True):
                compare(a, b)
            ar, af = pair.load(record["active_pair"]["router"], record["active_pair"]["fusion"])
            active, _ = evaluate(ar, af, p, x, y, columns, outage=cycle == 2)
            for a, b in zip(record["validation"]["active"], active, strict=True):
                compare(a, b)
    assert pair.current() == initial
    h = read(root / "historical-hashes.json")
    changed = [n for n, v in h.items() if not Path(n).exists() or digest(Path(n)) != v]
    assert not changed, changed
    write(
        root / "verification.json",
        dict(
            counts,
            historical_files=len(h),
            changed=changed,
            frozen_test_evaluations=0,
            frozen_test_sha256=digest(DATA / "frozen_test.npz"),
            physical_clients=results["physical_clients"],
        ),
    )
    print(json.dumps(read(root / "verification.json")), flush=True)


def summarize(root: Path) -> None:
    """Per-cycle matched population metrics and learning curves; no outer selection."""
    import csv

    result = read(root / "results.json")
    rows = []
    for cycle in result["cycles"]:
        for name, metrics in cycle["validation"].items():
            row = {"cycle": cycle["cycle"], "pipeline": name}
            for key in [
                "binary_cross_entropy",
                "auroc",
                "pr_auc",
                "brier",
                "ece",
                "accuracy",
                "calls",
                "cost",
                "stop",
                "objective",
            ]:
                row[key] = float(np.mean([m[key] for m in metrics]))
            row["actions"] = np.array([m["actions"] for m in metrics]).sum(0).tolist()
            rows.append(row)
    curves = []
    for cycle in (1, 2, 3):
        for component in ("router", "fusion"):
            for origin in range(3):
                record = read(root / f"cycle{cycle}" / component / f"local{origin}.json")
                curves.append({"cycle": cycle, "component": component, "origin": origin, **record})
    write(
        root / "summary.json",
        {
            "cycles": rows,
            "learning_curves": curves,
            "decisions": [
                {
                    k: c[k]
                    for k in [
                        "cycle",
                        "previous_pair",
                        "active_pair",
                        "component_eligible",
                        "pair_gate",
                        "feedback",
                        "drifts",
                    ]
                }
                for c in result["cycles"]
            ],
            "interpretation": "Different cycles use different patients, not a time trend/improvement claim. Router comparisons matched populations; fusion comparisons identical previously selected evidence. Deterministic behavior lacks unchosen action support. Retrospective simulated delay, not clinical online verification. Research-only activation.",
        },
    )
    with (root / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "learning-curves.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["cycle", "component", "origin", "epoch", "objective", "value"]
        )
        writer.writeheader()
        for curve in curves:
            entries = curve.get("curve", [{"loss": curve.get("loss")}])
            for epoch, entry in enumerate(entries, 1):
                writer.writerow(
                    {
                        "cycle": curve["cycle"],
                        "component": curve["component"],
                        "origin": curve["origin"],
                        "epoch": epoch if curve["component"] == "fusion" else 2,
                        "objective": "observed-set BCE"
                        if curve["component"] == "fusion"
                        else "final observed IPS surrogate",
                        "value": entry.get("bce", entry.get("loss")),
                    }
                )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
