"""CDC Flower compatibility and registry-backed private consultation adapters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from threadpoolctl import threadpool_limits

from ..collaboration.registry import (
    ConsultationSession,
    FeatureSchema,
    ModelAdapter,
    OutputSchema,
    RegistryError,
    Specialist,
    SpecialistRegistry,
    TaskSchema,
)
from ..devices import audit, select_device
from .cdc_data import DOMAINS, THREE_DOMAINS, THREE_SLOTS, expert_predict
from .cdc_routing import (
    CDCFusion,
    context,
    decide,
    design,
    gross_gains,
    metrics,
    router_network,
    target_vector,
    train_router,
)
from .m5_step03 import digest, write
from .m5_step035 import infer_values
from .routing_diagnostics import cosine, snapshot, vector

client_app, server_app = ClientApp(), ServerApp()
registry_server_app = ServerApp()
STAGE = os.environ.get("SWARMFL_CDC_STAGE", "pilot_fold0_outer")


def private_block() -> Path:
    """Explicit experimental binding; historical final checkpoint remains default."""
    block = os.environ.get("SWARMFL_CDC_PRIVATE_BLOCK", "final")
    if block not in {"final", "target0", "target1", "target2"}:
        raise ValueError("Invalid private checkpoint block")
    return root() / STAGE / block


CDC_TASK = TaskSchema(
    name="cdc891-binary-risk",
    labels=("neither", "prediabetes-or-diabetes"),
    label_definition="BRFSS cross-sectional Diabetes_binary; positive combines prediabetes and diabetes",
)
CDC_OUTPUT = OutputSchema(positive_label=CDC_TASK.labels[1])


def registry_declaration(client: int) -> Specialist:
    """Bind existing CDC provenance, ordered features and exact checkpoint to discovery."""
    provenance = json.loads((private_block() / "provenance.json").read_text())
    metadata = provenance["clients"][str(client)]
    features = tuple(metadata["columns"])
    return Specialist(
        specialist_id=metadata.get(
            "agent", "ABC"[client] if profile()[0] == THREE_DOMAINS else metadata["name"]
        ),
        model_adapter=ModelAdapter(
            adapter_id=f"cdc-histogram:{STAGE}:{client}",
            model_family=f"cdc891:{digest(private_block() / 'provenance.json')}",
            checkpoint_sha256=digest(private_block() / f"client_{client}/final.joblib"),
            feature_names=features,
            input_dim=len(features),
            alignment_namespace="cdc891-development-index",
        ),
        feature_schema=tuple(FeatureSchema(name=name, dtype="number", minimum=0) for name in features),
        task=CDC_TASK,
        output_schema=CDC_OUTPUT,
        capabilities=frozenset({"aligned_prediction", "private_inference"}),
        consultation_cost=0.02,
    )


@client_app.query("capabilities")
def capabilities(message: Message, ctx: Context) -> Message:
    """Discover actual physical-node adapter metadata; no feature values or private weights."""
    client = int(ctx.node_config["partition-id"])
    declaration = registry_declaration(client)
    return Message(
        RecordDict(
            {"response": ConfigRecord({"client": client, "declaration": declaration.model_dump_json()})}
        ),
        reply_to=message,
    )


@client_app.query("registry_predict")
def registry_predict(message: Message, ctx: Context) -> Message:
    """Recipient-validated aligned private inference for the registry path only."""
    client = int(ctx.node_config["partition-id"])
    request = json.loads(message.content["config"]["request"])
    allowed_keys = {
        "origin",
        "destination",
        "patient_ids",
        "adapter_id",
        "checkpoint_sha256",
        "task",
        "mode",
        "authorized_features",
    }
    if set(request) != allowed_keys:
        raise RegistryError("Unexpected fields or raw feature transfer in aligned request")
    declaration = registry_declaration(client)
    if request["destination"] != declaration.specialist_id:
        raise RegistryError("Wrong recipient")
    if (
        request["adapter_id"] != declaration.model_adapter.adapter_id
        or request["checkpoint_sha256"] != declaration.model_adapter.checkpoint_sha256
    ):
        raise RegistryError("Checkpoint/adapter identity mismatch")
    if TaskSchema.model_validate(request["task"]) != declaration.task:
        raise RegistryError("Incompatible request task/label")
    if request["mode"] not in {"aligned", "origin_local"}:
        raise RegistryError("CDC adapter accepts aligned IDs only")
    if (request["origin"] == request["destination"]) != (request["mode"] == "origin_local"):
        raise RegistryError("Self-consultation or incorrect origin mode")
    required = set(declaration.model_adapter.feature_names)
    if set(request["authorized_features"]) != required:
        raise RegistryError("Required fields not authorized")
    patient_ids = request["patient_ids"]
    if (
        not patient_ids
        or len(set(patient_ids)) != len(patient_ids)
        or any(
            not isinstance(value, str) or not value.isdecimal() or str(int(value)) != value
            for value in patient_ids
        )
    ):
        raise RegistryError("Invalid patient IDs")
    ids = np.array([int(value) for value in patient_ids])
    held = json.loads((private_block() / "provenance.json").read_text())["held"]
    if not set(ids).issubset(held):
        raise RegistryError("Outside development demonstration IDs")
    model = joblib.load(private_block() / f"client_{client}/final.joblib")
    columns = json.loads((root() / "dataset.json").read_text())["features"]
    if model["columns"] != [columns.index(name) for name in declaration.model_adapter.feature_names]:
        raise RegistryError("Private model feature mapping mismatch")
    if model["model"].n_features_in_ != declaration.model_adapter.input_dim:
        raise RegistryError("Private model input dimension mismatch")
    with np.load(root() / "development.npz") as data, threadpool_limits(limits=1):
        probabilities = expert_predict(model, data["x"][ids])
    return Message(
        RecordDict(
            {
                "response": ConfigRecord(
                    {
                        "json": json.dumps(
                            {
                                **{
                                    key: request[key]
                                    for key in (
                                        "origin",
                                        "destination",
                                        "patient_ids",
                                        "adapter_id",
                                        "checkpoint_sha256",
                                    )
                                },
                                "probabilities": probabilities.tolist(),
                                "client": client,
                            }
                        )
                    }
                )
            }
        ),
        reply_to=message,
    )


def root() -> Path:
    return Path(os.environ["SWARMFL_CDC_ROOT"])


def profile() -> tuple[dict, tuple[int, ...] | None]:
    """Saved historical/amended checkpoint bindings, separate from logical registry membership."""
    path = root() / "benchmark-config.json"
    if not path.exists():
        return DOMAINS, None
    config = json.loads(path.read_text())
    if config["clients"] != 3 or config["source_slots"] != list(THREE_SLOTS):
        raise ValueError("Unsupported fixed CDC amendment")
    if config["domains"] != {name: list(fields) for name, fields in THREE_DOMAINS.items()}:
        raise ValueError("Amended specialist fields mismatch")
    return THREE_DOMAINS, THREE_SLOTS


def training(client: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate only numerical pre-query context and offline supervised utility targets."""
    folder = root() / STAGE
    data = dict(np.load(folder / "stack.npz"))
    source = np.load(root() / "development.npz")
    columns = json.loads((root() / "dataset.json").read_text())["features"]
    domains, slots = profile()
    x = design(data["p"], source["ids"][data["fit"]], columns, data["rel"], True, domains, slots)
    target = target_vector(gross_gains(np.load(folder / "targets_original.npz")["train"], data["y"]))
    if client is not None:
        n = len(target) // len(domains)
        return x[client * n : (client + 1) * n], target[client * n : (client + 1) * n]
    return x, target


@client_app.query("evidence")
def evidence(message: Message, ctx: Context) -> Message:
    """Numeric full-information offline evidence only, no feature rows/private weights."""
    client = int(ctx.node_config["partition-id"])
    block = message.content["config"]["block"]
    if block not in ("final", "target0", "target1", "target2"):
        raise ValueError("Unknown block")
    folder = root() / STAGE / block
    arrays = np.load(folder / "probabilities.npz")
    provenance = json.loads((folder / "provenance.json").read_text())
    record = {
        "client": client,
        "fit_ids": provenance["train"],
        "held_ids": provenance["held"],
        "oof": arrays["train"][:, client].tolist(),
        "held": arrays["held"][:, client].tolist(),
        "private_hash": digest(folder / f"client_{client}/final.joblib"),
    }
    return Message(RecordDict({"response": ConfigRecord(record)}), reply_to=message)


@client_app.query("consult")
def consult(message: Message, ctx: Context) -> Message:
    """Classify selected aligned development IDs using this recipient's private subset."""
    client = int(ctx.node_config["partition-id"])
    ids = np.asarray(message.content["config"]["ids"], dtype=int)
    held = set(np.load(root() / STAGE / "stack.npz")["held"].tolist())
    if not set(ids).issubset(held):
        raise ValueError("Outside assessment")
    x = np.load(root() / "development.npz")["x"][ids]
    model = joblib.load(root() / STAGE / "final" / f"client_{client}/final.joblib")
    with threadpool_limits(limits=1):
        p = expert_predict(model, x)
    return Message(
        RecordDict({"response": ConfigRecord({"client": client, "ids": ids.tolist(), "p": p.tolist()})}),
        reply_to=message,
    )


@client_app.train()
def train(message: Message, ctx: Context) -> Message:
    """Origin-local shared numeric routing update, never private classifier/fusion arrays."""
    torch.set_num_threads(1)
    client = int(ctx.node_config["partition-id"])
    round_id = int(message.content["config"]["server-round"])
    x, y = training(client)
    model = router_network(x.shape[1])
    model.load_state_dict(message.content["arrays"].to_torch_state_dict())
    initial = vector(model)
    log = train_router(model, x, y, 2, 53530 + round_id, torch.device("cpu"))
    log.update(
        client=client,
        drift=float((vector(model) - initial).norm()),
        num_examples=len(y),
        payload_keys=["arrays", "metrics"],
    )
    folder = root() / "flower"
    torch.save(snapshot(model), folder / f"client_{client}_round_{round_id}.pt")
    write(folder / f"client_{client}_round_{round_id}.json", log)
    return Message(
        RecordDict(
            {
                "arrays": ArrayRecord(snapshot(model)),
                "metrics": MetricRecord({"client": client, "num-examples": len(y), "loss": log["loss"]}),
            }
        ),
        reply_to=message,
    )


@server_app.main()
def server(grid: Grid, ctx: Context) -> None:
    torch.set_num_threads(1)
    domains, slots = profile()
    clients = len(domains)
    start, deadline = time.perf_counter(), time.perf_counter() + 90
    while len(list(grid.get_node_ids())) != clients:
        if time.perf_counter() > deadline:
            raise TimeoutError(f"{clients} clients required")
        time.sleep(0.1)
    folder = root() / "flower"
    nodes, identities, evidence_log = list(grid.get_node_ids()), {}, []

    def query(node, content, topic):
        request = Message(
            RecordDict({"config": ConfigRecord(content)}), dst_node_id=node, message_type=f"query.{topic}"
        )
        replies = list(grid.send_and_receive([request], timeout=120))
        if len(replies) != 1 or replies[0].has_error() or replies[0].metadata.src_node_id != node:
            raise RuntimeError(f"Failed {topic} message")
        return dict(replies[0].content["response"])

    for block in ("final", "target0", "target1", "target2"):
        expected = np.load(root() / STAGE / block / "probabilities.npz")
        for node in nodes:
            r = query(node, {"block": block}, "evidence")
            client = int(r["client"])
            identities[client] = node
            np.testing.assert_array_equal(r["oof"], expected["train"][:, client])
            np.testing.assert_array_equal(r["held"], expected["held"][:, client])
            evidence_log.append({"block": block, **r})
    registry = SpecialistRegistry(
        CDC_TASK, CDC_OUTPUT, model_family=registry_declaration(0).model_adapter.model_family
    )
    specialist_ids = {}
    for slot, node in identities.items():
        response = query(node, {}, "capabilities")
        declaration = Specialist.model_validate_json(response["declaration"])
        if response["client"] != slot or declaration != registry_declaration(slot):
            raise RegistryError("Discovered adapter differs from saved model binding")
        registry.register(declaration, node_id=int(node))
        specialist_ids[slot] = declaration.specialist_id
    write(folder / "evidence.json", {"queries": evidence_log, "run_id": ctx.run_id})
    x, y = training()
    torch.manual_seed(53530)
    model = router_network(x.shape[1])
    initial = snapshot(model)
    history = {}

    def save(round_id, arrays):
        model.load_state_dict(arrays.to_torch_state_dict())
        state = snapshot(model)
        torch.save(state, folder / f"server_round_{round_id}.pt")
        gradients = []
        n = len(y) // clients
        for client in range(clients):
            model.zero_grad()
            inputs = torch.tensor(x[client * n : (client + 1) * n])
            labels = torch.tensor(y[client * n : (client + 1) * n])
            ((model(inputs).flatten() - labels) ** 2).mean().backward()
            gradients.append(vector(model, True))
        history[str(round_id)] = {
            "gradient_cosines": [
                cosine(gradients[a], gradients[b]) for a in range(clients) for b in range(a + 1, clients)
            ]
        }
        if round_id:
            updates = [
                torch.load(folder / f"client_{i}_round_{round_id}.pt", weights_only=True)
                for i in range(clients)
            ]
            for key, value in state.items():
                torch.testing.assert_close(
                    value, sum(c[key] for c in updates) / clients, atol=1e-6, rtol=1e-5
                )
            history[str(round_id)]["aggregation_verified"] = True
        return MetricRecord({"round": round_id})

    outcome = FedAvg(
        fraction_train=1, fraction_evaluate=0, min_train_nodes=clients, min_available_nodes=clients
    ).start(
        grid,
        initial_arrays=ArrayRecord(initial),
        num_rounds=3,
        train_config=ConfigRecord({"benchmark": "cdc891"}),
        evaluate_fn=save,
        timeout=300,
    )
    model.load_state_dict(outcome.arrays.to_torch_state_dict())
    central = router_network(x.shape[1])
    central.load_state_dict(initial)
    central_log = train_router(central, x, y, 6, 53530, torch.device("cpu"))
    torch.save(snapshot(central), folder / "central.pt")
    data = dict(np.load(root() / STAGE / "stack.npz"))
    source = np.load(root() / "development.npz")
    columns = json.loads((root() / "dataset.json").read_text())["features"]
    risks = np.load(root() / STAGE / "targets_original.npz")["held"]
    mean_gain = gross_gains(np.load(root() / STAGE / "targets_original.npz")["train"], data["y"]).mean(0)
    metrics_log, consultations, predictions = {}, [], {}
    for name, router in (("federated", model), ("central", central)):
        metrics_log[name] = {}
        for origin in range(clients):
            available, covered, prices = context(
                source["ids"][data["held"]], origin, columns, "variable", slots
            )
            benefit = infer_values(
                router,
                data["held_p"][:, origin],
                origin,
                available,
                covered,
                data["held_rel"],
                True,
                torch.device("cpu"),
                domains,
            )
            selected = decide(benefit, origin, available, prices, 0.02)
            metrics_log[name][str(origin)] = metrics(
                data["held_y"],
                risks[:, origin],
                selected,
                origin,
                available,
                prices,
                0.02,
                benefit,
                mean_gain,
            )
            predictions[f"{name}_{origin}"] = selected
            if name != "federated":
                continue
            p = np.full_like(data["held_p"], np.nan)
            p[:, origin] = data["held_p"][:, origin]
            for peer in range(clients):
                rows = np.flatnonzero(selected == peer + 1)
                if not len(rows):
                    continue
                ids = data["held"][rows]
                session = ConsultationSession(
                    registry,
                    specialist_ids[origin],
                    tuple(str(int(patient)) for patient in ids),
                    permitted=frozenset(specialist_ids.values()),
                    feature_permissions={
                        entry.specialist.specialist_id: frozenset(
                            entry.specialist.model_adapter.feature_names
                        )
                        for entry in registry.discover()
                    },
                    available_features=frozenset(columns),
                    alignment_namespace="cdc891-development-index",
                )

                def transport(entry, payload):
                    reply = query(entry.node_id, {"request": json.dumps(payload)}, "registry_predict")
                    return json.loads(reply["json"])

                reply = session.consult(specialist_ids[peer], transport)
                r = {
                    "client": reply["client"],
                    "ids": [int(value) for value in reply["patient_ids"]],
                    "p": reply["probabilities"],
                }
                np.testing.assert_array_equal(ids, r["ids"])
                p[rows, peer] = r["p"]
                consultations.append({"origin": origin, "peer": peer, **r})
            fusion = CDCFusion("original", clients)
            fusion.load_state_dict(torch.load(root() / STAGE / "fusion_original.pt", weights_only=True))
            with torch.no_grad():
                value = (
                    fusion(torch.tensor(p), torch.tensor(np.isfinite(p)), torch.full((len(p),), origin))
                    .sigmoid()
                    .numpy()
                )
            np.testing.assert_allclose(value, risks[np.arange(len(p)), origin, selected], atol=2e-6)
    np.savez_compressed(folder / "actions.npz", **predictions)
    drift = float((vector(model) - torch.cat([v.flatten() for v in initial.values()])).norm())
    assert drift > 1e-6
    write(
        folder / "results.json",
        {
            "run_id": ctx.run_id,
            "rounds": 3,
            "updates": 3 * clients,
            "clients": clients,
            "history": history,
            "metrics": metrics_log,
            "consultations": consultations,
            "drift": drift,
            "central_training": central_log,
            "seconds": time.perf_counter() - start,
        },
    )


@registry_server_app.main()
def registry_demo(grid: Grid, ctx: Context) -> None:
    """Exercise logical lifecycle through real, inference-only A/B/C Flower queries."""
    import tracemalloc

    torch.set_num_threads(1)
    folder = Path(os.environ["SWARMFL_CDC_REGISTRY_OUTPUT"])
    device = select_device(os.environ.get("SWARMFL_CDC_REGISTRY_DEVICE", "cpu"))
    start = time.perf_counter()
    tracemalloc.start()
    deadline = time.perf_counter() + 90
    while len(list(grid.get_node_ids())) != 3:
        if time.perf_counter() > deadline:
            raise TimeoutError("Registry demo requires the existing three physical clients")
        time.sleep(0.1)
    traffic, lifecycle = [], []

    def query(node: int, topic: str, content: dict) -> dict:
        replies = list(
            grid.send_and_receive(
                [
                    Message(
                        RecordDict({"config": ConfigRecord(content)}),
                        dst_node_id=node,
                        message_type=f"query.{topic}",
                    )
                ],
                timeout=120,
            )
        )
        if len(replies) != 1 or replies[0].has_error() or replies[0].metadata.src_node_id != node:
            raise RuntimeError(f"Invalid Flower query.{topic} reply")
        return dict(replies[0].content["response"])

    discovered = {}
    for node in grid.get_node_ids():
        response = query(int(node), "capabilities", {})
        slot = int(response["client"])
        declaration = Specialist.model_validate_json(response["declaration"])
        if slot in discovered or declaration != registry_declaration(slot):
            raise RegistryError("Incompatible or duplicate physical adapter discovery")
        discovered[slot] = (declaration, int(node))
    if set(discovered) != set(range(3)):
        raise RegistryError("Three saved fusion slots required")
    ordered = [discovered[slot] for slot in sorted(discovered)]
    registry = SpecialistRegistry(CDC_TASK, CDC_OUTPUT, model_family=ordered[0][0].model_adapter.model_family)
    ids = tuple(spec.specialist_id for spec, _ in ordered)
    if ids != ("A", "B", "C"):
        raise RegistryError("This demonstration is bound to the user-fixed CDC A/B/C checkpoints")
    slots = {name: slot for slot, name in enumerate(ids)}
    declarations = {spec.specialist_id: spec for spec, _ in ordered}
    for spec, node in ordered[:2]:
        registry.register(spec, node_id=node)

    def record(action: str) -> None:
        lifecycle.append(
            {
                "action": action,
                "registered": [entry.specialist.specialist_id for entry in registry.discover()],
                "available": [
                    entry.specialist.specialist_id for entry in registry.discover(available_only=True)
                ],
            }
        )

    record("register_first_two")
    registry.register(ordered[2][0], node_id=ordered[2][1])
    record("register_third")
    assert len(registry.discover()) == 3
    with np.load(root() / STAGE / "stack.npz") as stage:
        patient = str(int(stage["held"][0]))
        expected_private = stage["held_p"][0].copy()
    permissions = {key: frozenset(spec.model_adapter.feature_names) for key, spec in declarations.items()}
    available_features = frozenset(
        name for spec in declarations.values() for name in spec.model_adapter.feature_names
    )

    def session(*, budget: int = 1, permitted: frozenset[str] | None = None) -> ConsultationSession:
        return ConsultationSession(
            registry,
            ids[0],
            (patient,),
            permitted=frozenset(ids) if permitted is None else permitted,
            feature_permissions=permissions,
            available_features=available_features,
            max_calls=budget,
            alignment_namespace="cdc891-development-index",
        )

    def transport(entry, payload):
        reply = json.loads(query(entry.node_id, "registry_predict", {"request": json.dumps(payload)})["json"])
        if reply["client"] != slots[entry.specialist.specialist_id]:
            raise RegistryError("Reply client differs from frozen fusion binding")
        traffic.append(
            {"node_id": entry.node_id, "generation": entry.generation, "request": payload, "reply": reply}
        )
        return reply

    origin_spec = registry.get(ids[0])
    own_payload = {
        "origin": ids[0],
        "destination": ids[0],
        "patient_ids": [patient],
        "adapter_id": origin_spec.specialist.model_adapter.adapter_id,
        "checkpoint_sha256": origin_spec.specialist.model_adapter.checkpoint_sha256,
        "task": CDC_TASK.model_dump(mode="json"),
        "mode": "origin_local",
        "authorized_features": list(origin_spec.specialist.model_adapter.feature_names),
    }
    own = transport(origin_spec, own_payload)["probabilities"][0]
    np.testing.assert_allclose(own, expected_private[0], atol=2e-6)
    fusion = CDCFusion("original", 3).to(device)
    fusion.load_state_dict(
        torch.load(root() / STAGE / "fusion_original.pt", weights_only=True, map_location=device)
    )
    fused, rejections = [], []

    def consult_and_fuse(target: str, request: ConsultationSession) -> dict:
        reply = request.consult(target, transport)
        peer = slots[target]
        np.testing.assert_allclose(reply["probabilities"][0], expected_private[peer], atol=2e-6)
        p = torch.full((1, 3), 0.5, device=device)
        mask = torch.zeros((1, 3), dtype=torch.bool, device=device)
        p[0, 0], p[0, peer] = own, reply["probabilities"][0]
        mask[0, 0], mask[0, peer] = True, True
        with torch.no_grad():
            risk = float(fusion(p, mask, torch.tensor([0], device=device)).sigmoid()[0].cpu())
        expected = np.load(root() / STAGE / "targets_original.npz")["held"][0, 0, peer + 1]
        np.testing.assert_allclose(risk, expected, atol=2e-6)
        fused.append(
            {
                "origin": ids[0],
                "destination": target,
                "patient_ids": [patient],
                "evidence": p.cpu().tolist(),
                "mask": mask.cpu().tolist(),
                "risk": risk,
                "expected_saved_risk": float(expected),
            }
        )
        return reply

    def reject(target: str, request: ConsultationSession, reason: str) -> None:
        before = len(traffic), request.calls, request.cost
        try:
            request.consult(target, transport)
        except RegistryError as error:
            assert reason in str(error)
            assert before == (len(traffic), request.calls, request.cost)
            rejections.append({"target": target, "reason": str(error), "transport_called": False})
        else:
            raise AssertionError("Invalid consultation was not rejected")

    assert [entry.specialist.specialist_id for entry in session().eligible()] == list(ids[1:])
    consult_and_fuse(ids[1], session())
    registry.update(ids[1], available=False)
    record("disable_second")
    pending = session()
    reject(ids[1], pending, "unavailable")
    assert ids[1] not in [entry.specialist.specialist_id for entry in pending.eligible()]
    registry.update(ids[1], available=True, consultation_cost=0.03)
    record("restore_second_and_update_cost")
    consult_and_fuse(ids[1], pending)
    assert pending.cost == 0.03
    reject(ids[2], pending, "budget exhausted")
    stale = session()
    old = registry.unregister(ids[2])
    record("unregister_third")
    reject(ids[2], stale, "Unknown specialist")
    registry.register(old.specialist, node_id=old.node_id)
    record("reregister_third")
    assert registry.get(ids[2]).generation > old.generation
    consult_and_fuse(ids[2], stale)
    reject(ids[0], session(), "Self-consultation")
    reject(ids[1], session(permitted=frozenset({ids[2]})), "Unauthorized specialist")
    reject(ids[1], session(budget=0), "budget exhausted")
    # Preserve both existing policies: no optimization, no accuracy comparison.
    from ..routing import ContextualRouter

    graph = ContextualRouter(3, 1).to(device)
    graph.load_state_dict(
        torch.load(root() / STAGE / "router_original_graph.pt", weights_only=True, map_location=device)
    )
    availability = torch.tensor([[registry.get(name).specialist.available for name in ids]], device=device)
    with torch.no_grad():
        graph_mask, _ = graph.select(
            torch.tensor([0], device=device),
            torch.tensor([own], device=device),
            torch.tensor([2 * abs(own - 0.5)], device=device),
            availability.float(),
            availability,
        )
    graph_targets = [ids[index] for index in graph_mask[0].nonzero().flatten().cpu().tolist()]
    for target in graph_targets:
        consult_and_fuse(target, session())
    stage = dict(np.load(root() / STAGE / "stack.npz"))
    mean_gain = gross_gains(np.load(root() / STAGE / "targets_original.npz")["train"], stage["y"]).mean(0)
    affordable = session().eligible()
    fixed_scores = {
        entry.specialist.specialist_id: float(mean_gain[0, slots[entry.specialist.specialist_id]])
        - entry.specialist.consultation_cost
        for entry in affordable
    }
    fixed = (
        max(fixed_scores, key=fixed_scores.get) if fixed_scores and max(fixed_scores.values()) > 0 else None
    )
    if fixed is not None:
        consult_and_fuse(fixed, session())
    assert len(list(grid.get_node_ids())) == 3
    result = {
        "run_id": ctx.run_id,
        "physical_clients": 3,
        "logical_lifecycle": lifecycle,
        "traffic": traffic,
        "rejections": rejections,
        "fusion": fused,
        "graph_targets": graph_targets,
        "fixed_value_target": fixed,
        "fixed_value_scores": fixed_scores,
        "training_updates": 0,
        "seconds": time.perf_counter() - start,
        "server_python_current_bytes": tracemalloc.get_traced_memory()[0],
        "server_python_peak_bytes": tracemalloc.get_traced_memory()[1],
        "device": str(device),
        "frozen_test_evaluations": 0,
    }
    write(folder / "results.json", result)
    tracemalloc.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step035-three-agent-v1"))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--registry-demo", action="store_true", help="Inference-only logical registry lifecycle"
    )
    parser.add_argument("--evidence-output", type=Path, default=Path("artifacts/m5-step04-v1"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    os.environ["SWARMFL_CDC_ROOT"] = str(args.output.resolve())
    os.environ["SWARMFL_CDC_REGISTRY_OUTPUT"] = str(args.evidence_output.resolve())
    os.environ["SWARMFL_CDC_REGISTRY_DEVICE"] = str(select_device(args.device))
    if args.worker:
        from flwr.simulation import run_simulation

        run_simulation(
            server_app=registry_server_app if args.registry_demo else server_app,
            client_app=client_app,
            num_supernodes=len(profile()[0]),
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
        return
    if args.registry_demo:
        if profile()[0] != THREE_DOMAINS:
            parser.error("Registry demonstration requires the existing A/B/C source")
        folder = args.evidence_output.resolve() / "flower"
        folder.mkdir(parents=True)
        audit(folder / "environment.json")
        with (folder / "flower.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    __spec__.name,
                    "--worker",
                    "--registry-demo",
                    "--output",
                    str(root()),
                    "--evidence-output",
                    str(folder),
                    "--device",
                    args.device,
                ],
                env=os.environ | {"PYTHONUTF8": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        assert (folder / "results.json").exists()
        result = json.loads((folder / "results.json").read_text())
        print(
            json.dumps(
                {key: result[key] for key in ("run_id", "physical_clients", "training_updates", "seconds")},
                indent=2,
            )
        )
        return
    folder = root() / "flower"
    folder.mkdir()
    protected = {
        str(path): digest(path)
        for path in root().rglob("*")
        if path.suffix == ".joblib" or (path.suffix == ".pt" and "fusion_" in path.name)
    }
    preservation = root() / "private-before-federation.json"
    if preservation.exists():
        existing = {
            str(Path(path).resolve()): checksum
            for path, checksum in json.loads(preservation.read_text()).items()
        }
        assert protected == existing
    else:
        write(preservation, protected)
    with (folder / "flower.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-u", "-m", __spec__.name, "--worker", "--output", str(root())],
            env=os.environ | {"PYTHONUTF8": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert (folder / "results.json").exists()


if __name__ == "__main__":
    main()
