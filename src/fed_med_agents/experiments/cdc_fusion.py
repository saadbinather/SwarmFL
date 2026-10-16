"""Step6 fixed A/B/C adapter for reusable prediction-set fusion; routing is frozen."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from ..devices import audit, select_device, synchronize
from ..fusion.dynamic import DynamicFusion
from .cdc_routing import CDCFusion
from .cdc_scaling import read
from .m5_step03 import digest, write
from .routing_diagnostics import describe, snapshot

SOURCE = Path("artifacts/m5-step05-v1")
DATA = Path("artifacts/m5-step035-three-agent-v1")
DEFAULT = Path("artifacts/m5-step06-v1")
IDS = ["A", "B", "C"]  # Disease adapter binding only; DynamicFusion has no such list.


def declarations() -> list[dict]:
    """Use the existing CDC adapter's exact final checkpoint and schema contracts."""
    os.environ["SWARMFL_CDC_ROOT"] = str(SOURCE.resolve())
    os.environ["SWARMFL_CDC_STAGE"] = "fold0"
    os.environ["SWARMFL_CDC_PRIVATE_BLOCK"] = "final"
    from . import cdc_flower

    # Source handler STAGE is initialized at import; this runner uses only fold0.
    if cdc_flower.STAGE != "fold0":
        raise ValueError("Run fusion experiment in an isolated process")
    return [cdc_flower.registry_declaration(i).model_dump(mode="json") for i in range(3)]


def pairs(origin: int | None = None) -> list[tuple[int, tuple[int, ...]]]:
    """All nonempty subsets containing the origin; single evidence is exact fallback."""
    return [
        (o, subset)
        for n in range(1, 4)
        for subset in itertools.combinations(range(3), n)
        for o in subset
        if origin is None or origin == o
    ]


def fitting(
    model: DynamicFusion, p: np.ndarray, y: np.ndarray, *, seed: int, epochs: int, origin: int | None = None
) -> dict:
    """Only new shared fusion parameters; cached private OOF probabilities are fixed."""
    device = next(model.parameters()).device
    probabilities = torch.tensor(p, device=device)
    labels = torch.tensor(y, device=device)
    masks = pairs(origin)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    start = time.perf_counter()
    log = []
    identity = torch.tensor([v[0] for v in masks], device=device)
    allowed = torch.tensor([[i in subset for i in range(3)] for _, subset in masks], device=device)
    for epoch in range(epochs):
        losses = []
        norms = []
        for batch in torch.randperm(len(p) * len(masks), generator=generator).split(4096):
            b = batch.to(device)
            row = b % len(p)
            pair = b // len(p)
            optimizer.zero_grad()
            tokens = model.tokens(probabilities[row], IDS, identity[pair])
            logits = model(tokens, allowed[pair])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[row])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite fusion loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm))
        log.append(
            {
                "epoch": epoch + 1,
                "bce": float(np.mean(losses)),
                "max_gradient_norm": max(norms),
                "steps": len(losses),
            }
        )
    synchronize(device)
    return {"curve": log, "seconds": time.perf_counter() - start, "rows": len(p), "origin": origin}


def prediction(
    model: DynamicFusion, p: np.ndarray, origin: int, subset: tuple[int, ...], device: torch.device
) -> np.ndarray:
    """Trusted numeric evaluation for one explicit evidence subset, never routing."""
    ids = [IDS[i] for i in subset]
    with torch.no_grad():
        probs = torch.tensor(p[:, subset], device=device)
        tokens = model.tokens(probs, ids, torch.full((len(p),), subset.index(origin), device=device))
        if len(subset) == 1:
            return p[:, subset[0]].copy()
        return model(tokens, torch.ones_like(probs, dtype=torch.bool)).sigmoid().cpu().numpy()


def evaluation(
    model: DynamicFusion | CDCFusion, p: np.ndarray, y: np.ndarray, device: torch.device
) -> tuple[list[dict], dict]:
    rows = []
    predictions = {}
    for origin, subset in pairs():
        if isinstance(model, DynamicFusion):
            risk = prediction(model, p, origin, subset, device)
        else:
            mask = torch.zeros((len(p), 3), dtype=torch.bool, device=device)
            mask[:, subset] = True
            with torch.no_grad():
                risk = (
                    model(torch.tensor(p, device=device), mask, torch.full((len(p),), origin, device=device))
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
        key = f"{origin}-" + "".join(IDS[i] for i in subset)
        predictions[key] = risk
        rows.append(
            {
                "origin": IDS[origin],
                "subset": "".join(IDS[i] for i in subset),
                **describe(y, risk),
                "pr_auc": float(average_precision_score(y, risk)),
            }
        )
    return rows, predictions


def validation_gate(candidate: list[dict], reference: list[dict]) -> dict[str, bool]:
    """Predeclared research-only validation gate; all subset metrics retained."""
    a = {(r["origin"], r["subset"]): r["binary_cross_entropy"] for r in candidate}
    b = {(r["origin"], r["subset"]): r["binary_cross_entropy"] for r in reference}
    if set(a) != set(b):
        raise ValueError("Incompatible validation evidence sets")
    return {
        "mean_bce": float(np.mean(list(a.values()))) <= float(np.mean(list(b.values()))) + 0.01,
        "worst_subset_bce": all(a[k] <= b[k] + 0.03 for k in a),
        "finite": all(np.isfinite(v) for v in a.values()),
    }


def consistency(model: DynamicFusion, p: np.ndarray, device: torch.device) -> dict:
    """Tensor permutation invariance, masked NaNs and every cardinality singleton replay."""
    result = 0.0
    for origin, subset in pairs():
        original = prediction(model, p, origin, subset, device)
        reversed_subset = tuple(reversed(subset))
        reverse = prediction(model, p, origin, reversed_subset, device)
        result = max(result, float(np.max(np.abs(original - reverse))))
    probs = torch.tensor(p, device=device)
    tokens = model.tokens(probs, IDS, torch.zeros(len(p), dtype=torch.long, device=device))
    mask = torch.tensor([[True, False, True]], device=device).expand(len(p), -1)
    hidden = tokens.clone()
    hidden[:, 1] = float("nan")
    with torch.no_grad():
        np.testing.assert_allclose(model(tokens, mask).cpu(), model(hidden, mask).cpu(), atol=1e-7)
    assert result <= 2e-6
    return {"max_permutation_probability_difference": result, "masked_nan_unchanged": True}


def run(output: Path, device_name: str) -> None:
    assert digest(output / "protocol.json") == read(output / "execution.json")["protocol_sha256"]
    if (output / "config.json").exists():
        raise ValueError("Use fresh output; never overwrite an experimental fit")
    device = select_device(device_name)
    torch.set_num_threads(1)
    audit(output / "environment.json")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    stack = dict(np.load(SOURCE / "fold0/stack.npz"))
    private = dict(np.load(SOURCE / "fold0/final/probabilities.npz"))
    data = dict(np.load(DATA / "development.npz"))
    assert not set(data["ids"]) & set(np.load(DATA / "frozen_test_ids.npy"))
    for split in ("guard", "assessment"):
        assert not set(data["groups"][stack["fit"]]) & set(data["groups"][stack[split]])
    p = private["train"]
    y = data["y"][stack["fit"]]
    rel = {
        name: float(
            -np.mean(
                y * np.log(p[:, i].clip(1e-6, 1 - 1e-6)) + (1 - y) * np.log1p(-p[:, i].clip(1e-6, 1 - 1e-6))
            )
        )
        for i, name in enumerate(IDS)
    }
    specs = declarations()
    write(
        output / "config.json",
        {
            "declarations": specs,
            "reliability": rel,
            "device": str(device),
            "source": str(SOURCE),
            "seed": 56001,
        },
    )
    models = {kind: DynamicFusion(kind, specs, rel).to(device) for kind in ("mean", "weighted", "set")}
    torch.manual_seed(56001)
    models["set"] = DynamicFusion("set", specs, rel).to(device)
    models["set"].save(output / "initial.pt")
    log = fitting(models["set"], p, y, seed=56001, epochs=20)
    write(output / "fit.json", log)
    for name, model in models.items():
        model.save(output / f"{name}.pt")
    old = CDCFusion("original", 3).to(device)
    old.load_state_dict(
        torch.load(SOURCE / "fold0/final/fusion_original60.pt", weights_only=True, map_location=device)
    )
    records = []
    invariance = {}
    saved = {}
    for split in ("guard", "assessment"):
        begin = 0 if split == "guard" else len(stack["guard"])
        sl = slice(begin, begin + len(stack[split]))
        for name, model in dict(models, existing=old).items():
            scores, preds = evaluation(model, private["held"][sl], data["y"][stack[split]], device)
            for row in scores:
                records.append({"split": split, "kind": name, **row})
            saved.update({f"{split}-{name}-{key}": v for key, v in preds.items()})
            if isinstance(model, DynamicFusion):
                invariance[f"{split}-{name}"] = consistency(model, private["held"][sl][:128], device)
    write(output / "results.json", records)
    np.savez(output / "predictions.npz", **saved)
    write(output / "consistency.json", invariance)
    latency_report(output)
    write(
        output / "training.json",
        {
            "pipeline_seconds": time.perf_counter() - start,
            "device": str(device),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
            "private_fits": 0,
            "router_fits": 0,
            "frozen_test_evaluations": 0,
        },
    )
    print(json.dumps(read(output / "training.json")), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--flower", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.latency:
        latency_report(args.output)
    elif args.summary:
        summarize(args.output)
    elif args.worker:
        flower_worker(args.output)
    elif args.flower:
        launch_flower(args.output)
    elif args.verify:
        verify(args.output, args.device)
    else:
        run(args.output, args.device)


def flower_worker(output: Path) -> None:
    """Exactly three real Flower clients aggregate only compatible new fusion weights."""
    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp
    from flwr.serverapp.strategy import FedAvg
    from flwr.simulation import run_simulation

    from ..collaboration.registry import ConsultationSession, Specialist, SpecialistRegistry
    from ..feedback.versions import PolicyRegistry
    from ..fusion.dynamic import FusionEvidence
    from . import cdc_flower as adapter

    torch.set_num_threads(1)
    client, server = ClientApp(), ServerApp()
    folder = output / "flower"
    cfg = read(output / "config.json")
    stack = dict(np.load(SOURCE / "fold0/stack.npz"))
    cached = dict(np.load(SOURCE / "fold0/final/probabilities.npz"))
    development = dict(np.load(DATA / "development.npz"))
    client.query("capabilities")(adapter.capabilities)
    client.query("registry_predict")(adapter.registry_predict)

    @client.train()
    def train(message: Message, ctx: Context) -> Message:
        origin = int(ctx.node_config["partition-id"])
        round_id = int(message.content["config"]["server-round"])
        model = DynamicFusion.load(output / "initial.pt")
        if message.content["config"]["fusion-contract"] != model.fingerprint():
            raise ValueError("Incompatible shared fusion contract")
        assert set(message.content) == {"arrays", "config"}
        model.load_state_dict(message.content["arrays"].to_torch_state_dict(), strict=True)
        before = snapshot(model)
        with threadpool_limits(limits=1):
            log = fitting(
                model,
                cached["train"],
                development["y"][stack["fit"]],
                seed=56010 + round_id,
                epochs=2,
                origin=origin,
            )
        state = snapshot(model)
        log["drift"] = sum(float(((v - before[k]) ** 2).sum()) for k, v in state.items()) ** 0.5
        log["contract"] = model.fingerprint()
        torch.save(state, folder / f"local_{round_id}_{origin}.pt")
        write(folder / f"local_{round_id}_{origin}.json", log)
        return Message(
            RecordDict(
                {
                    "arrays": ArrayRecord(state),
                    "metrics": MetricRecord(
                        {"num-examples": len(cached["train"]), "loss": log["curve"][-1]["bce"]}
                    ),
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
                raise TimeoutError("Three physical clients required")
            time.sleep(0.1)
        trace = []

        def query(node, topic, payload):
            request = Message(
                RecordDict({"config": ConfigRecord(payload)}), dst_node_id=node, message_type=f"query.{topic}"
            )
            replies = list(grid.send_and_receive([request], timeout=300))
            if len(replies) != 1 or replies[0].has_error() or replies[0].metadata.src_node_id != node:
                raise RuntimeError(f"Failed {topic}: {replies}")
            trace.append({"topic": topic, "node": node, "request": payload})
            return dict(replies[0].content["response"])

        discovered = {}
        for node in grid.get_node_ids():
            response = query(node, "capabilities", {})
            spec = Specialist.model_validate_json(response["declaration"])
            discovered[spec.specialist_id] = (spec, node)
        assert set(discovered) == set(IDS)
        registry = SpecialistRegistry(adapter.CDC_TASK, adapter.CDC_OUTPUT)
        lifecycle = []

        def register(name):
            spec, node = discovered[name]
            registry.register(spec, node_id=node)

        def log(name):
            lifecycle.append(
                {
                    "event": name,
                    "registered": [e.specialist.specialist_id for e in registry.discover()],
                    "available": [e.specialist.specialist_id for e in registry.discover(available_only=True)],
                }
            )

        register("A")
        register("B")
        log("register-AB")
        patient = str(int(stack["guard"][0]))

        def session():
            return ConsultationSession(
                registry,
                "A",
                (patient,),
                permitted=frozenset(IDS),
                feature_permissions={
                    n: frozenset(s.model_adapter.feature_names) for n, (s, _) in discovered.items()
                },
                available_features=frozenset(
                    f for s, _ in discovered.values() for f in s.model_adapter.feature_names
                ),
                max_calls=2,
                max_cost=0.1,
                alignment_namespace="cdc891-development-index",
            )

        # Own evidence is local private inference over Flower, distinct from peer consultation.
        spec, node = discovered["A"]
        payload = {
            "origin": "A",
            "destination": "A",
            "patient_ids": [patient],
            "adapter_id": spec.model_adapter.adapter_id,
            "checkpoint_sha256": spec.model_adapter.checkpoint_sha256,
            "task": spec.task.model_dump(mode="json"),
            "mode": "origin_local",
            "authorized_features": list(spec.model_adapter.feature_names),
        }
        own_reply = json.loads(query(node, "registry_predict", {"request": json.dumps(payload)})["json"])
        own = FusionEvidence.from_reply(own_reply, patient)
        models = {name: DynamicFusion.load(output / f"{name}.pt") for name in ("mean", "weighted", "set")}
        fused = []

        def record(label, observations):
            request = session()
            for kind, model in models.items():
                probability = model.fuse(observations, request)
                reverse = model.fuse(reversed(observations), request)
                assert probability == reverse
                fused.append(
                    {
                        "event": label,
                        "kind": kind,
                        "origin": "A",
                        "evidence": [e.__dict__ for e in observations],
                        "probability": probability,
                        "reverse_probability": reverse,
                    }
                )

        record("STOP-A", [own])

        def consult(name):
            def transport(entry, payload):
                return json.loads(
                    query(entry.node_id, "registry_predict", {"request": json.dumps(payload)})["json"]
                )

            reply = session().consult(name, transport)
            return FusionEvidence.from_reply(reply, patient)

        b = consult("B")
        record("AB", [own, b])
        log("consult-B")
        register("C")
        log("register-C")
        c = consult("C")
        record("ABC", [own, b, c])
        log("consult-C")
        registry.update("B", available=False)
        log("disable-B")
        assert "B" not in [e.specialist.specialist_id for e in session().eligible()]
        before = len(trace)
        try:
            consult("B")
        except ValueError:
            pass
        else:
            raise AssertionError("Unavailable B admitted")
        assert len(trace) == before
        record("AC-B-disabled", [own, c])
        registry.update("B", available=True)
        log("restore-B")
        b = consult("B")
        record("AB-restored", [own, b])
        registry.unregister("C")
        log("unregister-C")
        record("AB-C-removed", [own, b])
        initial = DynamicFusion.load(output / "initial.pt")

        def save(round_id, arrays):
            torch.save(arrays.to_torch_state_dict(), folder / f"global_{round_id}.pt")

        outcome = FedAvg(
            fraction_train=1, fraction_evaluate=0, min_train_nodes=3, min_available_nodes=3
        ).start(
            grid,
            initial_arrays=ArrayRecord(snapshot(initial)),
            num_rounds=3,
            train_config=ConfigRecord({"fusion-contract": initial.fingerprint()}),
            evaluate_fn=save,
            timeout=600,
        )
        candidate = DynamicFusion.load(output / "initial.pt")
        candidate.load_state_dict(outcome.arrays.to_torch_state_dict())
        candidate.save(folder / "federated.pt")
        drift = (
            sum(float(((v - initial.state_dict()[k]) ** 2).sum()) for k, v in candidate.state_dict().items())
            ** 0.5
        )
        assert drift > 0
        mean = models["mean"]
        versions = PolicyRegistry(folder / "versions")
        versions.save(
            "v000",
            snapshot(mean),
            {
                "eligible": True,
                "fusion_config": mean.config(),
                "scope": "research mean reference, no production promotion",
            },
        )
        versions.approve("v000")
        guard_p = cached["held"][: len(stack["guard"])]
        guard_y = development["y"][stack["guard"]]
        reference, _ = evaluation(mean, guard_p, guard_y, torch.device("cpu"))
        decisions = {}
        for version, model in [("v001", models["set"]), ("v002", candidate)]:
            scores, _ = evaluation(model, guard_p, guard_y, torch.device("cpu"))
            gates = validation_gate(scores, reference)
            check = consistency(model, guard_p[:128], torch.device("cpu"))
            gates["permutation"] = check["max_permutation_probability_difference"] <= 2e-6
            gates["singleton"] = all(model.fuse([own], session()) == own.probability for _ in range(1))
            eligible = all(gates.values())
            versions.save(
                version,
                snapshot(model),
                {
                    "eligible": eligible,
                    "fusion_config": model.config(),
                    "gates": gates,
                    "scope": "research only",
                },
            )
            if eligible:
                versions.approve(version)
            else:
                try:
                    versions.approve(version)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Rejected fusion approved")
            decisions[version] = {"eligible": eligible, "gates": gates, "validation": scores}
        before_rollback = versions.current()
        versions.rollback("v000")
        assert versions.current() == "v000" and isinstance(versions.load(), DynamicFusion)
        results = []
        predictions = {}
        for split in ("guard", "assessment"):
            begin = 0 if split == "guard" else len(stack["guard"])
            sl = slice(begin, begin + len(stack[split]))
            scores, preds = evaluation(
                candidate, cached["held"][sl], development["y"][stack[split]], torch.device("cpu")
            )
            results.extend([{"split": split, "kind": "federated", **row} for row in scores])
            predictions.update({f"{split}-{key}": v for key, v in preds.items()})
        write(folder / "metrics.json", results)
        np.savez(folder / "predictions.npz", **predictions)
        write(
            folder / "results.json",
            {
                "run_id": ctx.run_id,
                "physical_clients": 3,
                "rounds": 3,
                "local_updates": 9,
                "drift": drift,
                "lifecycle": lifecycle,
                "traffic": trace,
                "fusion": fused,
                "validation": decisions,
                "before_rollback": before_rollback,
                "after_rollback": versions.current(),
                "seconds": time.perf_counter() - started,
                "private_fits": 0,
                "router_fits": 0,
                "frozen_test_evaluations": 0,
                "config_source": cfg["source"],
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
    """Fresh evidence directory; reuse CDC handlers with explicit frozen final binding."""
    import subprocess
    import sys

    folder = output / "flower"
    folder.mkdir(exist_ok=False)
    env = os.environ | {
        "SWARMFL_CDC_ROOT": str(SOURCE.resolve()),
        "SWARMFL_CDC_STAGE": "fold0",
        "SWARMFL_CDC_PRIVATE_BLOCK": "final",
        "PYTHONUTF8": "1",
    }
    with (folder / "flower.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-u", "-m", __spec__.name, "--output", str(output), "--worker"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    assert (folder / "results.json").exists()


def verify(output: Path, device_name: str) -> None:
    """Independent read-only private prediction, all subset metrics and optimizer replay."""
    import joblib

    from ..collaboration.registry import ConsultationSession, Specialist, SpecialistRegistry
    from ..feedback.versions import PolicyRegistry
    from ..fusion.dynamic import FusionEvidence
    from .cdc_data import expert_predict

    torch.set_num_threads(1)
    device = select_device(device_name)
    source = dict(np.load(DATA / "development.npz"))
    stack = dict(np.load(SOURCE / "fold0/stack.npz"))
    arrays = dict(np.load(SOURCE / "fold0/final/probabilities.npz"))
    provenance = read(SOURCE / "fold0/final/provenance.json")
    cfg = read(output / "config.json")
    assert digest(output / "protocol.json") == read(output / "execution.json")["protocol_sha256"]
    assert not set(source["ids"]) & set(np.load(DATA / "frozen_test_ids.npy"))
    for key in ("guard", "assessment"):
        assert not set(source["groups"][stack["fit"]]) & set(source["groups"][stack[key]])
    assert not set(source["groups"][stack["guard"]]) & set(source["groups"][stack["assessment"]])
    private_count = 0
    with threadpool_limits(limits=1):
        for origin in range(3):
            record = provenance["clients"][str(origin)]
            for name, fold in [(f"oof_{i}", f) for i, f in enumerate(record["folds"])] + [
                ("final", record["final"])
            ]:
                model = joblib.load(SOURCE / "fold0/final" / f"client_{origin}/{name}.joblib")
                ids = stack["held"] if name == "final" else np.array(fold["held"])
                assert not set(source["groups"][ids]) & set(
                    source["groups"][fold["model_fit"] + fold["calibration"]]
                )
                expected = (
                    arrays["held"][:, origin]
                    if name == "final"
                    else arrays["train"][fold["held_positions"], origin]
                )
                np.testing.assert_allclose(expert_predict(model, source["x"][ids]), expected, atol=1e-7)
                private_count += 1
    y = source["y"][stack["fit"]]
    p = arrays["train"]
    for i, name in enumerate(IDS):
        risk = p[:, i].clip(1e-6, 1 - 1e-6)
        assert np.isclose(
            cfg["reliability"][name], float(-np.mean(y * np.log(risk) + (1 - y) * np.log1p(-risk))), atol=1e-7
        )
    results = read(output / "results.json")
    predictions = dict(np.load(output / "predictions.npz"))
    field_count = 0

    def compare(saved, actual):
        nonlocal field_count
        assert saved.keys() == actual.keys()
        for key, value in saved.items():
            if isinstance(value, (int, float)):
                assert np.isclose(value, actual[key], atol=2e-6, rtol=2e-6), (key, value, actual[key])
                field_count += 1
            else:
                assert value == actual[key]

    for kind in ("mean", "weighted", "set", "existing"):
        if kind == "existing":
            model = CDCFusion("original", 3).to(device)
            model.load_state_dict(
                torch.load(
                    SOURCE / "fold0/final/fusion_original60.pt", weights_only=True, map_location=device
                )
            )
        else:
            model = DynamicFusion.load(output / f"{kind}.pt", str(device))
        for split in ("guard", "assessment"):
            begin = 0 if split == "guard" else len(stack["guard"])
            sl = slice(begin, begin + len(stack[split]))
            rows, pred = evaluation(model, arrays["held"][sl], source["y"][stack[split]], device)
            saved = [r for r in results if r["kind"] == kind and r["split"] == split]
            for a, b in zip(saved, rows, strict=True):
                compare({k: v for k, v in a.items() if k not in {"split", "kind"}}, b)
            for key, value in pred.items():
                np.testing.assert_allclose(value, predictions[f"{split}-{kind}-{key}"], atol=2e-6)
            if kind != "existing":
                consistency(model, arrays["held"][sl][:128], device)
            if kind in ("mean", "weighted"):
                for o, subset in pairs():
                    w = (
                        np.ones(len(subset))
                        if kind == "mean"
                        else np.exp(-np.array([cfg["reliability"][IDS[i]] for i in subset]))
                    )
                    expected = (arrays["held"][sl][:, subset] * w).sum(1) / w.sum()
                    np.testing.assert_allclose(
                        pred[f"{o}-" + "".join(IDS[i] for i in subset)], expected, atol=2e-6
                    )
    # Exact same seeded central optimization replay, ephemeral and never selected.
    central = DynamicFusion.load(output / "initial.pt", str(device))
    fitting(central, p, y, seed=56001, epochs=20)
    trained = DynamicFusion.load(output / "set.pt", str(device))
    for key, value in central.state_dict().items():
        torch.testing.assert_close(value, trained.state_dict()[key], rtol=0, atol=1e-7)
    folder = output / "flower"
    result = read(folder / "results.json")
    assert result["physical_clients"] == 3 and result["router_fits"] == 0 and result["private_fits"] == 0
    local_count = 0
    with threadpool_limits(limits=1):
        for round_id in range(1, 4):
            before = torch.load(folder / f"global_{round_id - 1}.pt", weights_only=True)
            local = []
            for origin in range(3):
                model = DynamicFusion.load(output / "initial.pt")
                model.load_state_dict(before)
                fitting(model, p, y, seed=56010 + round_id, epochs=2, origin=origin)
                state = torch.load(folder / f"local_{round_id}_{origin}.pt", weights_only=True)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=1e-7)
                local.append(state)
                local_count += 1
            after = torch.load(folder / f"global_{round_id}.pt", weights_only=True)
            for key, value in after.items():
                torch.testing.assert_close(
                    value, sum(state[key] for state in local) / 3, rtol=2e-6, atol=2e-7
                )
    model = DynamicFusion.load(folder / "federated.pt", str(device))
    flower_metrics = read(folder / "metrics.json")
    flower_predictions = dict(np.load(folder / "predictions.npz"))
    for split in ("guard", "assessment"):
        begin = 0 if split == "guard" else len(stack["guard"])
        sl = slice(begin, begin + len(stack[split]))
        rows, pred = evaluation(model, arrays["held"][sl], source["y"][stack[split]], device)
        for a, b in zip([r for r in flower_metrics if r["split"] == split], rows, strict=True):
            compare({k: v for k, v in a.items() if k not in {"split", "kind"}}, b)
        for key, value in pred.items():
            np.testing.assert_allclose(value, flower_predictions[f"{split}-{key}"], atol=2e-6)
    # Reconstruct actual observed lifecycle evidence independently from frozen final private outputs.
    specs = [Specialist.model_validate(s) for s in cfg["declarations"]]
    reg = SpecialistRegistry(specs[0].task, specs[0].output_schema)
    for i, s in enumerate(specs):
        reg.register(s, node_id=i + 1)
    permissions = {s.specialist_id: frozenset(s.model_adapter.feature_names) for s in specs}
    for row in result["fusion"]:
        evidence = [FusionEvidence(**e) for e in row["evidence"]]
        patient = evidence[0].patient_id
        request = ConsultationSession(
            reg,
            row["origin"],
            (patient,),
            permitted=frozenset(IDS),
            feature_permissions=permissions,
            available_features=frozenset(f for s in specs for f in s.model_adapter.feature_names),
            max_calls=2,
        )
        for e in evidence:
            index = int(np.flatnonzero(stack["held"] == int(patient))[0])
            assert np.isclose(e.probability, arrays["held"][index, IDS.index(e.specialist_id)], atol=1e-7)
        model = DynamicFusion.load(output / f"{row['kind']}.pt")
        assert np.isclose(model.fuse(evidence, request), row["probability"], atol=2e-6)
        assert row["probability"] == row["reverse_probability"]
    versions = PolicyRegistry(folder / "versions")
    assert versions.current() == "v000"
    mean = [r for r in results if r["kind"] == "mean" and r["split"] == "guard"]
    mean = [{k: v for k, v in r.items() if k not in {"kind", "split"}} for r in mean]
    for version, entry in result["validation"].items():
        gates = validation_gate(entry["validation"], mean)
        assert all(entry["gates"][key] == value for key, value in gates.items())
        assert read(folder / "versions" / f"{version}.json")["eligible"] == all(entry["gates"].values())
    hashes = read(output / "historical-hashes.json")
    changed = [name for name, h in hashes.items() if not Path(name).exists() or digest(Path(name)) != h]
    assert not changed, changed
    write(
        output / "verification.json",
        {
            "historical_files": len(hashes),
            "changed": changed,
            "private_prediction_replays": private_count,
            "central_optimizer_replays": 1,
            "local_optimizer_replays": local_count,
            "FedAvg_replays": 3,
            "metric_fields": field_count,
            "lifecycle_fusions": len(result["fusion"]),
            "frozen_test_evaluations": 0,
            "frozen_test_sha256": digest(DATA / "frozen_test.npz"),
        },
    )
    print(json.dumps(read(output / "verification.json")), flush=True)


def latency_report(output: Path) -> None:
    """CPU matched ABC evidence latency, metadata encoding included, transport excluded."""
    torch.set_num_threads(1)
    p = np.load(SOURCE / "fold0/final/probabilities.npz")["train"][:512]
    models = {kind: DynamicFusion.load(output / f"{kind}.pt") for kind in ("mean", "weighted", "set")}
    old = CDCFusion("original", 3)
    old.load_state_dict(torch.load(SOURCE / "fold0/final/fusion_original60.pt", weights_only=True))
    models["existing"] = old
    if (output / "flower/federated.pt").exists():
        models["federated"] = DynamicFusion.load(output / "flower/federated.pt")
    timings = {}
    for name, model in models.items():
        for batch in (1, 512):

            def predict(batch=batch, model=model):
                if isinstance(model, DynamicFusion):
                    return prediction(model, p[:batch], 0, (0, 1, 2), torch.device("cpu"))
                with torch.no_grad():
                    return (
                        model(
                            torch.tensor(p[:batch]),
                            torch.ones((batch, 3), dtype=torch.bool),
                            torch.zeros(batch, dtype=torch.long),
                        )
                        .sigmoid()
                        .numpy()
                    )

            predict()
            t = time.perf_counter()
            for _ in range(50):
                predict()
            timings[f"{name}-{batch}"] = {
                "batch": batch,
                "repeats": 50,
                "seconds_per_batch": (time.perf_counter() - t) / 50,
            }
    write(output / "latency.json", timings)


def summarize(output: Path) -> None:
    """All subset metrics, separately reported; no best-model/subset selection."""
    import csv

    rows = read(output / "results.json") + read(output / "flower/metrics.json")
    summary = []
    for subset in ["A", "B", "C", "AB", "AC", "BC", "ABC"]:
        for kind in ("existing", "mean", "weighted", "set", "federated"):
            selected = [
                r for r in rows if r["split"] == "assessment" and r["subset"] == subset and r["kind"] == kind
            ]
            summary.append(
                {
                    "subset": subset,
                    "kind": kind,
                    "origin_average": len(selected),
                    **{
                        k: float(np.mean([r[k] for r in selected]))
                        for k in ["binary_cross_entropy", "auroc", "pr_auc", "accuracy", "ece", "brier"]
                    },
                }
            )
    write(
        output / "summary.json",
        {
            "assessment": summary,
            "interpretation": "Identical evidence comparisons only. Existing residual60 trained on STOP/one-peer, not ABC mask. New singleton exact pass-through unlike old learned residual. No routing optimization or production promotion. One reused development split; no clinical/statistical superiority claim.",
            "latency": read(output / "latency.json"),
        },
    )
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
