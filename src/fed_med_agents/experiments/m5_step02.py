"""Isolated nested-training repair variants using the real Flower engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from ..devices import audit, peak_memory, select_device, synchronize
from .repair_models import CANDIDATES, evaluate, fit_bins, make_model, objective, select_inner, value_targets
from .routing_diagnostics import cosine, snapshot, vector
from .routing_diagnostics import fit as central_fit

client_app, server_app = ClientApp(), ServerApp()


def config() -> dict:
    return json.loads(os.environ["SWARMFL_REPAIR_CONFIG"])


def source(context_name: str, client: int) -> Path:
    if context_name not in config()["contexts"] or client not in range(3):
        raise ValueError("Unknown context/client")
    return Path(config()["output"]) / "contexts" / context_name / "seed_42" / f"client_{client}"


@client_app.query("evidence")
def evidence(message: Message, context: Context) -> Message:
    cfg = message.content["config"]
    client = int(context.node_config["partition-id"])
    folder = source(str(cfg["context"]), client)
    with np.load(folder / "local.npz") as local:
        record = {
            "client": client,
            "context": cfg["context"],
            "fit_ids": local["train"].tolist(),
            "assessment_ids": local["val"].tolist(),
            "oof": local["oof"].tolist(),
            "assessment": local["assessment"].tolist(),
            "private_hash": hashlib.sha256((folder / "specialist.pt").read_bytes()).hexdigest(),
        }
    return Message(RecordDict({"response": ConfigRecord(record)}), reply_to=message)


@client_app.query("cache")
def cache(message: Message, context: Context) -> Message:
    cfg = message.content["config"]
    client = int(context.node_config["partition-id"])
    folder = source(str(cfg["context"]), client)
    p = np.asarray(cfg["probabilities"], dtype=np.float32).reshape(-1, 3)
    with np.load(folder / "local.npz") as local:
        np.testing.assert_array_equal(local["train"], cfg["fit_ids"])
        np.testing.assert_allclose(local["oof"], p[:, client], atol=1e-7)
    np.save(folder / "shared_training.npy", p)
    return Message(RecordDict({"ack": ConfigRecord({"client": client, "cached": True})}), reply_to=message)


def local_fit(
    state: dict,
    p: np.ndarray,
    y: np.ndarray,
    origin: int,
    settings: dict,
    device: torch.device,
    optimizer_state: dict | None = None,
) -> tuple[dict, dict, dict]:
    """Adam reset/persistence or standard FedProx client loss on compatible weights."""
    model = make_model(settings.get("kind", "original")).to(device)
    model.load_state_dict(state)
    mode = settings.get("mode", "joint")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            not (mode == "uniform" and name.startswith("router."))
            and not (mode == "value" and name.startswith("fusion."))
        )
    initial = snapshot(model)
    reference = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad], lr=0.01)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    start_steps = max((float(v["step"]) for v in optimizer.state.values()), default=0.0)
    probs, labels = torch.tensor(p, device=device), torch.tensor(y, device=device)
    origins = torch.full((len(y),), origin, device=device, dtype=torch.long)
    targets = value_targets(model, probs, labels, origins) if mode == "value" else None
    generator = torch.Generator().manual_seed(int(settings["seed"]) + int(settings["round"]))
    norms = {"router.": [], "fusion.": []}
    mu = float(settings.get("mu", 0))
    start, steps, max_prox = time.perf_counter(), 0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(int(settings["epochs"])):
        for batch in torch.randperm(len(y), generator=generator).split(64):
            ids = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(
                model,
                probs[ids],
                labels[ids],
                origins[ids],
                mode,
                float(settings.get("cost", 0.02)),
                targets[ids] if targets is not None else None,
            )
            proximal = sum(
                ((v - reference[k]) ** 2).sum() for k, v in model.named_parameters() if v.requires_grad
            )
            loss = loss + 0.5 * mu * proximal
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite experimental objective")
            loss.backward()
            for prefix, values in norms.items():
                values.append(float(vector(model, True, prefix).norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            max_prox = max(max_prox, float(proximal.detach()))
            steps += 1
    synchronize(device)
    final = snapshot(model)
    report = {
        "steps": steps,
        "origin_loss_computations": steps,
        "optimizer_step_start": start_steps,
        "optimizer_step_end": max(float(v["step"]) for v in optimizer.state.values()),
        "mu": mu,
        "max_proximal_squared_norm": max_prox,
        "loss": float(loss.detach()),
        "gradient_norm": {k: float(np.mean(v)) for k, v in norms.items()},
        "drift": {
            prefix: sum(float(((final[k] - initial[k]) ** 2).sum()) for k in final if k.startswith(prefix))
            ** 0.5
            for prefix in norms
        },
        "runtime_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": peak_memory(device),
        "device": str(device),
    }
    return final, optimizer.state_dict(), report


@client_app.train()
def train(message: Message, context: Context) -> Message:
    cfg = dict(message.content["config"])
    client = int(context.node_config["partition-id"])
    folder = source(str(cfg["context"]), client)
    target = Path(config()["output"]) / "runs" / str(cfg["context"]) / str(cfg["candidate"])
    target.mkdir(parents=True, exist_ok=True)
    optimizer_path = target / f"optimizer_{client}.pt"
    previous = (
        torch.load(optimizer_path, weights_only=True)
        if cfg["persistent"] and optimizer_path.exists()
        else None
    )
    if float(cfg.get("proximal-mu", 0)) != float(cfg.get("mu", 0)):
        raise ValueError("Installed FedProx coefficient not delivered correctly")
    with np.load(folder / "local.npz") as local:
        y = local["train_y"].copy()
    settings = cfg | {"round": int(cfg["server-round"])}
    state, optimizer, report = local_fit(
        message.content["arrays"].to_torch_state_dict(),
        np.load(folder / "shared_training.npy"),
        y,
        client,
        settings,
        select_device(cfg["device"]),
        previous,
    )
    if cfg["persistent"]:
        torch.save(optimizer, optimizer_path)
    report.update(
        client=client,
        round=int(cfg["server-round"]),
        num_examples=len(y),
        payload_record_keys=["arrays", "metrics"],
    )
    name = f"client_{client}_round_{cfg['server-round']}"
    torch.save(state, target / f"{name}.pt")
    (target / f"{name}.json").write_text(json.dumps(report, indent=2))
    return Message(
        RecordDict(
            {
                "arrays": ArrayRecord(state),
                "metrics": MetricRecord(
                    {
                        "client": client,
                        "num-examples": len(y),
                        "steps": report["steps"],
                        "loss": report["loss"],
                    }
                ),
            }
        ),
        reply_to=message,
    )


def strategy(mu: float = 0.0) -> FedAvg:
    """Use installed Flower algorithms, not a hand-written aggregation substitute."""
    settings = {
        "fraction_train": 1.0,
        "fraction_evaluate": 0.0,
        "min_train_nodes": 3,
        "min_available_nodes": 3,
    }
    return FedProx(**settings, proximal_mu=mu) if mu else FedAvg(**settings)


def origin_gradients(model, p, y, settings, device) -> dict:
    gradients = {"router.": [], "fusion.": []}
    probs, labels = torch.tensor(p, device=device), torch.tensor(y, device=device)
    for origin in range(3):
        origins = torch.full((len(y),), origin, device=device, dtype=torch.long)
        model.zero_grad(set_to_none=True)
        targets = value_targets(model, probs, labels, origins) if settings.get("mode") == "value" else None
        objective(
            model, probs, labels, origins, settings.get("mode", "joint"), settings.get("cost", 0.02), targets
        ).backward()
        for prefix, values in gradients.items():
            values.append(vector(model, True, prefix))
    return {
        prefix: [cosine(v[0], v[1]), cosine(v[0], v[2]), cosine(v[1], v[2])]
        for prefix, v in gradients.items()
    }


@server_app.main()
def server(grid: Grid, context: Context) -> None:
    cfg = config()
    root, device = Path(cfg["output"]), select_device(cfg["device"])
    torch.set_num_threads(1)
    deadline = time.perf_counter() + 90
    while len(list(grid.get_node_ids())) != 3:
        if time.perf_counter() > deadline:
            raise TimeoutError("Three experimental clients required")
        time.sleep(0.1)
    nodes = list(grid.get_node_ids())
    caches, identities = {}, {}
    traffic = []
    for key in cfg["contexts"]:
        records = {}
        for node in nodes:
            request = Message(
                RecordDict({"config": ConfigRecord({"context": key})}),
                dst_node_id=node,
                message_type="query.evidence",
            )
            replies = list(grid.send_and_receive([request], timeout=120))
            if len(replies) != 1 or replies[0].has_error():
                raise RuntimeError("Evidence query failed")
            r = dict(replies[0].content["response"])
            client = int(r["client"])
            if client in records:
                raise ValueError("Duplicate origin")
            records[client], identities[client] = r, node
            traffic.append(
                {
                    "context": key,
                    "client": client,
                    "fit_ids": r["fit_ids"],
                    "assessment_ids": r["assessment_ids"],
                    "oof": r["oof"],
                    "assessment": r["assessment"],
                    "private_hash": r["private_hash"],
                }
            )
        if sorted(records) != [0, 1, 2]:
            raise ValueError("Missing origin")
        p = np.asarray([records[i]["oof"] for i in range(3)], dtype=np.float32).T
        held = np.asarray([records[i]["assessment"] for i in range(3)], dtype=np.float32).T
        for client in range(3):
            payload = {
                "context": key,
                "fit_ids": records[client]["fit_ids"],
                "probabilities": p.flatten().tolist(),
            }
            replies = list(
                grid.send_and_receive(
                    [
                        Message(
                            RecordDict({"config": ConfigRecord(payload)}),
                            dst_node_id=identities[client],
                            message_type="query.cache",
                        )
                    ],
                    timeout=60,
                )
            )
            if len(replies) != 1 or replies[0].has_error():
                raise RuntimeError("Training probability delivery failed")
        with np.load(root / "contexts" / key / "seed_42/cohort.npz") as data:
            y, held_y = data["y"][data["train"]], data["y"][data["val"]]
        caches[key] = (p, y, held, held_y)
    (root / "flower_evidence.json").write_text(json.dumps(traffic, indent=2))
    results = {key: {} for key in cfg["contexts"]}
    participation = []

    def run(key, name, profile, initial=None, rounds=5):
        settings = {
            "epochs": 10,
            "kind": "original",
            "mode": "joint",
            "persistent": False,
            "cost": 0.02,
            "mu": 0.0,
            **profile,
        }
        settings.update(context=key, candidate=name, seed=cfg["contexts"][key]["seed"], device=str(device))
        torch.manual_seed(settings["seed"])
        model = make_model(settings["kind"]).to(device)
        if initial is not None:
            model.load_state_dict(initial)
        target = root / "runs" / key / name
        target.mkdir(parents=True, exist_ok=True)
        history = {}
        p, y, held, held_y = caches[key]

        def save(round_id, arrays):
            state = arrays.to_torch_state_dict()
            model.load_state_dict(state)
            torch.save(snapshot(model), target / f"server_round_{round_id}.pt")
            record = {"origin_gradient_cosines": origin_gradients(model, p, y, settings, device)}
            if round_id:
                clients = [
                    torch.load(target / f"client_{i}_round_{round_id}.pt", weights_only=True)
                    for i in range(3)
                ]
                reports = [
                    json.loads((target / f"client_{i}_round_{round_id}.json").read_text()) for i in range(3)
                ]
                if sorted(r["client"] for r in reports) != [0, 1, 2]:
                    raise ValueError("Three distinct client updates required")
                previous = torch.load(target / f"server_round_{round_id - 1}.pt", weights_only=True)
                for k in state:
                    torch.testing.assert_close(
                        state[k].cpu(), sum(c[k] for c in clients) / 3, atol=2e-6, rtol=1e-5
                    )
                record["fedavg_verified"] = True
                record["reports"] = reports
                record["local_update_cosines"] = {}
                for prefix in ("router.", "fusion."):
                    vectors = [
                        torch.cat([(c[k] - previous[k]).flatten() for k in c if k.startswith(prefix)])
                        for c in clients
                    ]
                    record["local_update_cosines"][prefix] = [
                        cosine(vectors[0], vectors[1]),
                        cosine(vectors[0], vectors[2]),
                        cosine(vectors[1], vectors[2]),
                    ]
                participation.append(
                    {"context": key, "candidate": name, "round": round_id, "clients": [0, 1, 2]}
                )
            history[str(round_id)] = record
            (target / "history.json").write_text(json.dumps(history, indent=2))
            return MetricRecord({"saved-round": round_id})

        start = time.perf_counter()
        outcome = strategy(settings["mu"]).start(
            grid,
            initial_arrays=ArrayRecord(snapshot(model)),
            num_rounds=rounds,
            train_config=ConfigRecord(settings),
            evaluate_fn=save,
            timeout=300,
        )
        model.load_state_dict(outcome.arrays.to_torch_state_dict())
        result = {
            "settings": settings,
            "flower_run_id": context.run_id,
            "communication_rounds": rounds,
            "runtime_seconds": time.perf_counter() - start,
            "history": history,
        }
        if settings["mode"] != "uniform":
            mode = "value" if settings["mode"] == "value" else "joint"
            result["assessment"] = evaluate(model, held, held_y, mode)
            result["fitting"] = evaluate(model, p, y, mode)
        (target / "results.json").write_text(json.dumps(result, indent=2))
        print(f"M5 Step2 {key} {name} completed {rounds} real Flower rounds", flush=True)
        results[key][name] = result
        return snapshot(model)

    # Phase B finishes for every nested context before routing/fusion studies.
    for key in cfg["contexts"]:
        for name in ("avg10", "avg1", "avg2", "avg5", "persistent2", "prox2"):
            run(key, name, CANDIDATES[name])
        p, y, held, held_y = caches[key]
        for name in ("central_pooled", "central_interleaved"):
            torch.manual_seed(cfg["contexts"][key]["seed"])
            state = snapshot(make_model())
            target = root / "runs" / key / name
            target.mkdir()
            torch.save(state, target / "server_round_0.pt")
            reports = []
            for round_id in range(1, 6):
                state, report = central_fit(
                    state,
                    p,
                    y,
                    [0, 1, 2],
                    cfg["contexts"][key]["seed"],
                    round_id,
                    "central_pooled" if name == "central_pooled" else "joint",
                    device,
                )
                reports.append(report)
                torch.save(state, target / f"server_round_{round_id}.pt")
            model = make_model().to(device)
            model.load_state_dict(state)
            result = {
                "settings": {"kind": "original", "mode": "joint"},
                "communication_rounds": 0,
                "training_reports": reports,
                "assessment": evaluate(model, held, held_y),
                "fitting": evaluate(model, p, y),
            }
            results[key][name] = result
            (target / "results.json").write_text(json.dumps(result, indent=2))
    # Phase C: cost, pretrained original fusion, marginal value and utility bins.
    for key in cfg["contexts"]:
        for name in ("cost10", "cost30"):
            run(key, name, CANDIDATES[name])
        pretrained = run(key, "value_pretrain", {"epochs": 5, "mode": "uniform"}, rounds=3)
        run(key, "value", CANDIDATES["value"], pretrained)
        model = make_model().to(device)
        model.load_state_dict(pretrained)
        p, y, held, held_y = caches[key]
        table = fit_bins(model, p, y)
        np.save(root / "runs" / key / "value_pretrain/bins.npy", table)
        results[key]["utility_bins"] = {
            "assessment": evaluate(model, held, held_y, "bins", bins=table),
            "communication_rounds": 0,
            "source": "value_pretrain",
        }
        (root / "runs" / key / "value_pretrain/bin_results.json").write_text(
            json.dumps(results[key]["utility_bins"], indent=2)
        )
    # Phase D: fixed-capacity/linear fusion variants; then identical-evidence controls.
    for key in cfg["contexts"]:
        for name in ("logistic", "small"):
            run(key, name, CANDIDATES[name])
        p, y, held, held_y = caches[key]
        actions = [results[key]["avg10"]["assessment"][str(i)]["actions"] for i in range(3)]
        for name in ("avg10", "logistic", "small", "value"):
            profile = CANDIDATES[name]
            model = make_model(profile.get("kind", "original")).to(device)
            model.load_state_dict(
                torch.load(root / "runs" / key / name / "server_round_5.pt", weights_only=True)
            )
            results[key][name]["same_avg10_evidence"] = evaluate(model, held, held_y, fixed_actions=actions)
            if name == "avg10":
                results[key]["controls"] = {
                    mode: evaluate(model, held, held_y, mode) for mode in ("stop", "random", "cyclic")
                }
        folder = root / "contexts" / key / "seed_42"
        for i in range(3):
            meta = json.loads((folder / f"client_{i}/specialist.json").read_text())
            if (
                hashlib.sha256((folder / f"client_{i}/specialist.pt").read_bytes()).hexdigest()
                != meta["specialist_sha256"]
            ):
                raise ValueError("Frozen specialist changed")
    selected = select_inner(results)
    (root / "results.json").write_text(
        json.dumps(
            {
                "contexts": results,
                "selection": selected,
                "participation": participation,
                "flower_run_id": context.run_id,
            },
            indent=2,
        )
    )
    print("M5 Step2 B/C/D completed; no model promoted", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step02-v1"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Hash prior evidence and prepare fresh nested data"
    )
    parser.add_argument("--source", type=Path, default=Path("artifacts/milestone02/seed_42"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_only:
        from .repair_data import prepare_nested

        if output.exists():
            parser.error("Preparation requires a fresh output directory")
        output.mkdir(parents=True)
        preserved = {}
        for path in Path("artifacts").rglob("*"):
            if (
                path.is_file()
                and output not in path.resolve().parents
                and "ollama" not in path.parts
                and path.suffix in {".pt", ".npz", ".npy", ".json", ".jsonl", ".log", ".csv", ".txt"}
            ):
                preserved[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path("docs/experiments").glob("*.md"):
            preserved[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        (output / "preserved-before.json").write_text(json.dumps(preserved, indent=2))
        shutil.copyfile("docs/experiments/m5-step02-protocol.md", output / "protocol.md")
        baseline = {
            str(seed): json.loads(Path(f"artifacts/m5-step01/seed_{seed}/results.json").read_text())[
                "variants"
            ]["joint"]["history"]["5"]["validation"]
            for seed in (42, 43, 44)
        }
        (output / "historical-baseline.json").write_text(json.dumps(baseline, indent=2))
        prepare_nested(args.source, output)
        return
    cfg = {
        "output": str(output),
        "device": str(select_device(args.device)),
        "contexts": json.loads((output / "contexts.json").read_text()),
    }
    os.environ["SWARMFL_REPAIR_CONFIG"] = json.dumps(cfg)
    if args.worker:
        from flwr.simulation import run_simulation

        gpu = cfg["device"] == "cuda"
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=3,
            backend_config={
                "client_resources": {"num_cpus": 1, "num_gpus": int(gpu)},
                "init_args": {
                    "num_cpus": 1,
                    "num_gpus": int(gpu),
                    "include_dashboard": False,
                    "object_store_memory": 256 * 1024 * 1024,
                },
            },
        )
        return
    if (output / "runs").exists():
        parser.error("Training output exists; preserve it and choose a fresh prepared directory")
    audit(output / "environment.json")
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    with (output / "flower.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                __spec__.name,
                "--worker",
                "--output",
                str(output),
                "--device",
                cfg["device"],
            ],
            env=os.environ | {"PYTHONUTF8": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not (output / "results.json").exists():
        raise RuntimeError("Simulation failed before completing B/C/D")


if __name__ == "__main__":
    main()
