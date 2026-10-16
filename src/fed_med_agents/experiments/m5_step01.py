"""Isolated M5 routing/fusion ablations with genuine Flower training and queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from ..collaboration import ConsultationBroker
from ..devices import audit, select_device
from ..federated import client_app as specialist
from ..fusion import CollaborativeModel
from .routing_diagnostics import cosine, describe, evaluate, fit, gradient_audit, snapshot

client_app = ClientApp()
client_app.query("identity")(specialist.identity)
client_app.query("predict")(specialist.predict)
server_app = ServerApp()
VARIANTS = ("joint", "frozen_router", "frozen_fusion")


def config() -> dict:
    """Read this isolated experiment's worker configuration."""
    return json.loads(os.environ["SWARMFL_M5_CONFIG"])


@client_app.train()
def train(message: Message, context: Context) -> Message:
    """Return only compatible shared arrays and scalar metrics over Flower."""
    cfg = dict(message.content["config"])
    client = specialist.client_index(context)
    seed, round_id = int(cfg["seed"]), int(cfg["server-round"])
    local = specialist.local_data(seed, client)
    path = specialist.directory(seed) / f"client_{client}"
    p = np.load(path / "training_predictions.npy")
    state, report = fit(
        message.content["arrays"].to_torch_state_dict(),
        p,
        local["train_y"],
        [client],
        seed,
        round_id,
        str(cfg["variant"]),
        select_device(str(cfg["device"])),
    )
    target = Path(config()["output"]) / f"seed_{seed}" / str(cfg["variant"])
    target.mkdir(parents=True, exist_ok=True)
    name = f"client_{client}_round_{round_id}"
    torch.save(state, target / f"{name}.pt")
    (target / f"{name}.json").write_text(json.dumps(report, indent=2))
    return Message(
        RecordDict(
            {
                "arrays": ArrayRecord(state),
                "metrics": MetricRecord({"num-examples": len(local["train_y"]), "client": client}),
            }
        ),
        reply_to=message,
    )


def load_cohort(source: Path) -> tuple[np.ndarray, ...]:
    """Extract only training and validation labels; never index historical test labels."""
    with np.load(source / "cohort.npz") as data:
        train_ids, val_ids = data["train"], data["val"]
        train_y, val_y = data["y"][train_ids], data["y"][val_ids]
    return train_ids, train_y, val_ids, val_y


@server_app.main()
def server(grid: Grid, context: Context) -> None:
    """Execute all predeclared variants, saving every update and round evaluation."""
    cfg = config()
    seed = cfg["seed"]
    device = select_device(cfg["device"])
    torch.set_num_threads(1)
    target = Path(cfg["output"]) / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    source = Path(cfg["source"]) / f"seed_{seed}"
    train_ids, train_y, val_ids, val_y = load_cohort(source)
    deadline = time.monotonic() + 90
    while len(list(grid.get_node_ids())) != 3:
        if time.monotonic() > deadline:
            raise TimeoutError("Three Flower nodes did not connect")
        time.sleep(0.1)
    nodes, hashes = {}, {}
    for node in grid.get_node_ids():
        reply = next(
            iter(
                grid.send_and_receive(
                    [
                        Message(
                            RecordDict({"config": ConfigRecord({"seed": seed})}),
                            dst_node_id=node,
                            message_type="query.identity",
                        )
                    ],
                    timeout=60,
                )
            )
        )
        if reply.has_error():
            raise RuntimeError("Identity query failed")
        identity = reply.content["identity"]
        client = int(identity["client"])
        if client in nodes:
            raise ValueError("Duplicate Flower specialist")
        nodes[client], hashes[str(client)] = node, identity["specialist_sha256"]
    broker = ConsultationBroker(grid, nodes, target / "consultations.jsonl", seed)
    caches = {}
    for split, ids, mode in (
        ("train", train_ids, "oof"),
        ("validation", val_ids, "inference"),
        ("train_final", train_ids, "inference"),
    ):
        columns = []
        for destination in range(3):
            response = broker.predict(0, destination, ids.tolist(), mode, f"m5/diagnostic-cache/{split}")
            if any(status != "ok" for status in response["status"]):
                raise RuntimeError("Incomplete Flower diagnostic evidence")
            columns.append(response["probabilities"])
        caches[split] = np.array(columns, dtype=np.float32).T
    np.savez(
        target / "evidence.npz", train_ids=train_ids, val_ids=val_ids, train_y=train_y, val_y=val_y, **caches
    )
    # Explicitly check the existing origin-local training caches match Flower data.
    for client in range(3):
        private = source / f"client_{client}"
        with np.load(private / "local.npz") as local:
            np.testing.assert_array_equal(local["train"], train_ids)
            np.testing.assert_array_equal(local["train_y"], train_y)
        np.testing.assert_allclose(np.load(private / "training_predictions.npy"), caches["train"], atol=1e-7)
    initial = torch.load(source / "shared_initial.pt", weights_only=True, map_location="cpu")
    results = {
        "seed": seed,
        "flower_run_id": context.run_id,
        "device": str(device),
        "private_hashes": hashes,
        "variants": {},
        "historical": {},
    }
    model = CollaborativeModel().to(device)
    start = time.perf_counter()
    for variant in VARIANTS:
        run = target / variant
        run.mkdir(exist_ok=True)
        history = {}

        def validate(
            round_id: int,
            arrays: ArrayRecord,
            run: Path = run,
            history: dict = history,
            variant: str = variant,
        ) -> MetricRecord:
            state = arrays.to_torch_state_dict()
            model.load_state_dict(state)
            torch.save(snapshot(model), run / f"server_round_{round_id}.pt")
            record = {
                "train": evaluate(model, caches["train"], train_y, seed),
                "validation": evaluate(model, caches["validation"], val_y, seed),
                "gradient_audit": gradient_audit(model, caches["train"], train_y),
            }
            if round_id:
                previous = torch.load(run / f"server_round_{round_id - 1}.pt", weights_only=True)
                states = [
                    torch.load(run / f"client_{i}_round_{round_id}.pt", weights_only=True) for i in range(3)
                ]
                for key in state:
                    torch.testing.assert_close(
                        state[key].cpu(), sum(s[key] for s in states) / 3, atol=2e-6, rtol=1e-5
                    )
                record["fedavg_verified"] = True
                record["local_update_cosines"] = {}
                for prefix in ("router.", "fusion."):
                    deltas = [
                        torch.cat(
                            [
                                (s[k] - previous[k]).flatten()
                                for k in s
                                if k.startswith(prefix) and k != "router.actions"
                            ]
                        )
                        for s in states
                    ]
                    record["local_update_cosines"][prefix] = [
                        cosine(deltas[0], deltas[1]),
                        cosine(deltas[0], deltas[2]),
                        cosine(deltas[1], deltas[2]),
                    ]
            history[str(round_id)] = record
            (run / "history.json").write_text(json.dumps(history, indent=2))
            bce = float(
                np.mean(
                    [
                        v["policies"]["learned"]["learned_fusion"]["binary_cross_entropy"]
                        for v in record["validation"].values()
                    ]
                )
            )
            print(f"M5 seed={seed} {variant} round={round_id} validation_bce={bce:.6f}", flush=True)
            return MetricRecord({"bce": bce})

        strategy = FedAvg(fraction_train=1, fraction_evaluate=0, min_train_nodes=3, min_available_nodes=3)
        outcome = strategy.start(
            grid,
            initial_arrays=ArrayRecord(initial),
            num_rounds=5,
            train_config=ConfigRecord({"seed": seed, "variant": variant, "device": str(device)}),
            evaluate_fn=validate,
            timeout=300,
        )
        results["variants"][variant] = {"history": history, "rounds": len(outcome.train_metrics_clientapp)}
    # Centralized reference uses only training labels/probabilities, no feature rows.
    state = initial
    central = {}
    central_path = target / "central_joint"
    central_path.mkdir()
    for round_id in range(1, 6):
        state, report = fit(state, caches["train"], train_y, [0, 1, 2], seed, round_id, "joint", device)
        model.load_state_dict(state)
        torch.save(state, central_path / f"server_round_{round_id}.pt")
        central[str(round_id)] = {
            "train": evaluate(model, caches["train"], train_y, seed),
            "validation": evaluate(model, caches["validation"], val_y, seed),
            "training": report,
            "gradient_audit": gradient_audit(model, caches["train"], train_y),
        }
    results["variants"]["central_joint"] = {"history": central}
    for name in ("shared_initial", "shared_best", "shared_final"):
        model.load_state_dict(torch.load(source / f"{name}.pt", weights_only=True, map_location=device))
        results["historical"][name] = {
            split: evaluate(model, caches[split], labels, seed)
            for split, labels in (("train", train_y), ("validation", val_y), ("train_final", train_y))
        }
    results["specialists"] = {
        split: {str(i): describe(labels, caches[split][:, i]) for i in range(3)}
        for split, labels in (("train", train_y), ("validation", val_y), ("train_final", train_y))
    }
    results["probability_shift"] = {
        str(i): {
            "oof_to_final_mean_abs": float(
                np.abs(caches["train"][:, i] - caches["train_final"][:, i]).mean()
            ),
            "train_oof_std": float(caches["train"][:, i].std()),
            "train_final_std": float(caches["train_final"][:, i].std()),
            "validation_std": float(caches["validation"][:, i].std()),
        }
        for i in range(3)
    }
    results["runtime_seconds"] = time.perf_counter() - start
    results["specialists_unchanged"] = all(
        hashlib.sha256((source / f"client_{i}/specialist.pt").read_bytes()).hexdigest() == hashes[str(i)]
        for i in range(3)
    )
    (target / "results.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    """Launch fixed-protocol seed runs in fresh directories with captured Flower logs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/milestone02"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        from flwr.simulation import run_simulation

        torch.set_num_threads(1)
        gpu = config()["device"] == "cuda"
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
    if args.output.exists():
        parser.error("Use a fresh output path; earlier artifacts must be preserved")
    args.output.mkdir(parents=True)
    device = select_device(args.device)
    audit(args.output / "environment.json")
    cfg = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "device": str(device),
        "seeds": args.seeds,
        "rounds": 5,
        "epochs": 10,
        "lr": 0.01,
        "cost": 0.02,
        "batch": 64,
        "clip": 5,
        "variants": list(VARIANTS),
    }
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    for seed in args.seeds:
        env = os.environ | {
            "SWARMFL_M5_CONFIG": json.dumps(cfg | {"seed": seed}),
            "FED_MED_M2_DIR": str(args.source.resolve()),
            "FED_MED_DEVICE": str(device),
            "PYTHONUTF8": "1",
        }
        with (args.output / f"flower_seed_{seed}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-u", "-m", __spec__.name, "--worker", "--output", str(args.output)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not (args.output / f"seed_{seed}/results.json").exists():
            raise RuntimeError("Flower did not finish diagnostics; inspect saved log")
        print(f"Completed M5 diagnostic seed {seed}", flush=True)


if __name__ == "__main__":
    main()
