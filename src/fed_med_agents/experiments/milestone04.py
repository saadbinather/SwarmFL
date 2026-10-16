"""Run delayed-feedback learning on unique training patients, never historical test."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

import torch

from ..data import ARTIFACTS
from ..devices import audit, select_device
from ..feedback.versions import PolicyRegistry
from ..fusion import CollaborativeModel


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ARTIFACTS / "milestone02/seed_42")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "milestone04")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--episodes", type=int, default=364)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument(
        "--triggers", type=int, nargs="+", help="Explicit update episode numbers (including drain)"
    )
    parser.add_argument("--delay", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Stream seed; specialist artifacts use seed 42")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 12 <= args.episodes <= 364 or args.interval < 1 or args.delay < 1:
        parser.error("Use 12..364 unique episodes, positive interval and label delay")
    if args.seed != 42:
        parser.error("This adapter currently requires seed-42 specialist artifacts")
    if args.source.name != "seed_42" or not (args.source / "shared_best.pt").exists():
        parser.error("--source must be a prepared M2 seed_42 directory")
    if args.triggers and (
        len(set(args.triggers)) != len(args.triggers)
        or any(cutoff < 1 or cutoff > args.episodes + args.delay for cutoff in args.triggers)
    ):
        parser.error("Triggers must be unique episode positions within the stream and final drain")
    output = args.output.resolve()
    device = select_device(args.device)
    config = {
        "source": str(args.source.resolve()),
        "output": str(output),
        "device": str(device),
        "seed": args.seed,
        "episodes": args.episodes,
        "delay": args.delay,
        "triggers": args.triggers
        or list(range(args.interval, args.episodes + 1, args.interval)) + [args.episodes + args.delay],
        "epochs": 2,
        "learning_rate": 0.0003,
        "min_feedback": 4,
        "window": 128,
        "epsilon": 0.2,
        "cost": 0.02,
        "approval_tolerance": 0.02,
        "max_update_norm": 0.1,
    }
    os.environ["FED_MED_M4_CONFIG"] = json.dumps(config)
    os.environ["FED_MED_M2_DIR"] = str(args.source.resolve().parent)
    os.environ["FED_MED_DEVICE"] = str(device)
    if args.worker:
        from flwr.simulation import run_simulation

        from ..feedback.client_app import app as client_app
        from ..feedback.server_app import app as server_app

        gpu = device.type == "cuda"
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
    if (output / "config.json").exists():
        parser.error("Output already contains a run; choose a fresh --output to preserve evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "credentials.json").write_text(
        json.dumps(
            {
                str(i): {"prediction_key": secrets.token_hex(32), "outcome_key": secrets.token_hex(32)}
                for i in range(3)
            }
        )
    )
    audit(output / "environment.json")
    registry = PolicyRegistry(output / "versions")
    # The adaptive policy must not start from M2 weights fitted on future stream outcomes.
    torch.manual_seed(args.seed)
    state = CollaborativeModel().state_dict()
    registry.save(
        "v000",
        state,
        {"eligible": True, "source": "seeded-independent-initialization", "protocol": "partial-feedback"},
    )
    registry.approve("v000")
    registry.save(
        "offline",
        torch.load(args.source / "shared_best.pt", weights_only=True, map_location="cpu"),
        {
            "eligible": False,
            "source": str(args.source.resolve()),
            "protocol": "M2-full-information-reference",
        },
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "fed_med_agents.experiments.milestone04",
        "--worker",
        "--source",
        str(args.source.resolve()),
        "--output",
        str(output),
        "--device",
        str(device),
        "--episodes",
        str(args.episodes),
        "--interval",
        str(args.interval),
        "--delay",
        str(args.delay),
    ]
    if args.triggers:
        command += ["--triggers", *map(str, args.triggers)]
    with (output / "flower.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=os.environ | {"PYTHONUTF8": "1", "RAY_DEDUP_LOGS": "0"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if process.wait() or not (output / "metrics.json").exists():
            raise RuntimeError("M4 simulation failed; preserved flower.log")


if __name__ == "__main__":
    main()
