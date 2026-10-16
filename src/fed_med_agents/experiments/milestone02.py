"""Reproducible M2 entry point. M1 entry points and artifacts remain independent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..data import ARTIFACTS
from ..devices import audit, select_device
from .diabetes import prepare


def monitor(stop: threading.Event, samples: list[dict]) -> None:
    while not stop.is_set():
        try:
            data = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                text=True,
                timeout=10,
            ).strip()
            values = data.splitlines()[0].split(",")
            samples.append(
                {
                    "time": time.time(),
                    "gpu_utilization_percent": int(values[0]),
                    "board_memory_mib": int(values[1]),
                }
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(1.0)


def launch(config: dict) -> None:
    """Uses installed Flower 1.39 runtime; no handwritten parameter aggregation."""
    from flwr.simulation import run_simulation

    from ..federated.client_app import app as client_app
    from ..federated.server_app import app as server_app

    os.environ["FED_MED_M2_CONFIG"] = json.dumps(config)
    gpu = config["device"] == "cuda"
    # One actor executes all three logical clients sequentially; reserve the full GPU.
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=3,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": 1 if gpu else 0},
            "init_args": {"num_cpus": 1, "num_gpus": 1 if gpu else 0, "include_dashboard": False},
        },
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--budget", type=int, default=1)
    parser.add_argument("--cost", type=float, default=0.02)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "milestone02")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rounds < 1 or args.epochs < 1 or args.folds < 2 or args.cost < 0 or not 0 <= args.budget < 3:
        parser.error("Invalid rounds, epochs, folds, consultation cost, or budget")
    device = select_device(args.device)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["FED_MED_M2_DIR"] = str(output)
    os.environ["FED_MED_DEVICE"] = str(device)
    os.environ["RAY_DEDUP_LOGS"] = "0"
    config = {
        "rounds": args.rounds,
        "epochs": args.epochs,
        "budget": args.budget,
        "cost": args.cost,
        "device": str(device),
        "amp": args.amp,
        "learning_rate": 0.01,
    }
    if args.worker is not None:
        launch(config | {"seed": args.worker})
        return
    audit(output / "environment.json")
    stop, samples = threading.Event(), []
    watcher = threading.Thread(target=monitor, args=(stop, samples), daemon=True)
    watcher.start()
    started = time.perf_counter()
    try:
        for seed in args.seeds:
            path = output / f"seed_{seed}"
            if not args.skip_prepare:
                prepare(seed, path, str(device), args.folds)
            if not (path / "cohort.npz").exists():
                raise FileNotFoundError("Prepare the client-local dataset first")
            for old in path.glob("client_*/round_*.json"):
                old.unlink()
            (path / "metrics.json").unlink(missing_ok=True)
            command = [
                sys.executable,
                "-u",
                "-m",
                "fed_med_agents.experiments.milestone02",
                "--worker",
                str(seed),
                "--device",
                str(device),
                "--rounds",
                str(args.rounds),
                "--epochs",
                str(args.epochs),
                "--budget",
                str(args.budget),
                "--cost",
                str(args.cost),
                "--output",
                str(output),
            ]
            if args.amp:
                command.append("--amp")
            with (path / "flower.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=os.environ | {"PYTHONUTF8": "1"},
                )
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait() != 0 or not (path / "metrics.json").exists():
                    raise RuntimeError(f"Seed {seed} failed; inspect {path / 'flower.log'}")
    finally:
        stop.set()
        watcher.join(timeout=12)
        (output / "gpu_samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    results = [json.loads((output / f"seed_{seed}" / "metrics.json").read_text()) for seed in args.seeds]
    summary = {
        "seeds": args.seeds,
        "device": str(device),
        "runtime_seconds": time.perf_counter() - started,
        "config": config,
        "comparison_origin_A": {},
        "run_ids": [r["flower_run_id"] for r in results],
    }
    for policy in results[0]["test"]:
        summary["comparison_origin_A"][policy] = {
            key: {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
            for key in (
                "accuracy",
                "auroc",
                "f1",
                "binary_cross_entropy",
                "average_consultations",
                "collaboration_percent",
                "amortized_latency_ms",
            )
            for values in [[r["test"][policy]["0"][key] for r in results]]
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "requirements-lock.txt").open("w") as lock:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=lock, check=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
