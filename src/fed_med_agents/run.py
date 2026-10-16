"""Reproducible entry point; logs survive child-process console output."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .data import ARTIFACTS, ROOT


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--launcher",
        choices=["cli", "python"],
        default="cli",
        help="Current Flower CLI (default) or deprecated 1.39 Python launcher",
    )
    arguments = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    # Remove only previous per-round audit files, so participation cannot be stale.
    for path in ARTIFACTS.glob("client_*/round_*.json"):
        path.unlink()
    (ARTIFACTS / "federated_metrics.json").unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["RAY_DEDUP_LOGS"] = "0"
    environment["FED_MED_ARTIFACTS"] = str(ARTIFACTS)
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    commands = [
        (module, [sys.executable, "-u", "-m", f"fed_med_agents.{module}"])
        for module in ("data", "specialists")
    ]
    if arguments.launcher == "python":
        commands.append(("server_app", [sys.executable, "-u", "-m", "fed_med_agents.server_app"]))
    else:
        flwr = str(Path(sys.executable).parent / ("flwr.exe" if os.name == "nt" else "flwr"))
        commands.extend(
            [
                (
                    "simulation configuration",
                    [
                        flwr,
                        "federation",
                        "simulation-config",
                        "@none/default",
                        "local",
                        "--num-supernodes",
                        "3",
                        "--client-resources-num-cpus",
                        "1",
                        "--client-resources-num-gpus",
                        "0",
                        "--init-args-num-cpus",
                        "3",
                    ],
                ),
                ("Flower CLI", [flwr, "run", ".", "local", "--federation", "@none/default", "--stream"]),
            ]
        )
    with (ARTIFACTS / "run.log").open("w", encoding="utf-8") as log:
        for name, command in commands:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise SystemExit(f"{name} failed; inspect artifacts/run.log")
    if not (ARTIFACTS / "federated_metrics.json").exists():
        raise SystemExit("No verified federated result; inspect artifacts/run.log")
    with (ARTIFACTS / "requirements-lock.txt").open("w") as lock:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=lock, check=True)


if __name__ == "__main__":
    main()
