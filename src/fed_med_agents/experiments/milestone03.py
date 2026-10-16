"""Validation-only real-Qwen/Flower benchmark with per-patient round trips."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from sklearn.model_selection import train_test_split

from ..agents.provider import OllamaProvider
from ..agents.runtime import CLIENT_PROMPT
from ..agents.schemas import ClientAction, PatientRequest
from ..data import ARTIFACTS, FEATURES, GROUPS, ROOT, load
from ..devices import audit, select_device
from .milestone02 import monitor


def prepare_requests(config: dict, output: Path, patients: int, modes: list[str], demo: bool) -> None:
    """Use validation rows only; no new training or test-based policy tuning."""
    data = load()
    ids = data["val"]
    if patients < len(ids):
        ids, _ = train_test_split(ids, train_size=patients, stratify=data["y"][ids], random_state=42)
    scaler = json.loads((ARTIFACTS / "dataset_summary.json").read_text())
    requests = []
    labels = {}
    for patient in ids:
        patient_id = f"simulated-val-{int(patient):04d}"
        labels[patient_id] = int(data["y"][patient])
        raw = data["x"][patient].copy()
        raw[0] = round(float(raw[0]) * scaler["age_scale"] + scaler["age_mean"])
        fields = {name: float(raw[index]) for index, name in enumerate(FEATURES)}
        for mode in modes:
            for policy in ("slm", "learned_router", "feature_domain"):
                # The demo limits features to A+B so the requested A→B path is feasible.
                supported = GROUPS["A"] + GROUPS["B"] if demo else FEATURES
                features = {name: fields[name] for name in supported} if mode == "new_patient" else {}
                consent = (
                    {
                        name: [feature for feature in group if feature in features]
                        for name, group in GROUPS.items()
                    }
                    if mode == "new_patient"
                    else {}
                )
                request = PatientRequest(
                    request_id=f"{patient_id}-{mode}-{policy}",
                    patient_id=patient_id,
                    origin="A",
                    mode=mode,
                    alignment_id=int(patient) if mode == "aligned" else None,
                    features=features,
                    authorized_features=consent,
                    budget=config["budget"],
                )
                requests.append({"request": request.model_dump(), "policy": policy})
    (output / "requests.json").write_text(
        json.dumps({"labels": labels, "requests": requests, "source_split": "validation"}, indent=2)
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples/diabetes-agents.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--patients", type=int, default=24)
    parser.add_argument(
        "--modes", nargs="+", choices=["aligned", "new_patient"], default=["new_patient", "aligned"]
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--request-file",
        type=Path,
        help="Run a supplied new-patient PatientRequest JSON without labels or dataset lookup",
    )
    parser.add_argument(
        "--match-realized-budget",
        action="store_true",
        help="Grant controls the same per-patient call budget used by the SLM",
    )
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "milestone03")
    parser.add_argument(
        "--policy-registry", type=Path, help="Use newest approved M4 shared policy, inference only"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 2 <= args.patients <= 78:
        parser.error("Use 2–78 validation patients")
    config = json.loads(args.config.read_text())
    if args.policy_registry:
        config["policy_registry"] = str(args.policy_registry.resolve())
    config["match_realized_budget"] = args.match_realized_budget
    config["checkpoint_dir"] = str((ROOT / config["checkpoint_dir"]).resolve())
    os.environ["FED_MED_M3_CONFIG"] = json.dumps(config)
    os.environ["FED_MED_DEVICE"] = str(select_device(args.device))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["FED_MED_M3_DIR"] = str(output)
    if args.worker:
        from flwr.simulation import run_simulation

        from ..agents.client_app import app as client_app
        from ..agents.server_app import app as server_app

        gpu = os.environ["FED_MED_DEVICE"] == "cuda"
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
    audit(output / "tabular_environment.json")
    provider = OllamaProvider(**config["provider"])
    # A real schema generation warms the one Qwen instance and makes GPU residency measurable.
    raw, timing = provider.generate(
        [
            {"role": "system", "content": CLIENT_PROMPT},
            {
                "role": "user",
                "content": "All required features are present; return the appropriate tool action.",
            },
        ],
        ClientAction.model_json_schema(),
    )
    ClientAction.model_validate_json(raw)
    provider_audit = provider.audit()
    (output / "provider_before.json").write_text(json.dumps(provider_audit | {"warmup": timing}, indent=2))
    residents = provider_audit["running_models"].get("models", [])
    residents = [
        model
        for model in residents
        if model.get("name") == provider.model or model.get("model") == provider.model
    ]
    if not residents or not any(model.get("size_vram", 0) > 0 for model in residents):
        raise RuntimeError("Real Qwen GPU residency was not verified")
    if args.request_file:
        request = PatientRequest.model_validate_json(args.request_file.read_text())
        if request.mode != "new_patient":
            parser.error("--request-file accepts new-patient dictionaries only")
        (output / "requests.json").write_text(
            json.dumps(
                {
                    "source_split": "external_request",
                    "labels": {},
                    "requests": [{"request": request.model_dump(), "policy": "slm"}],
                },
                indent=2,
            )
        )
    else:
        prepare_requests(config, output, args.patients, args.modes, args.demo)
    (output / "traces.jsonl").write_text("")
    (output / "metrics.json").unlink(missing_ok=True)
    stop, samples = threading.Event(), []
    watcher = threading.Thread(target=monitor, args=(stop, samples), daemon=True)
    watcher.start()
    try:
        command = [
            sys.executable,
            "-u",
            "-m",
            "fed_med_agents.experiments.milestone03",
            "--worker",
            "--config",
            str(args.config.resolve()),
            "--device",
            args.device,
            "--output",
            str(output),
        ]
        if args.match_realized_budget:
            command.append("--match-realized-budget")
        if args.policy_registry:
            command += ["--policy-registry", str(args.policy_registry.resolve())]
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
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait() or not (output / "metrics.json").exists():
                raise RuntimeError("M3 simulation failed; inspect flower.log")
    finally:
        stop.set()
        watcher.join(timeout=12)
        (output / "gpu_samples.json").write_text(json.dumps(samples, indent=2))


if __name__ == "__main__":
    main()
