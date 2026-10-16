"""User-fixed three-agent CDC amendment; reuse private fits, retrain all shared models."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..devices import audit, peak_memory, select_device, synchronize
from .cdc_data import THREE_DOMAINS, THREE_SLOTS
from .cdc_routing import CDCFusion, fit_fusion, outcomes
from .m5_step03 import digest, write
from .m5_step035 import KINDS, gates, run_context, select_inner
from .routing_diagnostics import snapshot

PROTOCOL = Path("docs/experiments/m5-step035-three-agent-protocol.md")


def reuse_context(source: Path, target: Path, columns: list[str], records: list) -> None:
    """Copy compatible private states, IDs and probabilities; no shared weights copied."""
    target.mkdir(parents=True)
    shutil.copyfile(source / "target_provenance.json", target / "target_provenance.json")
    for block in ("final", "target0", "target1", "target2"):
        old, new = source / block, target / block
        new.mkdir()
        provenance = json.loads((old / "provenance.json").read_text())
        clients = {}
        for client, ((name, domain), slot) in enumerate(zip(THREE_DOMAINS.items(), THREE_SLOTS, strict=True)):
            metadata = provenance["clients"][str(slot)].copy()
            if metadata["columns"] != list(domain) or metadata["name"] != name:
                raise ValueError("Incompatible specialist provenance; do not silently remap")
            clients[str(client)] = metadata | {"source_client": slot, "agent": "ABC"[client]}
            folder = new / f"client_{client}"
            folder.mkdir()
            for original in sorted((old / f"client_{slot}").glob("*.joblib")):
                destination = folder / original.name
                checksum = digest(original)
                shutil.copyfile(original, destination)
                assert digest(destination) == checksum
                records.append(
                    {
                        "source": str(original.resolve()),
                        "destination": str(destination.resolve()),
                        "sha256": checksum,
                        "client": client,
                        "source_client": slot,
                    }
                )
        provenance["clients"] = clients
        write(new / "provenance.json", provenance)
        with np.load(old / "probabilities.npz") as data:
            np.savez(
                new / "probabilities.npz",
                train=data["train"][:, THREE_SLOTS],
                held=data["held"][:, THREE_SLOTS],
            )
    with np.load(source / "stack.npz") as data:
        remapped = {key: data[key].copy() for key in data.files}
    for key in ("p", "rel", "held_p", "held_rel"):
        remapped[key] = remapped[key][:, THREE_SLOTS]
    np.savez_compressed(target / "stack.npz", **remapped)


def fresh_fusions(folder: Path, y: np.ndarray, seed: int, device: torch.device) -> None:
    """New three-dimensional fusion fits for each independent upstream target stack."""
    stage = dict(np.load(folder / "stack.npz"))
    target_risks = {kind: np.zeros((len(stage["fit"]), 3, 4), dtype=np.float32) for kind in KINDS}
    logs = {}
    provenance = json.loads((folder / "target_provenance.json").read_text())["folds"]
    for block in ("final", "target0", "target1", "target2"):
        path = folder / block
        upstream = json.loads((path / "provenance.json").read_text())
        fit = upstream["train"]
        arrays = dict(np.load(path / "probabilities.npz"))
        for kind in KINDS:
            torch.manual_seed(seed)
            initial = snapshot(CDCFusion(kind, 3))
            destination = folder if block == "final" else path
            torch.save(initial, destination / f"fusion_initial_{kind}.pt")
            model, log = fit_fusion(arrays["train"], y[fit], kind, seed, device)
            torch.save(snapshot(model), destination / f"fusion_{kind}.pt")
            value = outcomes(model, arrays["held"], device)
            logs[f"{block}_{kind}"] = log | {
                "clients": 3,
                "initialization": "fresh_seeded",
                "initial_to_final_l2": sum(
                    float(((snapshot(model)[k] - v) ** 2).sum()) for k, v in initial.items()
                )
                ** 0.5,
            }
            if block == "final":
                np.save(folder / f"held_{kind}.npy", value)
            else:
                target_risks[kind][provenance[int(block[-1])]["positions"]] = value
    for kind, values in target_risks.items():
        np.savez_compressed(
            folder / f"targets_{kind}.npz", train=values, held=np.load(folder / f"held_{kind}.npy")
        )
    write(folder / "fusion_training.json", logs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/m5-step035-v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step035-three-agent-v1"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    root, source = args.output.resolve(), args.source.resolve()
    if root.exists():
        parser.error("Use a fresh output directory; do not overwrite experiments")
    root.mkdir(parents=True)
    preserved = {}
    for base in (Path("artifacts"), Path("docs/experiments")):
        for path in base.rglob("*"):
            if (
                path.is_file()
                and root not in path.resolve().parents
                and "ollama" not in path.parts
                and path.suffix
                in {".pt", ".npz", ".npy", ".json", ".jsonl", ".joblib", ".log", ".csv", ".md"}
                and "three-agent" not in path.name
            ):
                preserved[str(path)] = digest(path)
    write(root / "preserved-before.json", preserved)
    shutil.copyfile(PROTOCOL, root / "protocol.md")
    shutil.copyfile(source / "protocol.md", root / "parent-protocol.md")
    for filename in (
        "development.npz",
        "dataset.json",
        "pilot_ids.npy",
        "frozen_test_ids.npy",
        "frozen_test.npz",
        "contexts.json",
    ):
        shutil.copyfile(source / filename, root / filename)
        assert digest(source / filename) == digest(root / filename)
    write(
        root / "benchmark-config.json",
        {
            "clients": 3,
            "domains": THREE_DOMAINS,
            "source_slots": THREE_SLOTS,
            "source": str(source),
            "six_agent_weights_reused": False,
            "specialist_selection": "explicit_user_amendment",
        },
    )
    audit(root / "environment.json")
    device = select_device(args.device)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    metadata = json.loads((root / "dataset.json").read_text())
    contexts = json.loads((root / "contexts.json").read_text())
    data = dict(np.load(root / "development.npz"))
    records, results = [], {}
    with threadpool_limits(limits=1):
        for key, config in contexts.items():
            reuse_context(source / key, root / key, metadata["features"], records)
            fresh_fusions(root / key, data["y"], config["seed"], device)
            print(f"Reused specialists; fitted fresh fusion: {key}", flush=True)
        write(root / "reuse.json", {"models": records, "reused": len(records), "retrained_specialists": 0})
        for size in ("small", "pilot"):
            for stage in ("inner", "outer"):
                if stage == "outer":
                    write(root / f"selection_{size}.json", select_inner(results, size))
                for key, config in contexts.items():
                    if key.startswith(size) and key.endswith(stage):
                        results[key] = run_context(
                            root / key,
                            data["ids"],
                            metadata["features"],
                            config["seed"],
                            device,
                            THREE_DOMAINS,
                            THREE_SLOTS,
                        )
                        print(f"Completed {key}", flush=True)
            write(root / f"gates_{size}.json", gates(results, select_inner(results, size), size))
        write(root / "results.json", results)
    synchronize(device)
    write(
        root / "execution.json",
        {
            "seconds": time.perf_counter() - start,
            "device": str(device),
            "peak_allocated_bytes": peak_memory(device),
            "clients": 3,
            "reused_specialists": len(records),
            "six_agent_weights_reused": False,
            "frozen_test_evaluations": 0,
            "expanded_to_50k": False,
        },
    )
    print("Three-agent benchmark finished; stop before Step4", flush=True)


if __name__ == "__main__":
    main()
