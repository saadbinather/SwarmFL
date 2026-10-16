"""A global-step-matched central control to expose optimizer-count confounding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fed_med_agents.devices import audit, select_device
from fed_med_agents.experiments.routing_diagnostics import evaluate, fit, gradient_audit
from fed_med_agents.fusion import CollaborativeModel


def main() -> None:
    """Train on previously Flower-acquired OOF evidence; no new consultations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/m5-step01"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step01-central-pooled"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path")
    args.output.mkdir(parents=True)
    audit(args.output / "environment.json")
    cfg = json.loads((args.source / "config.json").read_text())
    device = select_device(args.device)
    for seed in cfg["seeds"]:
        with np.load(args.source / f"seed_{seed}/evidence.npz") as data:
            cache = {key: data[key] for key in data.files}
        state = torch.load(Path(cfg["source"]) / f"seed_{seed}/shared_initial.pt", weights_only=True)
        model = CollaborativeModel().to(device)
        history = {}
        root = args.output / f"seed_{seed}"
        root.mkdir()
        for round_id in range(1, 6):
            state, report = fit(
                state, cache["train"], cache["train_y"], [0, 1, 2], seed, round_id, "central_pooled", device
            )
            torch.save(state, root / f"server_round_{round_id}.pt")
            model.load_state_dict(state)
            history[str(round_id)] = {
                "train": evaluate(model, cache["train"], cache["train_y"], seed),
                "validation": evaluate(model, cache["validation"], cache["val_y"], seed),
                "training": report,
                "gradient_audit": gradient_audit(model, cache["train"], cache["train_y"]),
            }
        (root / "results.json").write_text(
            json.dumps(
                {"seed": seed, "device": str(device), "variant": "central_pooled", "history": history},
                indent=2,
            )
        )
        print(f"Completed pooled centralized diagnostic seed {seed}", flush=True)


if __name__ == "__main__":
    main()
