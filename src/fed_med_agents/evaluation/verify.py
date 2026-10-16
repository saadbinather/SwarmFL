"""Independent checks on saved M2 evidence; no training or test-driven selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..fusion import CollaborativeModel


def verify(directory: Path) -> dict:
    result = json.loads((directory / "metrics.json").read_text())
    with np.load(directory / "cohort.npz") as file:
        partitions = {key: set(file[key].tolist()) for key in ("train", "val", "test")}
    assert not partitions["train"] & (partitions["val"] | partitions["test"])
    assert not partitions["val"] & partitions["test"]
    initial = torch.load(directory / "shared_initial.pt", weights_only=True)
    final = torch.load(directory / "shared_final.pt", weights_only=True)
    best = torch.load(directory / "shared_best.pt", weights_only=True)
    for prefix in ("router.", "fusion."):
        if prefix == "router." and int(result["config"]["budget"]) == 0:
            continue
        assert any(not torch.equal(initial[key], final[key]) for key in initial if key.startswith(prefix))
    model = CollaborativeModel(3, int(result["config"]["budget"]))
    model.load_state_dict(best, strict=True)
    expected = list(range(1, int(result["config"]["rounds"]) + 1))
    assert result["verification"]["specialists_unchanged"]
    for client in range(3):
        assert result["verification"]["participation"][str(client)] == expected
        path = directory / f"client_{client}"
        with np.load(path / "local.npz") as file:
            assert "y" not in file and len(file["train_y"]) == len(partitions["train"])
            assert file["x"].shape[1] == [3, 5, 8][client]
        metadata = json.loads((path / "specialist.json").read_text())
        held_union = set()
        for fold in metadata["folds"]:
            held, fit, calibration = map(
                set, (fold["held_ids"], fold["model_fit_ids"], fold["calibration_ids"])
            )
            assert (held | fit | calibration) <= partitions["train"]
            assert not held & (fit | calibration) and not fit & calibration
            assert set(fold["preprocessing"]["fit_ids"]) == fit
            assert not held_union & held
            held_union |= held
        assert held_union == partitions["train"]
        for round_id in expected:
            audit = json.loads((path / f"round_{round_id}.json").read_text())
            assert audit["fusion-change"] > 0
            if int(result["config"]["budget"]) > 0:
                assert audit["router-change"] > 0
    events = [json.loads(line) for line in (directory / "consultations.jsonl").read_text().splitlines()]
    validation_indices, test_indices = [], []
    for index, event in enumerate(events):
        patient = int(event["patient_id"].split("-")[1])
        assert not {"label", "features", "specialist_weights"} & event.keys()
        if event["phase"].startswith("training"):
            assert patient in partitions["train"] and event["mode"] == "oof"
        elif event["phase"].startswith("validation"):
            assert patient in partitions["val"]
            validation_indices.append(index)
        else:
            assert event["phase"].startswith("test") and patient in partitions["test"]
            test_indices.append(index)
    assert min(test_indices) > max(validation_indices)
    assert all(event["status"] == "ok" for event in events)
    return {
        "seed": result["seed"],
        "flower_run_id": result["flower_run_id"],
        "client_updates": 3 * len(expected),
        "consultation_events": len(events),
        "oof_provenance_verified": True,
        "test_after_selection": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([verify(path) for path in args.directories], indent=2))
