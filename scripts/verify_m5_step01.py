"""Independent M5 cache, split, frozen-state, FedAvg and numerical replay checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fed_med_agents.devices import select_device
from fed_med_agents.experiments.routing_diagnostics import evaluate
from fed_med_agents.fusion import CollaborativeModel


def verify(path: Path) -> dict:
    """Fail on an altered cache, invalid mask/control, state or saved metric."""
    cfg = json.loads((path / "config.json").read_text())
    counts = {"states": 0, "aggregations": 0, "replayed_policy_metrics": 0, "trace_rows": 0}
    for seed in cfg["seeds"]:
        root = path / f"seed_{seed}"
        result = json.loads((root / "results.json").read_text())
        assert result["specialists_unchanged"]
        with np.load(root / "evidence.npz") as file:
            cache = {key: file[key] for key in file.files}
        source = Path(cfg["source"]) / f"seed_{seed}"
        with np.load(source / "cohort.npz") as cohort:
            np.testing.assert_array_equal(cache["train_ids"], cohort["train"])
            np.testing.assert_array_equal(cache["val_ids"], cohort["val"])
            np.testing.assert_array_equal(cache["train_y"], cohort["y"][cohort["train"]])
            np.testing.assert_array_equal(cache["val_y"], cohort["y"][cohort["val"]])
            assert not set(cache["train_ids"]) & set(cohort["test"])
            assert not set(cache["val_ids"]) & set(cohort["test"])
        trace = [json.loads(line) for line in (root / "consultations.jsonl").read_text().splitlines()]
        for split, ids in (
            ("train", cache["train_ids"]),
            ("validation", cache["val_ids"]),
            ("train_final", cache["train_ids"]),
        ):
            for client in range(3):
                rows = [
                    row
                    for row in trace
                    if row["phase"] == f"m5/diagnostic-cache/{split}" and row["destination"] == client
                ]
                assert [row["patient_id"] for row in rows] == [f"case-{i:04d}" for i in ids]
                assert all(row["status"] == "ok" for row in rows)
                np.testing.assert_allclose([row["prediction"] for row in rows], cache[split][:, client])
        assert len(trace) == 3 * (2 * len(cache["train_ids"]) + len(cache["val_ids"]))
        counts["trace_rows"] += len(trace)
        initial = torch.load(source / "shared_initial.pt", weights_only=True)
        variants = dict(result["variants"])
        pooled = path.parent / "m5-step01-central-pooled" / f"seed_{seed}"
        if (pooled / "results.json").exists():
            variants["central_pooled"] = json.loads((pooled / "results.json").read_text())
        for variant, record in variants.items():
            folder = pooled if variant == "central_pooled" else root / variant
            model = CollaborativeModel().to(select_device(record.get("device", cfg["device"])))
            for round_id, saved in record["history"].items():
                state = torch.load(folder / f"server_round_{round_id}.pt", weights_only=True)
                model.load_state_dict(state, strict=True)
                counts["states"] += 1
                if variant.startswith("frozen_"):
                    prefix = "router." if variant == "frozen_router" else "fusion."
                    for key, value in state.items():
                        if key.startswith(prefix):
                            # FedAvg floating-point weighting can round frozen arrays.
                            torch.testing.assert_close(value, initial[key], atol=1e-6, rtol=1e-6)
                if variant in {"joint", "frozen_router", "frozen_fusion"} and int(round_id) > 0:
                    updates = [
                        torch.load(folder / f"client_{i}_round_{round_id}.pt", weights_only=True)
                        for i in range(3)
                    ]
                    for key in state:
                        torch.testing.assert_close(
                            state[key], sum(u[key] for u in updates) / 3, atol=2e-6, rtol=1e-5
                        )
                    counts["aggregations"] += 1
                actual = evaluate(model, cache["validation"], cache["val_y"], seed)
                for origin, value in actual.items():
                    for policy, row in value["policies"].items():
                        previous = saved["validation"][origin]["policies"][policy]
                        assert row["action_frequencies"] == previous["action_frequencies"]
                        for fusion in ("learned_fusion", "mean_fusion"):
                            # Replay on the recorded device: near-tied ranking metrics
                            # can change on CPU despite close probability values.
                            for metric, score in row[fusion].items():
                                np.testing.assert_allclose(
                                    score, previous[fusion][metric], atol=2e-5, rtol=1e-4
                                )
                                counts["replayed_policy_metrics"] += 1
                        if policy.endswith("matched"):
                            assert row["calls"] == value["policies"]["learned"]["calls"]
                        assert row["action_frequencies"][int(origin) + 1] == 0
    return {"verified": True, **counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    output = verify(args.path)
    (args.path / "verification.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
