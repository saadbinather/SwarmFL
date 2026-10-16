"""Independent saved-run audit of provenance, partial evidence and FedAvg."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .events import FeedbackLedger, PredictionEvent
from .server_app import action_distribution
from .versions import PolicyRegistry


def verify(path: Path) -> dict:
    """Raise on leakage/association/aggregation violations; read no historical test labels."""
    result = json.loads((path / "metrics.json").read_text())
    config = result["config"]
    source = Path(config["source"])
    credentials = json.loads((path / "credentials.json").read_text())
    registry = PolicyRegistry(path / "versions")
    with np.load(source / "cohort.npz") as cohort:
        train = {f"case-{i:04d}" for i in cohort["train"]}
        held = {f"case-{i:04d}" for i in np.concatenate([cohort["val"], cohort["test"]])}
    stores = {}
    for origin in range(3):
        private = source / f"client_{origin}"
        assert (
            hashlib.sha256((private / "specialist.pt").read_bytes()).hexdigest()
            == result["private_hashes"][str(origin)]
        )
        for policy in ("adaptive", "frozen", "offline"):
            stores[origin, policy] = FeedbackLedger(
                path / f"client_{origin}" / policy, origin, train, **credentials[str(origin)]
            )
    events = [
        PredictionEvent.model_validate(row) for row in json.loads((path / "decisions.json").read_text())
    ]
    seen = set()
    for event in events:
        assert event.patient_id in train and event.patient_id not in held
        assert event.request_id not in seen
        seen.add(event.request_id)
        policy = event.request_id.split("-")[0]
        store = stores[event.origin, policy]
        assert event == store.predictions[event.request_id]
        assert event.request_id in store.outcomes
        with np.load(source / f"client_{event.origin}" / "local.npz") as local:
            patient = int(event.patient_id.removeprefix("case-"))
            index = np.flatnonzero(local["train"] == patient)[0]
            assert store.outcomes[event.request_id].label == int(local["train_y"][index])
            np.testing.assert_allclose(event.evidence[event.origin], local["oof"][index], atol=1e-7)
        model = registry.load(event.model_version)
        distribution, scores = action_distribution(
            model,
            event.origin,
            event.evidence[event.origin],
            event.available_domains,
            torch.device("cpu"),
            config["epsilon"],
        )
        assert model.router.actions[event.action].nonzero().flatten().tolist() == event.consulted
        assert torch.isfinite(scores[event.action])
        expected_propensity = float(distribution[event.action]) if policy != "offline" else 1.0
        np.testing.assert_allclose(event.propensity, expected_propensity, atol=2e-6)
        # Actual chosen evidence must appear in a real Flower consultation trace.
        values = torch.tensor([[0.5 if value is None else value for value in event.evidence]])
        mask = torch.tensor([[value is not None for value in event.evidence]])
        with torch.no_grad():
            prediction = torch.sigmoid(
                model.fusion(values, 2 * (values - 0.5).abs(), mask, torch.tensor([event.origin]))
            ).item()
        np.testing.assert_allclose(prediction, event.final_risk, atol=2e-6)
    trace = [json.loads(line) for line in (path / "consultations.jsonl").read_text().splitlines()]
    assert not any("test" in row["phase"] for row in trace)
    for event in events:
        policy = event.request_id.split("-")[0]
        actual = [
            row
            for row in trace
            if row["phase"] == f"m4/stream/{policy}"
            and row["patient_id"] == event.patient_id
            and row["origin"] == event.origin
        ]
        assert sorted(row["destination"] for row in actual) == event.consulted
        for row in actual:
            np.testing.assert_allclose(row["prediction"], event.evidence[row["destination"]], atol=1e-7)
    for update in result["history"]:
        if "skipped" in update:
            continue
        states, weights = [], []
        update_id = int(update["version"][1:])
        for origin in range(3):
            report = json.loads((path / f"client_{origin}" / f"update_{update_id}.json").read_text())
            rows = stores[origin, "adaptive"].replay(report["cutoff"], config["window"])
            assert [event.request_id for event, _ in rows] == report["request_ids"]
            assert len(rows) == report["num-examples"] >= config["min_feedback"]
            assert all(outcome.received_episode <= update["episode"] for _, outcome in rows)
            states.append(torch.load(path / f"client_{origin}" / f"update_{update_id}.pt", weights_only=True))
            weights.append(len(rows))
        candidate = registry.load(update["version"]).state_dict()
        for key in candidate:
            average = sum(state[key] * (weight / sum(weights)) for state, weight in zip(states, weights))
            torch.testing.assert_close(candidate[key], average, atol=2e-6, rtol=1e-5)
        assert update["shared_change_l2"] > 0
    releases = json.loads((path / "releases.json").read_text())
    assert all(row["released"] >= row["due"] > row["episode"] for row in releases)
    adaptive = [event for event in events if event.request_id.startswith("adaptive-")]
    assert len({event.patient_id for event in adaptive}) == len(adaptive) == config["episodes"]
    assert result["later_coordinator"]["version"] == registry.current() == result["approved_version"]
    assert result["specialists_unchanged"]
    return {
        "verified": True,
        "events": len(events),
        "unique_training_patients": len(adaptive),
        "updates": len([row for row in result["history"] if "version" in row]),
        "flower_run_id": result["flower_run_id"],
        "approved_version": registry.current(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    args = parser.parse_args()
    for directory in args.directories:
        print(json.dumps(verify(directory), indent=2))
