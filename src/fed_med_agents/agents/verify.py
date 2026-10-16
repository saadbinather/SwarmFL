"""Independent saved-evidence verification without invoking or trusting SLM risk scores."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..feedback.versions import PolicyRegistry
from ..fusion import CollaborativeModel


def verify(path: Path, checkpoint: Path) -> dict:
    metadata = json.loads((path / "metrics.json").read_text())
    manifest = json.loads((path / "requests.json").read_text())
    traces = [json.loads(line) for line in (path / "traces.jsonl").read_text().splitlines()]
    assert len(traces) == len(manifest["requests"])
    assert metadata["source_split"] == manifest["source_split"]
    assert metadata["source_split"] in {"validation", "external_request"}
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == metadata["fusion_checkpoint_sha256"]
    assert metadata["fusion_unchanged"]
    validation_ids = set()
    if metadata["source_split"] == "validation":
        with np.load(checkpoint.parent / "cohort.npz") as cohort:
            validation_ids = set(cohort["val"].tolist())
    capabilities = {cap["specialist"]: cap for cap in metadata["capabilities"]}
    model = CollaborativeModel(len(capabilities), 1)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    verified, consultations, failures = 0, 0, 0
    budgets = {}
    for item, trace in zip(manifest["requests"], traces, strict=True):
        if metadata.get("policy_registry"):
            model = PolicyRegistry(Path(metadata["policy_registry"])).load(trace["approved_policy_version"])
        request, final = item["request"], trace["final"]
        assert request["patient_id"] == final["patient_id"] and request["request_id"] == final["request_id"]
        if metadata["source_split"] == "validation":
            assert request["patient_id"].startswith("simulated-val-")
            assert int(request["patient_id"].rsplit("-", 1)[1]) in validation_ids
        else:
            assert request["mode"] == "new_patient" and not manifest["labels"]
        assert not request["features"] if request["mode"] == "aligned" else request["alignment_id"] is None
        assert request["origin"] not in final["consulted"]
        assert len(final["consulted"]) <= request["budget"] and len(set(final["consulted"])) == len(
            final["consulted"]
        )
        communication = trace["communication_trace"]
        assert any(event.get("action") == "return_to_origin" and event["received"] for event in communication)
        values = torch.full((1, len(capabilities)), 0.5)
        evidence = torch.zeros_like(values, dtype=torch.bool)
        for event in communication:
            if not isinstance(event.get("action"), str) or event["action"] not in {
                "origin_intake",
                "specialist_prediction",
            }:
                continue
            response = event["output"]
            assert response["patient_id"] == request["patient_id"]
            assert response["request_id"] == request["request_id"]
            assert response["specialist"] == event["destination"]
            if event["action"] == "specialist_prediction" and request["mode"] == "new_patient":
                names = set(response["received_feature_names"])
                assert names == set(capabilities[response["specialist"]]["required_features"])
                assert names <= set(request["authorized_features"][response["specialist"]])
            if response["status"] == "ok":
                index = capabilities[response["specialist"]]["index"]
                values[0, index] = response["probability"]
                evidence[0, index] = True
        if final["probability"] is not None:
            with torch.no_grad():
                origin = torch.tensor([capabilities[request["origin"]]["index"]])
                score = torch.sigmoid(model.fusion(values, 2 * (values - 0.5).abs(), evidence, origin)).item()
            assert abs(score - final["probability"]) < 2e-6
            verified += 1
        else:
            assert final["status"] in {"agent_error", "no_prediction"}
            failures += 1
        consultations += len(final["consulted"])
        key = (request["patient_id"], request["mode"])
        if item["policy"] == "slm":
            budgets[key] = len(final["consulted"])
            assert (
                any(
                    event.get("role") == "server_coordinator"
                    and event.get("provider", {}).get("model", "").startswith("qwen2.5")
                    for event in trace["coordinator_trace"]
                )
                or final["status"] != "ok"
            )
        elif metadata.get("match_realized_budget"):
            assert final["consultation_budget"] == budgets[key]
    assert any(
        model_info.get("size_vram", 0) > 0
        and metadata["provider"]["requested_model"] in {model_info.get("name"), model_info.get("model")}
        for model_info in metadata["provider"]["running_models"]["models"]
    )
    return {
        "flower_run_id": metadata["flower_run_id"],
        "requests": len(traces),
        "fusion_predictions_recomputed": verified,
        "no_prediction_responses": failures,
        "peer_consultations": consultations,
        "authorized_subsets_verified": True,
        "return_hops_verified": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/milestone02/seed_42/shared_best.pt")
    )
    args = parser.parse_args()
    print(json.dumps([verify(path, args.checkpoint) for path in args.directories], indent=2))
