"""Independent nested-data, Flower arithmetic and policy replay checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from fed_med_agents.devices import select_device
from fed_med_agents.experiments.diabetes import generate_oof
from fed_med_agents.experiments.repair_data import training_source
from fed_med_agents.experiments.repair_models import CANDIDATES, evaluate, make_model
from fed_med_agents.feedback.verify import verify as verify_feedback
from fed_med_agents.models import Classifier


def verify(root: Path, feedback: bool = True, refit_oof: bool = False) -> dict:
    """Read source evidence; write nothing to previous milestones."""
    torch.set_num_threads(1)
    config = json.loads((root / "config.json").read_text())
    result = json.loads((root / "results.json").read_text())
    with np.load("artifacts/milestone02/seed_42/client_0/local.npz") as old:
        historical_training = set(old["train"].tolist())
    evidence = json.loads((root / "flower_evidence.json").read_text())
    counts = {"aggregations": 0, "client_updates": 0, "contexts": 0, "assessment_replays": 0}
    outer_held = []
    raw, all_y, _ = (
        training_source(Path("artifacts/milestone02/seed_42")) if refit_oof else (None, None, None)
    )
    oof_reproduced = 0
    for key, ctx in config["contexts"].items():
        folder = root / "contexts" / key / "seed_42"
        with np.load(folder / "cohort.npz") as cohort:
            fitting, held = cohort["train"].copy(), cohort["val"].copy()
            originals = cohort["original_ids"].copy()
            fit_y, held_y = cohort["y"][fitting], cohort["y"][held]
        assert set(originals.tolist()) == historical_training
        assert set(fitting).isdisjoint(held)
        assert fitting.tolist() == ctx["fit_ids"] and held.tolist() == ctx["assessment_ids"]
        if ctx["stage"] == "outer":
            outer_held.extend(held.tolist())
        else:
            outer = config["contexts"][key.replace("inner", "outer")]
            assert set(fitting) | set(held) == set(outer["fit_ids"])
            assert (set(fitting) | set(held)).isdisjoint(outer["assessment_ids"])
        held_p = []
        for origin in range(3):
            local = folder / f"client_{origin}"
            meta = json.loads((local / "specialist.json").read_text())
            assert (
                hashlib.sha256((local / "specialist.pt").read_bytes()).hexdigest()
                == meta["specialist_sha256"]
            )
            assert set(meta["preprocessing"]["fit_ids"]) == set(fitting)
            oof_rows = []
            for split in meta["folds"]:
                fit_ids, cal_ids, target_ids = (
                    set(split[n]) for n in ("model_fit_ids", "calibration_ids", "held_ids")
                )
                assert (
                    fit_ids.isdisjoint(cal_ids)
                    and fit_ids.isdisjoint(target_ids)
                    and cal_ids.isdisjoint(target_ids)
                )
                assert fit_ids | cal_ids | target_ids == set(fitting)
                assert set(split["preprocessing"]["fit_ids"]) == fit_ids
                oof_rows.extend(target_ids)
            assert sorted(oof_rows) == sorted(fitting.tolist())
            with np.load(local / "local.npz") as data:
                np.testing.assert_array_equal(data["train_y"], fit_y)
                held_p.append(data["assessment"].copy())
                delivered = np.load(local / "shared_training.npy")
                np.testing.assert_allclose(delivered[:, origin], data["oof"], atol=1e-7)
                own = [r for r in evidence if r["context"] == key and r["client"] == origin]
                assert len(own) == 1
                np.testing.assert_allclose(own[0]["assessment"], data["assessment"])
                np.testing.assert_allclose(own[0]["oof"], data["oof"])
                assert own[0]["fit_ids"] == fitting.tolist() and own[0]["assessment_ids"] == held.tolist()
                classifier = Classifier(data["x"].shape[1])
                classifier.load_state_dict(
                    torch.load(local / "specialist.pt", weights_only=True, map_location="cpu")
                )
                with torch.no_grad():
                    actual = torch.sigmoid(
                        classifier(torch.tensor(data["x"][held])) / meta["temperature"]
                    ).numpy()
                np.testing.assert_allclose(actual, data["assessment"], atol=1e-7, rtol=1e-6)
                if refit_oof:
                    reproduction, _, _ = generate_oof(
                        raw[origin],
                        all_y,
                        fitting,
                        ctx["seed"],
                        origin,
                        torch.device("cpu"),
                        origin == 0,
                        folds=3,
                    )
                    np.testing.assert_allclose(reproduction, data["oof"], atol=1e-7, rtol=1e-6)
                    oof_reproduced += 1
        held_p = np.asarray(held_p, dtype=np.float32).T
        for name, record in result["contexts"][key].items():
            if name in {"controls", "utility_bins"}:
                continue
            target = root / "runs" / key / name
            if "history" in record:
                mode = record["settings"].get("mode", "joint")
                epochs = record["settings"]["epochs"]
                for round_key in record["history"]:
                    round_id = int(round_key)
                    if not round_id:
                        continue
                    state = torch.load(
                        target / f"server_round_{round_id}.pt", weights_only=True, map_location="cpu"
                    )
                    previous = torch.load(
                        target / f"server_round_{round_id - 1}.pt", weights_only=True, map_location="cpu"
                    )
                    clients = []
                    for origin in range(3):
                        audit = json.loads((target / f"client_{origin}_round_{round_id}.json").read_text())
                        update = torch.load(
                            target / f"client_{origin}_round_{round_id}.pt",
                            weights_only=True,
                            map_location="cpu",
                        )
                        assert audit["client"] == origin and audit["num_examples"] == len(fitting)
                        assert audit["steps"] == math.ceil(len(fitting) / 64) * epochs
                        assert audit["payload_record_keys"] == ["arrays", "metrics"]
                        expected_start = (round_id - 1) * audit["steps"] if name == "persistent2" else 0
                        assert audit["optimizer_step_start"] == expected_start
                        assert audit["optimizer_step_end"] == expected_start + audit["steps"]
                        assert audit["mu"] == (0.01 if name == "prox2" else 0.0)
                        frozen = "router." if mode == "uniform" else "fusion." if mode == "value" else None
                        if frozen:
                            assert all(
                                torch.equal(v, previous[k]) for k, v in update.items() if k.startswith(frozen)
                            )
                        assert any(not torch.equal(v, previous[k]) for k, v in update.items())
                        clients.append(update)
                        counts["client_updates"] += 1
                    for k, v in state.items():
                        torch.testing.assert_close(v, sum(c[k] for c in clients) / 3, atol=2e-6, rtol=1e-5)
                    counts["aggregations"] += 1
            if "assessment" not in record:
                continue
            kind = record["settings"].get("kind", "original")
            model = make_model(kind).to(select_device(config["device"]))
            model.load_state_dict(
                torch.load(target / "server_round_5.pt", weights_only=True, map_location="cpu")
            )
            actual = evaluate(
                model, held_p, held_y, "value" if record["settings"].get("mode") == "value" else "joint"
            )
            for origin in range(3):
                old = record["assessment"][str(origin)]
                assert actual[str(origin)]["actions"] == old["actions"]
                for metric in (
                    "accuracy",
                    "auroc",
                    "f1",
                    "binary_cross_entropy",
                    "primary",
                    "ece",
                    "brier",
                    "calls",
                ):
                    np.testing.assert_allclose(actual[str(origin)][metric], old[metric], atol=2e-5, rtol=1e-4)
                assert old["action_frequencies"][origin + 1] == 0 and old["calls"] <= 1
                counts["assessment_replays"] += 1
        inner_names = list(CANDIDATES) + ["utility_bins"]
        if ctx["stage"] == "inner":
            objectives = {
                name: float(
                    np.mean([r["primary"] for r in result["contexts"][key][name]["assessment"].values()])
                )
                for name in inner_names
            }
            assert result["selection"][str(ctx["outer_fold"])]["candidate"] == min(
                objectives, key=objectives.get
            )
        counts["contexts"] += 1
    assert sorted(outer_held) == list(range(364))
    feedback_result = {}
    if feedback:
        for steps in (2, 8):
            feedback_result[str(steps)] = verify_feedback(root / f"feedback_steps{steps}")
        a = json.loads((root / "feedback_steps2/decisions.json").read_text())
        b = json.loads((root / "feedback_steps8/decisions.json").read_text())
        assert [(e["patient_id"], e["origin"], e["episode"]) for e in a] == [
            (e["patient_id"], e["origin"], e["episode"]) for e in b
        ]
        for x, y in zip(a, b):
            if x["policy_version"] == "frozen":
                assert x["action"] == y["action"] and x["final_risk"] == y["final_risk"]
    manifest = json.loads((root / "preserved-before.json").read_text())
    changed = [p for p, h in manifest.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert not changed, changed
    return {
        "verified": True,
        **counts,
        "preserved_files": len(manifest),
        "changed": changed,
        "feedback": feedback_result,
        "flower_run_id": result["flower_run_id"],
        "oof_columns_independently_refitted": oof_reproduced,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--without-feedback", action="store_true")
    parser.add_argument(
        "--refit-oof",
        action="store_true",
        help="Refit all nested OOF experts as an additional reproducibility check",
    )
    args = parser.parse_args()
    evidence = verify(args.path, not args.without_feedback, args.refit_oof)
    (args.path / "verification.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
