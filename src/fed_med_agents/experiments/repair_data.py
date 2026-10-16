"""Training-partition-only nested specialist preparation for M5 Step 2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..data import GROUPS
from .diabetes import fit_specialist, generate_oof, logits, preprocess, temperature


def training_source(source: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Recover original fields only at original training IDs; read no held-out labels."""
    features, labels, patient_ids = [], None, None
    for client in range(3):
        folder = source / f"client_{client}"
        metadata = json.loads((folder / "specialist.json").read_text())
        with np.load(folder / "local.npz") as data:
            if patient_ids is None:
                patient_ids, labels = data["train"].copy(), data["train_y"].copy()
            np.testing.assert_array_equal(patient_ids, data["train"])
            np.testing.assert_array_equal(labels, data["train_y"])
            raw = data["x"][patient_ids].copy()
        if "age" in metadata["features"]:
            scaling = metadata["preprocessing"]
            raw[:, 0] = np.rint(raw[:, 0] * scaling["scale"] + scaling["mean"])
        features.append(raw.astype(np.float32))
    assert len(labels) == 364
    return features, labels, patient_ids


def prepare_context(
    path: Path,
    local: list[np.ndarray],
    y: np.ndarray,
    original_ids: np.ndarray,
    fit_ids: np.ndarray,
    assessment_ids: np.ndarray,
    seed: int,
) -> None:
    """Fit/calibrate only fit_ids, with nested OOF evidence for shared training."""
    if path.exists():
        raise ValueError("Context already exists; preserve previous preparation")
    folder = path / "seed_42"
    folder.mkdir(parents=True)
    assert not set(fit_ids) & set(assessment_ids)
    np.savez(
        folder / "cohort.npz",
        train=fit_ids,
        val=assessment_ids,
        test=np.array([], dtype=int),
        y=y,
        original_ids=original_ids,
    )
    reports = {}
    for client, (name, feature_names) in enumerate(GROUPS.items()):
        target = folder / f"client_{client}"
        target.mkdir()
        oof, raw_oof, folds = generate_oof(
            local[client], y, fit_ids, seed, client, torch.device("cpu"), client == 0, folds=3
        )
        x, scaling = preprocess(local[client], fit_ids, client == 0)
        model = fit_specialist(x[fit_ids], y[fit_ids], seed + client, torch.device("cpu"))
        fitted_temperature = temperature(raw_oof, y[fit_ids], torch.device("cpu"))
        assessment = torch.sigmoid(
            torch.tensor(logits(model, x[assessment_ids], torch.device("cpu")) / fitted_temperature)
        ).numpy()
        torch.save(model.state_dict(), target / "specialist.pt")
        np.savez(
            target / "local.npz",
            x=x,
            ids=np.arange(len(y)),
            train=fit_ids,
            train_y=y[fit_ids],
            val=assessment_ids,
            test=np.array([], dtype=int),
            oof=oof,
            assessment=assessment,
        )
        metadata = {
            "client": name,
            "features": feature_names,
            "preprocessing": scaling,
            "temperature": fitted_temperature,
            "folds": folds,
            "specialist_sha256": hashlib.sha256((target / "specialist.pt").read_bytes()).hexdigest(),
        }
        (target / "specialist.json").write_text(json.dumps(metadata, indent=2))
        reports[str(client)] = {
            "fit": len(fit_ids),
            "assessment": len(assessment_ids),
            "temperature": fitted_temperature,
        }
    (path / "preparation.json").write_text(json.dumps(reports, indent=2))


def prepare_nested(source: Path, output: Path) -> dict:
    """Three outer assessments and independently prepared inner selection contexts."""
    torch.set_num_threads(1)
    local, y, original_ids = training_source(source)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261014)
    contexts = {}
    for fold, (fitting, assessment) in enumerate(splitter.split(original_ids, y)):
        inner_fit, selection = train_test_split(
            fitting, test_size=0.2, stratify=y[fitting], random_state=9100 + fold
        )
        for stage, fit_ids, held_ids in (("inner", inner_fit, selection), ("outer", fitting, assessment)):
            key = f"fold{fold}_{stage}"
            prepare_context(output / "contexts" / key, local, y, original_ids, fit_ids, held_ids, 4200 + fold)
            contexts[key] = {
                "outer_fold": fold,
                "stage": stage,
                "seed": 4200 + fold,
                "fit_ids": fit_ids.tolist(),
                "assessment_ids": held_ids.tolist(),
            }
            print(f"Prepared {key}: {len(fit_ids)} fit / {len(held_ids)} assessment", flush=True)
    (output / "contexts.json").write_text(json.dumps(contexts, indent=2))
    return contexts
