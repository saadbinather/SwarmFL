"""One patient-aligned cohort. All fitted statistics use training IDs only."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("FED_MED_ARTIFACTS", ROOT / "artifacts")).resolve()
FEATURES = [
    "age",
    "gender",
    "polyuria",
    "polydipsia",
    "sudden_weight_loss",
    "weakness",
    "polyphagia",
    "genital_thrush",
    "visual_blurring",
    "itching",
    "irritability",
    "delayed_healing",
    "partial_paresis",
    "muscle_stiffness",
    "alopecia",
    "obesity",
]
GROUPS = {
    "A": ["age", "gender", "obesity"],
    "B": ["polyuria", "polydipsia", "polyphagia", "sudden_weight_loss", "weakness"],
}
GROUPS["C"] = [f for f in FEATURES if f not in GROUPS["A"] + GROUPS["B"]]


def normalize(frame):
    frame = frame.copy()
    frame.columns = [c.strip().lower().replace(" ", "_") for c in frame.columns]
    if list(frame.columns) != FEATURES or frame.shape != (520, 16):
        raise ValueError(f"Unexpected UCI schema: {frame.shape}, {list(frame.columns)}")
    for col in FEATURES[1:]:
        mapping = {"Male": 1, "Female": 0} if col == "gender" else {"Yes": 1, "No": 0}
        frame[col] = frame[col].map(mapping)
    frame["age"] = pd.to_numeric(frame["age"])
    if frame.isna().any().any():
        raise ValueError("Unexpected missing value or category")
    return frame.to_numpy(dtype=np.float32)


def prepare_arrays(x, y):
    ids = np.arange(len(y))
    train, holdout = train_test_split(ids, test_size=0.30, stratify=y, random_state=42)
    val, test = train_test_split(holdout, test_size=0.5, stratify=y[holdout], random_state=42)
    scaler = StandardScaler().fit(x[train, :1])
    transformed = x.copy()
    transformed[:, :1] = scaler.transform(x[:, :1])
    arrays = {
        "x": transformed.astype(np.float32),
        "y": y.astype(np.float32),
        "train": train,
        "val": val,
        "test": test,
    }
    metadata = {
        "rows": len(y),
        "features": FEATURES,
        "groups": GROUPS,
        "seed": 42,
        "split_sizes": {k: len(arrays[k]) for k in ("train", "val", "test")},
        "age_mean": float(scaler.mean_[0]),
        "age_scale": float(scaler.scale_[0]),
        "preprocessing_fit_ids": train.tolist(),
        "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }
    return arrays, metadata


def prepare():
    ARTIFACTS.mkdir(exist_ok=True)
    dataset = fetch_ucirepo(id=529)
    x = normalize(dataset.data.features)
    target = dataset.data.targets.iloc[:, 0].map({"Positive": 1, "Negative": 0})
    if target.isna().any():
        raise ValueError("Unexpected target")
    arrays, metadata = prepare_arrays(x, target.to_numpy())
    np.savez(ARTIFACTS / "dataset.npz", **arrays)
    (ARTIFACTS / "dataset_summary.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata | {"preprocessing_fit_ids": "saved in summary"}, indent=2))


def load():
    with np.load(ARTIFACTS / "dataset.npz") as data:
        return {key: data[key] for key in data.files}


def view(data, client, split, shared=False):
    ids = data[split]
    indices = [FEATURES.index(f) for f in GROUPS[client]]
    local = data["x"][ids][:, indices].copy()
    if not shared:
        return local, data["y"][ids].copy()
    values = np.zeros((len(ids), 16), dtype=np.float32)
    mask = np.zeros_like(values)
    values[:, indices] = local
    mask[:, indices] = 1
    return np.concatenate([values, mask], axis=1), data["y"][ids].copy()


if __name__ == "__main__":
    prepare()
