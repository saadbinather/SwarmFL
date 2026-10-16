"""UCI891 benchmark configuration and group-safe private specialist preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from .m5_step03 import digest, write

DOMAINS = {
    "demographics": ("Age", "Sex", "Education", "Income"),
    "metabolic": ("BMI", "HighBP", "HighChol", "CholCheck"),
    "cardiovascular": ("Stroke", "HeartDiseaseorAttack", "HighBP", "Age"),
    "lifestyle": ("Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"),
    "general_health": ("GenHlth", "MentHlth", "PhysHlth", "DiffWalk"),
    "access": ("AnyHealthcare", "NoDocbcCost", "Education", "Income", "Age"),
}

# User-selected amendment; order and membership are not selected from outcomes.
THREE_SLOTS = (1, 3, 4)
THREE_DOMAINS = {name: DOMAINS[name] for name in ("metabolic", "lifestyle", "general_health")}


def group_ids(x: np.ndarray) -> np.ndarray:
    """Label-independent identical-feature grouping, stable little-endian float32 bytes."""
    return np.array(
        [int.from_bytes(hashlib.sha256(row.astype("<f4").tobytes()).digest()[:8], "little") for row in x],
        dtype=np.uint64,
    )


def acquire(root: Path) -> dict:
    """Fetch official data once and freeze separate test/development files before fitting."""
    from ucimlrepo import fetch_ucirepo

    cached = Path("artifacts/m5-step035-fetch/uci891.pkl")
    if cached.exists():
        import pickle

        with cached.open("rb") as handle:
            dataset = pickle.load(handle)  # Locally fetched trusted ucimlrepo object.
    else:
        dataset = fetch_ucirepo(id=891)
    frame, target = dataset.data.features, dataset.data.targets
    assert frame.shape == (253680, 21)
    assert target.columns.tolist() == ["Diabetes_binary"]
    x, y = frame.to_numpy(dtype=np.float32), target.iloc[:, 0].to_numpy(dtype=np.float32)
    assert set(np.unique(y)) == {0.0, 1.0} and np.isfinite(x).all()
    assert set(frame.columns) == {column for domain in DOMAINS.values() for column in domain}
    groups = group_ids(x)
    test = groups % np.uint64(5) == 0
    ids = np.arange(len(y))
    np.savez_compressed(root / "frozen_test.npz", x=x[test], y=y[test], ids=ids[test], groups=groups[test])
    np.savez_compressed(
        root / "development.npz", x=x[~test], y=y[~test], ids=ids[~test], groups=groups[~test]
    )
    np.save(root / "frozen_test_ids.npy", ids[test])
    dev_y = y[~test]
    pilot, _ = train_test_split(np.arange(len(dev_y)), train_size=15000, stratify=dev_y, random_state=53500)
    np.save(root / "pilot_ids.npy", pilot)
    summary = {
        "rows": len(y),
        "features": frame.columns.tolist(),
        "target": "Diabetes_binary",
        "semantics": "0=neither;1=prediabetes or diabetes",
        "target_counts": np.bincount(y.astype(int)).tolist(),
        "development_rows": int((~test).sum()),
        "test_rows": int(test.sum()),
        "unique_feature_groups": len(np.unique(groups)),
        "pilot_rows": len(pilot),
        "frozen_test_sha256": digest(root / "frozen_test.npz"),
        "source": "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
        "metadata": dataset.metadata,
        "test_evaluations": 0,
    }
    # ucimlrepo metadata is a JSON-compatible dotdict.
    write(root / "dataset.json", summary)
    return summary


def grouped_split(
    ids: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Independent group-held calibration/selection boundary, no assessment group fitting."""
    a, b = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(ids, y[ids], groups[ids])
    )
    return ids[a], ids[b]


def fit_expert(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, train: np.ndarray, columns: list[int], seed: int
) -> tuple[dict, dict]:
    """Independent histogram classifier plus held-group sigmoid calibration, no class weighting."""
    fit, calibration = grouped_split(train, y, groups, seed)
    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=0.1,
        learning_rate=0.1,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(x[fit][:, columns], y[fit])
    raw = model.predict_proba(x[calibration][:, columns])[:, 1].clip(1e-6, 1 - 1e-6)
    sigmoid = LogisticRegression(C=1, max_iter=200).fit(np.log(raw / (1 - raw))[:, None], y[calibration])
    return {"model": model, "calibrator": sigmoid, "columns": columns}, {
        "model_fit": fit.tolist(),
        "calibration": calibration.tolist(),
        "seed": seed,
    }


def expert_predict(model: dict, x: np.ndarray) -> np.ndarray:
    """Private inference uses only this specialist's legitimate configured columns."""
    raw = model["model"].predict_proba(x[:, model["columns"]])[:, 1].clip(1e-6, 1 - 1e-6)
    return model["calibrator"].predict_proba(np.log(raw / (1 - raw))[:, None])[:, 1].astype(np.float32)


def prepare_stack(
    root: Path,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    columns: list[str],
    train: np.ndarray,
    held: np.ndarray,
    seed: int,
    domain_config: dict[str, tuple[str, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Specialist OOF/final stack; save every upstream model and exact group provenance."""
    root.mkdir(parents=True)
    assert not set(groups[train]) & set(groups[held])
    domains = DOMAINS if domain_config is None else domain_config
    p = np.zeros((len(train), len(domains)), dtype=np.float32)
    hp = np.zeros((len(held), len(domains)), dtype=np.float32)
    provenance = {"train": train.tolist(), "held": held.tolist(), "clients": {}}
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    folds = list(splitter.split(train, y[train], groups[train]))
    for client, (name, domain) in enumerate(domains.items()):
        local = root / f"client_{client}"
        local.mkdir()
        indexes = [columns.index(c) for c in domain]
        records = []
        for fold, (a, b) in enumerate(folds):
            model, record = fit_expert(x, y, groups, train[a], indexes, seed + fold + client * 100)
            p[b, client] = expert_predict(model, x[train[b]])
            joblib.dump(model, local / f"oof_{fold}.joblib")
            record.update(held=train[b].tolist(), held_positions=b.tolist())
            records.append(record)
        model, final = fit_expert(x, y, groups, train, indexes, seed + client * 100 + 10)
        hp[:, client] = expert_predict(model, x[held])
        joblib.dump(model, local / "final.joblib")
        provenance["clients"][str(client)] = {
            "name": name,
            "columns": list(domain),
            "folds": records,
            "final": final,
            "sha256": digest(local / "final.joblib"),
        }
    np.savez(root / "probabilities.npz", train=p, held=hp)
    write(root / "provenance.json", provenance)
    return p, hp
