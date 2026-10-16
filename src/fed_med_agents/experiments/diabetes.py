"""Client-local OOF specialists with nested train-only temperature calibration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from ucimlrepo import fetch_ucirepo

from ..data import FEATURES, GROUPS, load, normalize
from ..devices import peak_memory, select_device, synchronize
from ..models import Classifier


def fit_specialist(
    x: np.ndarray, y: np.ndarray, seed: int, device: torch.device, epochs: int = 30
) -> Classifier:
    torch.manual_seed(seed)
    model = Classifier(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    xt, yt = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for ids in torch.randperm(len(y), generator=generator).split(32):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xt[ids.to(device)]), yt[ids.to(device)]
            )
            loss.backward()
            optimizer.step()
    return model


def logits(model: Classifier, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, device=device)).float().cpu().numpy()


def temperature(values: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    """Fit only on a declared calibration set; bounded temperature prevents overflow."""
    parameter = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    v, y = torch.as_tensor(values, device=device), torch.as_tensor(labels, device=device)
    for _ in range(80):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(v / parameter.exp(), y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(-2.3, 2.3)
    return float(parameter.exp().detach().cpu())


def preprocess(x: np.ndarray, fit_ids: np.ndarray, has_age: bool) -> tuple[np.ndarray, dict]:
    output = x.copy()
    metadata = {"mean": 0.0, "scale": 1.0, "fit_ids": fit_ids.tolist()}
    if has_age:
        mean, scale = float(x[fit_ids, 0].mean()), float(x[fit_ids, 0].std())
        metadata.update(mean=mean, scale=max(scale, 1e-8))
        output[:, 0] = (x[:, 0] - mean) / metadata["scale"]
    return output, metadata


def generate_oof(
    local: np.ndarray,
    y: np.ndarray,
    train_ids: np.ndarray,
    seed: int,
    client_index: int,
    device: torch.device,
    has_age: bool,
    folds: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """OOF probabilities use neither held labels nor held-fitted preprocessing/calibration."""
    oof = np.zeros(len(train_ids), dtype=np.float32)
    raw_oof = np.zeros_like(oof)
    provenance = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_pos, held_pos) in enumerate(splitter.split(train_ids, y[train_ids])):
        outer_fit, held = train_ids[fit_pos], train_ids[held_pos]
        inner_fit, calibration = train_test_split(
            outer_fit, test_size=0.2, stratify=y[outer_fit], random_state=seed + fold
        )
        transformed, scaling = preprocess(local, inner_fit, has_age)
        model = fit_specialist(transformed[inner_fit], y[inner_fit], seed + client_index * 100 + fold, device)
        fitted_temperature = temperature(
            logits(model, transformed[calibration], device), y[calibration], device
        )
        held_logits = logits(model, transformed[held], device)
        raw_oof[held_pos] = held_logits
        oof[held_pos] = torch.sigmoid(torch.tensor(held_logits / fitted_temperature)).numpy()
        provenance.append(
            {
                "fold": fold,
                "model_fit_ids": inner_fit.tolist(),
                "calibration_ids": calibration.tolist(),
                "held_ids": held.tolist(),
                "preprocessing": scaling,
                "temperature": fitted_temperature,
            }
        )
    return oof, raw_oof, provenance


def prepare(seed: int, directory: Path, device_request: str = "auto", folds: int = 4) -> dict:
    """Fetch raw data once, split by immutable M1 IDs, write separate local feature files."""
    directory.mkdir(parents=True, exist_ok=True)
    device = select_device(device_request)
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    dataset = fetch_ucirepo(id=529)
    raw = normalize(dataset.data.features)
    y = dataset.data.targets.iloc[:, 0].map({"Positive": 1, "Negative": 0}).to_numpy(dtype=np.float32)
    baseline = load()
    np.testing.assert_array_equal(y, baseline["y"])
    train_ids = baseline["train"]
    np.savez(directory / "cohort.npz", train=train_ids, val=baseline["val"], test=baseline["test"], y=y)
    reports = {}
    for client_index, (client, features) in enumerate(GROUPS.items()):
        local = raw[:, [FEATURES.index(f) for f in features]]
        client_dir = directory / f"client_{client_index}"
        client_dir.mkdir(exist_ok=True)
        oof, raw_oof, provenance = generate_oof(
            local, y, train_ids, seed, client_index, device, "age" in features, folds
        )
        transformed, scaling = preprocess(local, train_ids, "age" in features)
        final = fit_specialist(transformed[train_ids], y[train_ids], seed + client_index, device)
        final_temperature = temperature(raw_oof, y[train_ids], device)
        torch.save({k: v.cpu() for k, v in final.state_dict().items()}, client_dir / "specialist.pt")
        np.savez(
            client_dir / "local.npz",
            x=transformed,
            ids=np.arange(len(y)),
            train=train_ids,
            train_y=y[train_ids],
            val=baseline["val"],
            test=baseline["test"],
            oof=oof,
        )
        metadata = {
            "client": client,
            "client_index": client_index,
            "features": features,
            "preprocessing": scaling,
            "temperature": final_temperature,
            "folds": provenance,
        }
        (client_dir / "specialist.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        reports[client] = {"temperature": final_temperature, "features": features, "oof_cases": len(oof)}
        print(
            f"OOF prepared {client}: {len(oof)} cases, device={device}, temperature={final_temperature:.4f}",
            flush=True,
        )
    synchronize(device)
    result = {
        "seed": seed,
        "folds": folds,
        "device": str(device),
        "runtime_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": peak_memory(device),
        "clients": reports,
    }
    (directory / "preparation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
