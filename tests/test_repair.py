"""Guards for isolated Step 2 optimization, masking and pre-query decisions."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
from flwr.serverapp.strategy import FedProx
from flwr.supercore.task_identity import TaskIdentity

from fed_med_agents.experiments.m5_step02 import local_fit, strategy, train
from fed_med_agents.experiments.repair_models import (
    CANDIDATES,
    evaluate,
    make_model,
    objective,
    query_scores,
    select_inner,
    value_targets,
)
from fed_med_agents.experiments.routing_diagnostics import snapshot


@pytest.fixture(autouse=True)
def message_task_identity(monkeypatch):
    """Supply the runtime identity normally set by Flower, only in unit tests."""
    monkeypatch.setattr(TaskIdentity, "_run_id", 1)
    monkeypatch.setattr(TaskIdentity, "_task_id", 1)
    monkeypatch.setattr(TaskIdentity, "_node_id", 0)


def sample():
    torch.set_num_threads(1)
    torch.manual_seed(123)
    return (
        snapshot(make_model()),
        np.random.default_rng(1).uniform(0.1, 0.9, (16, 3)).astype(np.float32),
        np.tile([0.0, 1.0], 8).astype(np.float32),
    )


@pytest.mark.parametrize("epochs", [1, 2, 5, 10])
def test_local_step_count_and_compatible_reload(epochs):
    initial, p, y = sample()
    final, _, report = local_fit(
        initial, p, y, 0, {"epochs": epochs, "seed": 42, "round": 1}, torch.device("cpu")
    )
    assert report["steps"] == epochs
    assert all(value > 0 for value in report["drift"].values())
    make_model().load_state_dict(ArrayRecord(final).to_torch_state_dict())


def test_persistent_adam_really_carries_moments_across_broadcasts():
    initial, p, y = sample()
    settings = {"epochs": 1, "seed": 42, "round": 1}
    first, moments, audit = local_fit(initial, p, y, 0, settings, torch.device("cpu"))
    persisted, _, carried = local_fit(
        first, p, y, 0, settings | {"round": 2}, torch.device("cpu"), copy.deepcopy(moments)
    )
    reset, _, fresh = local_fit(first, p, y, 0, settings | {"round": 2}, torch.device("cpu"))
    assert audit["optimizer_step_end"] == carried["optimizer_step_start"] == 1
    assert carried["optimizer_step_end"] == 2 and fresh["optimizer_step_end"] == 1
    assert any(not torch.equal(persisted[k], reset[k]) for k in reset)


def test_official_fedprox_injection_and_standard_proximal_loss():
    initial, p, y = sample()
    settings = {"epochs": 1, "seed": 42, "round": 1}
    plain, _, _ = local_fit(initial, p, y, 0, settings, torch.device("cpu"))
    proximal, _, _ = local_fit(initial, p, y, 0, settings | {"mu": 100.0}, torch.device("cpu"))
    assert all(torch.equal(plain[k], proximal[k]) for k in plain)  # zero proximal gradient at broadcast
    plain, _, _ = local_fit(initial, p, y, 0, settings | {"epochs": 2}, torch.device("cpu"))
    proximal, _, audit = local_fit(
        initial, p, y, 0, settings | {"epochs": 2, "mu": 100.0}, torch.device("cpu")
    )
    assert audit["max_proximal_squared_norm"] > 0
    assert any(not torch.equal(plain[k], proximal[k]) for k in plain)
    official = strategy(0.01)
    assert isinstance(official, FedProx)
    messages = list(
        official.configure_train(
            1, ArrayRecord(initial), ConfigRecord(), SimpleNamespace(get_node_ids=lambda: [1, 2, 3])
        )
    )
    assert len(messages) == 3 and all(m.content["config"]["proximal-mu"] == 0.01 for m in messages)


@pytest.mark.parametrize("kind", ["original", "small", "logistic"])
def test_missing_evidence_is_masked_in_every_experimental_fusion(kind):
    model = make_model(kind)
    p = torch.tensor([[0.3, 0.8, 0.4]])
    hidden = p.clone()
    hidden[0, 2] = float("nan")
    mask = torch.tensor([[True, True, False]])
    origin = torch.tensor([0])
    torch.testing.assert_close(
        model.fusion(p, 2 * (p - 0.5).abs(), mask, origin),
        model.fusion(hidden, 2 * (hidden - 0.5).abs(), mask, origin),
    )


def test_marginal_policy_stop_cost_budget_and_availability():
    model = make_model()
    with torch.no_grad():
        for parameter in model.router.parameters():
            parameter.zero_()
        model.router.context[-1].bias[:3] = 0.05
    origin, own, available = torch.tensor([0]), torch.tensor([0.7]), torch.ones(1, 3, dtype=torch.bool)
    cheap = query_scores(model, origin, own, available, True, 0.02)
    costly = query_scores(model, origin, own, available, True, 0.1)
    assert cheap.argmax(-1).item() in [2, 3] and costly.argmax(-1).item() == 0
    assert cheap[0, 0] == 0 and torch.isneginf(cheap[0, 1])
    selected = model.router.actions[cheap.argmax(-1)]
    assert selected.sum() <= 1 and not selected[0, 0]
    available[0, 1:] = False
    assert query_scores(model, origin, own, available, True, 0.02).argmax(-1).item() == 0
    assert "p" not in query_scores.__annotations__  # no peer probabilities in routing API


def test_uniform_pretraining_and_value_training_freeze_correct_parameters():
    initial, p, y = sample()
    fusion, _, audit = local_fit(
        initial, p, y, 0, {"epochs": 1, "seed": 1, "round": 1, "mode": "uniform"}, torch.device("cpu")
    )
    assert audit["drift"]["router."] == 0 and audit["drift"]["fusion."] > 0
    value, _, audit = local_fit(
        fusion, p, y, 0, {"epochs": 1, "seed": 1, "round": 1, "mode": "value"}, torch.device("cpu")
    )
    assert audit["drift"]["fusion."] == 0 and audit["drift"]["router."] > 0
    assert all(torch.equal(v, value[k]) for k, v in fusion.items() if k.startswith("fusion."))
    with pytest.raises(ValueError, match="targets"):
        objective(
            make_model(), torch.tensor(p), torch.tensor(y), torch.zeros(len(y), dtype=torch.long), "value"
        )
    targets = value_targets(
        make_model(), torch.tensor(p), torch.tensor(y), torch.zeros(len(y), dtype=torch.long)
    )
    assert not targets.requires_grad and targets.abs().max() <= 3


def test_flower_training_reply_contains_shared_arrays_only(tmp_path, monkeypatch):
    initial, p, y = sample()
    folder = tmp_path / "contexts/unit/seed_42/client_0"
    folder.mkdir(parents=True)
    np.savez(folder / "local.npz", train_y=y)
    np.save(folder / "shared_training.npy", p)
    (folder / "specialist.pt").write_bytes(b"private-parameters-not-readable-by-shared-training")
    before = hashlib.sha256((folder / "specialist.pt").read_bytes()).hexdigest()
    monkeypatch.setenv(
        "SWARMFL_REPAIR_CONFIG", json.dumps({"output": str(tmp_path), "contexts": {"unit": {}}})
    )
    settings = {
        "context": "unit",
        "candidate": "avg1",
        "persistent": False,
        "epochs": 1,
        "seed": 42,
        "device": "cpu",
        "server-round": 1,
        "mu": 0.0,
    }
    message = Message(
        RecordDict({"arrays": ArrayRecord(initial), "config": ConfigRecord(settings)}),
        dst_node_id=1,
        message_type="train",
    )
    reply = train(message, SimpleNamespace(node_config={"partition-id": 0}))
    assert set(reply.content) == {"arrays", "metrics"}
    assert set(reply.content["arrays"].to_torch_state_dict()) == set(initial)
    assert before == hashlib.sha256((folder / "specialist.pt").read_bytes()).hexdigest()


def test_selection_is_unchanged_when_outer_assessment_labels_or_metrics_change():
    results = {}
    for fold in range(3):
        results[f"fold{fold}_inner"] = {
            name: {"assessment": {str(i): {"primary": float(j)} for i in range(3)}}
            for j, name in enumerate([*CANDIDATES, "utility_bins"])
        }
        results[f"fold{fold}_outer"] = {"arbitrary": {"labels": [1, 0], "primary": -999}}
    before = select_inner(results)
    for fold in range(3):
        results[f"fold{fold}_outer"] = {"changed_labels": [0, 1], "primary": 999}
    assert select_inner(results) == before
    assert all(row["candidate"] == "avg10" for row in before.values())


def test_nested_specialist_provenance_excludes_inner_and_outer_assessment():
    from pathlib import Path

    root = Path("artifacts/m5-step02-v1")
    if not (root / "contexts.json").exists():
        pytest.skip("Prepared nested experiment unavailable")
    contexts = json.loads((root / "contexts.json").read_text())
    for name, ctx in contexts.items():
        fitting, held = set(ctx["fit_ids"]), set(ctx["assessment_ids"])
        assert fitting.isdisjoint(held)
        if ctx["stage"] == "inner":
            outer = contexts[name.replace("inner", "outer")]
            assert (fitting | held).isdisjoint(outer["assessment_ids"])
        for origin in range(3):
            meta = json.loads(
                (root / "contexts" / name / "seed_42" / f"client_{origin}/specialist.json").read_text()
            )
            assert set(meta["preprocessing"]["fit_ids"]) == fitting
            for fold in meta["folds"]:
                fit, cal, oof = (set(fold[key]) for key in ("model_fit_ids", "calibration_ids", "held_ids"))
                assert fit.isdisjoint(cal) and fit.isdisjoint(oof) and cal.isdisjoint(oof)
                assert fit | cal | oof == fitting
                assert (fit | cal).isdisjoint(held)


@pytest.mark.parametrize("mode", ["joint", "value"])
def test_new_routing_decisions_ignore_unqueried_probabilities_and_assessment_labels(mode):
    _, p, y = sample()
    model = make_model()
    before = evaluate(model, p, y, mode)["0"]["actions"]
    changed = p.copy()
    changed[:, 1:] = 1 - changed[:, 1:]
    after = evaluate(model, changed, 1 - y, mode)["0"]["actions"]
    assert before == after


def test_cost_objective_adds_exact_expected_consultation_penalty():
    _, p, y = sample()
    model = make_model()
    p = torch.tensor(p)
    y = torch.tensor(y)
    o = torch.zeros(len(y), dtype=torch.long)
    available = torch.ones_like(p, dtype=torch.bool)
    q = model.router(o, p[:, 0], 2 * (p[:, 0] - 0.5).abs(), available.float(), available).softmax(-1)
    difference = objective(model, p, y, o, cost=0.3) - objective(model, p, y, o, cost=0.02)
    torch.testing.assert_close(difference, 0.28 * (1 - q[:, 0]).mean(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", ["original", "small", "logistic"])
def test_experimental_checkpoint_and_array_compatibility(kind, tmp_path):
    state = snapshot(make_model(kind))
    torch.save(state, tmp_path / "model.pt")
    restored = ArrayRecord(torch.load(tmp_path / "model.pt", weights_only=True)).to_torch_state_dict()
    model = make_model(kind)
    model.load_state_dict(restored)
    assert all(torch.equal(v, restored[k]) for k, v in state.items())
    if kind != "original":
        with pytest.raises(RuntimeError):
            make_model().load_state_dict(restored)
