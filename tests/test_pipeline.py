import numpy as np
import torch
from flwr.app import ArrayRecord

from fed_med_agents.data import FEATURES, GROUPS, prepare_arrays, view
from fed_med_agents.models import Classifier, shared_model, train


def fixture_data():
    rng = np.random.default_rng(42)
    x = rng.integers(0, 2, (520, 16)).astype(np.float32)
    x[:, 0] = np.arange(520)
    return x, np.tile([0, 1], 260)


def test_no_leakage_and_reproducible_splits():
    x, y = fixture_data()
    data, summary = prepare_arrays(x, y)
    other, _ = prepare_arrays(x, y)
    for split in ("train", "val", "test"):
        np.testing.assert_array_equal(data[split], other[split])
        assert abs(y[data[split]].mean() - 0.5) < 0.02
    sets = [set(data[s]) for s in ("train", "val", "test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert len(set.union(*sets)) == 520
    changed = x.copy()
    changed[np.concatenate([data["val"], data["test"]]), 0] = 100000
    changed_data, changed_summary = prepare_arrays(changed, y)
    assert changed_summary["age_mean"] == summary["age_mean"]
    assert changed_summary["age_scale"] == summary["age_scale"]
    assert set(summary["preprocessing_fit_ids"]) == sets[0]
    np.testing.assert_array_equal(data["x"][data["train"]], changed_data["x"][data["train"]])


def test_feature_isolation_and_alignment():
    x, y = fixture_data()
    data, _ = prepare_arrays(x, y)
    assert sorted(feature for group in GROUPS.values() for feature in group) == sorted(FEATURES)
    assert [len(v) for v in GROUPS.values()] == [3, 5, 8]
    for client, features in GROUPS.items():
        indices = [FEATURES.index(f) for f in features]
        private, labels = view(data, client, "train")
        shared, shared_labels = view(data, client, "train", True)
        np.testing.assert_array_equal(labels, shared_labels)
        np.testing.assert_array_equal(labels, data["y"][data["train"]])
        np.testing.assert_array_equal(shared[:, indices], private)
        excluded = [i for i in range(16) if i not in indices]
        assert np.all(shared[:, excluded] == 0)
        assert np.all(shared[:, np.array(indices) + 16] == 1)
        assert np.all(shared[:, np.array(excluded) + 16] == 0)
        changed = {k: v.copy() for k, v in data.items()}
        changed["x"][:, excluded] = 999
        np.testing.assert_array_equal(view(changed, client, "train", True)[0], shared)


def test_shared_compatibility_and_private_weights():
    x, y = fixture_data()
    data, _ = prepare_arrays(x, y)
    torch.manual_seed(42)
    shared = shared_model()
    record = ArrayRecord(shared.state_dict())
    initial = {k: v.clone() for k, v in shared.state_dict().items()}
    for client, features in GROUPS.items():
        model = shared_model()
        model.load_state_dict(record.to_torch_state_dict(), strict=True)
        private = Classifier(len(features))
        private_before = {k: v.clone() for k, v in private.state_dict().items()}
        local_x, local_y = view(data, client, "train", True)
        train(model, local_x, local_y, epochs=1)
        assert any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
        assert all(torch.equal(private_before[k], v) for k, v in private.state_dict().items())
        assert {k: tuple(v.shape) for k, v in model.state_dict().items()} == {
            k: tuple(v.shape) for k, v in initial.items()
        }
        assert private.net[0].weight.shape != model.net[0].weight.shape


def test_client_reply_contains_only_shared_weights_and_scalars(tmp_path, monkeypatch):
    from flwr.app import ConfigRecord, Context, Message, RecordDict
    from flwr.supercore.task_identity import TaskIdentity

    from fed_med_agents import client_app

    monkeypatch.setattr(TaskIdentity, "_run_id", 1)
    monkeypatch.setattr(TaskIdentity, "_node_id", 0)
    monkeypatch.setattr(TaskIdentity, "_task_id", 1)
    x, y = fixture_data()
    data, _ = prepare_arrays(x, y)
    monkeypatch.setattr(client_app, "load", lambda: data)
    monkeypatch.setattr(client_app, "ARTIFACTS", tmp_path)
    model = shared_model()
    request = Message(
        content=RecordDict(
            {"arrays": ArrayRecord(model.state_dict()), "config": ConfigRecord({"server-round": 1})}
        ),
        dst_node_id=1,
        message_type="train",
    )
    context = Context(run_id=1, node_id=1, node_config={"partition-id": 0}, state=RecordDict(), run_config={})
    reply = client_app.train_client(request, context)
    assert set(reply.content) == {"arrays", "metrics"}
    state = reply.content["arrays"].to_torch_state_dict()
    assert set(state) == set(model.state_dict())
    assert all(state[k].shape == model.state_dict()[k].shape for k in state)
    assert all(isinstance(value, (int, float)) for value in reply.content["metrics"].values())
    assert reply.content["metrics"]["num-examples"] == 364
    assert reply.content["metrics"]["weight-change"] > 0
