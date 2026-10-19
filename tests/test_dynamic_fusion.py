"""Variable-cardinality fusion, live admission and explicit version compatibility."""

from __future__ import annotations

import copy
import itertools

import pytest
import torch

from fed_med_agents.feedback.versions import PolicyRegistry
from fed_med_agents.fusion.dynamic import DynamicFusion, FusionEvidence
from tests.test_registry import registry, session, specialist


def model(kind):
    return DynamicFusion(
        kind, [specialist(n).model_dump(mode="json") for n in "ABC"], {"A": 0.3, "B": 0.5, "C": 0.7}
    )


def evidence(name, p=0.3):
    return FusionEvidence(name, "123", p, specialist(name).model_adapter.checkpoint_sha256)


@pytest.mark.parametrize("kind", ["mean", "weighted", "set"])
def test_cardinality_permutation_missing_and_stop(kind, tmp_path):
    torch.manual_seed(1)
    fusion = model(kind)
    request = session(registry())
    for peers in [(), ("B",), ("C",), ("B", "C")]:
        records = [evidence("A", 0.2)] + [evidence(name, 0.4 + i * 0.1) for i, name in enumerate(peers)]
        expected = fusion.fuse(records, request)
        for order in itertools.permutations(records):
            assert fusion.fuse(order, request) == expected
        assert 0 <= expected <= 1
    for p in (0.0, 1.0, 0.123):
        assert fusion.fuse([evidence("A", p)], request) == p
    assert request.calls == 0  # Fusion performs no consultation or additional inference.
    path = tmp_path / "fusion.pt"
    fusion.save(path)
    restored = DynamicFusion.load(path)
    assert restored.config() == fusion.config()
    assert restored.fingerprint() == fusion.fingerprint()
    config = fusion.config()
    for spec in config["declarations"]:
        spec["capabilities"].reverse()
    assert DynamicFusion(**config).fingerprint() == fusion.fingerprint()
    assert restored.fuse([evidence("A"), evidence("B")], request) == fusion.fuse(
        [evidence("A"), evidence("B")], request
    )
    bad = tmp_path / "historical.pt"
    torch.save({"model.network.0.weight": torch.zeros(24, 12)}, bad)
    with pytest.raises(KeyError):
        DynamicFusion.load(bad)


def test_mean_and_training_only_reliability_subset_normalization():
    request = session(registry())
    records = [evidence("A", 0.2), evidence("C", 0.8)]
    assert model("mean").fuse(records, request) == pytest.approx(0.5)
    w = torch.tensor([-0.3, -0.7], dtype=torch.float64).softmax(0)
    assert model("weighted").fuse(records, request) == pytest.approx(
        float((w * torch.tensor([0.2, 0.8], dtype=torch.float64)).sum())
    )
    with pytest.raises(ValueError, match="reliability"):
        DynamicFusion("weighted", [specialist("A").model_dump(mode="json")], {})


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1, None, True])
def test_invalid_observed_probability_rejected(probability):
    with pytest.raises(ValueError, match="probability"):
        model("mean").fuse([evidence("A", probability)], session(registry()))


def test_duplicate_unknown_missing_unauthorized_and_stale_predictions():
    reg = registry()
    request = session(reg)
    fusion = model("mean")
    for records in ([], [evidence("B")], [evidence("A"), evidence("A")]):
        with pytest.raises(ValueError):
            fusion.fuse(records, request)
    small = DynamicFusion("mean", [specialist(n).model_dump(mode="json") for n in "AB"])
    with pytest.raises(ValueError, match="Unknown"):
        small.fuse([evidence("A"), evidence("C")], request)
    bad = FusionEvidence("A", "wrong", 0.2, specialist("A").model_adapter.checkpoint_sha256)
    with pytest.raises(ValueError, match="Misaligned"):
        fusion.fuse([bad], request)
    denied = session(reg, permitted=frozenset({"C"}))
    with pytest.raises(ValueError, match="Unauthorized"):
        fusion.fuse([evidence("A"), evidence("B")], denied)
    reg.update("B", available=False)
    assert not any(e.specialist.specialist_id == "B" for e in request.eligible())
    with pytest.raises(ValueError, match="Unavailable"):
        fusion.fuse([evidence("A"), evidence("B")], request)
    reg.update("B", available=True)
    assert fusion.fuse([evidence("A"), evidence("B")], request) == pytest.approx(0.3)
    removed = reg.unregister("C")
    assert fusion.fuse([evidence("A"), evidence("B")], request) == pytest.approx(0.3)
    reg.register(removed.specialist, node_id=removed.node_id)
    assert fusion.fuse([evidence("A"), evidence("C")], request) == pytest.approx(0.3)


def test_changed_checkpoint_task_and_output_rejected():
    reg = registry()
    fusion = model("set")
    request = session(reg)
    spec = specialist("B").model_dump(mode="json")
    spec["model_adapter"]["checkpoint_sha256"] = "e" * 64
    from fed_med_agents.collaboration.registry import Specialist

    reg.unregister("B")
    reg.register(Specialist.model_validate(spec), node_id=2)
    with pytest.raises(ValueError, match="Incompatible"):
        fusion.fuse([evidence("A"), evidence("B")], request)
    spec = specialist("C").model_dump(mode="json")
    spec["task"]["name"] = "incompatible-task"
    with pytest.raises(ValueError, match="Incompatible"):
        DynamicFusion("mean", [specialist("A").model_dump(mode="json"), spec])
    spec = specialist("C").model_dump(mode="json")
    spec["output_schema"]["dtype"] = "float64"
    with pytest.raises(ValueError, match="Incompatible"):
        DynamicFusion("mean", [specialist("A").model_dump(mode="json"), spec])


def test_tensor_mask_and_permutation_and_real_gradient():
    torch.manual_seed(2)
    fusion = model("set")
    p = torch.tensor([[0.2, 0.4, 0.8], [0.7, 0.6, 0.3]])
    x = fusion.tokens(p, list("ABC"), torch.tensor([0, 1]))
    mask = torch.tensor([[True, False, True], [False, True, True]])
    original = fusion(x, mask)
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(fusion(x[:, perm], mask[:, perm]), original, atol=1e-7, rtol=1e-6)
    hidden = x.clone()
    hidden[~mask] = float("nan")
    assert torch.equal(fusion(hidden, mask), original)
    before = copy.deepcopy(fusion.state_dict())
    opt = torch.optim.Adam(fusion.parameters(), lr=0.003)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(original, torch.tensor([0.0, 1.0]))
    loss.backward()
    opt.step()
    assert any(not torch.equal(v, fusion.state_dict()[k]) for k, v in before.items())
    with pytest.raises(ValueError, match="Empty"):
        fusion(x, torch.zeros_like(mask))


def test_existing_version_registry_fusion_approval_rejection_and_rollback(tmp_path):
    fusion = model("set")
    versions = PolicyRegistry(tmp_path)
    versions.save("v000", fusion.state_dict(), {"eligible": True, "fusion_config": fusion.config()})
    versions.approve("v000")
    with torch.no_grad():
        next(fusion.parameters()).add_(0.01)
    versions.save("v001", fusion.state_dict(), {"eligible": False, "fusion_config": fusion.config()})
    with pytest.raises(ValueError, match="Rejected"):
        versions.approve("v001")
    versions.save("v002", fusion.state_dict(), {"eligible": True, "fusion_config": fusion.config()})
    versions.approve("v002")
    versions.rollback("v000")
    assert versions.current() == "v000"
    assert isinstance(versions.load(), DynamicFusion)
    assert not torch.equal(next(versions.load("v002").parameters()), next(versions.load().parameters()))
    with pytest.raises(ValueError, match="Ambiguous"):
        versions.save(
            "v003",
            fusion.state_dict(),
            {"eligible": True, "fusion_config": fusion.config(), "router_config": {}},
        )


def test_reliability_weight_normalization_is_stable_for_large_losses():
    fusion = DynamicFusion(
        "weighted",
        [specialist(n).model_dump(mode="json") for n in "ABC"],
        {"A": 1000.0, "B": 1001.0, "C": 1002.0},
    )
    value = fusion.fuse([evidence("A", 0.2), evidence("B", 0.8)], session(registry()))
    assert 0.2 < value < 0.8
    p = torch.tensor([[0.2, 0.8]])
    logits = fusion(fusion.tokens(p, list("AB"), torch.tensor([0])), torch.ones(1, 2, dtype=torch.bool))
    assert float(logits.sigmoid()[0]) == pytest.approx(value, abs=1e-6)
