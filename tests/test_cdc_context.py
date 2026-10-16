"""Authorization, whole-stack isolation, two-stage masking and observed-only feedback."""

import numpy as np
import pytest
import torch

from fed_med_agents.experiments.cdc_context import (
    ValueRouter,
    fit_router,
    local_fields,
    observed_records,
    paired_gains,
    query_features,
    select_router,
    two_stage,
)
from fed_med_agents.experiments.cdc_data import DOMAINS, THREE_DOMAINS
from fed_med_agents.experiments.cdc_routing import gross_gains


@pytest.mark.parametrize("origin", [0, 1, 2])
def test_origin_context_is_invariant_to_unauthorized_fields(origin):
    columns = list(dict.fromkeys(field for domain in DOMAINS.values() for field in domain))
    x = np.ones((8, len(columns)), dtype=np.float32)
    expected = local_fields(x, columns, origin)
    forbidden = [i for i, name in enumerate(columns) if name not in list(THREE_DOMAINS.values())[origin]]
    x[:, forbidden] = 9999
    actual = local_fields(x, columns, origin)
    for first, second in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(first, second)
    inputs = {
        "own": np.full(8, 0.2),
        "origin": origin,
        "peer": (origin + 1) % 3,
        "available": np.ones((8, 3)),
        "covered": np.ones((8, 21)),
        "reliability": np.ones((8, 3)),
    }
    feature = query_features(**inputs, local=actual)
    assert feature.shape == (8, 63)
    bad = actual[0].copy()
    bad[actual[1] == 0] = 1
    with pytest.raises(ValueError, match="Unauthorized"):
        query_features(**inputs, local=(bad, actual[1]))


def test_two_stage_masks_peers_and_stops_on_selected_expected_net_gain():
    value = np.array([[9, 0.01, 0.2], [9, 0.01, 0.2], [9, 0.5, 0.2]], dtype=np.float32)
    rank = np.array([[9, 0.9, 0.8]] * 3, dtype=np.float32)
    available = np.array([[True, True, True], [True, False, True], [True, False, False]])
    actions = two_stage(value, rank, 0, available, np.ones((3, 3)), 0.02)
    # Ranking picks B, but stage one stops because B's expected benefit is too small.
    np.testing.assert_array_equal(actions, [0, 3, 0])
    assert not np.any(actions == 1)


def test_bandit_records_ignore_all_unselected_risks(tmp_path):
    columns = list(dict.fromkeys(field for domain in DOMAINS.values() for field in domain))
    paired = np.random.default_rng(8).normal(size=(3, 12, 2, 63)).astype(np.float32)
    risks = np.random.default_rng(9).uniform(0.1, 0.9, size=(12, 3, 4)).astype(np.float32)
    p = np.full((12, 3), 0.4, dtype=np.float32)
    labels = np.tile([0.0, 1.0], 6).astype(np.float32)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    records = observed_records(paired, risks, labels, p, np.arange(12), columns, 42, first)
    changed = 1 - risks.copy()
    changed[records["row"], records["origin"], records["action"]] = records["risk"]
    other = observed_records(paired, changed, labels, p, np.arange(12), columns, 42, second)
    for key in records:
        np.testing.assert_array_equal(records[key], other[key])
    assert ((records["propensity"] >= 1 / 3) & (records["propensity"] <= 1)).all()
    assert all(
        np.isfinite(row).sum() == 1 + int(action > 0)
        for row, action in zip(records["evidence"], records["action"], strict=True)
    )


def test_gain_pair_alignment_and_separate_compatible_router_heads():
    risks = np.random.default_rng(5).uniform(0.1, 0.9, (10, 3, 4)).astype(np.float32)
    gain = gross_gains(risks, np.tile([0.0, 1.0], 5))
    paired = paired_gains(gain)
    assert paired.shape == (3, 10, 2)
    np.testing.assert_array_equal(paired[1, :, 0], gain[:, 1, 0])
    model = ValueRouter(63, True)
    assert model(torch.zeros(7, 2, 63)).shape == (7, 2, 2)
    with pytest.raises(RuntimeError):
        ValueRouter(37).load_state_dict(ValueRouter(63).state_dict())


def test_inner_selection_has_no_outer_scores():
    result = {"variable": {"0.02": {"a": {"0": {"objective": 0.4}}, "b": {"0": {"objective": 0.3}}}}}
    assert select_router(result, ("a", "b")) == "b"


def test_pairwise_cpu_fit_updates_value_and_rank_heads():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = ValueRouter(63, True)
    before = model.network[-1].weight.detach().clone()
    rng = np.random.default_rng(4)
    inputs = rng.normal(size=(3, 16, 2, 63)).astype(np.float32)
    gain = rng.normal(size=(3, 16, 2)).astype(np.float32)
    log = fit_router(model, inputs, gain, 4, epochs=2)
    assert log["steps"] == 2 and np.isfinite(log["loss"])
    assert all(
        (after - initial).abs().sum() > 0
        for after, initial in zip(model.network[-1].weight, before, strict=True)
    )
