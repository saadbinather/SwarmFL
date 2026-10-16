"""Guard diagnostic controls, frozen components and patient/evidence isolation."""

import numpy as np
import pytest
import torch

from fed_med_agents.experiments.routing_diagnostics import evaluate, fit, prediction_table, snapshot
from fed_med_agents.fusion import CollaborativeModel


@pytest.mark.parametrize("variant,prefix", [("frozen_router", "router."), ("frozen_fusion", "fusion.")])
def test_frozen_ablation_changes_only_trainable_component(variant, prefix):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    state = snapshot(CollaborativeModel())
    untouched = {key: value.clone() for key, value in state.items()}
    rng = np.random.default_rng(7)
    p = rng.uniform(0.1, 0.9, (16, 3)).astype(np.float32)
    y = np.tile([0.0, 1.0], 8).astype(np.float32)
    final, audit = fit(state, p, y, [0], 7, 1, variant, torch.device("cpu"))
    assert all(torch.equal(value, final[key]) for key, value in state.items() if key.startswith(prefix))
    other = "fusion." if prefix == "router." else "router."
    assert audit["drift"][prefix] == 0 and audit["drift"][other] > 0
    assert all(torch.equal(value, untouched[key]) for key, value in state.items())


def test_router_never_sees_unconsulted_probabilities_and_fusion_masks_them():
    model = CollaborativeModel()
    p = torch.tensor([[0.3, 0.4, 0.8], [0.7, 0.2, 0.9]])
    scores, outputs, means = prediction_table(model, p, 0)
    changed = p.clone()
    changed[:, 2] = 1 - changed[:, 2]
    other_scores, other_outputs, other_means = prediction_table(model, changed, 0)
    torch.testing.assert_close(scores, other_scores)
    # Empty action and consult-B have no C evidence.
    torch.testing.assert_close(outputs[:, [0, 2]], other_outputs[:, [0, 2]])
    torch.testing.assert_close(means[:, [0, 2]], other_means[:, [0, 2]])


def test_matched_controls_have_identical_call_counts_and_no_self_action():
    p = np.random.default_rng(3).uniform(0.1, 0.9, (20, 3)).astype(np.float32)
    y = np.tile([0.0, 1.0], 10).astype(np.float32)
    model = CollaborativeModel()
    report = evaluate(model, p, y, 3)
    for origin in range(3):
        rows = report[str(origin)]["policies"]
        for control in ("random_matched", "cyclic_matched"):
            assert rows[control]["calls"] == rows["learned"]["calls"]
            assert rows[control]["action_frequencies"][origin + 1] == 0
        assert rows["stop"]["calls"] == 0
        assert rows["random_max"]["calls"] == rows["cyclic_max"]["calls"] == 1
        assert rows["stop"]["mean_fusion"] == report[str(origin)]["raw_origin"]
