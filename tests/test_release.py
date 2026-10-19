"""Release-surface tests that do not access CDC data or historical artifacts."""

from __future__ import annotations

import json

import fed_med_agents
from fed_med_agents import ConsultationSession, DynamicFusion, RoutingPolicy, SpecialistRegistry
from fed_med_agents.quickstart import IDS, feedback_smoke, prepare, versioning_smoke


def test_public_release_imports() -> None:
    assert fed_med_agents.__version__ == "0.1.0"
    assert all(
        value is not None for value in (SpecialistRegistry, ConsultationSession, RoutingPolicy, DynamicFusion)
    )


def test_quickstart_preparation_and_rollback(tmp_path) -> None:
    output = tmp_path / "demo"
    prepare(output)
    declarations = json.loads((output / "declarations.json").read_text(encoding="utf-8"))
    cases = json.loads((output / "cases.json").read_text(encoding="utf-8"))
    assert [record["specialist_id"] for record in declarations] == list(IDS)
    assert len(cases) == 3
    assert versioning_smoke(output) == {"approved": "v001", "rollback": "v000"}
    result = {
        "predictions": [{"patient_id": "synthetic-0", "prediction": 0.5}],
    }
    assert feedback_smoke(output, result) == {
        "authenticated": True,
        "pending_before_release": 0,
        "verified_after_release": 1,
    }
