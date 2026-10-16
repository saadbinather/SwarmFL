"""Read-only M4 replay diagnostics and preservation audit for M5 Step 1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fed_med_agents.experiments.routing_diagnostics import evaluate, loss_rows
from fed_med_agents.feedback.events import FeedbackLedger
from fed_med_agents.feedback.learning import feedback_loss
from fed_med_agents.feedback.verify import verify
from fed_med_agents.feedback.versions import PolicyRegistry


def audit_feedback(path: Path, evidence: Path) -> dict:
    """Audit authenticated selected-only replay without acquiring missing outcomes."""
    verification = verify(path)
    result = json.loads((path / "metrics.json").read_text())
    cfg = result["config"]
    source = Path(cfg["source"])
    keys = json.loads((path / "credentials.json").read_text())
    registry = PolicyRegistry(path / "versions")
    with np.load(source / "cohort.npz") as cohort:
        train = {f"case-{i:04d}" for i in cohort["train"]}
    stores = [FeedbackLedger(path / f"client_{i}" / "adaptive", i, train, **keys[str(i)]) for i in range(3)]
    replay = []
    for update in result["history"]:
        if "version" not in update:
            continue
        metadata = json.loads((path / "versions" / f"{update['version']}.json").read_text())
        parent = registry.load(metadata["parent"])
        for origin, store in enumerate(stores):
            rows = store.replay(update["episode"], cfg["window"])
            p = torch.tensor([[0.5 if v is None else v for v in e.evidence] for e, _ in rows])
            masks = torch.tensor([[v is not None for v in e.evidence] for e, _ in rows])
            own = p[:, origin]
            origins = torch.full((len(rows),), origin, dtype=torch.long)
            domains = torch.tensor([e.available_domains for e, _ in rows])
            with torch.no_grad():
                q = parent.router(origins, own, 2 * (own - 0.5).abs(), domains.float(), domains).softmax(-1)
                actions = torch.tensor([e.action for e, _ in rows])
                weights = q.gather(1, actions[:, None]).flatten() / torch.tensor(
                    [e.propensity for e, _ in rows]
                )
                current = torch.sigmoid(parent.fusion(p, 2 * (p - 0.5).abs(), masks, origins)).numpy()
            y = np.array([o.label for _, o in rows])
            logged = np.array([e.final_risk for e, _ in rows])
            clips = {}
            for clip in (0.5, 1.0, 2.0, 10.0):
                parent.zero_grad(set_to_none=True)
                loss, stats = feedback_loss(parent, rows, torch.device("cpu"), clip=clip)
                loss.backward()
                w = weights.clamp(max=clip)
                clips[str(clip)] = stats | {
                    "effective_sample_size": float(w.sum().square() / w.square().sum()),
                    "router_gradient_norm": float(
                        torch.cat(
                            [parameter.grad.flatten() for parameter in parent.router.parameters()]
                        ).norm()
                    ),
                }
            replay.append(
                {
                    "version": update["version"],
                    "origin": origin,
                    "n": len(rows),
                    "actions": np.bincount([e.action for e, _ in rows], minlength=4).tolist(),
                    "max_age_episodes": max(update["episode"] - e.episode for e, _ in rows),
                    "fraction_old_version": float(
                        np.mean([e.model_version != metadata["parent"] for e, _ in rows])
                    ),
                    "current_minus_logged_bce": float((loss_rows(y, current) - loss_rows(y, logged)).mean()),
                    "mean_abs_risk_change": float(np.abs(current - logged).mean()),
                    "clipping": clips,
                }
            )
    with np.load(evidence) as data:
        validation = {
            v: evaluate(registry.load(v), data["validation"], data["val_y"], 42)
            for v in ("v000", registry.current())
        }
    return {
        "verification": verification,
        "replay": replay,
        "validation": validation,
        "historical_metrics": {"stream": result["stream"]},
    }


def main() -> None:
    """Write fresh audit output; source files are never modified."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("artifacts/milestone04"))
    parser.add_argument("--evidence", type=Path, default=Path("artifacts/m5-step01/seed_42/evidence.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step01-audit/feedback.json"))
    args = parser.parse_args()
    report = audit_feedback(args.source, args.evidence)
    args.output.write_text(json.dumps(report, indent=2))
    manifest = Path("artifacts/m5-step01-audit/preserved-before.json")
    if manifest.exists():
        before = json.loads(manifest.read_text(encoding="utf-8-sig"))
        changed = [
            row["Path"]
            for row in before
            if hashlib.sha256(Path(row["Path"]).read_bytes()).hexdigest().upper() != row["Hash"]
        ]
        evidence = {"files_checked": len(before), "changed": changed}
        (args.output.parent / "preservation.json").write_text(json.dumps(evidence, indent=2))
        assert not changed, changed
    print(json.dumps(report["verification"], indent=2))


if __name__ == "__main__":
    main()
