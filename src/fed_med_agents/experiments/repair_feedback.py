"""Reuse M4 Apps unchanged on training-only specialists and guarded holdout."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ..feedback.events import FeedbackLedger
from ..feedback.learning import feedback_loss
from ..feedback.versions import PolicyRegistry


def audit_steps(path: Path) -> dict:
    """Replay each actually used local step from authenticated observed evidence."""
    result = json.loads((path / "metrics.json").read_text())
    cfg = result["config"]
    keys = json.loads((path / "credentials.json").read_text())
    registry = PolicyRegistry(path / "versions")
    with np.load(Path(cfg["source"]) / "cohort.npz") as data:
        allowed = {f"case-{i:04d}" for i in data["train"]}
    stores = [FeedbackLedger(path / f"client_{i}/adaptive", i, allowed, **keys[str(i)]) for i in range(3)]
    reports = []
    for update in result["history"]:
        if "version" not in update:
            continue
        version = update["version"]
        parent = json.loads((path / "versions" / f"{version}.json").read_text())["parent"]
        for origin, store in enumerate(stores):
            model = registry.load(parent)
            model.train()
            rows = store.replay(update["episode"], cfg["window"])
            labels = np.asarray([outcome.label for _, outcome in rows], dtype=float)
            risks = np.clip(np.asarray([event.final_risk for event, _ in rows]), 1e-6, 1 - 1e-6)
            reward_cost = -(labels * np.log(risks) + (1 - labels) * np.log1p(-risks)) + np.asarray(
                [event.consultation_cost for event, _ in rows]
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            for step in range(cfg["epochs"]):
                optimizer.zero_grad()
                loss, stats = feedback_loss(model, rows, torch.device("cpu"))
                loss.backward()
                p = torch.tensor([[0.5 if v is None else v for v in e.evidence] for e, _ in rows])
                own = p[:, origin]
                origins = torch.full((len(rows),), origin)
                available = torch.tensor([e.available_domains for e, _ in rows])
                with torch.no_grad():
                    pi = model.router(
                        origins, own, 2 * (own - 0.5).abs(), available.float(), available
                    ).softmax(-1)
                    indices = torch.tensor([e.action for e, _ in rows])
                    ratio = pi.gather(1, indices[:, None]).flatten() / torch.tensor(
                        [e.propensity for e, _ in rows]
                    )
                    w = ratio.clamp(max=2)
                gradients = {
                    prefix: float(
                        torch.cat(
                            [
                                v.grad.flatten()
                                for name, v in model.named_parameters()
                                if name.startswith(prefix) and v.grad is not None
                            ]
                        ).norm()
                    )
                    for prefix in ("router.", "fusion.")
                }
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5))
                optimizer.step()
                reports.append(
                    {
                        "update": version,
                        "origin": origin,
                        "step": step + 1,
                        "records": len(rows),
                        "loss": float(loss.detach()),
                        "preclip_norm": norm,
                        "component_gradient_norms": gradients,
                        "ess": float(w.sum() ** 2 / w.square().sum()),
                        "policy_entropy": float(-(pi * pi.clamp_min(1e-30).log()).sum(-1).mean()),
                        "min_propensity": min(e.propensity for e, _ in rows),
                        "mean_reward_age": float(np.mean([update["episode"] - e.episode for e, _ in rows])),
                        "logged_reward_cost_mean": float(reward_cost.mean()),
                        "logged_reward_cost_std": float(reward_cost.std()),
                        "older_version_fraction": float(
                            np.mean([e.model_version != parent for e, _ in rows])
                        ),
                        "actions": np.bincount([e.action for e, _ in rows], minlength=4).tolist(),
                        **stats,
                    }
                )
            recorded = torch.load(
                path / f"client_{origin}/update_{int(version[1:])}.pt", weights_only=True, map_location="cpu"
            )
            max_difference = max(
                float((value.cpu() - recorded[key]).abs().max()) for key, value in model.state_dict().items()
            )
            if max_difference > 2e-5:
                raise ValueError(f"Actual local feedback steps did not reproduce: {max_difference}")
    return {"steps": reports, "source": str(path), "epochs": cfg["epochs"], "observed_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/m5-step02-v1"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.output.resolve()
    torch.set_num_threads(1)
    if args.worker:
        from flwr.simulation import run_simulation

        from ..feedback.client_app import app as client_app
        from ..feedback.server_app import app as server_app

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=3,
            backend_config={
                "client_resources": {"num_cpus": 1, "num_gpus": 0},
                "init_args": {
                    "num_cpus": 1,
                    "num_gpus": 0,
                    "include_dashboard": False,
                    "object_store_memory": 256 * 1024 * 1024,
                },
            },
        )
        return
    source = root / "contexts/fold0_outer/seed_42"
    baseline = torch.load(root / "runs/fold0_outer/avg10/server_round_5.pt", weights_only=True)
    # New, training-only source, never a historical checkpoint/pointer.
    torch.save(baseline, source / "shared_best.pt")
    audits = {"historical_local_steps": audit_steps(Path("artifacts/milestone04"))}
    for epochs in (2, 8):
        target = root / f"feedback_steps{epochs}"
        target.mkdir(exist_ok=False)
        cfg = {
            "source": str(source),
            "output": str(target),
            "device": "cpu",
            "seed": 42,
            "episodes": 120,
            "delay": 3,
            "triggers": [60, 120, 123],
            "epochs": epochs,
            "learning_rate": 0.0003,
            "min_feedback": 4,
            "window": 128,
            "epsilon": 0.2,
            "cost": 0.02,
            "approval_tolerance": 0.02,
            "max_update_norm": 0.1,
        }
        (target / "config.json").write_text(json.dumps(cfg, indent=2))
        (target / "credentials.json").write_text(
            json.dumps(
                {
                    str(i): {"prediction_key": secrets.token_hex(32), "outcome_key": secrets.token_hex(32)}
                    for i in range(3)
                }
            )
        )
        registry = PolicyRegistry(target / "versions")
        torch.manual_seed(42)
        from ..fusion import CollaborativeModel

        registry.save(
            "v000", CollaborativeModel().state_dict(), {"eligible": True, "source": "seeded-cold-start"}
        )
        registry.approve("v000")
        registry.save(
            "offline", baseline, {"eligible": False, "protocol": "fitting-full-information-reference"}
        )
        environment = os.environ | {
            "FED_MED_M4_CONFIG": json.dumps(cfg),
            "FED_MED_M2_DIR": str(source.parent),
            "FED_MED_DEVICE": "cpu",
            "PYTHONUTF8": "1",
        }
        with (target / "flower.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-u", "-m", __spec__.name, "--worker", "--output", str(root)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not (target / "metrics.json").exists():
            raise RuntimeError("Feedback study did not complete")
        audits[f"steps{epochs}"] = audit_steps(target)
        (root / "feedback_audit.json").write_text(json.dumps(audits, indent=2))
        print(f"Completed genuine feedback stream with {epochs} local steps", flush=True)


if __name__ == "__main__":
    main()
