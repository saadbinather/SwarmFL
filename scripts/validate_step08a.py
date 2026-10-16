"""External, CPU-only synthetic validation of existing SwarmFL public modules.

The synthetic event-within-30-days label has no clinical meaning. This adapter
is deliberately outside the core package and never imports CDC implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fed_med_agents.collaboration.registry import (
    ConsultationSession,
    FeatureSchema,
    ModelAdapter,
    OutputSchema,
    RegistryError,
    Specialist,
    SpecialistRegistry,
    TaskSchema,
)
from fed_med_agents.feedback.continual import ContinualController
from fed_med_agents.feedback.events import FeedbackLedger, VerifiedOutcome, digest, sign
from fed_med_agents.feedback.learning import fusion_feedback_update
from fed_med_agents.feedback.versions import PolicyPair, PolicyRegistry, routing_update_gates
from fed_med_agents.fusion.dynamic import DynamicFusion, FusionEvidence
from fed_med_agents.routing.policy import RoutingPolicy

ROOT = Path("artifacts/m5-step08a-v1")
IDS = ("external-linear", "external-tree", "external-linear-alt")
KEYS = ("18" * 32, "81" * 32)  # Public simulation credentials, never deployment keys.
TASK = TaskSchema(
    name="synthetic-event-30d",
    labels=("synthetic-no-event", "synthetic-event"),
    label_definition="Artificial Bernoulli event within 30 synthetic days; no clinical meaning",
)
OUTPUT = OutputSchema(positive_label=TASK.labels[1])


def case_id(index: int) -> str:
    """Return a deterministic opaque identifier without exposing a row number."""
    return "syn-" + hashlib.sha256(f"swarmfl-step08a:{index}".encode()).hexdigest()[:20]


def planned_specialists() -> list[dict[str, Any]]:
    """Identities frozen before fitting; checkpoint hashes are recorded afterward."""
    return [
        {
            "specialist_id": IDS[0],
            "version": "synthetic-a-v1",
            "model_family": "sklearn-standard-scaler-logistic-regression",
            "features": ["synthetic_0", "synthetic_1"],
            "preprocessing": "StandardScaler fit on private training rows only",
            "cost": 0.02,
        },
        {
            "specialist_id": IDS[1],
            "version": "synthetic-b-v1",
            "model_family": "sklearn-decision-tree-depth-3",
            "features": ["synthetic_2", "synthetic_3"],
            "preprocessing": "none",
            "cost": 0.02,
        },
        {
            "specialist_id": IDS[2],
            "version": "synthetic-c-v1",
            "model_family": "sklearn-standard-scaler-logistic-regression",
            "features": ["synthetic_4", "synthetic_5"],
            "preprocessing": "StandardScaler fit on private training rows only",
            "cost": 0.02,
        },
    ]


def read(path: Path) -> Any:
    """Read a local UTF-8 evidence file."""
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    """Write finite local evidence, never overwrite an old experiment directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path: Path) -> str:
    """Hash permitted files; frozen test containers are explicitly forbidden."""
    if path.name.startswith("frozen_test"):
        raise ValueError("Frozen CDC test files must not be opened, including for hashing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def distance(a: dict, b: dict) -> float:
    return sum(float(((a[k].double() - b[k].double()) ** 2).sum()) for k in a) ** 0.5


def generated() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic artificial independent cases; no historical data access."""
    rng = np.random.default_rng(58001)
    x = rng.normal(size=(600, 6)).astype(np.float64)
    logits = x @ np.array([1.1, -0.8, 0.7, 0.5, -0.9, 0.6]) - 0.2
    y = (rng.random(600) < 1 / (1 + np.exp(-logits))).astype(np.int64)
    return x, y


def partitions(root: Path) -> dict:
    """Fail closed on overlapping, missing, duplicated or relabeled population IDs."""
    cfg = read(root / "protocol.json")
    sets = [set(cfg[k]) for k in ("training", "feedback", "validation")]
    assert [len(s) for s in sets] == [360, 120, 120]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    assert set.union(*sets) == {case_id(i) for i in range(600)}
    assert all(len(cfg[k]) == len(set(cfg[k])) for k in ("training", "feedback", "validation"))
    return cfg


def initialize(root: Path, *, preserve_history: bool = True) -> None:
    """Freeze settings before any classifier fitting; save role-separated datasets."""
    if root.exists() and any(path.name != "attempts" for path in root.iterdir()):
        raise FileExistsError("Use a fresh evidence directory; retain failed attempts")
    root.mkdir(parents=True, exist_ok=True)
    attempt_manifest = {}
    attempts = root / "attempts"
    if attempts.exists():
        attempt_manifest = {str(path): sha(path) for path in attempts.rglob("*") if path.is_file()}
        write(root / "attempt-hashes.json", attempt_manifest)
    order = np.random.default_rng(58002).permutation(600)
    splits = {
        "training": [case_id(int(i)) for i in order[:360]],
        "feedback": [case_id(int(i)) for i in order[360:480]],
        "validation": [case_id(int(i)) for i in order[480:]],
    }
    cfg = {
        "protocol_version": "m5-step08a-v1",
        "frozen_before_fitting": True,
        "prior_incomplete_attempt": {
            "path": "attempts/initialization-only-20261015",
            "files": len(attempt_manifest),
            "manifest_sha256": sha(root / "attempt-hashes.json") if attempt_manifest else None,
            "failure": "Flower node discovery ran before all three nodes connected",
        },
        "seed": 58001,
        "seeds": {"data": 58001, "split": 58002, "models": 58003, "shared": 58003},
        "synthetic_outcome": TASK.model_dump(mode="json"),
        "horizon_days": 30,
        "positive_class": "synthetic-event means an artificial event within 30 simulated days",
        "features": [f"synthetic_{i}" for i in range(6)],
        "generation": "six standard-normal inputs; Bernoulli(sigmoid(x@[1.1,-.8,.7,.5,-.9,.6]-.2))",
        "specialists": planned_specialists(),
        **splits,
        "split_counts": {"training": 360, "feedback": 120, "validation": 120},
        "consultation": {
            "maximum_calls": 1,
            "maximum_cost": 0.02,
            "specialist_cost": 0.02,
            "selected_only": True,
            "aligned_payload": "opaque IDs and contract metadata; no raw feature values",
            "new_patient_payload": "only selected recipient required explicitly authorized features",
        },
        "feedback_protocol": {
            "minimum_per_origin": 64,
            "missing_every": 10,
            "prediction_episode": 1,
            "outcome_episode": 2,
            "delay_episodes": 1,
            "simulated_horizon_days": 30,
        },
        "federation": {
            "physical_clients": 3,
            "device": "cpu",
            "aggregation": "FedAvg",
            "components": ["router", "DeepSets fusion"],
            "rounds_per_component": 1,
            "local_epochs": 2,
            "shared_parameters_only": True,
        },
        "minimum_feedback_per_origin": 64,
        "missing_every": 10,
        "prediction_episode": 1,
        "outcome_episode": 2,
        "local_epochs": 2,
        "rounds_per_component": 1,
        "cost": 0.02,
        "metrics": [
            "accuracy",
            "AUROC",
            "PR-AUC",
            "binary cross-entropy",
            "Brier score",
            "ECE-10",
            "BCE plus consultation cost",
            "STOP rate",
            "recipient distribution",
            "application JSON bytes",
            "shared parameter bytes",
            "latency",
        ],
        "acceptance_gates": {
            "candidate_objective": "no worse than incumbent and appropriate fixed/mean control",
            "accuracy_drop_max": 0.005,
            "per_origin_bce_increase_max": 0.01,
            "finite_parameters": True,
            "schema_compatible": True,
            "minimum_verified_feedback_per_origin": 64,
            "deepsets_permutation_invariance": True,
        },
        "accuracy_drop_max": 0.005,
        "origin_bce_increase_max": 0.01,
        "gate": "candidate objective <= incumbent and fixed/mean; finite; sufficient; permutation",
        "routing": "existing MLP width3 with deterministic constant positive gain, no search",
        "bootstrap_fusion": "seed58003 cold DeepSets; no initial supervised fusion fitting",
        "validation_access": "Only after both candidates trained; no selection/tuning/refitting",
        "privacy": "Tested trusted local simulation only; no DP/secure aggregation/isolation",
        "forbidden_data": "Never open, load, hash, modify or evaluate the frozen CDC test file",
    }
    write(root / "protocol.json", cfg)
    write(
        root / "execution.json",
        {
            "status": "started",
            "protocol_sha256": sha(root / "protocol.json"),
            "retained_failures": [
                "Archived initialization-only attempt reached Flower but queried nodes before all three connected",
                "Initial focused-test amendment used an overly narrow output-schema regex and omitted RegistryError from the expected compatibility exception set",
                "Initial Ruff amendment found late-bound replay lambda variables and quadratic test-list construction",
            ],
        },
    )
    files = (
        [p for p in Path("artifacts").rglob("*") if p.is_file() and root not in p.parents]
        if preserve_history
        else []
    )
    if preserve_history:
        files += list(Path("docs/experiments").rglob("*.md")) + list(Path("docs/devlog").rglob("*.md"))
    protected = {str(p): sha(p) for p in files if not p.name.startswith("frozen_test")}
    write(root / "historical-hashes.json", protected)
    write(root / "core-hashes.json", {str(p): sha(p) for p in Path("src").rglob("*.py")})
    partitions(root)
    x, y = generated()
    index = {case_id(i): i for i in range(600)}
    for role in ("training", "feedback", "validation"):
        ids = cfg[role]
        indices = np.array([index[patient] for patient in ids], dtype=int)
        folder = root / role
        folder.mkdir()
        np.savez(folder / "features.npz", ids=np.array(ids), x=x[indices])
        # Outcomes stored separately so inference never opens a label file.
        np.save(folder / "outcomes.npy", y[indices])
    declarations = []
    train_x = np.load(root / "training/features.npz")["x"]
    train_y = np.load(root / "training/outcomes.npy")
    for i, identity in enumerate(cfg["specialists"]):
        name = identity["specialist_id"]
        model = (
            DecisionTreeClassifier(max_depth=3, min_samples_leaf=15, random_state=58003)
            if i == 1
            else make_pipeline(StandardScaler(), LogisticRegression(C=1.0, random_state=58003))
        )
        model.fit(train_x[:, 2 * i : 2 * i + 2], train_y)
        path = root / "private" / f"{name}.joblib"
        path.parent.mkdir(exist_ok=True)
        joblib.dump(model, path)
        features = tuple(identity["features"])
        declaration = Specialist(
            specialist_id=name,
            task=TASK,
            output_schema=OUTPUT,
            model_adapter=ModelAdapter(
                adapter_id=f"external-sklearn-v1:{identity['version']}",
                model_family=identity["model_family"],
                checkpoint_sha256=sha(path),
                feature_names=features,
                input_dim=2,
                alignment_namespace="synthetic-case-v1",
            ),
            feature_schema=tuple(FeatureSchema(name=n, minimum=-10, maximum=10) for n in features),
            capabilities=frozenset({"aligned_prediction", "new_patient_prediction"}),
            consultation_cost=cfg["cost"],
        ).model_dump(mode="json")
        declaration["capabilities"].sort()
        declarations.append(declaration)
    write(root / "declarations.json", declarations)
    write(
        root / "checkpoint-identities.json",
        {
            declaration["specialist_id"]: {
                "version": cfg["specialists"][i]["version"],
                "checkpoint_sha256": declaration["model_adapter"]["checkpoint_sha256"],
            }
            for i, declaration in enumerate(declarations)
        },
    )
    torch.manual_seed(58003)
    router = RoutingPolicy("mlp", IDS, width=3)
    with torch.no_grad():
        for p in router.parameters():
            p.zero_()
        router.network[-1].bias.fill_(0.15)
    fusion = DynamicFusion("set", declarations, dict.fromkeys(IDS, 0.7))
    pair = bundle(root)
    pair.routers.save("r000", snapshot(router), {"eligible": True, "router_config": router.config()})
    pair.fusions.save("f000", snapshot(fusion), {"eligible": True, "fusion_config": fusion.config()})
    pair.activate("r000", "f000", reason="deterministic synthetic bootstrap, no score selection")
    write(root / "private-hashes.json", {str(p): sha(p) for p in (root / "private").glob("*")})


def bundle(root: Path) -> PolicyPair:
    return PolicyPair(root / "pair", PolicyRegistry(root / "routers"), PolicyRegistry(root / "fusions"))


def registry(root: Path, nodes: dict | None = None) -> SpecialistRegistry:
    reg = SpecialistRegistry(TASK, OUTPUT)  # Heterogeneous model families permitted.
    for i, declaration in enumerate(read(root / "declarations.json")):
        reg.register(Specialist.model_validate(declaration), node_id=(nodes or {}).get(IDS[i], i + 1))
    return reg


def session(
    reg: SpecialistRegistry, origin: str, ids: tuple[str, ...], **changes: Any
) -> ConsultationSession:
    values = {
        "permitted": frozenset(n for n in IDS if n != origin),
        "feature_permissions": {
            n: frozenset(reg.get(n).specialist.model_adapter.feature_names)
            for n in IDS
            if n != origin and n in {e.specialist.specialist_id for e in reg.discover()}
        },
        "available_features": frozenset(f"synthetic_{i}" for i in range(6)),
        "alignment_namespace": "synthetic-case-v1",
        "max_calls": 1,
        "max_cost": 0.02,
    }
    return ConsultationSession(reg, origin, ids, **(values | changes))


class ExternalAdapter:
    """Developer-owned sklearn inference adapter; all preprocessing is checkpointed.

    Public ModelAdapter is declaration metadata, not an executable plugin loader.
    This adapter validates the receiving endpoint before loading private data.
    """

    def __init__(self, root: Path, index: int) -> None:
        self.root, self.index = root, index
        self.spec = Specialist.model_validate(read(root / "declarations.json")[index])

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        path = self.root / "private" / f"{self.spec.specialist_id}.joblib"
        if sha(path) != self.spec.model_adapter.checkpoint_sha256:
            raise ValueError("Private checkpoint integrity failure")
        model = joblib.load(path)
        return model.predict_proba(x)[:, list(model.classes_).index(1)]

    def local(self, role: str, ids: list[str]) -> list[float]:
        if role not in {"feedback", "validation"}:
            raise ValueError("Private training rows cannot be consulted")
        cfg = partitions(self.root)
        if not set(ids) <= set(cfg[role]):
            raise ValueError("Wrong data population")
        with np.load(self.root / role / "features.npz") as data:
            lookup = {v: i for i, v in enumerate(data["ids"].tolist())}
            x = data["x"][[lookup[n] for n in ids], 2 * self.index : 2 * self.index + 2]
        return self.probabilities(x).tolist()

    def answer(self, request: dict) -> dict:
        fields = {
            "origin",
            "destination",
            "patient_ids",
            "adapter_id",
            "checkpoint_sha256",
            "task",
            "mode",
            "authorized_features",
        }
        if request.get("mode") == "new_patient":
            fields.add("features")
        if set(request) != fields:
            raise RegistryError("Unexpected payload fields")
        if (
            request["destination"] != self.spec.specialist_id
            or request["origin"] not in IDS
            or request["origin"] == request["destination"]
        ):
            raise RegistryError("Wrong recipient/origin")
        if (
            request["adapter_id"] != self.spec.model_adapter.adapter_id
            or request["checkpoint_sha256"] != self.spec.model_adapter.checkpoint_sha256
        ):
            raise RegistryError("Incompatible checkpoint identity")
        if TaskSchema.model_validate(request["task"]) != self.spec.task:
            raise RegistryError("Incompatible task/horizon")
        names = self.spec.model_adapter.feature_names
        if request["authorized_features"] != list(names):
            raise RegistryError("Unauthorized required feature schema")
        ids = request["patient_ids"]
        if not ids or len(ids) != len(set(ids)):
            raise RegistryError("Invalid patient alignment")
        if request["mode"] == "new_patient":
            if len(ids) != 1 or set(request["features"]) != set(names):
                raise RegistryError("Wrong/missing/unauthorized input fields")
            for feature in self.spec.feature_schema:
                feature.validate_value(request["features"][feature.name])
            p = self.probabilities(np.array([[request["features"][n] for n in names]])).tolist()
        elif request["mode"] == "aligned":
            cfg = partitions(self.root)
            role = "feedback" if set(ids) <= set(cfg["feedback"]) else "validation"
            p = self.local(role, ids)
        else:
            raise RegistryError("Unsupported request mode")
        return {
            k: request[k] for k in ("origin", "destination", "patient_ids", "adapter_id", "checkpoint_sha256")
        } | {"probabilities": p}


def controller(root: Path, origin: int) -> ContinualController:
    cfg = partitions(root)
    local = root / f"origin{origin}"
    return ContinualController(
        local, bundle(root), FeedbackLedger(local / "ledger", origin, set(cfg["feedback"]), *KEYS)
    )


def historical_compatibility(root: Path) -> dict[str, str]:
    """Exercise configured loading against retained CDC A/B/C identities.

    The raw CDC test is never touched. These checks load only Step 7 shared policy
    metadata/weights and require the existing pair compatibility guard to reject
    mixing CDC and synthetic identities, tasks, adapters and checkpoints.
    """
    cdc = Path("artifacts/m5-step07-v1")
    if (
        not all((cdc / component / "r000.pt").exists() for component in ("routers",))
        or not (cdc / "fusions/f000.pt").exists()
    ):
        raise FileNotFoundError("Retained Step 7 policy checkpoints required for compatibility test")
    checks = {}
    combinations = {
        "cdc_router_synthetic_fusion": PolicyPair(
            root / "compatibility/cdc-router",
            PolicyRegistry(cdc / "routers"),
            PolicyRegistry(root / "fusions"),
        ),
        "synthetic_router_cdc_fusion": PolicyPair(
            root / "compatibility/cdc-fusion",
            PolicyRegistry(root / "routers"),
            PolicyRegistry(cdc / "fusions"),
        ),
    }
    for name, pair in combinations.items():
        checks[name] = expect_rejection(lambda pair=pair: pair.load("r000", "f000"))
    cdc_declaration = read(cdc / "fusions/f000.json")["fusion_config"]["declarations"][0]
    checks["cdc_task_registration"] = expect_rejection(
        lambda: registry(root).register(Specialist.model_validate(cdc_declaration), node_id=99)
    )
    write(root / "compatibility.json", checks)
    return checks


def controlled_lifecycle(root: Path) -> dict[str, Any]:
    """Exercise approval, rejection, restart recovery and exact rollback fixtures."""
    target = root / "controlled-fixtures"
    pair = PolicyPair(target / "pair", PolicyRegistry(target / "routers"), PolicyRegistry(target / "fusions"))
    router, fusion = bundle(root).load("r000", "f000")
    pair.routers.save("r000", snapshot(router), {"eligible": True, "router_config": router.config()})
    pair.fusions.save("f000", snapshot(fusion), {"eligible": True, "fusion_config": fusion.config()})
    pair.activate("r000", "f000", reason="controlled compatible baseline")
    pair.fusions.save("fRejected", snapshot(fusion), {"eligible": False, "fusion_config": fusion.config()})
    rejected = expect_rejection(
        lambda: pair.activate("r000", "fRejected", reason="controlled rejection fixture")
    )
    approved = copy.deepcopy(fusion)
    with torch.no_grad():
        approved.head[-1].bias.add_(0.001)
    pair.fusions.save("fApproved", snapshot(approved), {"eligible": True, "fusion_config": approved.config()})
    pair.activate("r000", "fApproved", reason="controlled approval fixture; no quality claim")
    restarted = PolicyPair(
        target / "pair", PolicyRegistry(target / "routers"), PolicyRegistry(target / "fusions")
    )
    recovered = restarted.current()
    restarted.rollback("r000", "f000")
    restored_router, restored_fusion = restarted.load()
    result = {
        "rejection": rejected,
        "recovered": recovered,
        "rollback": restarted.current(),
        "router_exact": distance(snapshot(router), snapshot(restored_router)) == 0,
        "fusion_exact": distance(snapshot(fusion), snapshot(restored_fusion)) == 0,
    }
    write(root / "controlled-fixtures.json", result)
    return result


def replay_local(root: Path, origin: int, component: str, initial: dict) -> tuple[Any, dict]:
    """Replay an authenticated local update after production-style consumption."""
    ctrl = controller(root, origin)
    records = ctrl.ledger.replay(2, window=max(1, len(ctrl.ledger.predictions)))
    contexts_ = {}
    provenance = {}
    for event, _ in records:
        saved = ctrl._read(event.request_id)
        if saved.get("event_digest") != digest(event):
            raise ValueError("Prediction/context digest mismatch")
        contexts_[event.request_id] = saved["context"]
        provenance[event.request_id] = saved
    router, fusion = ctrl.pair.load()
    model = router if component == "router" else fusion
    model.load_state_dict(initial, strict=True)
    log = (
        model.update(records, contexts_, minimum=64, epochs=2)
        if component == "router"
        else fusion_feedback_update(model, records, provenance, minimum=64, epochs=2)
    )
    return model, log | {"request_ids": [event.request_id for event, _ in records]}


def contexts(own: float) -> dict:
    """Only own prediction and public cost; no hidden peer prediction or label."""
    return {n: [own, 2 * abs(own - 0.5), 0.02] for n in IDS}


def audit_record(request: dict) -> dict:
    """Default public trace omits feature values, labels, predictions and weights.

    IDs remain pseudonymous identifiers, not anonymous. Signed origin ledgers
    are private training storage, not public logs, and necessarily hold outcomes.
    """
    return {
        "origin": request["origin"],
        "destination": request["destination"],
        "ids": request["patient_ids"],
        "mode": request["mode"],
        "keys": sorted(request),
        "feature_names": sorted(request.get("features", {})),
    }


def network_audit(root: Path) -> dict:
    """Observe listeners owned by this simulation subtree on Windows.

    Ray's node IP setting is not treated as proof of loopback binding. Wildcard
    listeners are reported as a failed local-only service check, never hidden.
    """
    command = (
        "$taskProcesses = @(Get-CimInstance Win32_Process); "
        f"$taskIds = @({os.getpid()}); "
        "for ($taskPass = 0; $taskPass -lt 8; $taskPass++) { "
        "$taskIds += @($taskProcesses | Where-Object { $_.ParentProcessId -in $taskIds } | "
        "Select-Object -ExpandProperty ProcessId); $taskIds = @($taskIds | Select-Object -Unique) }; "
        "@(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | "
        "Where-Object { $_.OwningProcess -in $taskIds } | "
        "Select-Object LocalAddress,LocalPort,OwningProcess) | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=False
    )
    listeners = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(listeners, dict):
        listeners = [listeners]
    audit = {
        "listeners": listeners,
        "loopback_only": bool(listeners)
        and all(r["LocalAddress"] in {"127.0.0.1", "::1"} for r in listeners),
        "command_returncode": result.returncode,
        "scope": "Simulation driver and its process descendants; OS firewall reachability not tested",
    }
    write(root / "network.json", audit)
    return audit


def expect_rejection(action: Any) -> str:
    try:
        action()
    except (ValueError, KeyError, TypeError, TimeoutError, RuntimeError) as error:
        return type(error).__name__
    raise AssertionError("Invalid operation was accepted")


def evaluate(
    root: Path, router: RoutingPolicy, fusion: DynamicFusion, *, actions: list | None = None
) -> tuple[list, list]:
    """Held-out evaluation, called only after candidate training in the real run."""
    cfg = partitions(root)
    ids = cfg["validation"]
    y = np.load(root / "validation/outcomes.npy")
    p = np.array([ExternalAdapter(root, i).local("validation", ids) for i in range(3)]).T
    reg = registry(root)
    rows, all_actions = [], []
    for origin, name in enumerate(IDS):
        risks, selected = [], []
        for j, patient in enumerate(ids):
            req = session(reg, name, (patient,))
            target = (
                actions[origin][j]
                if actions is not None
                else router.route(
                    req, contexts(p[j, origin]), float(p[j, origin]), version="evaluation"
                ).destination
            )
            names = [name] + ([] if target is None else [target])
            observed = [
                FusionEvidence(
                    n,
                    patient,
                    float(p[j, IDS.index(n)]),
                    reg.get(n).specialist.model_adapter.checkpoint_sha256,
                )
                for n in names
            ]
            risks.append(fusion.fuse(observed, req))
            selected.append(target)
        risk = np.array(risks)
        calls = np.array([n is not None for n in selected])
        bins = np.minimum((risk * 10).astype(int), 9)
        ece = sum(
            float(np.mean(bins == b) * abs(y[bins == b].mean() - risk[bins == b].mean()))
            for b in range(10)
            if (bins == b).any()
        )
        bce = float(log_loss(y, risk, labels=[0, 1]))
        rows.append(
            {
                "origin": origin,
                "binary_cross_entropy": bce,
                "auroc": float(roc_auc_score(y, risk)),
                "pr_auc": float(average_precision_score(y, risk)),
                "brier": float(brier_score_loss(y, risk)),
                "ece": ece,
                "accuracy": float(((risk >= 0.5) == y).mean()),
                "calls": float(calls.mean()),
                "stop": float((~calls).mean()),
                "cost": float(calls.mean() * 0.02),
                "objective": bce + float(calls.mean() * 0.02),
                "selections": {"STOP" if n is None else n: selected.count(n) for n in (None, *IDS)},
            }
        )
        all_actions.append(selected)
    return rows, all_actions


def flower_worker(root: Path) -> None:
    """Run real three-node Flower transport and compatible shared FedAvg on CPU."""
    from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
    from flwr.serverapp import ServerApp
    from flwr.serverapp.strategy import FedAvg
    from flwr.simulation import run_simulation

    torch.set_num_threads(1)
    cfg = partitions(root)
    client, server = ClientApp(), ServerApp()

    @client.query("capabilities")
    def capabilities(message: Message, ctx: Context) -> Message:
        i = int(ctx.node_config["partition-id"])
        return Message(
            RecordDict(
                {
                    "response": ConfigRecord(
                        {"index": i, "declaration": json.dumps(read(root / "declarations.json")[i])}
                    )
                }
            ),
            reply_to=message,
        )

    @client.query("predict")
    def predict(message: Message, ctx: Context) -> Message:
        i = int(ctx.node_config["partition-id"])
        request = json.loads(message.content["config"]["request"])
        answer = ExternalAdapter(root, i).answer(request)
        receipt = audit_record(request)
        with (root / f"receipts{i}.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(receipt) + "\n")
        return Message(RecordDict({"response": ConfigRecord({"json": json.dumps(answer)})}), reply_to=message)

    @client.query("prepare")
    def prepare(message: Message, ctx: Context) -> Message:
        i = int(ctx.node_config["partition-id"])
        ctrl = controller(root, i)
        reg = registry(root)
        own = ExternalAdapter(root, i).local("feedback", cfg["feedback"])
        plans = [
            ctrl.prepare(f"o{i}-{patient}", session(reg, IDS[i], (patient,)), contexts(p), p, episode=1)
            for patient, p in zip(cfg["feedback"], own, strict=True)
        ]
        return Message(RecordDict({"response": ConfigRecord({"plans": json.dumps(plans)})}), reply_to=message)

    @client.query("finish")
    def finish(message: Message, ctx: Context) -> Message:
        i = int(ctx.node_config["partition-id"])
        replies = json.loads(message.content["config"]["replies"])
        ctrl = controller(root, i)
        reg = registry(root)

        def actual_receipt(entry: Any, request: dict) -> dict:
            response = replies[entry.specialist.specialist_id]
            if any(
                response[k] != request[k]
                for k in ("origin", "destination", "adapter_id", "checkpoint_sha256")
            ):
                raise ValueError("Misassociated batched Flower receipt")
            ids = request["patient_ids"]
            return {
                k: request[k]
                for k in ("origin", "destination", "patient_ids", "adapter_id", "checkpoint_sha256")
            } | {"probabilities": [response["probabilities"][response["patient_ids"].index(n)] for n in ids]}

        for patient in cfg["feedback"]:
            ctrl.finish(f"o{i}-{patient}", session(reg, IDS[i], (patient,)), actual_receipt)
        return Message(
            RecordDict({"response": ConfigRecord({"predictions": len(ctrl.ledger.predictions)})}),
            reply_to=message,
        )

    @client.query("release")
    def release(message: Message, ctx: Context) -> Message:
        i = int(ctx.node_config["partition-id"])
        ctrl = controller(root, i)
        labels = np.load(root / "feedback/outcomes.npy")
        for j, patient in enumerate(cfg["feedback"]):
            if j % 10 == 0:
                continue
            event = ctrl.ledger.predictions[f"o{i}-{patient}"]
            outcome = VerifiedOutcome(
                request_id=event.request_id,
                patient_id=patient,
                prediction_digest=digest(event),
                label=int(labels[j]),
                authority="simulation-training-label",
                received_episode=2,
                verified_timestamp=(
                    datetime.fromisoformat(event.event_timestamp) + timedelta(seconds=1)
                ).isoformat(),
            )
            ctrl.receive(sign(outcome, KEYS[1]))
        assert not ctrl.pending("fusion", 1, minimum=1)[0]
        return Message(
            RecordDict({"response": ConfigRecord({"verified": len(ctrl.ledger.outcomes)})}), reply_to=message
        )

    @client.query("commit")
    def commit(message: Message, ctx: Context) -> Message:
        i = int(ctx.node_config["partition-id"])
        component = str(message.content["config"]["component"])
        request_ids = json.loads(message.content["config"]["request_ids"])
        controller(root, i).commit(component, request_ids)
        return Message(
            RecordDict({"response": ConfigRecord({"component": component, "consumed": len(request_ids)})}),
            reply_to=message,
        )

    @client.query("slow")
    def slow(message: Message, ctx: Context) -> Message:
        time.sleep(float(message.content["config"]["seconds"]))
        return Message(
            RecordDict(
                {"response": ConfigRecord({"index": int(ctx.node_config["partition-id"]), "status": "late"})}
            ),
            reply_to=message,
        )

    @client.train()
    def train(message: Message, ctx: Context) -> Message:
        torch.set_num_threads(1)
        i = int(ctx.node_config["partition-id"])
        component = message.content["config"]["component"]
        ctrl = controller(root, i)
        model = ctrl.pair.load()[component == "fusion"]
        if json.loads(message.content["config"]["contract"]) != model.config():
            raise ValueError("Incompatible shared parameter contract")
        initial = message.content["arrays"].to_torch_state_dict()
        start = time.perf_counter()
        model, log = ctrl.train_local(component, 2, minimum=64, epochs=2, device="cpu", state=initial)
        final = snapshot(model)
        folder = root / component
        folder.mkdir(exist_ok=True)
        torch.save(final, folder / f"local{i}.pt")
        write(
            folder / f"local{i}.json",
            log
            | {
                "seconds": time.perf_counter() - start,
                "drift": distance(initial, final),
                "device": "cpu",
                "shared_bytes": sum(v.numel() * v.element_size() for v in final.values()),
            },
        )
        return Message(
            RecordDict(
                {
                    "arrays": ArrayRecord(final),
                    "metrics": MetricRecord({"num-examples": len(log["request_ids"]), "origin": i}),
                }
            ),
            reply_to=message,
        )

    @server.main()
    def main(grid: Any, ctx: Context) -> None:
        tracemalloc.start()
        start = time.perf_counter()
        traffic = []

        def query(node: int, topic: str, content: dict, *, timeout: float = 60) -> dict:
            then = time.perf_counter()
            replies = list(
                grid.send_and_receive(
                    [
                        Message(
                            RecordDict({"config": ConfigRecord(content)}),
                            dst_node_id=node,
                            message_type=f"query.{topic}",
                        )
                    ],
                    timeout=timeout,
                )
            )
            valid = (
                len(replies) == 1 and not replies[0].has_error() and replies[0].metadata.src_node_id == node
            )
            answer = dict(replies[0].content["response"]) if valid else {}
            traffic.append(
                {
                    "node": node,
                    "topic": topic,
                    "keys": sorted(content),
                    "seconds": time.perf_counter() - then,
                    "request_json_bytes": len(json.dumps(content).encode()),
                    "reply_json_bytes": len(json.dumps(answer).encode()),
                    "status": "ok" if valid else "timeout_or_error",
                }
            )
            if not valid:
                raise TimeoutError("Flower query failed or timed out; no fallback consultation")
            return answer

        nodes = {}
        deadline = time.perf_counter() + 60
        while len(list(grid.get_node_ids())) < 3:
            if time.perf_counter() >= deadline:
                raise RuntimeError("Three Flower clients did not connect")
            time.sleep(0.05)
        for node in grid.get_node_ids():
            info = query(node, "capabilities", {})
            assert json.loads(info["declaration"]) == read(root / "declarations.json")[info["index"]]
            nodes[IDS[info["index"]]] = node
        assert len(nodes) == 3
        network = network_audit(root)
        if not network["loopback_only"]:
            raise RuntimeError("Simulation process listeners are not exclusively loopback-bound")
        reg = registry(root, nodes)
        lifecycle = []

        def mark(action: str) -> None:
            lifecycle.append(
                {
                    "action": action,
                    "registered": [e.specialist.specialist_id for e in reg.discover()],
                    "eligible": [
                        e.specialist.specialist_id
                        for e in session(reg, IDS[0], (cfg["validation"][0],)).eligible()
                    ],
                }
            )

        def transport(entry: Any, payload: dict) -> dict:
            return json.loads(query(entry.node_id, "predict", {"request": json.dumps(payload)})["json"])

        removed = reg.unregister(IDS[2])
        mark("register first two")
        reg.register(removed.specialist, node_id=removed.node_id)
        mark("register third")
        patient = cfg["validation"][0]
        own = ExternalAdapter(root, 0).local("validation", [patient])[0]
        mean = DynamicFusion("mean", read(root / "declarations.json"))
        req = session(reg, IDS[0], (patient,))
        origin_evidence = FusionEvidence(
            IDS[0], patient, own, reg.get(IDS[0]).specialist.model_adapter.checkpoint_sha256
        )
        assert mean.fuse([origin_evidence], req) == own
        stop = RoutingPolicy("fixed", IDS, 3)
        assert stop.route(req, {}, own, version="stop").destination is None
        response = req.consult(IDS[1], transport)
        observed = [origin_evidence, FusionEvidence.from_reply(response, patient)]
        assert mean.fuse(observed, req) == sum(e.probability for e in observed) / 2
        assert mean.fuse(reversed(observed), req) == mean.fuse(observed, req)
        lifecycle_rejections = {
            "maximum_calls": expect_rejection(lambda: req.consult(IDS[2], transport)),
            "maximum_cost": expect_rejection(
                lambda: session(reg, IDS[0], (patient,), max_cost=0).consult(IDS[1], transport)
            ),
            "authorization": expect_rejection(
                lambda: session(reg, IDS[0], (patient,), permitted=frozenset()).consult(IDS[1], transport)
            ),
        }
        reg.update(IDS[1], available=False)
        mark("disable second")
        lifecycle_rejections["unavailable"] = expect_rejection(
            lambda: session(reg, IDS[0], (patient,)).consult(IDS[1], transport)
        )
        reg.update(IDS[1], available=True)
        mark("restore second")
        session(reg, IDS[0], (patient,)).consult(IDS[1], transport)
        reg.unregister(IDS[2])
        mark("unregister third")
        lifecycle_rejections["unregistered"] = expect_rejection(
            lambda: session(reg, IDS[0], (patient,)).consult(IDS[2], transport)
        )
        reg.register(removed.specialist, node_id=removed.node_id)
        mark("reregister third")
        # Authorized synthetic new-patient values only; never a training row.
        with np.load(root / "validation/features.npz") as data:
            values = {f"synthetic_{j}": float(data["x"][0, j]) for j in range(6)}
        values["unrelated-secret"] = 999
        new_reply = session(reg, IDS[0], (patient,)).consult(IDS[2], transport, features=values)
        assert new_reply["probabilities"] == ExternalAdapter(root, 2).local("validation", [patient])
        lifecycle_rejections["timeout"] = expect_rejection(
            lambda: query(nodes[IDS[1]], "slow", {"seconds": 0.25}, timeout=0.01)
        )
        write(root / "lifecycle.json", lifecycle)
        # Only outcome-free plans and actual selected-peer replies cross Flower.
        plans = {n: json.loads(query(nodes[n], "prepare", {})["plans"]) for n in IDS}
        action_counts = {"STOP": 0, **dict.fromkeys(IDS, 0)}
        for origin in IDS:
            for plan in plans[origin]:
                action_counts["STOP" if plan["selected"] is None else plan["selected"]] += 1
        for origin in IDS:
            responses = {}
            for peer in IDS:
                selected = tuple(p["patient_id"] for p in plans[origin] if p["selected"] == peer)
                if selected:
                    responses[peer] = session(reg, origin, selected).consult(peer, transport)
            query(nodes[origin], "finish", {"replies": json.dumps(responses)})
        counts = [query(nodes[n], "release", {})["verified"] for n in IDS]
        assert min(counts) >= 64
        pair = bundle(root)
        router, fusion = pair.load()
        candidates = {}
        for component, model in (("router", router), ("fusion", fusion)):
            folder = root / component
            folder.mkdir(exist_ok=True)
            torch.save(snapshot(model), folder / "initial.pt")
            result = FedAvg(
                fraction_train=1, fraction_evaluate=0, min_train_nodes=3, min_available_nodes=3
            ).start(
                grid,
                initial_arrays=ArrayRecord(snapshot(model)),
                num_rounds=1,
                train_config=ConfigRecord(
                    {"component": component, "contract": json.dumps(model.config(), sort_keys=True)}
                ),
                timeout=120,
            )
            state = result.arrays.to_torch_state_dict()
            assert all(torch.isfinite(v).all() for v in state.values())
            candidate = copy.deepcopy(model)
            candidate.load_state_dict(state, strict=True)
            torch.save(state, folder / "aggregate.pt")
            candidates[component] = candidate
            for i, name in enumerate(IDS):
                local_log = read(folder / f"local{i}.json")
                committed = query(
                    nodes[name],
                    "commit",
                    {"component": component, "request_ids": json.dumps(local_log["request_ids"])},
                )
                assert committed == {
                    "component": component,
                    "consumed": len(local_log["request_ids"]),
                }
        # Both candidates are now frozen. This is the first validation-label read.
        previous, actions = evaluate(root, router, fusion)
        fixed = RoutingPolicy("fixed", IDS, 3)
        fixed.gains.fill_(0.15)
        baseline, _ = evaluate(root, fixed, fusion)
        mean_scores, _ = evaluate(root, router, mean, actions=actions)
        router_scores, _ = evaluate(root, candidates["router"], fusion)
        fusion_scores, _ = evaluate(root, router, candidates["fusion"], actions=actions)
        gates = {
            "router": routing_update_gates(previous, baseline, router_scores, counts, minimum_feedback=64),
            "fusion": routing_update_gates(previous, mean_scores, fusion_scores, counts, minimum_feedback=64),
        }
        gates["fusion"]["permutation"] = permutation_check(root, candidates["fusion"])
        r, f = "r000", "f000"
        for component, version, key in (
            ("router", "r001", "router_config"),
            ("fusion", "f001", "fusion_config"),
        ):
            eligible = all(gates[component].values())
            store = pair.routers if component == "router" else pair.fusions
            store.save(
                version,
                snapshot(candidates[component]),
                {"eligible": eligible, key: candidates[component].config(), "gates": gates[component]},
            )
            if eligible:
                if component == "router":
                    r = version
                else:
                    f = version
        joint, _ = evaluate(root, pair.routers.load(r), pair.fusions.load(f))
        joint_gates = routing_update_gates(previous, baseline, joint, counts, minimum_feedback=64)
        if all(joint_gates.values()):
            pair.activate(r, f, reason="frozen synthetic held-out gates")
        activated = pair.current()
        pair.rollback("r000", "f000")
        restored, _ = evaluate(root, *pair.load())
        assert restored == previous
        fixture = controlled_lifecycle(root)
        request_bytes = sum(row["request_json_bytes"] for row in traffic)
        reply_bytes = sum(row["reply_json_bytes"] for row in traffic)
        latencies = sorted(row["seconds"] for row in traffic)
        local_logs = [
            read(root / component / f"local{i}.json") for component in ("router", "fusion") for i in range(3)
        ]
        write(
            root / "results.json",
            {
                "run_id": ctx.run_id,
                "physical_clients": 3,
                "cpu_workers": 3,
                "physical_node_ids": nodes,
                "FedAvg_rounds": 2,
                "local_updates": 6,
                "verified_per_origin": counts,
                "predictions": 360,
                "missing_outcomes": 36,
                "routing_actions": action_counts,
                "stop_rate": action_counts["STOP"] / 360,
                "recipient_distribution": {name: action_counts[name] for name in IDS},
                "consultation_cost": (360 - action_counts["STOP"]) * cfg["cost"],
                "selected_only_feedback_batches": sum(
                    any(plan["selected"] == peer for plan in plans[origin]) for origin in IDS for peer in IDS
                ),
                "lifecycle_rejections": lifecycle_rejections,
                "gates": gates,
                "joint_gates": joint_gates,
                "activated": activated,
                "rollback": pair.current(),
                "validation": {
                    "previous": previous,
                    "fixed": baseline,
                    "mean_matched": mean_scores,
                    "router_candidate": router_scores,
                    "fusion_candidate": fusion_scores,
                    "joint": joint,
                },
                "traffic": traffic,
                "application_bytes": {"request": request_bytes, "reply": reply_bytes},
                "latency_seconds": {
                    "minimum": latencies[0],
                    "median": latencies[len(latencies) // 2],
                    "maximum": latencies[-1],
                    "scope": "whole batched application request/reply including Flower and local inference",
                },
                "shared_parameter_bytes": {
                    component: read(root / component / "local0.json")["shared_bytes"]
                    for component in ("router", "fusion")
                },
                "local_update_seconds": sum(log["seconds"] for log in local_logs),
                "controlled_fixtures": fixture,
                "network": network,
                "seconds": time.perf_counter() - start,
                "server_python_peak_bytes": tracemalloc.get_traced_memory()[1],
            },
        )
        tracemalloc.stop()

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=3,
        backend_config={
            "client_resources": {"num_cpus": 1, "num_gpus": 0},
            "init_args": {
                "num_cpus": 3,
                "num_gpus": 0,
                "include_dashboard": False,
                "_node_ip_address": "127.0.0.1",
                "object_store_memory": 256 * 1024 * 1024,
            },
        },
    )


def permutation_check(root: Path, model: DynamicFusion) -> bool:
    reg = registry(root)
    patient = partitions(root)["validation"][0]
    req = session(reg, IDS[0], (patient,))
    observed = [
        FusionEvidence(n, patient, 0.2 + 0.2 * i, reg.get(n).specialist.model_adapter.checkpoint_sha256)
        for i, n in enumerate(IDS)
    ]
    return model.fuse(observed, req) == model.fuse(reversed(observed), req)


def verify(root: Path) -> dict:
    """Independent optimizer, aggregate, numerical, event and preservation replay."""
    torch.set_num_threads(1)
    cfg = partitions(root)
    assert sha(root / "protocol.json") == read(root / "execution.json")["protocol_sha256"]
    results = read(root / "results.json")
    generated_x, generated_y = generated()
    generated_index = {case_id(i): i for i in range(600)}
    for role in ("training", "feedback", "validation"):
        indexes = np.array([generated_index[patient] for patient in cfg[role]])
        with np.load(root / role / "features.npz") as values:
            assert values["ids"].tolist() == cfg[role]
            np.testing.assert_array_equal(values["x"], generated_x[indexes])
        np.testing.assert_array_equal(np.load(root / role / "outcomes.npy"), generated_y[indexes])
    labels = np.load(root / "feedback/outcomes.npy")
    replayed = 0
    for i, name in enumerate(IDS):
        ctrl = controller(root, i)
        assert len(ctrl.ledger.predictions) == 120 and len(ctrl.ledger.outcomes) == 108
        adapter = ExternalAdapter(root, i)
        own = adapter.local("feedback", cfg["feedback"])
        for j, patient in enumerate(cfg["feedback"]):
            event = ctrl.ledger.predictions[f"o{i}-{patient}"]
            saved = ctrl._read(event.request_id)
            router, fusion = ctrl.pair.load(event.policy_version, event.model_version)
            req = session(registry(root), name, (patient,))
            decision = router.route(req, contexts(own[j]), own[j], version=event.policy_version)
            assert decision.destination == saved["selected"]
            np.testing.assert_allclose(event.evidence[i], own[j], rtol=0, atol=1e-15)
            observed = [
                FusionEvidence(
                    name,
                    patient,
                    float(event.evidence[i]),
                    adapter.spec.model_adapter.checkpoint_sha256,
                )
            ]
            for peer in event.consulted:
                p = ExternalAdapter(root, peer).local("feedback", [patient])[0]
                np.testing.assert_allclose(event.evidence[peer], p, rtol=0, atol=1e-15)
                observed.append(
                    FusionEvidence(
                        IDS[peer],
                        patient,
                        float(event.evidence[peer]),
                        registry(root).get(IDS[peer]).specialist.model_adapter.checkpoint_sha256,
                    )
                )
            assert fusion.fuse(observed, req) == event.final_risk
            if j % 10 == 0:
                assert event.request_id not in ctrl.ledger.outcomes
            else:
                outcome = ctrl.ledger.outcomes[event.request_id]
                assert outcome.label == int(labels[j]) and outcome.prediction_digest == digest(event)
            replayed += 1
    for component in ("router", "fusion"):
        folder = root / component
        initial = torch.load(folder / "initial.pt", weights_only=True)
        locals_ = []
        for i in range(3):
            ctrl = controller(root, i)
            assert not ctrl.pending(component, 2, minimum=1)[0]
            model, log = replay_local(root, i, component, initial)
            saved = torch.load(folder / f"local{i}.pt", weights_only=True)
            assert distance(snapshot(model), saved) < 1e-7
            assert distance(initial, saved) > 0
            assert log["request_ids"] == read(folder / f"local{i}.json")["request_ids"]
            assert (
                expect_rejection(
                    lambda ctrl=ctrl, component=component, request_ids=log["request_ids"]: ctrl.commit(
                        component, request_ids
                    )
                )
                == "ValueError"
            )
            locals_.append(saved)
        averaged = {k: sum(s[k] / 3 for s in locals_) for k in initial}
        aggregate = torch.load(folder / "aggregate.pt", weights_only=True)
        assert distance(averaged, aggregate) < 1e-6
    pair = bundle(root)
    router, fusion = pair.load("r000", "f000")
    previous, actions = evaluate(root, router, fusion)
    fixed = RoutingPolicy("fixed", IDS, 3)
    fixed.gains.fill_(0.15)
    baseline, _ = evaluate(root, fixed, fusion)
    mean, _ = evaluate(root, router, DynamicFusion("mean", read(root / "declarations.json")), actions=actions)
    rc, _ = evaluate(root, pair.routers.load("r001"), fusion)
    fc, _ = evaluate(root, router, pair.fusions.load("f001"), actions=actions)
    for key, value in (
        ("previous", previous),
        ("fixed", baseline),
        ("mean_matched", mean),
        ("router_candidate", rc),
        ("fusion_candidate", fc),
    ):
        assert value == results["validation"][key]
    counts = results["verified_per_origin"]
    assert (
        routing_update_gates(previous, baseline, rc, counts, minimum_feedback=64)
        == results["gates"]["router"]
    )
    assert (
        routing_update_gates(previous, mean, fc, counts, minimum_feedback=64)
        | {"permutation": permutation_check(root, pair.fusions.load("f001"))}
        == results["gates"]["fusion"]
    )
    # Every actual receipt must belong to a selected batch or one named lifecycle call.
    receipts = [
        json.loads(line)
        for i in range(3)
        for line in (root / f"receipts{i}.jsonl").read_text().splitlines()
        if line
    ]
    for receipt in receipts:
        assert receipt["origin"] != receipt["destination"]
        assert not {"label", "labels", "weights", "x", "y"} & set(receipt["keys"])
        if receipt["mode"] == "aligned":
            assert not receipt["feature_names"]
        for patient in set(receipt["ids"]) & set(cfg["feedback"]):
            origin = IDS.index(receipt["origin"])
            event = controller(root, origin).ledger.predictions[f"o{origin}-{patient}"]
            assert IDS.index(receipt["destination"]) in event.consulted
    compatibility = read(root / "compatibility.json")
    assert set(compatibility) == {
        "cdc_router_synthetic_fusion",
        "synthetic_router_cdc_fusion",
        "cdc_task_registration",
    }
    assert all(value in {"ValueError", "RuntimeError", "RegistryError"} for value in compatibility.values())
    fixture = read(root / "controlled-fixtures.json")
    assert fixture["rollback"] == {"router": "r000", "fusion": "f000"}
    assert fixture["router_exact"] and fixture["fusion_exact"]
    changed = []
    for manifest in (
        "historical-hashes.json",
        "core-hashes.json",
        "private-hashes.json",
        "attempt-hashes.json",
    ):
        changed += [p for p, h in read(root / manifest).items() if not Path(p).exists() or sha(Path(p)) != h]
    assert not changed
    result = {
        "decision_fusion_replays": replayed,
        "verified_origin_records": 324,
        "local_optimizer_replays": 6,
        "FedAvg_replays": 2,
        "receipt_batches": len(receipts),
        "deterministic_split_and_data_replays": 600,
        "compatibility_rejections": compatibility,
        "feedback_consumed_once_per_component": 648,
        "controlled_fixture": fixture,
        "historical_files": len(read(root / "historical-hashes.json")),
        "prior_attempt_files": len(read(root / "attempt-hashes.json")),
        "changed": changed,
        "core_source_unchanged": True,
        "frozen_test_opened": False,
        "frozen_test_evaluations": 0,
        "privacy_scope": "Trusted local simulation only",
    }
    write(root / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--qa", action="store_true")
    args = parser.parse_args()
    if args.qa:
        run_checks(args.output)
        execution = read(args.output / "execution.json")
        execution.update(status="complete", qa=read(args.output / "qa/checks.json"))
        write(args.output / "execution.json", execution)
    elif args.verify:
        verification = verify(args.output)
        execution = read(args.output / "execution.json")
        execution.update(status="experiment-and-replay-complete", verification=verification)
        write(args.output / "execution.json", execution)
        print(json.dumps(verification, indent=2))
    elif args.worker:
        flower_worker(args.output.resolve())
    else:
        initialize(args.output)
        historical_compatibility(args.output)
        with (args.output / "flower.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.validate_step08a",
                    "--worker",
                    "--output",
                    str(args.output.resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ | {"PYTHONUTF8": "1"},
                check=True,
            )
        execution = read(args.output / "execution.json")
        execution.update(
            status="experiment-complete-pending-independent-replay",
            completed_experiment=datetime.now().astimezone().isoformat(),
            retained_failures=execution["retained_failures"]
            + ["attempts/initialization-only-20261015/flower.log"],
            result=read(args.output / "results.json"),
        )
        write(args.output / "execution.json", execution)
        print(json.dumps(read(args.output / "results.json"), indent=2))


def run_checks(root: Path) -> None:
    """Run full regression without reopening historical runtime experiments."""
    commands = {
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "ruff": [sys.executable, "-m", "ruff", "check", "--no-cache", "src", "tests", "scripts"],
        "format": [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--no-cache",
            "--check",
            "src",
            "tests",
            "scripts",
        ],
        "dependencies": [sys.executable, "-m", "pip", "check"],
        "freeze": [sys.executable, "-m", "pip", "freeze"],
    }
    reports = {}
    folder = root / "qa"
    folder.mkdir(exist_ok=True)
    env = os.environ | {
        "SWARMFL_REAL_STEP08A": "1",
        "SWARMFL_REAL_REGISTRY": "0",
        "SWARMFL_REAL_FEEDBACK": "0",
        "PYTHONUTF8": "1",
    }
    for name, cmd in commands.items():
        start = time.perf_counter()
        result = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace", env=env, check=False
        )
        (folder / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        reports[name] = {
            "command": cmd,
            "returncode": result.returncode,
            "seconds": time.perf_counter() - start,
        }
        write(folder / "checks.json", reports)
        print(name, result.returncode, flush=True)
    assert all(v["returncode"] == 0 for v in reports.values())


if __name__ == "__main__":
    main()
