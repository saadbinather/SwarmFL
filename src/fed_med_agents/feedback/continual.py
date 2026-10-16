"""Registry-authoritative prediction, authenticated feedback and consumption orchestration.

Single trusted writer per origin. No remote clinical verification or privacy claim.
M4 ledger schema and digests remain unchanged; signed local sidecars carry new
identity/context provenance without breaking historical signed events.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ..collaboration.registry import ConsultationSession, RegisteredSpecialist
from ..fusion.dynamic import DynamicFusion, FusionEvidence
from ..routing.policy import RoutingPolicy
from .events import Event, FeedbackLedger, PredictionEvent, authenticate, digest, sign, timestamp
from .versions import PolicyPair


class RequestSnapshot(Event):
    """Signed local provenance linked to the exact legacy PredictionEvent digest."""

    request_id: str
    data: dict


class ContinualController:
    """Connect approved policies, selected-only transport, M4 feedback and training.

    Context construction and independent ground-truth adjudication are supplied
    by the application. This controller never obtains a label during prediction.
    """

    def __init__(self, path: Path, pair: PolicyPair, ledger: FeedbackLedger) -> None:
        self.path, self.pair, self.ledger = path, pair, ledger
        path.mkdir(parents=True, exist_ok=True)
        (path / "requests").mkdir(exist_ok=True)
        self._recover()

    def _file(self, request_id: str) -> Path:
        import hashlib

        return self.path / "requests" / f"{hashlib.sha256(request_id.encode()).hexdigest()}.json"

    def _read(self, request_id: str) -> dict:
        record = RequestSnapshot.model_validate(
            authenticate(
                json.loads(self._file(request_id).read_text(encoding="utf-8")), self.ledger.prediction_key
            )
        )
        if record.request_id != request_id:
            raise ValueError("Misassociated context")
        return record.data

    def _write(self, request_id: str, data: dict) -> None:
        file = self._file(request_id)
        temporary = file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                sign(RequestSnapshot(request_id=request_id, data=data), self.ledger.prediction_key),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(file)

    def _recover(self) -> None:
        for request_id, event in self.ledger.predictions.items():
            saved = self._read(request_id)
            if saved.get("event_digest") != digest(event):
                raise ValueError("Feedback context integrity failure")
        if (self.path / "consumed.json").exists():
            data = authenticate(
                json.loads((self.path / "consumed.json").read_text(encoding="utf-8")),
                self.ledger.prediction_key,
            )
            for component, ids in data["data"].items():
                if component not in {"router", "fusion"} or not set(ids) <= set(self.ledger.outcomes):
                    raise ValueError("Invalid consumption state")

    def prepare(
        self,
        request_id: str,
        session: ConsultationSession,
        peer_contexts: dict[str, list[float]],
        own_probability: float,
        *,
        episode: int,
    ) -> dict:
        """Route with active approved pair and sign local pre-consultation context.

        Returns only decision/identity metadata suitable for coordinating batched
        selected-only transport; raw clinical context stays in the local sidecar.
        """
        if self._file(request_id).exists():
            raise ValueError("Duplicate request")
        if len(session.patient_ids) != 1 or session.patient_ids[0] not in self.ledger.allowed_patients:
            raise ValueError("Unauthorized patient")
        if any(e.patient_id == session.patient_ids[0] for e in self.ledger.predictions.values()):
            raise ValueError("Duplicate feedback patient")
        versions = self.pair.current()
        router, fusion = self.pair.load(versions["router"], versions["fusion"])
        identities = list(router.specialist_ids)
        if identities[self.ledger.origin] != session.origin:
            raise ValueError("Wrong local origin")
        decision = router.route(session, peer_contexts, own_probability, version=versions["router"])
        eligibility = {e.specialist.specialist_id: e for e in session.eligible()}
        origin = session.registry.get(session.origin).specialist
        # fuse() validates exact origin contract even for STOP before sidecar persistence.
        fusion.fuse(
            [
                FusionEvidence(
                    session.origin,
                    session.patient_ids[0],
                    own_probability,
                    origin.model_adapter.checkpoint_sha256,
                )
            ],
            session,
        )
        contexts = [[0.0] * router.width for _ in identities]
        costs = [0.0] * len(identities)
        available = [False] * len(identities)
        for name, entry in eligibility.items():
            index = identities.index(name)
            contexts[index] = peer_contexts.get(name, [0.0] * router.width)
            costs[index] = entry.specialist.consultation_cost
            available[index] = True
        available[self.ledger.origin] = True
        data = {
            "versions": versions,
            "patient_id": session.patient_ids[0],
            "origin": session.origin,
            "episode": episode,
            "own": own_probability,
            "selected": decision.destination,
            "scores": decision.scores,
            "propensity": 1.0,
            "context": {"features": contexts, "costs": costs},
            "available": available,
            "identities": identities,
            "fusion_contract": fusion.fingerprint(),
            "checkpoints": {
                name: s.model_adapter.checkpoint_sha256 for name, s in fusion.declarations.items()
            },
            "event_timestamp": timestamp(),
            "registry_generations": decision.registry_generations,
        }
        self._write(request_id, data)
        return {
            "request_id": request_id,
            "patient_id": data["patient_id"],
            "origin": session.origin,
            "selected": decision.destination,
            "versions": versions,
        }

    def finish(
        self,
        request_id: str,
        session: ConsultationSession,
        transport: Callable[[RegisteredSpecialist, dict], dict],
    ) -> PredictionEvent:
        """Consult only the selected peer, fuse, and sign the outcome-free M4 event."""
        if request_id in self.ledger.predictions:
            raise ValueError("Duplicate completed request")
        saved = self._read(request_id)
        if saved["versions"] != self.pair.current():
            raise ValueError("Policy changed while request pending")
        if saved["patient_id"] != session.patient_ids[0] or saved["origin"] != session.origin:
            raise ValueError("Misaligned request")
        _, fusion = self.pair.load(saved["versions"]["router"], saved["versions"]["fusion"])
        identities = saved["identities"]
        origin = self.ledger.origin
        observed = [
            FusionEvidence(
                session.origin, saved["patient_id"], saved["own"], saved["checkpoints"][session.origin]
            )
        ]
        selected = saved["selected"]
        before = session.cost
        if selected is not None:
            reply = session.consult(selected, transport)
            observed.append(FusionEvidence.from_reply(reply, saved["patient_id"]))
        probability = fusion.fuse(observed, session)
        evidence = [None] * len(identities)
        for value in observed:
            evidence[identities.index(value.specialist_id)] = value.probability
        action = 0 if selected is None else identities.index(selected) + 1
        event = PredictionEvent(
            request_id=request_id,
            patient_id=saved["patient_id"],
            origin=origin,
            episode=saved["episode"],
            available_domains=saved["available"],
            action=action,
            consulted=[] if not action else [action - 1],
            propensity=saved["propensity"],
            action_score=saved["scores"]["STOP" if selected is None else selected],
            evidence=evidence,
            final_risk=probability,
            origin_only_risk=saved["own"],
            model_version=saved["versions"]["fusion"],
            policy_version=saved["versions"]["router"],
            consultation_cost=session.cost - before,
            event_timestamp=saved["event_timestamp"],
        )
        # Sidecar first: interruption leaves an orphan, never unsigned trainable feedback.
        saved["event_digest"] = digest(event)
        self._write(request_id, saved)
        self.ledger.prediction(sign(event, self.ledger.prediction_key))
        return event

    def predict(
        self,
        request_id: str,
        session: ConsultationSession,
        peer_contexts: dict,
        own_probability: float,
        transport: Callable,
        *,
        episode: int,
    ) -> PredictionEvent:
        """One-call route/selected Flower transport/fuse/log orchestration."""
        self.prepare(request_id, session, peer_contexts, own_probability, episode=episode)
        return self.finish(request_id, session, transport)

    def receive(self, envelope: dict) -> None:
        """Delegate signature/authority/request/delay/duplicate checks to the M4 ledger."""
        self.ledger.outcome(envelope)

    def pending(self, component: str, cutoff: int, *, minimum: int = 32) -> tuple[list, dict, dict]:
        """Return fresh verified patient-origin records only, plus signed local contexts.

        A missing outcome remains pending forever and never contributes a label.
        Successful training consumes each record separately for router/fusion.
        """
        if component not in {"router", "fusion"} or minimum < 1:
            raise ValueError("Invalid training requirements")
        used = []
        if (self.path / "consumed.json").exists():
            used = authenticate(
                json.loads((self.path / "consumed.json").read_text(encoding="utf-8")),
                self.ledger.prediction_key,
            )["data"].get(component, [])
        records = [
            (e, o)
            for e, o in self.ledger.replay(cutoff, window=max(1, len(self.ledger.predictions)))
            if e.request_id not in used
        ]
        contexts = {}
        provenance = {}
        patients = set()
        for event, _ in records:
            saved = self._read(event.request_id)
            if saved.get("event_digest") != digest(event):
                raise ValueError("Prediction/context digest mismatch")
            if event.patient_id in patients:
                raise ValueError("Duplicated feedback patient")
            patients.add(event.patient_id)
            contexts[event.request_id] = saved["context"]
            provenance[event.request_id] = saved
        if len(records) < minimum:
            return [], {}, {}
        return records, contexts, provenance

    def commit(self, component: str, request_ids: list[str]) -> None:
        """After successful local/federated update, atomically consume feedback once.

        A rejected quality candidate still consumed its fitting observations.
        Pending/invalid IDs cannot be consumed. Single trusted writer required.
        """
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Duplicate feedback consumption")
        file = self.path / "consumed.json"
        data = {}
        if file.exists():
            data = authenticate(json.loads(file.read_text(encoding="utf-8")), self.ledger.prediction_key)[
                "data"
            ]
        if component not in {"router", "fusion"} or not set(request_ids) <= set(self.ledger.outcomes):
            raise ValueError("Unverified consumption")
        if set(request_ids) & set(data.get(component, [])):
            raise ValueError("Feedback already consumed")
        data[component] = data.get(component, []) + request_ids
        payload = sign(RequestSnapshot(request_id="consumption", data=data), self.ledger.prediction_key)
        temporary = file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(file)

    def train_local(
        self,
        component: str,
        cutoff: int,
        *,
        minimum: int = 32,
        epochs: int = 2,
        device: str = "cpu",
        state: dict | None = None,
    ) -> tuple[RoutingPolicy | DynamicFusion, dict]:
        """Train one independently enabled component from fresh verified records.

        Flower supplies compatible state after checking its deployment contract.
        No active version/consumption changes; commit after successful federation.
        Insufficient feedback and nontrainable baselines retain exact weights.
        """
        import torch

        from .learning import fusion_feedback_update

        records, contexts, provenance = self.pending(component, cutoff, minimum=minimum)
        router, fusion = self.pair.load()
        model = (router if component == "router" else fusion).to(device)
        if state is not None:
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Nonfinite shared candidate")
            model.load_state_dict(state, strict=True)
        log = (
            model.update(records, contexts, minimum=minimum, epochs=epochs)
            if component == "router"
            else fusion_feedback_update(model, records, provenance, minimum=minimum, epochs=epochs)
        )
        return model, log | {"request_ids": [e.request_id for e, _ in records]}
