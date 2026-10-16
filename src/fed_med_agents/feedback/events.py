"""Strict prediction/outcome separation and an origin-local authenticated ledger."""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PredictionEvent(Event):
    request_id: str = Field(min_length=1, max_length=100)
    patient_id: str
    origin: int = Field(ge=0)
    episode: int = Field(ge=1)
    available_domains: list[bool]
    action: int = Field(ge=0)
    consulted: list[int]
    propensity: float = Field(gt=0, le=1)
    action_score: float
    # None means unobserved: there is no hidden candidate probability in this event.
    evidence: list[float | None]
    final_risk: float = Field(ge=0, le=1)
    origin_only_risk: float = Field(ge=0, le=1)
    model_version: str
    policy_version: str
    consultation_cost: float = Field(ge=0)
    event_timestamp: str

    @model_validator(mode="after")
    def valid_evidence(self) -> PredictionEvent:
        n = len(self.available_domains)
        if len(self.evidence) != n or self.origin >= n or self.evidence[self.origin] is None:
            raise ValueError("Origin evidence and compatible capability dimensions required")
        if len(set(self.consulted)) != len(self.consulted) or self.origin in self.consulted:
            raise ValueError("Duplicate/self consultation")
        if any(i < 0 or i >= n or not self.available_domains[i] for i in self.consulted):
            raise ValueError("Unavailable candidate")
        permitted = {self.origin, *self.consulted}
        for i, value in enumerate(self.evidence):
            if (i not in permitted and value is not None) or (value is not None and not 0 <= value <= 1):
                raise ValueError("Unconsulted evidence or invalid probability")
        if any(self.evidence[i] is None for i in self.consulted):
            raise ValueError("Completed feedback requires observed selected evidence")
        if datetime.fromisoformat(self.event_timestamp).tzinfo is None:
            raise ValueError("Timezone required")
        return self


class VerifiedOutcome(Event):
    request_id: str
    patient_id: str
    prediction_digest: str
    label: int = Field(ge=0, le=1)
    authority: str
    received_episode: int
    verified_timestamp: str

    @model_validator(mode="after")
    def valid_timestamp(self) -> VerifiedOutcome:
        if datetime.fromisoformat(self.verified_timestamp).tzinfo is None:
            raise ValueError("Timezone required")
        return self


def canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(event: PredictionEvent) -> str:
    return hashlib.sha256(canonical(event.model_dump())).hexdigest()


def sign(event: Event, key: str) -> dict[str, Any]:
    payload = event.model_dump()
    return {
        "payload": payload,
        "signature": hmac.new(bytes.fromhex(key), canonical(payload), "sha256").hexdigest(),
    }


def authenticate(envelope: dict[str, Any], key: str) -> dict[str, Any]:
    if set(envelope) != {"payload", "signature"}:
        raise ValueError("Invalid signed envelope")
    expected = hmac.new(bytes.fromhex(key), canonical(envelope["payload"]), "sha256").hexdigest()
    if not hmac.compare_digest(expected, envelope["signature"]):
        raise ValueError("Unauthorized/tampered event")
    return envelope["payload"]


class FeedbackLedger:
    """Persist only authenticated, request-bound outcomes; reject duplicate replay.

    Keys are trusted simulator credentials, not a deployed consent or clinical
    verification service. Outcome authority must supply a real verified label.
    """

    def __init__(
        self,
        path: Path,
        origin: int,
        allowed_patients: set[str],
        prediction_key: str,
        outcome_key: str,
        authority: str = "simulation-training-label",
    ) -> None:
        self.path, self.origin, self.allowed_patients = path, origin, allowed_patients
        self.prediction_key, self.outcome_key, self.authority = prediction_key, outcome_key, authority
        self.predictions: dict[str, PredictionEvent] = {}
        self.outcomes: dict[str, VerifiedOutcome] = {}
        path.mkdir(parents=True, exist_ok=True)
        for kind in ("predictions", "outcomes"):
            file = path / f"{kind}.jsonl"
            if file.exists():
                for line in file.read_text().splitlines():
                    self._accept(kind, json.loads(line), persist=False)

    def _accept(self, kind: str, envelope: dict[str, Any], persist: bool = True) -> None:
        if kind == "predictions":
            event = PredictionEvent.model_validate(authenticate(envelope, self.prediction_key))
            if event.origin != self.origin or event.patient_id not in self.allowed_patients:
                raise ValueError("Wrong origin or unauthorized/non-training patient")
            if event.request_id in self.predictions:
                raise ValueError("Duplicate prediction")
            self.predictions[event.request_id] = event
        elif kind == "outcomes":
            event = VerifiedOutcome.model_validate(authenticate(envelope, self.outcome_key))
            prediction = self.predictions.get(event.request_id)
            if (
                prediction is None
                or event.patient_id != prediction.patient_id
                or event.prediction_digest != digest(prediction)
            ):
                raise ValueError("Outcome does not match original request")
            if event.authority != self.authority:
                raise ValueError("Untrusted outcome authority")
            if event.request_id in self.outcomes:
                raise ValueError("Duplicate outcome")
            if event.received_episode <= prediction.episode or datetime.fromisoformat(
                event.verified_timestamp
            ) <= datetime.fromisoformat(prediction.event_timestamp):
                raise ValueError("Outcome must arrive after decision")
            self.outcomes[event.request_id] = event
        else:
            raise ValueError("Unknown event kind")
        if persist:
            with (self.path / f"{kind}.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(envelope, allow_nan=False) + "\n")

    def prediction(self, envelope: dict[str, Any]) -> None:
        self._accept("predictions", envelope)

    def outcome(self, envelope: dict[str, Any]) -> None:
        self._accept("outcomes", envelope)

    def replay(self, cutoff: int, window: int = 128) -> list[tuple[PredictionEvent, VerifiedOutcome]]:
        """Exclude pending and future outcomes; bounded accumulated local replay."""
        return [
            (self.predictions[key], value)
            for key, value in self.outcomes.items()
            if value.received_episode <= cutoff
        ][-window:]
