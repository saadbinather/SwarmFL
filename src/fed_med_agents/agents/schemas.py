"""Strict typed request, authorization, capability, action and response contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class PatientRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=80)
    patient_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    origin: str
    destination: str | None = None
    mode: Literal["aligned", "new_patient"]
    alignment_id: int | None = Field(default=None, ge=0)
    features: dict[str, float | int | str] = Field(default_factory=dict)
    authorized_features: dict[str, list[str]] = Field(default_factory=dict)
    budget: int = Field(default=1, ge=0, le=8)

    @model_validator(mode="after")
    def valid_mode(self) -> PatientRequest:
        if self.mode == "aligned" and (
            self.alignment_id is None or self.features or self.authorized_features
        ):
            raise ValueError("Aligned mode uses only an alignment ID, never feature transfer")
        if self.mode == "new_patient" and self.alignment_id is not None:
            raise ValueError("New-patient inference cannot look up an aligned record")
        if any(len(set(names)) != len(names) for names in self.authorized_features.values()):
            raise ValueError("Duplicate authorized feature")
        return self


class Capability(StrictModel):
    specialist: str
    index: int = Field(ge=0)
    domain: str
    required_features: list[str]
    available: bool = True


class EmptyArguments(StrictModel):
    pass


class SpecialistArguments(StrictModel):
    specialist: str


class PredictArguments(StrictModel):
    specialist: str


class StatusArguments(StrictModel):
    consultation_id: str


class ListCall(StrictModel):
    name: Literal["list_specialists"]
    arguments: EmptyArguments


class CapabilitiesCall(StrictModel):
    name: Literal["get_specialist_capabilities"]
    arguments: SpecialistArguments


class PredictCall(StrictModel):
    name: Literal["request_specialist_prediction"]
    arguments: PredictArguments


class StatusCall(StrictModel):
    name: Literal["get_consultation_status"]
    arguments: StatusArguments


class FinalizeCall(StrictModel):
    name: Literal["finalize_prediction"]
    arguments: EmptyArguments


class RoutingCall(StrictModel):
    name: Literal["get_routing_recommendation"]
    arguments: EmptyArguments


ToolCall = RootModel[
    Annotated[
        ListCall | CapabilitiesCall | PredictCall | StatusCall | FinalizeCall | RoutingCall,
        Field(discriminator="name"),
    ]
]


class ClientAction(StrictModel):
    name: Literal["run_local_inference", "return_insufficient_features"]
    arguments: EmptyArguments


class SpecialistResponse(StrictModel):
    request_id: str
    patient_id: str
    specialist: str
    status: Literal[
        "ok", "insufficient_features", "unavailable", "invalid_features", "agent_error", "timeout"
    ]
    probability: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    missing_features: list[str] = Field(default_factory=list)
    received_feature_names: list[str] = Field(default_factory=list)
    agent_action: str | None = None
    provider_seconds: float = 0.0
    agent_failures: int = Field(default=0, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def valid_probability(self) -> SpecialistResponse:
        if self.status == "ok" and (self.probability is None or self.confidence is None):
            raise ValueError("Successful inference requires measured classifier outputs")
        if self.status != "ok" and (self.probability is not None or self.confidence is not None):
            raise ValueError("Failed inference must not fabricate a probability")
        return self


class FinalPrediction(StrictModel):
    request_id: str
    patient_id: str
    origin: str
    status: Literal["ok", "partial_evidence", "no_prediction", "agent_error"]
    probability: float | None = Field(default=None, ge=0, le=1)
    consulted: list[str]
    tool_failures: int
    routing_valid: bool
    latency_seconds: float = Field(ge=0)
    policy: str
    consultation_budget: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def valid_score(self) -> FinalPrediction:
        if self.status in {"ok", "partial_evidence"} and self.probability is None:
            raise ValueError("Successful fusion requires a measured probability")
        if self.status in {"no_prediction", "agent_error"} and self.probability is not None:
            raise ValueError("Failed coordination must not fabricate a score")
        return self
