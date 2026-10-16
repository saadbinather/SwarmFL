"""Typed SLM tool dispatcher over real Flower requests and trained fusion."""

from __future__ import annotations

import json
import time
from typing import Any

import torch
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.serverapp import Grid

from ..feedback.versions import PolicyRegistry
from ..fusion import CollaborativeModel
from .provider import ModelProvider, ProviderError
from .runtime import COORDINATOR_PROMPT, ToolAgent
from .schemas import Capability, FinalPrediction, PatientRequest, SpecialistResponse, ToolCall


class AgentTransport:
    """Only interface the coordinator has to private clients."""

    def __init__(self, grid: Grid, nodes: dict[str, int], timeout: float = 90.0) -> None:
        self.grid, self.nodes, self.timeout = grid, nodes, timeout
        self.trace: list[dict[str, Any]] = []

    def exchange(self, destination: str, action: str, record: str, value: str) -> ConfigRecord:
        if destination not in self.nodes:
            raise RuntimeError("Unknown/unavailable specialist")
        message = Message(
            RecordDict({record: ConfigRecord({"json": value})}),
            dst_node_id=self.nodes[destination],
            message_type=f"query.{action}",
            group_id="slm-agents",
        )
        replies = list(self.grid.send_and_receive([message], timeout=self.timeout))
        if len(replies) != 1 or replies[0].has_error():
            raise RuntimeError("Flower agent request failed or timed out")
        if replies[0].metadata.src_node_id != self.nodes[destination]:
            raise ValueError("Wrong Flower response node")
        return replies[0].content["ack" if action == "agent_result" else "response"]

    def predict(self, destination: str, request: PatientRequest, intake: bool = False) -> SpecialistResponse:
        started = time.perf_counter()
        try:
            content = self.exchange(
                destination,
                "agent_intake" if intake else "agent_predict",
                "request",
                request.model_dump_json(),
            )
            response = SpecialistResponse.model_validate_json(content["json"])
            if (
                response.request_id != request.request_id
                or response.patient_id != request.patient_id
                or response.specialist != destination
            ):
                raise ValueError("Misassociated agent consultation")
            client_trace = json.loads(content["trace"])
            response.agent_failures = sum(bool(event.get("error")) for event in client_trace)
            self.trace.extend(client_trace)
        except RuntimeError as error:
            response = SpecialistResponse(
                request_id=request.request_id,
                patient_id=request.patient_id,
                specialist=destination,
                status="timeout",
                error=str(error),
            )
        self.trace.append(
            {
                "hop": "Flower",
                "action": "origin_intake" if intake else "specialist_prediction",
                "destination": destination,
                "origin": request.origin,
                "patient_id": request.patient_id,
                "request_id": request.request_id,
                "authorized_feature_names": sorted(request.features),
                "mode": request.mode,
                "output": response.model_dump(),
                "seconds": time.perf_counter() - started,
            }
        )
        return response

    def return_to_origin(self, result: FinalPrediction) -> None:
        reply = self.exchange(result.origin, "agent_result", "result", result.model_dump_json())
        if (reply["request_id"], reply["patient_id"], reply["origin"], reply["received"]) != (
            result.request_id,
            result.patient_id,
            result.origin,
            True,
        ):
            raise ValueError("Final reply acknowledgement mismatch")
        self.trace.append(
            {
                "hop": "Flower",
                "action": "return_to_origin",
                "request_id": result.request_id,
                "patient_id": result.patient_id,
                "origin": result.origin,
                "received": True,
            }
        )


class CoordinationSession:
    """Patient-local evidence and permissions, inaccessible through arbitrary SLM tools."""

    def __init__(
        self,
        request: PatientRequest,
        capabilities: list[Capability],
        transport: AgentTransport,
        model: CollaborativeModel,
        device: torch.device,
        policy: str,
        policy_registry: PolicyRegistry | None = None,
    ) -> None:
        self.request, self.transport, self.model, self.device, self.policy = (
            request,
            transport,
            model,
            device,
            policy,
        )
        self.capabilities = {cap.specialist: cap for cap in capabilities}
        self.policy_registry = policy_registry
        self.policy_version = None
        if request.origin not in self.capabilities:
            raise ValueError("Unknown origin")
        if request.budget > model.router.max_consultations:
            raise ValueError("Budget exceeds the evidence masks supported by the trained model")
        if request.mode == "new_patient":
            known = {name for cap in capabilities for name in cap.required_features}
            if not set(request.features) <= known:
                raise ValueError("Unrecognized patient feature names")
            if not set(request.authorized_features) <= self.capabilities.keys():
                raise ValueError("Authorization references an unknown specialist")
        self.responses: dict[str, SpecialistResponse] = {}
        self.consultation_ids: dict[str, str] = {}
        self.calls: list[str] = []
        self.tool_trace: list[dict[str, Any]] = []
        self.tool_failures = 0
        self.routing_valid = True
        self.finished: FinalPrediction | None = None
        self.started = time.perf_counter()

    def eligibility(self, cap: Capability) -> bool:
        if not cap.available or cap.specialist not in self.transport.nodes:
            return False
        if self.request.mode == "aligned":
            return True
        allowed = set(self.request.authorized_features.get(cap.specialist, []))
        return set(cap.required_features) <= allowed & self.request.features.keys()

    def observation(self) -> dict[str, Any]:
        observation = {
            "origin": self.request.origin,
            "mode": self.request.mode,
            "budget_remaining": self.request.budget - len(self.calls),
            "specialists": [
                cap.model_dump()
                | {"eligible": self.eligibility(cap), "attempted": cap.specialist in self.calls}
                for cap in self.capabilities.values()
            ],
            "origin_status": self.responses[self.request.origin].status,
            "origin_probability": self.responses[self.request.origin].probability,
            "available_feature_names": sorted(self.request.features),
            "consultations": {
                name: {"consultation_id": self.consultation_ids[name], "status": self.responses[name].status}
                for name in self.calls
            },
            "last_tool_result": self.tool_trace[-1] if self.tool_trace else None,
        }
        if self.policy_registry:
            observation["policy_registry"] = True
        return observation

    def intake(self) -> None:
        self.responses[self.request.origin] = self.transport.predict(
            self.request.origin, self.request, intake=True
        )
        self.tool_failures += self.responses[self.request.origin].agent_failures
        if self.responses[self.request.origin].status != "ok":
            self.tool_failures += 1

    def consult(self, specialist: str) -> dict[str, Any]:
        if specialist not in self.capabilities or specialist == self.request.origin:
            raise ValueError("Unknown specialist or prohibited self-consultation")
        if specialist in self.calls or len(self.calls) >= self.request.budget:
            raise ValueError("Duplicate consultation or exhausted budget")
        cap = self.capabilities[specialist]
        if not self.eligibility(cap):
            raise ValueError("Specialist unavailable, missing required features, or lacking authorization")
        allowed = set(self.request.authorized_features.get(specialist, []))
        features = (
            {
                name: self.request.features[name]
                for name in cap.required_features
                if name in self.request.features and name in allowed
            }
            if self.request.mode == "new_patient"
            else {}
        )
        envelope = PatientRequest.model_validate(
            self.request.model_dump()
            | {
                "destination": specialist,
                "features": features,
                "authorized_features": {specialist: sorted(allowed & set(features))}
                if self.request.mode == "new_patient"
                else {},
            }
        )
        self.calls.append(specialist)
        self.consultation_ids[specialist] = f"{self.request.request_id}-{specialist}"
        self.responses[specialist] = self.transport.predict(specialist, envelope)
        self.tool_failures += self.responses[specialist].agent_failures
        if self.responses[specialist].status != "ok":
            self.tool_failures += 1
        return self.responses[specialist].model_dump() | {
            "consultation_id": self.consultation_ids[specialist]
        }

    def finalize(self) -> FinalPrediction:
        own = self.responses[self.request.origin]
        probability = None
        status = "no_prediction"
        if own.status == "ok":
            values = torch.full((1, len(self.capabilities)), 0.5, device=self.device)
            evidence = torch.zeros_like(values, dtype=torch.bool)
            for name, response in self.responses.items():
                if response.status == "ok":
                    index = self.capabilities[name].index
                    values[0, index] = response.probability
                    evidence[0, index] = True
            origin = torch.tensor([self.capabilities[self.request.origin].index], device=self.device)
            with torch.no_grad():
                probability = float(
                    torch.sigmoid(
                        self.model.fusion(values, 2 * (values - 0.5).abs(), evidence, origin)
                    ).cpu()[0]
                )
            status = (
                "ok"
                if all(self.responses[name].status == "ok" for name in self.calls)
                else "partial_evidence"
            )
        self.finished = FinalPrediction(
            request_id=self.request.request_id,
            patient_id=self.request.patient_id,
            origin=self.request.origin,
            status=status,
            probability=probability,
            consulted=self.calls.copy(),
            tool_failures=self.tool_failures,
            routing_valid=self.routing_valid,
            latency_seconds=time.perf_counter() - self.started,
            policy=self.policy,
            consultation_budget=self.request.budget,
        )
        return self.finished

    def execute(self, call: ToolCall) -> dict[str, Any]:
        action = call.root
        try:
            if action.name == "list_specialists":
                output = {"specialists": [cap.model_dump() for cap in self.capabilities.values()]}
            elif action.name == "get_specialist_capabilities":
                if action.arguments.specialist not in self.capabilities:
                    raise ValueError("Unknown specialist")
                output = self.capabilities[action.arguments.specialist].model_dump()
            elif action.name == "get_routing_recommendation":
                if self.policy_registry is None:
                    raise ValueError("No approved policy registry configured")
                self.policy_version = self.policy_registry.current()
                self.model = self.policy_registry.load(self.policy_version, self.device)
                self.model.requires_grad_(False)
                own = torch.tensor([self.responses[self.request.origin].probability], device=self.device)
                origin = torch.tensor([self.capabilities[self.request.origin].index], device=self.device)
                ordered = sorted(self.capabilities.values(), key=lambda cap: cap.index)
                available = torch.tensor([[self.eligibility(cap) for cap in ordered]], device=self.device)
                with torch.no_grad():
                    selected, scores = self.model.router.select(
                        origin, own, 2 * (own - 0.5).abs(), available.float(), available
                    )
                output = {
                    "version": self.policy_version,
                    "selected": [cap.specialist for cap in ordered if selected[0, cap.index]][
                        : self.request.budget
                    ],
                    "scores": [float(score) if torch.isfinite(score) else None for score in scores[0].cpu()],
                }
            elif action.name == "request_specialist_prediction":
                output = self.consult(action.arguments.specialist)
            elif action.name == "get_consultation_status":
                names = [
                    name
                    for name, identity in self.consultation_ids.items()
                    if identity == action.arguments.consultation_id
                ]
                if not names:
                    raise ValueError("Unknown consultation")
                output = self.responses[names[0]].model_dump()
            else:
                output = self.finalize().model_dump()
            entry = {"tool": action.name, "arguments": action.arguments.model_dump(), "output": output}
        except ValueError as error:
            self.tool_failures += 1
            self.routing_valid = False
            entry = {"tool": action.name, "arguments": action.arguments.model_dump(), "error": str(error)}
        self.tool_trace.append(entry)
        return entry


def coordinate(
    session: CoordinationSession, provider: ModelProvider, max_steps: int = 6
) -> list[dict[str, Any]]:
    """Separate SLM policy, deterministic feature policy and learned neural routing."""
    session.intake()
    agent_trace = []
    if session.responses[session.request.origin].status != "ok":
        session.finalize()
    elif session.policy == "slm":
        prompt = COORDINATOR_PROMPT
        if session.policy_registry:
            prompt += " First call get_routing_recommendation to consult the newest approved neural policy. "
            prompt += (
                "Its selected list may be empty; respect the budget and finalize when it recommends no peers."
            )
        agent = ToolAgent(provider, "server_coordinator", prompt, ToolCall)
        try:
            for _ in range(max_steps):
                action = agent.choose(session.observation())
                session.execute(action)
                if session.finished is not None:
                    break
            if session.finished is None:
                raise ProviderError("Coordinator exceeded maximum tool steps")
        except ProviderError as error:
            session.tool_failures += 1
            session.routing_valid = False
            session.finished = FinalPrediction(
                request_id=session.request.request_id,
                patient_id=session.request.patient_id,
                origin=session.request.origin,
                status="agent_error",
                probability=None,
                consulted=session.calls,
                tool_failures=session.tool_failures,
                routing_valid=False,
                latency_seconds=time.perf_counter() - session.started,
                policy=session.policy,
                consultation_budget=session.request.budget,
            )
            agent.trace.append({"role": agent.role, "error": str(error)})
        agent_trace = agent.trace
        if session.finished.status != "agent_error":
            recovered_errors = sum(bool(entry.get("error")) for entry in agent.trace)
            if recovered_errors:
                session.tool_failures += recovered_errors
                session.routing_valid = False
                session.finished.tool_failures = session.tool_failures
                session.finished.routing_valid = False
    else:
        candidates = [
            cap
            for cap in session.capabilities.values()
            if cap.specialist != session.request.origin and session.eligibility(cap)
        ]
        if session.policy == "feature_domain":
            # Public feature-domain coverage only; no candidate prediction/label is inspected.
            selected = sorted(candidates, key=lambda cap: (-len(cap.required_features), cap.index))[
                : session.request.budget
            ]
        elif session.policy == "learned_router":
            names = {cap.specialist for cap in candidates} | {session.request.origin}
            available = torch.tensor(
                [[cap.specialist in names for cap in session.capabilities.values()]], device=session.device
            )
            probability = torch.tensor(
                [session.responses[session.request.origin].probability], device=session.device
            )
            origin = torch.tensor([session.capabilities[session.request.origin].index], device=session.device)
            with torch.no_grad():
                actions, scores = session.model.router.select(
                    origin, probability, 2 * (probability - 0.5).abs(), available.float(), available
                )
            selected = [cap for cap in candidates if actions[0, cap.index]][: session.request.budget]
            session.tool_trace.append(
                {
                    "policy": "learned_router",
                    "selected": [cap.specialist for cap in selected],
                    "scores": [float(value) if torch.isfinite(value) else None for value in scores[0].cpu()],
                }
            )
        else:
            raise ValueError("Unknown coordination policy")
        for cap in selected:
            session.consult(cap.specialist)
        session.finalize()
    session.transport.return_to_origin(session.finished)
    # End-to-end latency includes the real final Flower return hop.
    session.finished.latency_seconds = time.perf_counter() - session.started
    return agent_trace + session.tool_trace
