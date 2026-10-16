"""Isolated role state, bounded malformed-action retries and redacted traces."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from .provider import ModelProvider, ProviderError


class ToolAgent:
    def __init__(
        self, provider: ModelProvider, role: str, prompt: str, schema: type[BaseModel], retries: int = 1
    ) -> None:
        self.provider, self.role, self.schema, self.retries = provider, role, schema, retries
        self.messages = [{"role": "system", "content": prompt}]
        self.trace: list[dict[str, Any]] = []

    def choose(self, observation: dict[str, Any]) -> BaseModel:
        # Caller supplies capability names/status only, not clinical feature values/labels.
        self.messages.append({"role": "user", "content": json.dumps(observation, allow_nan=False)})
        schema = self.schema.model_json_schema()
        if "oneOf" in schema and not observation.get("policy_registry"):
            schema["oneOf"] = [
                variant for variant in schema["oneOf"] if variant.get("$ref") != "#/$defs/RoutingCall"
            ]
            schema.get("$defs", {}).pop("RoutingCall", None)
            schema.get("discriminator", {}).get("mapping", {}).pop("get_routing_recommendation", None)
        if "specialists" in observation and "$defs" in schema:
            known = [cap["specialist"] for cap in observation["specialists"]]
            eligible = [
                cap["specialist"]
                for cap in observation["specialists"]
                if cap["eligible"] and not cap["attempted"] and cap["specialist"] != observation["origin"]
            ]
            schema["$defs"]["SpecialistArguments"]["properties"]["specialist"]["enum"] = known
            if eligible and observation["budget_remaining"] > 0:
                schema["$defs"]["PredictArguments"]["properties"]["specialist"]["enum"] = eligible
            else:
                schema["oneOf"] = [
                    variant for variant in schema["oneOf"] if variant.get("$ref") != "#/$defs/PredictCall"
                ]
            consultations = [value["consultation_id"] for value in observation["consultations"].values()]
            if consultations:
                schema["$defs"]["StatusArguments"]["properties"]["consultation_id"]["enum"] = consultations
            else:
                schema["oneOf"] = [
                    variant for variant in schema["oneOf"] if variant.get("$ref") != "#/$defs/StatusCall"
                ]
        for attempt in range(self.retries + 1):
            try:
                raw, metadata = self.provider.generate(self.messages, schema)
                action = self.schema.model_validate_json(raw)
                self.messages.append({"role": "assistant", "content": raw})
                self.trace.append(
                    {
                        "role": self.role,
                        "observation": observation,
                        "action": action.model_dump(),
                        "provider": metadata,
                        "attempt": attempt,
                    }
                )
                return action
            except (ProviderError, ValidationError, ValueError) as error:
                self.trace.append({"role": self.role, "error": str(error), "attempt": attempt})
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Invalid action. Return one action matching the provided JSON schema. "
                        + str(error)[:160],
                    }
                )
        raise ProviderError(f"{self.role} exhausted valid-action retries")


CLIENT_PROMPT = """You are a specialist interface for a private trained binary classifier.
You cannot diagnose, generate probabilities, access files or modify weights.
The observation names your required and received fields or an existing aligned record.
For a complete request call run_local_inference. If required fields are absent call
return_insufficient_features. Never invent missing fields. Output one JSON tool action,
arguments {}. The application executes the tool and supplies the classifier score."""

COORDINATOR_PROMPT = """You are the SwarmFL server coordinator. You have no file access and
must never invent risk scores, consultations or feature values. Use only listed tools:
list_specialists {}, get_specialist_capabilities {specialist},
request_specialist_prediction {specialist}, get_consultation_status {consultation_id},
finalize_prediction {}. Output exactly one JSON tool action.
Choose complementary fully eligible specialist domains using availability and consent
metadata. You must not consult the origin, unknown specialists, incomplete domains,
already attempted specialists or exceed the consultation budget. When there is a fully
eligible unconsulted peer and budget remains, request a peer prediction; otherwise call
finalize_prediction. You may inspect capabilities first. Finalize after your budget is
used; finalize invokes the trained fusion model, not a free-text medical diagnosis.
Every observation is per-request state, not training feedback. Weights remain frozen."""
