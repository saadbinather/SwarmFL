"""Private classifier adapter and frozen-SLM specialist interface."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..devices import select_device, synchronize
from ..models import Classifier
from .provider import ModelProvider, ProviderError
from .runtime import CLIENT_PROMPT, ToolAgent
from .schemas import Capability, ClientAction, PatientRequest, SpecialistResponse


class PrivateClassifier:
    """Loads only its own checkpoint/features; new cases never query stored patients."""

    def __init__(self, path: Path, index: int, config: dict[str, Any], device: str = "auto") -> None:
        self.path, self.config, self.device = path, config, select_device(device)
        self.metadata = json.loads((path / "specialist.json").read_text())
        entry = config["clients"][index]
        self.capability = Capability(
            specialist=entry["specialist"],
            index=index,
            domain=entry["domain"],
            required_features=self.metadata["features"],
            available=bool(self.metadata.get("domain_available", True)),
        )
        self.model: Classifier | None = None

    def _model(self) -> Classifier:
        if self.model is None:
            self.model = Classifier(len(self.capability.required_features)).to(self.device)
            self.model.load_state_dict(
                torch.load(self.path / "specialist.pt", weights_only=True, map_location=self.device)
            )
            self.model.eval()
            self.model.requires_grad_(False)
        return self.model

    def encode(self, features: dict[str, float | int | str]) -> np.ndarray:
        names = self.capability.required_features
        if set(features) != set(names):
            raise ValueError("Classifier requires its exact full feature subset")
        values = []
        for name in names:
            value = features[name]
            if name in self.config["numeric_fields"]:
                if isinstance(value, (str, bool)) or not math.isfinite(value):
                    raise ValueError(f"Invalid numeric field {name}")
                # Apply the original training-fitted scaler; never fit on a new patient.
                value = (float(value) - self.metadata["preprocessing"]["mean"]) / self.metadata[
                    "preprocessing"
                ]["scale"]
            else:
                if isinstance(value, str):
                    mapping = self.config["encodings"].get(name, self.config["encodings"]["symptom"])
                    if value not in mapping:
                        raise ValueError(f"Unknown category in {name}")
                    value = mapping[value]
                if isinstance(value, bool) or value not in (0, 1):
                    raise ValueError(f"Invalid binary field {name}")
            values.append(value)
        return np.asarray([values], dtype=np.float32)

    def probability(self, request: PatientRequest) -> float:
        if request.mode == "new_patient":
            x = self.encode(request.features)
        else:
            with np.load(self.path / "local.npz") as local:
                matches = np.flatnonzero(local["ids"] == request.alignment_id)
                if len(matches) != 1:
                    raise ValueError("Aligned patient unavailable")
                x = local["x"][matches]
        with torch.no_grad():
            p = torch.sigmoid(
                self._model()(torch.as_tensor(x, device=self.device)) / self.metadata["temperature"]
            )
        synchronize(self.device)
        return float(p.cpu()[0])


class SpecialistAgent:
    def __init__(self, adapter: PrivateClassifier, provider: ModelProvider) -> None:
        self.adapter, self.provider = adapter, provider

    def predict(self, request: PatientRequest) -> tuple[SpecialistResponse, list[dict[str, Any]]]:
        capability = self.adapter.capability
        if (request.destination or request.origin) != capability.specialist:
            raise ValueError("Private inference envelope destination mismatch")
        names = capability.required_features
        missing = (
            [name for name in names if name not in request.features] if request.mode == "new_patient" else []
        )
        base = {
            "request_id": request.request_id,
            "patient_id": request.patient_id,
            "specialist": capability.specialist,
            "received_feature_names": sorted(request.features),
            "missing_features": missing,
        }
        if not capability.available:
            return SpecialistResponse(**base, status="unavailable"), []
        if request.mode == "new_patient":
            authorized = set(request.authorized_features.get(capability.specialist, []))
            if not set(request.features) <= authorized or not set(request.features) <= set(names):
                raise ValueError("Recipient received unauthorized or extraneous fields")
        agent = ToolAgent(self.provider, f"client_{capability.specialist}", CLIENT_PROMPT, ClientAction)
        started = time.perf_counter()
        try:
            action = agent.choose(
                {
                    "specialist": capability.model_dump(),
                    "mode": request.mode,
                    "received_feature_names": sorted(request.features),
                    "missing_features": missing,
                    "aligned_record_requested": request.mode == "aligned",
                }
            )
            if missing:
                return SpecialistResponse(
                    **base,
                    status="insufficient_features",
                    agent_action=action.name,
                    provider_seconds=time.perf_counter() - started,
                ), agent.trace
            if action.name != "run_local_inference":
                return SpecialistResponse(
                    **base,
                    status="agent_error",
                    agent_action=action.name,
                    error="SLM declined a complete feature set",
                ), agent.trace
            p = self.adapter.probability(request)
            response = SpecialistResponse(
                **base,
                status="ok",
                probability=p,
                confidence=2 * abs(p - 0.5),
                agent_action=action.name,
                provider_seconds=time.perf_counter() - started,
            )
            agent.trace.append(
                {"role": agent.role, "tool": "run_local_inference", "output": response.model_dump()}
            )
            return response, agent.trace
        except ProviderError as error:
            return SpecialistResponse(
                **base, status="agent_error", error=str(error), provider_seconds=time.perf_counter() - started
            ), agent.trace
        except ValueError as error:
            return SpecialistResponse(**base, status="invalid_features", error=str(error)), agent.trace
