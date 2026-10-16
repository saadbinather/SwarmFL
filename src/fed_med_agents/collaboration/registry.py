"""Runtime specialist metadata, authorization and transport-independent admission.

Logical registration does not start/stop physical Flower nodes. Permissions are
trusted application metadata, not authenticated patient consent or a privacy proof.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from threading import RLock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RegistryError(ValueError):
    """Rejected registry mutation, compatibility, authorization or consultation."""


class Declaration(BaseModel):
    """Immutable, closed declarations suitable for capability discovery over Flower."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class TaskSchema(Declaration):
    """Task and exact ordered label meanings; matching names alone is insufficient."""

    name: str = Field(min_length=1)
    labels: tuple[str, str]
    label_definition: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_labels(self) -> TaskSchema:
        if len(set(self.labels)) != 2 or any(not label.strip() for label in self.labels):
            raise ValueError("Two distinct nonempty label meanings required")
        return self


class FeatureSchema(Declaration):
    """Ordered model feature requirement, including numeric/category constraints."""

    name: str = Field(min_length=1)
    dtype: Literal["number", "integer", "boolean", "category"] = "number"
    minimum: float | None = None
    maximum: float | None = None
    categories: tuple[str, ...] = ()
    unit: str = ""

    @model_validator(mode="after")
    def valid_bounds(self) -> FeatureSchema:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("Reversed feature bounds")
        if self.dtype == "category" and (
            not self.categories or len(set(self.categories)) != len(self.categories)
        ):
            raise ValueError("Category schema requires distinct allowed values")
        return self

    def validate_value(self, value: object) -> None:
        """Reject incomplete, incorrectly typed, nonfinite or out-of-schema inputs."""
        if self.dtype == "category":
            if not isinstance(value, str) or value not in self.categories:
                raise RegistryError(f"Invalid category for {self.name}")
            return
        if self.dtype == "boolean":
            if not isinstance(value, bool):
                raise RegistryError(f"Invalid boolean for {self.name}")
            return
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise RegistryError(f"Invalid finite number for {self.name}")
        if self.dtype == "integer" and not isinstance(value, Integral):
            raise RegistryError(f"Invalid integer for {self.name}")
        if (self.minimum is not None and value < self.minimum) or (
            self.maximum is not None and value > self.maximum
        ):
            raise RegistryError(f"Out-of-range feature {self.name}")


class OutputSchema(Declaration):
    """A scalar calibrated-risk interface, not a clinical guarantee of calibration."""

    kind: Literal["binary_probability"] = "binary_probability"
    dtype: Literal["float32", "float64"] = "float32"
    positive_label: str = Field(min_length=1)


class ModelAdapter(Declaration):
    """Local adapter identity/binding; never private model weights or a model path."""

    adapter_id: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    feature_names: tuple[str, ...]
    input_dim: int = Field(gt=0, strict=True)
    alignment_namespace: str = Field(min_length=1)


class Specialist(Declaration):
    """Complete discovery declaration, independent of any disease configuration."""

    specialist_id: str = Field(min_length=1)
    model_adapter: ModelAdapter
    feature_schema: tuple[FeatureSchema, ...]
    output_schema: OutputSchema
    task: TaskSchema
    capabilities: frozenset[str]
    available: bool = Field(default=True, strict=True)
    consultation_cost: float = Field(default=0.02, ge=0, strict=True)

    @model_validator(mode="after")
    def compatible_adapter(self) -> Specialist:
        names = tuple(feature.name for feature in self.feature_schema)
        if not names or len(set(names)) != len(names):
            raise ValueError("Nonempty distinct feature names required")
        if names != self.model_adapter.feature_names or len(names) != self.model_adapter.input_dim:
            raise ValueError("Adapter feature order/dimension mismatch")
        if self.output_schema.positive_label != self.task.labels[1]:
            raise ValueError("Output positive label differs from task label meaning")
        if not self.capabilities or any(not value.strip() for value in self.capabilities):
            raise ValueError("Nonempty capabilities required")
        return self


@dataclass(frozen=True)
class RegisteredSpecialist:
    """Immutable logical metadata and optional physical transport binding."""

    specialist: Specialist
    node_id: int | None
    generation: int


class SpecialistRegistry:
    """Thread-safe in-memory logical membership with an explicit compatibility contract."""

    def __init__(
        self, task: TaskSchema, output_schema: OutputSchema, *, model_family: str | None = None
    ) -> None:
        if output_schema.positive_label != task.labels[1]:
            raise RegistryError("Registry task/output label mismatch")
        self.task, self.output_schema, self.model_family = task, output_schema, model_family
        self._entries: dict[str, RegisteredSpecialist] = {}
        self._generation = 0
        self._lock = RLock()

    def register(self, specialist: Specialist, *, node_id: int | None = None) -> RegisteredSpecialist:
        """Register a validated declaration; IDs and live node bindings must be unique."""
        specialist = Specialist.model_validate(specialist.model_dump())
        with self._lock:
            if specialist.specialist_id in self._entries:
                raise RegistryError("Duplicate specialist ID")
            if specialist.task != self.task or specialist.output_schema != self.output_schema:
                raise RegistryError("Incompatible task, labels or output schema")
            if self.model_family is not None and specialist.model_adapter.model_family != self.model_family:
                raise RegistryError("Incompatible checkpoint family")
            known_features = {
                feature.name: feature
                for entry in self._entries.values()
                for feature in entry.specialist.feature_schema
            }
            if any(
                feature.name in known_features and feature != known_features[feature.name]
                for feature in specialist.feature_schema
            ):
                raise RegistryError("Incompatible shared feature schema")
            if node_id is not None:
                if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id <= 0:
                    raise RegistryError("Invalid physical node binding")
                if any(entry.node_id == node_id for entry in self._entries.values()):
                    raise RegistryError("Duplicate physical node binding")
            self._generation += 1
            entry = RegisteredSpecialist(specialist, node_id, self._generation)
            self._entries[specialist.specialist_id] = entry
            return entry

    def unregister(self, specialist_id: str) -> RegisteredSpecialist:
        """Remove logical membership only; never terminate its physical Flower node."""
        with self._lock:
            entry = self.get(specialist_id)
            del self._entries[specialist_id]
            self._generation += 1
            return entry

    def get(self, specialist_id: str) -> RegisteredSpecialist:
        """Return an immutable snapshot or reject unknown/removed identities."""
        with self._lock:
            if specialist_id not in self._entries:
                raise RegistryError(f"Unknown specialist {specialist_id}")
            return self._entries[specialist_id]

    def discover(
        self, *, capability: str | None = None, available_only: bool = False
    ) -> tuple[RegisteredSpecialist, ...]:
        """Discover current declarations without exposing private models or values."""
        with self._lock:
            return tuple(
                entry
                for entry in self._entries.values()
                if (capability is None or capability in entry.specialist.capabilities)
                and (not available_only or entry.specialist.available)
            )

    def update(
        self, specialist_id: str, *, available: bool | None = None, consultation_cost: float | None = None
    ) -> RegisteredSpecialist:
        """Atomically update availability/cost, validating finite nonnegative costs."""
        with self._lock:
            entry = self.get(specialist_id)
            values = entry.specialist.model_dump()
            if available is not None:
                if not isinstance(available, bool):
                    raise RegistryError("Availability must be boolean")
                values["available"] = available
            if consultation_cost is not None:
                if isinstance(consultation_cost, bool):
                    raise RegistryError("Cost must be numeric")
                values["consultation_cost"] = consultation_cost
            specialist = Specialist.model_validate(values)
            self._generation += 1
            updated = RegisteredSpecialist(specialist, entry.node_id, self._generation)
            self._entries[specialist_id] = updated
            return updated

    def validate(
        self,
        origin: str,
        destination: str,
        *,
        task: TaskSchema,
        permitted: frozenset[str],
        feature_permissions: Mapping[str, frozenset[str]],
        available_features: frozenset[str],
        capability: str = "aligned_prediction",
        alignment_namespace: str | None = None,
        features: Mapping[str, Any] | None = None,
    ) -> tuple[RegisteredSpecialist, dict[str, Any] | None]:
        """Authorize a peer and prepare only its required feature subset, if supplied.

        Aligned mode requires public feature availability/permission metadata, not
        feature values. New-patient values are separately authorized and filtered;
        this does not authorize sending private aligned training rows.
        """
        with self._lock:
            source, target = self.get(origin), self.get(destination)
            spec = target.specialist
            if origin == destination:
                raise RegistryError("Self-consultation rejected")
            if not source.specialist.available or not spec.available:
                raise RegistryError("Specialist unavailable")
            if destination not in permitted:
                raise RegistryError("Unauthorized specialist")
            if task != self.task or spec.task != task:
                raise RegistryError("Incompatible request task/labels")
            if capability not in spec.capabilities:
                raise RegistryError("Unsupported capability")
            if target.node_id is None:
                raise RegistryError("Specialist has no physical transport binding")
            required = frozenset(spec.model_adapter.feature_names)
            if not required.issubset(feature_permissions.get(destination, frozenset())):
                raise RegistryError("Unauthorized required features")
            if not required.issubset(available_features):
                raise RegistryError("Missing required features")
            if (
                alignment_namespace is not None
                and spec.model_adapter.alignment_namespace != alignment_namespace
            ):
                raise RegistryError("Incompatible patient alignment namespace")
            if features is None and (
                source.specialist.model_adapter.alignment_namespace != spec.model_adapter.alignment_namespace
            ):
                raise RegistryError("Incompatible origin/recipient alignment namespace")
            if features is None:
                return target, None
            subset = {}
            for feature in spec.feature_schema:
                if feature.name not in features:
                    raise RegistryError("Missing required feature value")
                feature.validate_value(features[feature.name])
                subset[feature.name] = features[feature.name]
            return target, subset

    def eligible(self, origin: str, **requirements: Any) -> tuple[RegisteredSpecialist, ...]:
        """Filter discovery by the same admission checks used for actual dispatch."""
        result = []
        for entry in self.discover():
            try:
                target, _ = self.validate(origin, entry.specialist.specialist_id, **requirements)
            except RegistryError:
                continue
            result.append(target)
        return tuple(result)


class ConsultationSession:
    """Enforce one case/batch's call and cost budgets across live registry changes.

    A batch has identical selected recipient(s) for its case IDs. One dispatched
    peer call consumes one budget unit per case; rejection consumes none. Failed
    transport attempts consume reserved budgets. Availability changes gate future
    dispatches, not an already admitted in-flight call. No retries are implicit.
    """

    def __init__(
        self,
        registry: SpecialistRegistry,
        origin: str,
        patient_ids: tuple[str, ...],
        *,
        permitted: frozenset[str],
        feature_permissions: Mapping[str, frozenset[str]],
        available_features: frozenset[str],
        max_calls: int = 1,
        max_cost: float | None = None,
        alignment_namespace: str | None = None,
    ) -> None:
        if (
            not patient_ids
            or len(set(patient_ids)) != len(patient_ids)
            or any(not isinstance(p, str) or not p for p in patient_ids)
        ):
            raise RegistryError("Nonempty unique patient IDs required")
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0:
            raise RegistryError("Invalid consultation budget")
        if max_cost is not None and (
            isinstance(max_cost, bool)
            or not isinstance(max_cost, Real)
            or not math.isfinite(max_cost)
            or max_cost < 0
        ):
            raise RegistryError("Invalid cost budget")
        registry.get(origin)
        self.registry, self.origin, self.patient_ids = registry, origin, patient_ids
        self.permitted = frozenset(permitted)
        self.feature_permissions = {key: frozenset(value) for key, value in feature_permissions.items()}
        self.available_features, self.max_calls, self.max_cost = (
            frozenset(available_features),
            max_calls,
            max_cost,
        )
        self.alignment_namespace = alignment_namespace
        self.calls, self.cost = 0, 0.0
        self._lock = RLock()

    def requirements(self) -> dict[str, Any]:
        """Return a defensive admission-metadata snapshot for eligibility discovery."""
        return {
            "task": self.registry.task,
            "permitted": self.permitted,
            "feature_permissions": dict(self.feature_permissions),
            "available_features": self.available_features,
            "alignment_namespace": self.alignment_namespace,
        }

    def consult(
        self,
        destination: str,
        transport: Callable[[RegisteredSpecialist, dict[str, Any]], dict[str, Any]],
        *,
        features: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve live membership, authorize, reserve budgets, dispatch and validate replies."""
        with self._lock:
            capability = "new_patient_prediction" if features is not None else "aligned_prediction"
            entry, subset = self.registry.validate(
                self.origin, destination, **self.requirements(), capability=capability, features=features
            )
            if features is not None and len(self.patient_ids) != 1:
                raise RegistryError("New-patient feature dictionaries require one case")
            if self.calls >= self.max_calls:
                raise RegistryError("Consultation budget exhausted")
            cost = entry.specialist.consultation_cost
            if self.max_cost is not None and self.cost + cost > self.max_cost + 1e-12:
                raise RegistryError("Consultation cost budget exceeded")
            payload = {
                "origin": self.origin,
                "destination": destination,
                "patient_ids": list(self.patient_ids),
                "adapter_id": entry.specialist.model_adapter.adapter_id,
                "checkpoint_sha256": entry.specialist.model_adapter.checkpoint_sha256,
                "task": self.registry.task.model_dump(mode="json"),
                "mode": "aligned",
                "authorized_features": list(entry.specialist.model_adapter.feature_names),
            }
            if subset is not None:
                payload.update(mode="new_patient", features=subset)
            self.calls += 1
            self.cost += cost
            expected = {
                key: list(value) if isinstance(value, list) else value
                for key, value in payload.items()
                if key in {"origin", "destination", "patient_ids", "adapter_id", "checkpoint_sha256"}
            }
            reply = transport(entry, payload)
            if not isinstance(reply, dict) or set(reply) - (set(expected) | {"probabilities", "client"}):
                raise RegistryError("Unexpected prediction output fields")
            for key in ("origin", "destination", "patient_ids", "adapter_id", "checkpoint_sha256"):
                if reply.get(key) != expected[key]:
                    raise RegistryError(f"Misassociated reply: {key}")
            values = reply.get("probabilities")
            if not isinstance(values, list) or len(values) != len(self.patient_ids):
                raise RegistryError("Invalid prediction output shape")
            if any(
                isinstance(p, bool) or not isinstance(p, Real) or not math.isfinite(p) or not 0 <= p <= 1
                for p in values
            ):
                raise RegistryError("Invalid probability output")
            return reply

    def eligible(self) -> tuple[RegisteredSpecialist, ...]:
        """Discover peers affordable within this session's remaining live budgets."""
        with self._lock:
            if self.calls >= self.max_calls:
                return ()
            return tuple(
                entry
                for entry in self.registry.eligible(self.origin, **self.requirements())
                if self.max_cost is None
                or self.cost + entry.specialist.consultation_cost <= self.max_cost + 1e-12
            )
