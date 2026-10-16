"""SwarmFL public API for federated specialist consultation research."""

from .collaboration.registry import (
    ConsultationSession,
    FeatureSchema,
    ModelAdapter,
    OutputSchema,
    RegisteredSpecialist,
    RegistryError,
    Specialist,
    SpecialistRegistry,
    TaskSchema,
)
from .feedback.continual import ContinualController
from .feedback.events import FeedbackLedger, PredictionEvent, VerifiedOutcome
from .feedback.versions import PolicyPair, PolicyRegistry
from .fusion.dynamic import DynamicFusion, FusionEvidence
from .routing.policy import RoutingDecision, RoutingPolicy

__version__ = "0.1.0a1"

__all__ = [
    "ConsultationSession",
    "ContinualController",
    "DynamicFusion",
    "FeatureSchema",
    "FeedbackLedger",
    "FusionEvidence",
    "ModelAdapter",
    "OutputSchema",
    "PolicyPair",
    "PolicyRegistry",
    "PredictionEvent",
    "RegisteredSpecialist",
    "RegistryError",
    "RoutingDecision",
    "RoutingPolicy",
    "Specialist",
    "SpecialistRegistry",
    "TaskSchema",
    "VerifiedOutcome",
    "__version__",
]
