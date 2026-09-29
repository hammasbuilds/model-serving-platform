"""Multi-model serving: registry, canary, shadow, SLOs and auto-rollback."""

from .observe.slo import SLO, SLOMonitor
from .registry.store import Registry, RegistryError
from .server import ServingError, ServingPlatform
from .types import ModelVersion, Prediction, Stage

__version__ = "0.1.0"

__all__ = [
    "SLO",
    "ModelVersion",
    "Prediction",
    "Registry",
    "RegistryError",
    "SLOMonitor",
    "ServingError",
    "ServingPlatform",
    "Stage",
]
