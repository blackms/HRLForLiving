"""Models package for API request and response schemas"""

from .requests import (
    EnvironmentConfig,
    TrainingConfig,
    RewardConfig,
    ScenarioConfig,
    TrainingRequest,
    SimulationRequest,
    ReportRequest,
)

from .responses import (
    TrainingProgress,
    TrainingStatus,
    EpisodeResult,
    SimulationResults,
    ScenarioSummary,
    ModelSummary,
    ScenarioListResponse,
    ModelListResponse,
    SimulationHistoryResponse,
    ReportResponse,
    HealthCheckResponse,
    ErrorResponse,
)

__all__ = [
    # Request models
    "EnvironmentConfig",
    "TrainingConfig",
    "RewardConfig",
    "ScenarioConfig",
    "TrainingRequest",
    "SimulationRequest",
    "ReportRequest",
    # Response models
    "TrainingProgress",
    "TrainingStatus",
    "EpisodeResult",
    "SimulationResults",
    "ScenarioSummary",
    "ModelSummary",
    "ScenarioListResponse",
    "ModelListResponse",
    "SimulationHistoryResponse",
    "ReportResponse",
    "HealthCheckResponse",
    "ErrorResponse",
]
