# API Data Models Documentation

This document provides a comprehensive reference for all Pydantic models used in the HRL Finance System backend API.

## Overview

The API uses Pydantic v2 models for request validation and response serialization. All models include:
- Comprehensive field validation with constraints
- Descriptive field documentation
- Type safety with Python type hints
- Automatic OpenAPI schema generation

## Request Models

### EnvironmentConfig

Financial simulation environment configuration.

```python
class EnvironmentConfig(BaseModel):
    income: float                      # Monthly income (>0)
    fixed_expenses: float              # Fixed monthly expenses (≥0)
    variable_expense_mean: float       # Mean of variable expenses (≥0)
    variable_expense_std: float        # Std dev of variable expenses (≥0)
    inflation: float                   # Monthly inflation rate [-1, 1]
    safety_threshold: float            # Minimum cash buffer (≥0)
    max_months: int                    # Max simulation duration (>0)
    initial_cash: float = 0            # Starting cash balance (≥0)
    risk_tolerance: float              # Risk tolerance [0, 1]
    investment_return_mean: float = 0.005    # Mean monthly return
    investment_return_std: float = 0.02      # Return volatility
    investment_return_type: str = "stochastic"  # fixed|stochastic|none
```

**Validation Rules:**
- `income` must be positive (gt=0)
- All expense fields must be non-negative (ge=0)
- `inflation` must be in range [-1, 1]
- `risk_tolerance` must be in range [0, 1]
- `investment_return_type` must match pattern: `^(fixed|stochastic|none)$`

**Example:**
```json
{
  "income": 3200,
  "fixed_expenses": 1400,
  "variable_expense_mean": 700,
  "variable_expense_std": 100,
  "inflation": 0.02,
  "safety_threshold": 1000,
  "max_months": 60,
  "initial_cash": 0,
  "risk_tolerance": 0.5,
  "investment_return_mean": 0.005,
  "investment_return_std": 0.02,
  "investment_return_type": "stochastic"
}
```

### TrainingConfig

HRL system training parameters.

```python
class TrainingConfig(BaseModel):
    num_episodes: int = 5000           # Number of training episodes (>0)
    gamma_low: float = 0.95            # Low-level discount factor (0, 1]
    gamma_high: float = 0.99           # High-level discount factor (0, 1]
    high_period: int = 6               # Strategic planning interval (>0)
    batch_size: int = 32               # Batch size for training (>0)
    learning_rate_low: float = 3e-4    # Low-level learning rate (>0)
    learning_rate_high: float = 1e-4   # High-level learning rate (>0)
```

**Validation Rules:**
- All integer fields must be positive (gt=0)
- Gamma values must be in range (0, 1]
- Learning rates must be positive (gt=0)

**Example:**
```json
{
  "num_episodes": 1000,
  "gamma_low": 0.95,
  "gamma_high": 0.99,
  "high_period": 6,
  "batch_size": 32,
  "learning_rate_low": 0.0003,
  "learning_rate_high": 0.0001
}
```

### RewardConfig

Reward function coefficients.

```python
class RewardConfig(BaseModel):
    alpha: float = 10.0                # Investment reward coefficient
    beta: float = 0.1                  # Stability penalty coefficient
    gamma: float = 5.0                 # Overspend penalty coefficient
    delta: float = 20.0                # Debt penalty coefficient
    lambda_: float = 1.0               # Wealth growth coefficient (alias: "lambda")
    mu: float = 0.5                    # Stability bonus coefficient
```

**Special Configuration:**
- `lambda_` uses `alias="lambda"` for JSON serialization
- `populate_by_name = True` allows both "lambda" and "lambda_" in JSON

**Example:**
```json
{
  "alpha": 10.0,
  "beta": 0.1,
  "gamma": 5.0,
  "delta": 20.0,
  "lambda": 1.0,
  "mu": 0.5
}
```

### ScenarioConfig

Complete scenario configuration combining environment, training, and reward settings.

```python
class ScenarioConfig(BaseModel):
    name: str                          # Scenario name (1-100 chars)
    description: Optional[str] = None  # Description (max 500 chars)
    environment: EnvironmentConfig     # Environment configuration
    training: TrainingConfig = ...     # Training config (uses defaults)
    reward: RewardConfig = ...         # Reward config (uses defaults)
```

**Validation Rules:**
- `name` must be 1-100 characters
- `description` is optional, max 500 characters
- `training` and `reward` use default_factory if not provided

**Example:**
```json
{
  "name": "Bologna Coppia",
  "description": "Young couple in Bologna with rental expenses",
  "environment": { ... },
  "training": { ... },
  "reward": { ... }
}
```

### TrainingRequest

Request to start model training.

```python
class TrainingRequest(BaseModel):
    scenario_name: str                 # Scenario to train on (min 1 char)
    num_episodes: int = 1000           # Number of episodes (>0)
    save_interval: int = 100           # Checkpoint interval (>0)
    eval_episodes: int = 10            # Evaluation episodes (>0)
    seed: Optional[int] = None         # Random seed for reproducibility
```

**Example:**
```json
{
  "scenario_name": "bologna_coppia",
  "num_episodes": 1000,
  "save_interval": 100,
  "eval_episodes": 10,
  "seed": 42
}
```

### SimulationRequest

Request to run a simulation.

```python
class SimulationRequest(BaseModel):
    model_name: str                    # Trained model name (min 1 char)
    scenario_name: str                 # Scenario name (min 1 char)
    num_episodes: int = 10             # Number of episodes (>0)
    seed: Optional[int] = None         # Random seed
```

**Special Configuration:**
- `model_config = {"protected_namespaces": ()}` to allow `model_name` field

**Example:**
```json
{
  "model_name": "bologna_coppia_trained",
  "scenario_name": "bologna_coppia",
  "num_episodes": 10,
  "seed": 42
}
```

### ReportRequest

Request to generate a report.

```python
class ReportRequest(BaseModel):
    simulation_id: str                 # Simulation ID (min 1 char)
    report_type: str                   # Format: "pdf" or "html"
    include_sections: Optional[list[str]] = None  # Sections to include
    title: Optional[str] = None        # Custom title (max 200 chars)
```

**Validation Rules:**
- `report_type` must match pattern: `^(pdf|html)$`

**Example:**
```json
{
  "simulation_id": "sim_20251106_123456",
  "report_type": "pdf",
  "include_sections": ["summary", "charts", "strategy"],
  "title": "Bologna Coppia Financial Analysis"
}
```

## Response Models

### TrainingProgress

Real-time training progress update (for WebSocket streaming).

```python
class TrainingProgress(BaseModel):
    episode: int                       # Current episode number
    total_episodes: int                # Total episodes planned
    avg_reward: float                  # Average reward (recent episodes)
    avg_duration: float                # Average duration in months
    avg_cash: float                    # Average final cash balance
    avg_invested: float                # Average final invested amount
    stability: float                   # Stability metric [0, 1]
    goal_adherence: float              # Goal adherence [0, 1]
    elapsed_time: float                # Elapsed time in seconds
```

### TrainingStatus

Current training status.

```python
class TrainingStatus(BaseModel):
    is_training: bool                  # Whether training is active
    scenario_name: Optional[str]       # Scenario being trained
    current_episode: Optional[int]     # Current episode number
    total_episodes: Optional[int]      # Total episodes planned
    start_time: Optional[datetime]     # Training start timestamp
    latest_progress: Optional[TrainingProgress]  # Latest progress
```

### EpisodeResult

Results from a single simulation episode.

```python
class EpisodeResult(BaseModel):
    episode_id: int                    # Episode identifier
    duration: int                      # Duration in months
    final_cash: float                  # Final cash balance
    final_invested: float              # Final invested amount
    final_portfolio_value: float       # Final portfolio value
    total_wealth: float                # Total wealth (cash + portfolio)
    investment_gains: float            # Investment gains/losses
    months: List[int]                  # Month numbers
    cash_history: List[float]          # Cash balance over time
    invested_history: List[float]      # Invested amount over time
    portfolio_history: List[float]     # Portfolio value over time
    actions: List[List[float]]         # Actions [invest%, save%, consume%]
```

### SimulationResults

Aggregated results from multiple simulation episodes.

```python
class SimulationResults(BaseModel):
    simulation_id: str                 # Unique simulation ID
    scenario_name: str                 # Scenario used
    model_name: str                    # Model used
    num_episodes: int                  # Number of episodes run
    timestamp: datetime                # Completion timestamp
    
    # Summary statistics
    duration_mean: float               # Mean duration
    duration_std: float                # Std dev of duration
    final_cash_mean: float             # Mean final cash
    final_invested_mean: float         # Mean final invested
    final_portfolio_mean: float        # Mean final portfolio
    total_wealth_mean: float           # Mean total wealth
    total_wealth_std: float            # Std dev of wealth
    investment_gains_mean: float       # Mean investment gains
    
    # Strategy metrics
    avg_invest_pct: float              # Average investment %
    avg_save_pct: float                # Average save %
    avg_consume_pct: float             # Average consume %
    
    # Detailed data
    episodes: List[EpisodeResult]      # Individual episode results
```

### ScenarioSummary

Summary information about a scenario.

```python
class ScenarioSummary(BaseModel):
    name: str                          # Scenario name
    description: Optional[str]         # Description
    created_at: Optional[datetime]     # Creation timestamp
    updated_at: Optional[datetime]     # Last update timestamp
    income: float                      # Monthly income
    fixed_expenses: float              # Fixed expenses
    available_income_pct: float        # % available after fixed expenses
    risk_tolerance: float              # Risk tolerance level
```

### ModelSummary

Summary information about a trained model.

```python
class ModelSummary(BaseModel):
    name: str                          # Model name
    scenario_name: str                 # Training scenario
    episodes: int                      # Training episodes
    final_reward: float                # Final average reward
    final_stability: float             # Final stability metric
    trained_at: datetime               # Training completion timestamp
    file_size_mb: Optional[float]      # Model file size in MB
```

### List Response Models

```python
class ScenarioListResponse(BaseModel):
    scenarios: List[ScenarioSummary]   # List of scenarios
    total: int                         # Total count

class ModelListResponse(BaseModel):
    models: List[ModelSummary]         # List of models
    total: int                         # Total count

class SimulationHistoryResponse(BaseModel):
    simulations: List[Dict[str, Any]]  # List of simulation summaries
    total: int                         # Total count
```

### ReportResponse

Response for report generation.

```python
class ReportResponse(BaseModel):
    report_id: str                     # Unique report ID
    report_type: str                   # Format (pdf or html)
    file_path: str                     # Path to generated file
    file_size_mb: float                # File size in MB
    generated_at: datetime             # Generation timestamp
```

### System Response Models

```python
class HealthCheckResponse(BaseModel):
    status: str                        # Service status
    version: str                       # API version
    timestamp: datetime                # Current server timestamp

class ErrorResponse(BaseModel):
    error: str                         # Error type or code
    message: str                       # Human-readable message
    details: Optional[Dict[str, Any]]  # Additional error details
    timestamp: datetime                # Error timestamp
```

## Usage in FastAPI Endpoints

### Request Validation

```python
from backend.models import ScenarioConfig, TrainingRequest

@app.post("/api/scenarios")
async def create_scenario(scenario: ScenarioConfig):
    # Pydantic automatically validates the request body
    # Invalid data returns 422 Unprocessable Entity
    return {"message": "Scenario created", "name": scenario.name}

@app.post("/api/training/start")
async def start_training(request: TrainingRequest):
    # All fields are validated according to Field constraints
    return {"message": "Training started", "scenario": request.scenario_name}
```

### Response Serialization

```python
from backend.models import SimulationResults, ErrorResponse

@app.get("/api/simulation/results/{id}", response_model=SimulationResults)
async def get_results(id: str):
    # Return value is automatically serialized to JSON
    # Pydantic ensures all required fields are present
    results = load_simulation_results(id)
    return results

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    # Return structured error responses
    return ErrorResponse(
        error="ValueError",
        message=str(exc),
        timestamp=datetime.now()
    )
```

## Validation Error Handling

When validation fails, Pydantic returns a 422 Unprocessable Entity response with detailed error information:

```json
{
  "detail": [
    {
      "loc": ["body", "environment", "income"],
      "msg": "Input should be greater than 0",
      "type": "greater_than",
      "ctx": {"gt": 0}
    }
  ]
}
```

## Best Practices

1. **Always use type hints**: Enables IDE autocomplete and type checking
2. **Provide descriptive Field descriptions**: Improves API documentation
3. **Use appropriate validation constraints**: Prevents invalid data at API boundary
4. **Handle optional fields properly**: Use `Optional[T]` and provide defaults
5. **Use aliases for reserved keywords**: Like `lambda_` with `alias="lambda"`
6. **Configure model settings**: Use `model_config` for Pydantic v2 settings
7. **Document examples**: Include example JSON in docstrings

## Related Documentation

- [FastAPI Pydantic Models](https://fastapi.tiangolo.com/tutorial/body/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/latest/)
- [Backend API Design](../README.md)
- [API Requirements](.kiro/specs/hrl-finance-ui/requirements.md)
