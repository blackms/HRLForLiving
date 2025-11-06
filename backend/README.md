# HRL Finance System Backend

FastAPI backend for the HRL Finance System.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --port 8000
```

## API Endpoints

### Core Endpoints

- `GET /` - Root endpoint returning API information
- `GET /health` - Health check endpoint for monitoring

### Scenarios API

- `GET /api/scenarios` - List all scenarios with summary information
- `GET /api/scenarios/{name}` - Get detailed scenario configuration
- `POST /api/scenarios` - Create a new scenario
- `PUT /api/scenarios/{name}` - Update an existing scenario
- `DELETE /api/scenarios/{name}` - Delete a scenario
- `GET /api/scenarios/templates` - Get preset scenario templates

### API Documentation

Once running, visit:
- API Root: http://localhost:8000
- Health Check: http://localhost:8000/health
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Data Models

### Request Models (`backend/models/requests.py`)

All request models include comprehensive validation using Pydantic Field constraints.

#### EnvironmentConfig
Configuration for financial simulation environment:
- `income` (float, >0): Monthly income in currency units
- `fixed_expenses` (float, ≥0): Fixed monthly expenses
- `variable_expense_mean` (float, ≥0): Mean of variable expenses
- `variable_expense_std` (float, ≥0): Standard deviation of variable expenses
- `inflation` (float, [-1, 1]): Monthly inflation rate
- `safety_threshold` (float, ≥0): Minimum cash buffer threshold
- `max_months` (int, >0): Maximum simulation duration in months
- `initial_cash` (float, ≥0, default=0): Starting cash balance
- `risk_tolerance` (float, [0, 1]): Risk tolerance level
- `investment_return_mean` (float, default=0.005): Mean monthly investment return
- `investment_return_std` (float, default=0.02): Standard deviation of investment returns
- `investment_return_type` (str, default="stochastic"): Type of investment returns (fixed|stochastic|none)

#### TrainingConfig
Configuration for HRL system training:
- `num_episodes` (int, >0, default=5000): Number of training episodes
- `gamma_low` (float, (0, 1], default=0.95): Discount factor for low-level agent
- `gamma_high` (float, (0, 1], default=0.99): Discount factor for high-level agent
- `high_period` (int, >0, default=6): Planning horizon for high-level agent in months
- `batch_size` (int, >0, default=32): Batch size for training
- `learning_rate_low` (float, >0, default=3e-4): Learning rate for low-level agent
- `learning_rate_high` (float, >0, default=1e-4): Learning rate for high-level agent

#### RewardConfig
Reward function configuration:
- `alpha` (float, default=10.0): Investment reward coefficient
- `beta` (float, default=0.1): Stability penalty coefficient
- `gamma` (float, default=5.0): Overspend penalty coefficient
- `delta` (float, default=20.0): Debt penalty coefficient
- `lambda_` (float, default=1.0): Wealth growth coefficient (alias: "lambda")
- `mu` (float, default=0.5): Stability bonus coefficient

#### ScenarioConfig
Complete scenario configuration:
- `name` (str, 1-100 chars): Scenario name
- `description` (str, optional, max 500 chars): Scenario description
- `environment` (EnvironmentConfig): Environment configuration
- `training` (TrainingConfig, optional): Training configuration (uses defaults if not provided)
- `reward` (RewardConfig, optional): Reward configuration (uses defaults if not provided)

#### TrainingRequest
Request to start model training:
- `scenario_name` (str, min 1 char): Name of the scenario to train on
- `num_episodes` (int, >0, default=1000): Number of training episodes
- `save_interval` (int, >0, default=100): Save checkpoint every N episodes
- `eval_episodes` (int, >0, default=10): Number of evaluation episodes
- `seed` (int, optional): Random seed for reproducibility

#### SimulationRequest
Request to run a simulation:
- `model_name` (str, min 1 char): Name of the trained model to use
- `scenario_name` (str, min 1 char): Name of the scenario to simulate
- `num_episodes` (int, >0, default=10): Number of simulation episodes
- `seed` (int, optional): Random seed for reproducibility

#### ReportRequest
Request to generate a report:
- `simulation_id` (str, min 1 char): ID of the simulation results
- `report_type` (str, "pdf"|"html"): Report format
- `include_sections` (list[str], optional): Sections to include in report
- `title` (str, optional, max 200 chars): Custom report title

### Response Models (`backend/models/responses.py`)

See `backend/models/responses.py` for complete response model definitions including:
- `TrainingProgress`: Real-time training progress updates
- `TrainingStatus`: Current training status
- `EpisodeResult`: Single simulation episode results
- `SimulationResults`: Aggregated simulation results
- `ScenarioSummary`, `ModelSummary`: Summary information
- `HealthCheckResponse`, `ErrorResponse`: System responses

## File Management Utilities

The `backend/utils/file_manager.py` module provides comprehensive file management for the HRL Finance System. See [FILE_MANAGER_README.md](utils/FILE_MANAGER_README.md) for detailed documentation.

**Key Features:**
- YAML configuration file operations (read, write, delete, list)
- PyTorch model management (save, load, delete, list)
- JSON results storage (save, read, list)
- Security features: filename sanitization and path validation
- Automatic directory creation and management

**Example Usage:**
```python
from backend.utils.file_manager import read_yaml_config, list_pytorch_models

# Read a scenario configuration
config = read_yaml_config('bologna_coppia.yaml', scenarios=True)

# List all trained models
models = list_pytorch_models()
for model in models:
    print(f"{model['name']}: {model['size_mb']} MB")
```

## Current Implementation Status

✅ **Completed:**
- FastAPI application initialization
- Root endpoint with API information
- Health check endpoint
- CORS middleware configuration (ready for frontend integration)
- OpenAPI documentation (auto-generated)
- **Pydantic request models with comprehensive validation**
- **Pydantic response models for all API endpoints**
- **File management utilities with security features**
- **Scenarios API (complete CRUD operations)**
- **Scenario service layer with business logic**
- **Scenario templates (5 preset profiles)**

🚧 **In Progress:**
- Training API and WebSocket support
- Simulation API
- Models API
- Reports API
