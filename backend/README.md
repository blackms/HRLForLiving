# HRL Finance System Backend

FastAPI backend for the HRL Finance System.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server (with WebSocket support)
uvicorn backend.main:socket_app --reload --port 8000
```

## Testing

The backend includes a comprehensive test suite using pytest.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=backend --cov-report=html

# Run specific test file
pytest backend/tests/test_api_scenarios.py

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_create_scenario"
```

### Test Dependencies

The following testing libraries are included:
- **pytest** (7.4.3) - Testing framework
- **pytest-asyncio** (0.21.1) - Async test support for FastAPI
- **httpx** (0.25.2) - HTTP client for testing API endpoints

### Test Structure

```
backend/tests/
├── conftest.py              # Shared fixtures and configuration
├── test_api_scenarios.py    # Scenarios API tests
├── test_api_training.py     # Training API tests
├── test_api_simulation.py   # Simulation API tests
├── test_api_models.py       # Models API tests
├── test_api_reports.py      # Reports API tests
├── test_services.py         # Service layer tests
├── test_file_manager.py     # File management tests
└── test_integration.py      # End-to-end integration tests
```

### Writing Tests

Example test for API endpoint:

```python
import pytest
from httpx import AsyncClient
from backend.main import app

@pytest.mark.asyncio
async def test_list_scenarios():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/scenarios")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
```

### Test Coverage

Run tests with coverage to ensure code quality:

```bash
pytest --cov=backend --cov-report=term-missing
```

This will show which lines of code are not covered by tests.

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

### Training API ✅ **IMPLEMENTED**

- `POST /api/training/start` - Start training a model on a scenario
- `POST /api/training/stop` - Stop the current training process
- `GET /api/training/status` - Get current training status

### WebSocket ✅ **IMPLEMENTED**

- `ws://localhost:8000/socket.io` - WebSocket endpoint for real-time training updates
- Real-time progress updates during training
- Events: `training_started`, `training_progress`, `training_completed`, `training_stopped`, `training_error`

See [TRAINING_API.md](api/TRAINING_API.md) for detailed Training API documentation.

### Simulation API ✅ **IMPLEMENTED**

- `POST /api/simulation/run` - Run a simulation with a trained model
- `GET /api/simulation/results/{simulation_id}` - Get simulation results
- `GET /api/simulation/history` - List all past simulations

See [SIMULATION_API.md](api/SIMULATION_API.md) for detailed Simulation API documentation.

### Models API ✅ **IMPLEMENTED**

- `GET /api/models` - List all trained models with summary information
- `GET /api/models/{name}` - Get detailed model information including training history
- `DELETE /api/models/{name}` - Delete a trained model

See [MODELS_API.md](api/MODELS_API.md) for detailed Models API documentation.

**Key Features:**
- Automatic metadata extraction from training history
- Training metrics aggregation (rewards, duration, cash, invested)
- Processed history with statistics (count, mean, min, max)
- Graceful handling of missing metadata/history files
- NaN/Infinity filtering for robust metrics
- Complete model file deletion (weights, metadata, history)

### Simulation API ✅ **IMPLEMENTED**

- `POST /api/simulation/run` - Run a simulation with a trained model
- `GET /api/simulation/results/{id}` - Get results for a specific simulation
- `GET /api/simulation/history` - List all past simulations

See [SIMULATION_API.md](api/SIMULATION_API.md) for detailed Simulation API documentation.

**Key Features:**
- Deterministic policy evaluation (no exploration)
- Comprehensive trajectory data collection
- Aggregate statistics calculation
- Results persistence to JSON files
- Simulation history management

### Reports API ✅ **IMPLEMENTED**

- `POST /api/reports/generate` - Generate PDF or HTML report from simulation results
- `GET /api/reports/{report_id}` - Download a generated report file
- `GET /api/reports/list` - List all generated reports
- `GET /api/reports/{report_id}/metadata` - Get report metadata

See [REPORTS_API.md](api/REPORTS_API.md) for detailed Reports API documentation.

**Key Features:**
- HTML and PDF report generation
- Customizable report sections (summary, scenario, training, results, strategy, charts)
- Professional styled HTML templates
- Responsive design for web viewing
- Report metadata storage and retrieval
- File download support

### API Documentation

Once running, visit:
- **API Root**: http://localhost:8000 - Overview with navigation links and endpoint listing
- **Health Check**: http://localhost:8000/health - Service health status with timestamp
- **Swagger UI**: http://localhost:8000/docs - Interactive API documentation with try-it-out functionality
- **ReDoc**: http://localhost:8000/redoc - Alternative documentation view with better readability
- **OpenAPI Schema**: http://localhost:8000/openapi.json - Machine-readable API specification

### Enhanced OpenAPI Documentation ✅

The API now includes comprehensive OpenAPI metadata:

**API Tags:**
- `scenarios` - Operations for managing financial scenarios
- `training` - Operations for training AI models on scenarios
- `simulation` - Operations for running simulations with trained models
- `models` - Operations for managing trained models
- `reports` - Operations for generating and downloading reports
- `general` - Root and health check endpoints

**Features:**
- Detailed descriptions for each endpoint group
- Complete request/response examples
- WebSocket connection documentation
- Getting started workflow guide
- Authentication and rate limiting notes
- Contact information and license details

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

## Scenarios API Usage

### List All Scenarios

```bash
curl http://localhost:8000/api/scenarios
```

**Response:**
```json
[
  {
    "name": "bologna_coppia",
    "description": "Young couple in Bologna with rental expenses",
    "income": 3200,
    "fixed_expenses": 1800,
    "variable_expenses": 800,
    "available_monthly": 600,
    "available_pct": 18.8,
    "risk_tolerance": 0.55,
    "updated_at": "2025-11-06T10:30:00",
    "size": 1234
  }
]
```

### Get Scenario Details

```bash
curl http://localhost:8000/api/scenarios/bologna_coppia
```

**Response:**
```json
{
  "name": "bologna_coppia",
  "description": "Young couple in Bologna with rental expenses",
  "environment": {
    "income": 3200,
    "fixed_expenses": 1800,
    "variable_expense_mean": 800,
    "variable_expense_std": 150,
    "inflation": 0.02,
    "safety_threshold": 3500,
    "max_months": 120,
    "initial_cash": 10000,
    "risk_tolerance": 0.55,
    "investment_return_mean": 0.005,
    "investment_return_std": 0.02,
    "investment_return_type": "stochastic"
  },
  "training": { ... },
  "reward": { ... },
  "created_at": "2025-11-05T15:20:00",
  "updated_at": "2025-11-06T10:30:00",
  "size": 1234
}
```

### Create New Scenario

```bash
curl -X POST http://localhost:8000/api/scenarios \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_scenario",
    "description": "My custom financial scenario",
    "environment": {
      "income": 3000,
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
  }'
```

**Response (201 Created):**
```json
{
  "name": "my_scenario",
  "description": "My custom financial scenario",
  "path": "configs/scenarios/my_scenario.yaml",
  "created_at": "2025-11-06T12:00:00",
  "updated_at": "2025-11-06T12:00:00",
  "message": "Scenario created successfully"
}
```

### Update Scenario

```bash
curl -X PUT http://localhost:8000/api/scenarios/my_scenario \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_scenario",
    "description": "Updated description",
    "environment": { ... }
  }'
```

### Delete Scenario

```bash
curl -X DELETE http://localhost:8000/api/scenarios/my_scenario
```

**Response:**
```json
{
  "name": "my_scenario",
  "message": "Scenario deleted successfully"
}
```

### Get Templates

```bash
curl http://localhost:8000/api/scenarios/templates
```

**Response:**
```json
[
  {
    "name": "conservative",
    "display_name": "Conservative Profile",
    "description": "Low-risk profile with high savings buffer",
    "environment": { ... },
    "training": { ... },
    "reward": { ... }
  },
  {
    "name": "balanced",
    "display_name": "Balanced Profile",
    "description": "Moderate risk with balanced savings and investment",
    "environment": { ... },
    "training": { ... },
    "reward": { ... }
  },
  ...
]
```

**Available Templates:**
- `conservative` - Low-risk profile with high savings buffer
- `balanced` - Moderate risk with balanced savings and investment
- `aggressive` - High-risk profile focused on investment growth
- `young_professional` - Single professional with owned home
- `young_couple` - Dual income couple with rental

## Training API Usage

### Start Training

```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_name": "bologna_coppia",
    "num_episodes": 1000,
    "save_interval": 100,
    "eval_episodes": 10,
    "seed": 42
  }'
```

**Response (202 Accepted):**
```json
{
  "message": "Training started successfully",
  "data": {
    "status": "started",
    "scenario_name": "bologna_coppia",
    "num_episodes": 1000,
    "start_time": "2025-11-06T10:30:00"
  }
}
```

### Get Training Status

```bash
curl http://localhost:8000/api/training/status
```

**Response:**
```json
{
  "is_training": true,
  "scenario_name": "bologna_coppia",
  "current_episode": 450,
  "total_episodes": 1000,
  "start_time": "2025-11-06T10:30:00",
  "latest_progress": {
    "episode": 450,
    "total_episodes": 1000,
    "avg_reward": 168.5,
    "avg_duration": 118.2,
    "avg_cash": 5234.67,
    "avg_invested": 12500.00,
    "stability": 0.985,
    "goal_adherence": 0.0234,
    "elapsed_time": 323.45
  }
}
```

### Stop Training

```bash
curl -X POST http://localhost:8000/api/training/stop
```

**Response:**
```json
{
  "message": "Training stopped successfully",
  "data": {
    "status": "stopped",
    "scenario_name": "bologna_coppia",
    "episodes_completed": 450,
    "total_episodes": 1000
  }
}
```

### WebSocket Connection (Python)

```python
import socketio

sio = socketio.Client()

@sio.on('training_progress')
def on_progress(data):
    print(f"Episode {data['episode']}/{data['total_episodes']}")
    print(f"Reward: {data['avg_reward']:.2f}, Stability: {data['stability']:.2%}")

@sio.on('training_completed')
def on_complete(data):
    print(f"Training completed: {data['scenario_name']}")

sio.connect('http://localhost:8000', socketio_path='/socket.io')
sio.wait()
```

### WebSocket Connection (JavaScript)

```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:8000', { path: '/socket.io' });

socket.on('connect', () => {
  console.log('Connected to training updates');
});

socket.on('training_progress', (data) => {
  console.log(`Episode ${data.episode}/${data.total_episodes}`);
  console.log(`Reward: ${data.avg_reward.toFixed(2)}`);
});

socket.on('training_completed', (data) => {
  console.log(`Training completed: ${data.scenario_name}`);
});
```

## Simulation API Usage

### Run Simulation

```bash
curl -X POST http://localhost:8000/api/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "bologna_coppia",
    "scenario_name": "bologna_coppia",
    "num_episodes": 10,
    "seed": 42
  }'
```

**Response (202 Accepted):**
```json
{
  "status": "completed",
  "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
  "message": "Simulation completed with 10 episodes",
  "results": {
    "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
    "scenario_name": "bologna_coppia",
    "model_name": "bologna_coppia",
    "num_episodes": 10,
    "timestamp": "2025-11-06T12:00:00",
    "seed": 42,
    "duration_mean": 27.3,
    "total_wealth_mean": 20020.8,
    "avg_invest_pct": 0.333,
    "avg_save_pct": 0.333,
    "avg_consume_pct": 0.334,
    "episodes": [...]
  }
}
```

### Get Simulation Results

```bash
curl http://localhost:8000/api/simulation/results/bologna_coppia_bologna_coppia_1730901234
```

**Response:**
```json
{
  "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
  "scenario_name": "bologna_coppia",
  "model_name": "bologna_coppia",
  "num_episodes": 10,
  "duration_mean": 27.3,
  "total_wealth_mean": 20020.8,
  "episodes": [
    {
      "episode_id": 0,
      "duration": 27,
      "final_cash": 842.5,
      "total_wealth": 20020.8,
      "months": [1, 2, 3, ...],
      "cash_history": [10000, 9500, ...],
      "actions": [[0.33, 0.33, 0.34], ...]
    }
  ]
}
```

### List Simulation History

```bash
curl http://localhost:8000/api/simulation/history
```

**Response:**
```json
{
  "simulations": [
    {
      "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
      "scenario_name": "bologna_coppia",
      "model_name": "bologna_coppia",
      "num_episodes": 10,
      "timestamp": "2025-11-06T12:00:00",
      "total_wealth_mean": 20020.8,
      "duration_mean": 27.3
    }
  ],
  "total": 1
}
```

### Python Client

```python
import requests

# Run simulation
response = requests.post('http://localhost:8000/api/simulation/run', json={
    'model_name': 'bologna_coppia',
    'scenario_name': 'bologna_coppia',
    'num_episodes': 10,
    'seed': 42
})

results = response.json()
simulation_id = results['simulation_id']
print(f"Mean wealth: ${results['results']['total_wealth_mean']:.2f}")
print(f"Investment strategy: {results['results']['avg_invest_pct']:.1%}")

# Get results later
response = requests.get(
    f'http://localhost:8000/api/simulation/results/{simulation_id}'
)
results = response.json()

# List all simulations
response = requests.get('http://localhost:8000/api/simulation/history')
history = response.json()
print(f"Total simulations: {history['total']}")
```

## Reports API Usage

### Generate Report

```bash
curl -X POST http://localhost:8000/api/reports/generate \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
    "report_type": "html",
    "include_sections": ["summary", "scenario", "results", "strategy"],
    "title": "Financial Analysis Report - Bologna Couple"
  }'
```

**Response (202 Accepted):**
```json
{
  "report_id": "report_bologna_coppia_bologna_coppia_1730901234_1730901300",
  "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
  "report_type": "html",
  "title": "Financial Analysis Report - Bologna Couple",
  "generated_at": "2025-11-06T12:05:00",
  "file_path": "reports/report_bologna_coppia_bologna_coppia_1730901234_1730901300.html",
  "file_size_kb": 125.5,
  "sections": ["summary", "scenario", "results", "strategy"],
  "status": "completed",
  "message": "Report generated successfully: report_bologna_coppia_bologna_coppia_1730901234_1730901300"
}
```

**Available Sections:**
- `summary` - Summary statistics (duration, wealth, gains)
- `scenario` - Scenario configuration details
- `training` - Training configuration
- `results` - Detailed results breakdown
- `strategy` - Strategy learned visualization
- `charts` - Episode data and visualizations

### Download Report

```bash
curl -X GET http://localhost:8000/api/reports/report_bologna_coppia_bologna_coppia_1730901234_1730901300 \
  -o financial_report.html
```

### List All Reports

```bash
curl http://localhost:8000/api/reports/list
```

**Response:**
```json
{
  "reports": [
    {
      "report_id": "report_bologna_coppia_bologna_coppia_1730901234_1730901300",
      "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
      "report_type": "html",
      "title": "Financial Analysis Report - Bologna Couple",
      "generated_at": "2025-11-06T12:05:00",
      "file_path": "reports/report_bologna_coppia_bologna_coppia_1730901234_1730901300.html",
      "file_size_kb": 125.5,
      "sections": ["summary", "scenario", "results", "strategy"]
    }
  ],
  "total": 1
}
```

### Get Report Metadata

```bash
curl http://localhost:8000/api/reports/report_bologna_coppia_bologna_coppia_1730901234_1730901300/metadata
```

### Python Client

```python
import requests

# Generate HTML report
response = requests.post('http://localhost:8000/api/reports/generate', json={
    'simulation_id': 'bologna_coppia_bologna_coppia_1730901234',
    'report_type': 'html',
    'title': 'My Financial Report'
})

report_data = response.json()
report_id = report_data['report_id']

# Download the report
report_file = requests.get(
    f'http://localhost:8000/api/reports/{report_id}'
)

with open('my_report.html', 'wb') as f:
    f.write(report_file.content)

print(f"Report saved: my_report.html ({report_data['file_size_kb']} KB)")

# List all reports
response = requests.get('http://localhost:8000/api/reports/list')
reports = response.json()
print(f"Total reports: {reports['total']}")
```

**Note:** PDF generation requires the WeasyPrint library. If not installed, the API will generate an HTML report instead and return an error message with installation instructions:

```bash
pip install weasyprint
```

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
- **Scenarios API (complete CRUD operations)** ⭐
- **Scenario service layer with business logic** ⭐
- **Scenario templates (5 preset profiles)** ⭐
- **Training API with WebSocket support** ⭐
- **Training service layer with HRL orchestration** ⭐
- **Real-time training progress updates via WebSocket** ⭐
- **Asynchronous training execution with progress callbacks** ⭐
- **Automatic model checkpointing and persistence** ⭐
- **Simulation API (complete evaluation system)** ⭐
- **Simulation service layer with deterministic policy** ⭐
- **Simulation results storage and retrieval** ⭐
- **Models API (complete model management)** ⭐
- **Model service layer with metadata extraction** ⭐
- **Training history processing and statistics** ⭐

✅ **Completed:**
- **Reports API (PDF and HTML report generation)** ⭐
- **Report service layer with comprehensive formatting** ⭐
- **HTML report generation with styled templates** ⭐
- **PDF report generation (requires WeasyPrint)** ⭐
- **Report metadata storage and retrieval** ⭐
