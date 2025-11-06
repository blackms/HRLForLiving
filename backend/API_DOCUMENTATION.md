# HRL Finance System API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
   - [Scenarios](#scenarios)
   - [Training](#training)
   - [Simulation](#simulation)
   - [Models](#models)
   - [Reports](#reports)
5. [WebSocket Events](#websocket-events)
6. [Request/Response Examples](#requestresponse-examples)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)

## Overview

The HRL Finance System API provides a RESTful interface for managing financial scenarios, training AI models, running simulations, and generating reports using Hierarchical Reinforcement Learning.

**Base URL:** `http://localhost:8000`

**API Version:** 1.0.0

**Interactive Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Schema: `http://localhost:8000/openapi.json`

## Getting Started

### Installation

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Start the server:
```bash
uvicorn main:app --reload --port 8000
```

3. Access the API documentation at `http://localhost:8000/docs`

### Quick Start Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Create a scenario
scenario = {
    "name": "my_first_scenario",
    "description": "A simple financial scenario",
    "environment": {
        "income": 3000,
        "fixed_expenses": 1200,
        "variable_expense_mean": 500,
        "variable_expense_std": 100,
        "inflation": 0.002,
        "safety_threshold": 5000,
        "max_months": 120,
        "initial_cash": 10000,
        "risk_tolerance": 0.5,
        "investment_return_mean": 0.005,
        "investment_return_std": 0.02,
        "investment_return_type": "stochastic"
    }
}

response = requests.post(f"{BASE_URL}/api/scenarios", json=scenario)
print(response.json())

# 2. Start training
training_request = {
    "scenario_name": "my_first_scenario",
    "num_episodes": 100,
    "save_interval": 50,
    "eval_episodes": 10
}

response = requests.post(f"{BASE_URL}/api/training/start", json=training_request)
print(response.json())

# 3. Check training status
response = requests.get(f"{BASE_URL}/api/training/status")
print(response.json())

# 4. Run simulation (after training completes)
simulation_request = {
    "model_name": "my_first_scenario",
    "scenario_name": "my_first_scenario",
    "num_episodes": 10
}

response = requests.post(f"{BASE_URL}/api/simulation/run", json=simulation_request)
print(response.json())
```

## Authentication

**Current Status:** No authentication required.

**Future Implementation:** For production deployments, consider implementing:
- API Key authentication
- OAuth 2.0
- JWT tokens

Add authentication headers to requests:
```python
headers = {
    "Authorization": "Bearer YOUR_API_KEY"
}
```

## API Endpoints

### Scenarios

Manage financial scenarios with customizable parameters.

#### List All Scenarios

```http
GET /api/scenarios
```

**Response:** `200 OK`
```json
[
  {
    "name": "benedetta_case",
    "description": "Young professional with owned home",
    "income": 2000,
    "fixed_expenses": 770,
    "variable_expenses": 500,
    "available_monthly": 730,
    "available_pct": 36.5,
    "risk_tolerance": 0.8,
    "updated_at": "2024-01-15T10:30:00Z",
    "size": 1024
  }
]
```

#### Get Scenario Details

```http
GET /api/scenarios/{name}
```

**Parameters:**
- `name` (path, required): Scenario name

**Response:** `200 OK`
```json
{
  "name": "benedetta_case",
  "description": "Young professional with owned home",
  "environment": {
    "income": 2000,
    "fixed_expenses": 770,
    "variable_expense_mean": 500,
    "variable_expense_std": 120,
    "inflation": 0.002,
    "safety_threshold": 5000,
    "max_months": 120,
    "initial_cash": 5000,
    "risk_tolerance": 0.8,
    "investment_return_mean": 0.005,
    "investment_return_std": 0.02,
    "investment_return_type": "stochastic"
  },
  "training": {
    "num_episodes": 5000,
    "gamma_low": 0.95,
    "gamma_high": 0.99,
    "high_period": 6,
    "batch_size": 32,
    "learning_rate_low": 0.0003,
    "learning_rate_high": 0.0001
  },
  "reward": {
    "alpha": 10.0,
    "beta": 0.1,
    "gamma": 5.0,
    "delta": 20.0,
    "lambda": 1.0,
    "mu": 0.5
  },
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "size": 1024
}
```

**Errors:**
- `404 Not Found`: Scenario does not exist
- `500 Internal Server Error`: Failed to read scenario

#### Create Scenario

```http
POST /api/scenarios
```

**Request Body:**
```json
{
  "name": "my_scenario",
  "description": "Optional description",
  "environment": {
    "income": 3000,
    "fixed_expenses": 1200,
    "variable_expense_mean": 500,
    "variable_expense_std": 100,
    "inflation": 0.002,
    "safety_threshold": 5000,
    "max_months": 120,
    "initial_cash": 10000,
    "risk_tolerance": 0.5,
    "investment_return_mean": 0.005,
    "investment_return_std": 0.02,
    "investment_return_type": "stochastic"
  },
  "training": {
    "num_episodes": 5000,
    "gamma_low": 0.95,
    "gamma_high": 0.99,
    "high_period": 6,
    "batch_size": 32,
    "learning_rate_low": 0.0003,
    "learning_rate_high": 0.0001
  },
  "reward": {
    "alpha": 10.0,
    "beta": 0.1,
    "gamma": 5.0,
    "delta": 20.0,
    "lambda": 1.0,
    "mu": 0.5
  }
}
```

**Response:** `201 Created`
```json
{
  "name": "my_scenario",
  "description": "Optional description",
  "path": "configs/my_scenario.yaml",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "message": "Scenario created successfully"
}
```

**Errors:**
- `400 Bad Request`: Invalid parameters
- `409 Conflict`: Scenario name already exists
- `500 Internal Server Error`: Failed to create scenario

#### Update Scenario

```http
PUT /api/scenarios/{name}
```

**Parameters:**
- `name` (path, required): Current scenario name

**Request Body:** Same as Create Scenario

**Response:** `200 OK`
```json
{
  "name": "my_scenario",
  "description": "Updated description",
  "path": "configs/my_scenario.yaml",
  "updated_at": "2024-01-15T11:00:00Z",
  "message": "Scenario updated successfully"
}
```

**Errors:**
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Scenario does not exist
- `409 Conflict`: New name already exists
- `500 Internal Server Error`: Failed to update scenario

#### Delete Scenario

```http
DELETE /api/scenarios/{name}
```

**Parameters:**
- `name` (path, required): Scenario name

**Response:** `200 OK`
```json
{
  "name": "my_scenario",
  "message": "Scenario deleted successfully"
}
```

**Errors:**
- `404 Not Found`: Scenario does not exist
- `500 Internal Server Error`: Failed to delete scenario

#### Get Templates

```http
GET /api/scenarios/templates
```

**Response:** `200 OK`
```json
[
  {
    "name": "conservative",
    "display_name": "Conservative Profile",
    "description": "Low risk tolerance with focus on savings",
    "environment": { ... },
    "training": { ... },
    "reward": { ... }
  },
  {
    "name": "balanced",
    "display_name": "Balanced Profile",
    "description": "Moderate risk with balanced approach",
    "environment": { ... },
    "training": { ... },
    "reward": { ... }
  }
]
```

### Training

Train AI models using hierarchical reinforcement learning.

#### Start Training

```http
POST /api/training/start
```

**Request Body:**
```json
{
  "scenario_name": "my_scenario",
  "num_episodes": 1000,
  "save_interval": 100,
  "eval_episodes": 10,
  "seed": 42
}
```

**Response:** `202 Accepted`
```json
{
  "message": "Training started successfully",
  "data": {
    "scenario_name": "my_scenario",
    "num_episodes": 1000,
    "start_time": "2024-01-15T10:30:00Z"
  }
}
```

**Errors:**
- `400 Bad Request`: Training already in progress
- `404 Not Found`: Scenario not found
- `500 Internal Server Error`: Failed to start training

#### Stop Training

```http
POST /api/training/stop
```

**Response:** `200 OK`
```json
{
  "message": "Training stopped successfully",
  "data": {
    "scenario_name": "my_scenario",
    "episodes_completed": 450
  }
}
```

**Errors:**
- `400 Bad Request`: No training in progress
- `500 Internal Server Error`: Failed to stop training

#### Get Training Status

```http
GET /api/training/status
```

**Response:** `200 OK`
```json
{
  "is_training": true,
  "scenario_name": "my_scenario",
  "current_episode": 450,
  "total_episodes": 1000,
  "start_time": "2024-01-15T10:30:00Z",
  "latest_progress": {
    "episode": 450,
    "total_episodes": 1000,
    "avg_reward": 168.5,
    "avg_duration": 118.2,
    "avg_cash": 5234.5,
    "avg_invested": 12500.0,
    "stability": 0.985,
    "goal_adherence": 0.92,
    "elapsed_time": 323.5
  }
}
```

### Simulation

Run simulations with trained models.

#### Run Simulation

```http
POST /api/simulation/run
```

**Request Body:**
```json
{
  "model_name": "my_scenario",
  "scenario_name": "my_scenario",
  "num_episodes": 10,
  "seed": 42
}
```

**Response:** `202 Accepted`
```json
{
  "status": "completed",
  "simulation_id": "my_scenario_bologna_coppia_1762437888",
  "message": "Simulation completed with 10 episodes",
  "results": {
    "simulation_id": "my_scenario_bologna_coppia_1762437888",
    "scenario_name": "my_scenario",
    "model_name": "my_scenario",
    "num_episodes": 10,
    "timestamp": "2024-01-15T10:30:00Z",
    "duration_mean": 27.5,
    "duration_std": 1.3,
    "final_cash_mean": 842.5,
    "final_invested_mean": 18000.0,
    "final_portfolio_mean": 18000.0,
    "total_wealth_mean": 18842.5,
    "total_wealth_std": 234.5,
    "investment_gains_mean": 1178.3,
    "avg_invest_pct": 33.3,
    "avg_save_pct": 33.3,
    "avg_consume_pct": 33.4,
    "episodes": [ ... ]
  }
}
```

**Errors:**
- `404 Not Found`: Model or scenario not found
- `500 Internal Server Error`: Simulation failed

#### Get Simulation Results

```http
GET /api/simulation/results/{simulation_id}
```

**Parameters:**
- `simulation_id` (path, required): Simulation identifier

**Response:** `200 OK`
```json
{
  "simulation_id": "my_scenario_bologna_coppia_1762437888",
  "scenario_name": "my_scenario",
  "model_name": "my_scenario",
  "num_episodes": 10,
  "timestamp": "2024-01-15T10:30:00Z",
  "duration_mean": 27.5,
  "duration_std": 1.3,
  "final_cash_mean": 842.5,
  "final_invested_mean": 18000.0,
  "final_portfolio_mean": 18000.0,
  "total_wealth_mean": 18842.5,
  "total_wealth_std": 234.5,
  "investment_gains_mean": 1178.3,
  "avg_invest_pct": 33.3,
  "avg_save_pct": 33.3,
  "avg_consume_pct": 33.4,
  "episodes": [
    {
      "episode_id": 0,
      "duration": 27,
      "final_cash": 842.5,
      "final_invested": 18000.0,
      "final_portfolio_value": 18000.0,
      "total_wealth": 18842.5,
      "investment_gains": 1178.3,
      "months": [0, 1, 2, ..., 27],
      "cash_history": [5000, 4500, ...],
      "invested_history": [0, 500, ...],
      "portfolio_history": [0, 502.5, ...],
      "actions": [[0.33, 0.33, 0.34], ...]
    }
  ]
}
```

**Errors:**
- `404 Not Found`: Simulation not found
- `500 Internal Server Error`: Failed to retrieve results

#### Get Simulation History

```http
GET /api/simulation/history
```

**Response:** `200 OK`
```json
{
  "simulations": [
    {
      "simulation_id": "my_scenario_bologna_coppia_1762437888",
      "scenario_name": "my_scenario",
      "model_name": "my_scenario",
      "num_episodes": 10,
      "timestamp": "2024-01-15T10:30:00Z",
      "total_wealth_mean": 18842.5
    }
  ],
  "total": 1
}
```

### Models

Manage trained models.

#### List Models

```http
GET /api/models
```

**Response:** `200 OK`
```json
{
  "models": [
    {
      "name": "my_scenario",
      "scenario_name": "my_scenario",
      "size_mb": 2.5,
      "trained_at": "2024-01-15T10:30:00Z",
      "has_metadata": true,
      "episodes": 1000,
      "income": 2000,
      "risk_tolerance": 0.8,
      "final_reward": 168.5,
      "avg_reward": 145.2,
      "max_reward": 175.3,
      "final_duration": 118.2,
      "final_cash": 5234.5,
      "final_invested": 12500.0
    }
  ],
  "total": 1
}
```

#### Get Model Details

```http
GET /api/models/{name}
```

**Parameters:**
- `name` (path, required): Model name

**Response:** `200 OK`
```json
{
  "name": "my_scenario",
  "scenario_name": "my_scenario",
  "high_agent_path": "models/my_scenario_high_agent.pt",
  "low_agent_path": "models/my_scenario_low_agent.pt",
  "size_mb": 2.5,
  "trained_at": "2024-01-15T10:30:00Z",
  "has_metadata": true,
  "has_history": true,
  "episodes": 1000,
  "metadata": { ... },
  "environment_config": { ... },
  "training_config": { ... },
  "reward_config": { ... },
  "training_history": { ... },
  "final_metrics": { ... }
}
```

**Errors:**
- `404 Not Found`: Model not found
- `500 Internal Server Error`: Failed to retrieve model

#### Delete Model

```http
DELETE /api/models/{name}
```

**Parameters:**
- `name` (path, required): Model name

**Response:** `200 OK`
```json
{
  "message": "Model 'my_scenario' deleted successfully",
  "deleted_at": "2024-01-15T11:00:00Z"
}
```

**Errors:**
- `404 Not Found`: Model not found
- `500 Internal Server Error`: Failed to delete model

### Reports

Generate and download reports.

#### Generate Report

```http
POST /api/reports/generate
```

**Request Body:**
```json
{
  "simulation_id": "my_scenario_bologna_coppia_1762437888",
  "report_type": "html",
  "include_sections": [
    "summary",
    "scenario",
    "training",
    "results",
    "strategy",
    "charts"
  ],
  "title": "My Financial Analysis Report"
}
```

**Response:** `202 Accepted`
```json
{
  "report_id": "report_my_scenario_bologna_coppia_1762437888_1762439948",
  "simulation_id": "my_scenario_bologna_coppia_1762437888",
  "report_type": "html",
  "title": "My Financial Analysis Report",
  "generated_at": "2024-01-15T10:30:00Z",
  "file_path": "reports/report_my_scenario_bologna_coppia_1762437888_1762439948.html",
  "file_size_kb": 125.5,
  "sections": ["summary", "scenario", "training", "results", "strategy", "charts"],
  "status": "completed",
  "message": "Report generated successfully: report_my_scenario_bologna_coppia_1762437888_1762439948"
}
```

**Errors:**
- `400 Bad Request`: Invalid report type
- `404 Not Found`: Simulation not found
- `500 Internal Server Error`: Report generation failed

#### List Reports

```http
GET /api/reports/list
```

**Response:** `200 OK`
```json
{
  "reports": [
    {
      "report_id": "report_my_scenario_bologna_coppia_1762437888_1762439948",
      "simulation_id": "my_scenario_bologna_coppia_1762437888",
      "report_type": "html",
      "title": "My Financial Analysis Report",
      "generated_at": "2024-01-15T10:30:00Z",
      "file_size_kb": 125.5
    }
  ],
  "total": 1
}
```

#### Download Report

```http
GET /api/reports/{report_id}
```

**Parameters:**
- `report_id` (path, required): Report identifier

**Response:** `200 OK` (File download)

**Headers:**
- `Content-Type`: `text/html` or `application/pdf`
- `Content-Disposition`: `attachment; filename="report_id.html"`

**Errors:**
- `404 Not Found`: Report not found
- `500 Internal Server Error`: Failed to retrieve report

#### Get Report Metadata

```http
GET /api/reports/{report_id}/metadata
```

**Parameters:**
- `report_id` (path, required): Report identifier

**Response:** `200 OK`
```json
{
  "report_id": "report_my_scenario_bologna_coppia_1762437888_1762439948",
  "simulation_id": "my_scenario_bologna_coppia_1762437888",
  "report_type": "html",
  "title": "My Financial Analysis Report",
  "generated_at": "2024-01-15T10:30:00Z",
  "file_path": "reports/report_my_scenario_bologna_coppia_1762437888_1762439948.html",
  "file_size_kb": 125.5,
  "sections": ["summary", "scenario", "training", "results", "strategy", "charts"]
}
```

## WebSocket Events

Connect to the WebSocket server for real-time training updates.

**Connection URL:** `ws://localhost:8000/socket.io`

### Client Connection

```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:8000', {
  path: '/socket.io',
  transports: ['websocket']
});

socket.on('connect', () => {
  console.log('Connected to training updates');
});

socket.on('disconnect', () => {
  console.log('Disconnected from training updates');
});
```

### Events

#### training_started

Emitted when training begins.

**Payload:**
```json
{
  "scenario_name": "my_scenario",
  "num_episodes": 1000,
  "start_time": "2024-01-15T10:30:00Z"
}
```

#### training_progress

Emitted periodically during training (every episode).

**Payload:**
```json
{
  "episode": 450,
  "total_episodes": 1000,
  "avg_reward": 168.5,
  "avg_duration": 118.2,
  "avg_cash": 5234.5,
  "avg_invested": 12500.0,
  "stability": 0.985,
  "goal_adherence": 0.92,
  "elapsed_time": 323.5
}
```

#### training_completed

Emitted when training finishes successfully.

**Payload:**
```json
{
  "scenario_name": "my_scenario",
  "episodes_completed": 1000,
  "final_metrics": {
    "avg_reward": 168.5,
    "avg_duration": 118.2,
    "stability": 0.985
  }
}
```

#### training_stopped

Emitted when training is stopped by user.

**Payload:**
```json
{
  "scenario_name": "my_scenario",
  "episodes_completed": 450
}
```

#### training_error

Emitted when training encounters an error.

**Payload:**
```json
{
  "message": "Training failed",
  "details": "Error details here"
}
```

## Request/Response Examples

### Complete Workflow Example

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Create a scenario
print("Creating scenario...")
scenario = {
    "name": "example_scenario",
    "description": "Example for documentation",
    "environment": {
        "income": 3000,
        "fixed_expenses": 1200,
        "variable_expense_mean": 500,
        "variable_expense_std": 100,
        "inflation": 0.002,
        "safety_threshold": 5000,
        "max_months": 120,
        "initial_cash": 10000,
        "risk_tolerance": 0.5,
        "investment_return_mean": 0.005,
        "investment_return_std": 0.02,
        "investment_return_type": "stochastic"
    }
}

response = requests.post(f"{BASE_URL}/api/scenarios", json=scenario)
print(f"Scenario created: {response.json()}")

# 2. Start training
print("\nStarting training...")
training_request = {
    "scenario_name": "example_scenario",
    "num_episodes": 100,
    "save_interval": 50,
    "eval_episodes": 10,
    "seed": 42
}

response = requests.post(f"{BASE_URL}/api/training/start", json=training_request)
print(f"Training started: {response.json()}")

# 3. Monitor training status
print("\nMonitoring training...")
while True:
    response = requests.get(f"{BASE_URL}/api/training/status")
    status = response.json()
    
    if not status['is_training']:
        print("Training completed!")
        break
    
    progress = status.get('latest_progress', {})
    print(f"Episode {progress.get('episode', 0)}/{status['total_episodes']} - "
          f"Reward: {progress.get('avg_reward', 0):.2f}")
    
    time.sleep(5)

# 4. Run simulation
print("\nRunning simulation...")
simulation_request = {
    "model_name": "example_scenario",
    "scenario_name": "example_scenario",
    "num_episodes": 10,
    "seed": 42
}

response = requests.post(f"{BASE_URL}/api/simulation/run", json=simulation_request)
results = response.json()
print(f"Simulation completed: {results['simulation_id']}")

# 5. Generate report
print("\nGenerating report...")
report_request = {
    "simulation_id": results['simulation_id'],
    "report_type": "html",
    "include_sections": ["summary", "scenario", "results", "strategy", "charts"],
    "title": "Example Financial Analysis"
}

response = requests.post(f"{BASE_URL}/api/reports/generate", json=report_request)
report = response.json()
print(f"Report generated: {report['report_id']}")

# 6. Download report
print("\nDownloading report...")
response = requests.get(f"{BASE_URL}/api/reports/{report['report_id']}")
with open(f"{report['report_id']}.html", 'wb') as f:
    f.write(response.content)
print("Report downloaded successfully!")
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "additional": "context"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

- `200 OK`: Request succeeded
- `201 Created`: Resource created successfully
- `202 Accepted`: Request accepted for processing
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Common Error Scenarios

#### Scenario Already Exists

```json
{
  "detail": "Scenario 'my_scenario' already exists"
}
```

**Solution:** Use a different name or update the existing scenario with PUT.

#### Training Already in Progress

```json
{
  "detail": "Training is already in progress for scenario 'other_scenario'"
}
```

**Solution:** Wait for current training to complete or stop it first.

#### Model Not Found

```json
{
  "detail": "Model 'nonexistent_model' not found"
}
```

**Solution:** Verify the model name and ensure training has completed.

#### Invalid Parameters

```json
{
  "detail": [
    {
      "loc": ["body", "environment", "income"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

**Solution:** Check parameter constraints and fix invalid values.

## Best Practices

### 1. Scenario Naming

- Use descriptive, lowercase names with underscores
- Avoid special characters except underscores and hyphens
- Examples: `young_professional`, `family_with_kids`, `retirement_planning`

### 2. Training Configuration

- Start with fewer episodes (100-500) for testing
- Use larger episode counts (1000-5000) for production models
- Set save_interval to save checkpoints regularly
- Use consistent seeds for reproducible results

### 3. Simulation

- Run multiple episodes (10-50) for statistical significance
- Use the same seed for comparing different models
- Check simulation results before generating reports

### 4. Error Handling

- Always check HTTP status codes
- Implement retry logic for transient errors
- Log errors for debugging
- Validate inputs before sending requests

### 5. WebSocket Connections

- Implement reconnection logic
- Handle connection timeouts
- Buffer updates to avoid UI overload
- Close connections when not needed

### 6. Performance

- Limit concurrent training sessions
- Clean up old models and reports periodically
- Use appropriate episode counts for your use case
- Monitor server resources during training

### 7. Data Management

- Back up important scenarios and models
- Use descriptive names and descriptions
- Document custom configurations
- Version control scenario configurations

## Support and Resources

- **API Documentation:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`
- **OpenAPI Schema:** `http://localhost:8000/openapi.json`
- **GitHub:** [Project Repository](https://github.com/yourusername/hrl-finance-system)
- **Issues:** [Report Issues](https://github.com/yourusername/hrl-finance-system/issues)

## Changelog

### Version 1.0.0 (2024-01-15)

- Initial API release
- Scenarios CRUD operations
- Training with real-time updates
- Simulation execution
- Model management
- Report generation (HTML/PDF)
- WebSocket support for training updates
