# Models API Documentation

The Models API provides endpoints for managing trained HRL models.

## Base URL

```
/api/models
```

## Endpoints

### 1. List All Models

Get a list of all trained models with summary information.

**Endpoint:** `GET /api/models`

**Response:** `200 OK`

```json
{
  "models": [
    {
      "name": "benedetta_case",
      "scenario_name": "benedetta_case",
      "size_mb": 0.14,
      "trained_at": "2025-11-04T22:07:30.707963",
      "has_metadata": false,
      "episodes": null,
      "income": null,
      "risk_tolerance": null,
      "final_reward": 159.79,
      "avg_reward": 170.47,
      "max_reward": 197.58,
      "final_duration": 25.0,
      "final_cash": -304.66,
      "final_invested": 16666.67
    }
  ],
  "total": 1
}
```

**Fields:**
- `name`: Model identifier
- `scenario_name`: Scenario used for training
- `size_mb`: Total model file size in MB
- `trained_at`: Training completion timestamp (ISO 8601)
- `has_metadata`: Whether metadata file exists
- `episodes`: Number of training episodes (if metadata available)
- `income`: Income from environment config (if metadata available)
- `risk_tolerance`: Risk tolerance from environment config (if metadata available)
- `final_reward`: Final average reward from training
- `avg_reward`: Average reward across all training episodes
- `max_reward`: Maximum reward achieved during training
- `final_duration`: Final episode duration in months
- `final_cash`: Final cash balance
- `final_invested`: Final invested amount

### 2. Get Model Details

Get detailed information about a specific trained model.

**Endpoint:** `GET /api/models/{name}`

**Parameters:**
- `name` (path): Model name

**Response:** `200 OK`

```json
{
  "name": "benedetta_case",
  "scenario_name": "benedetta_case",
  "high_agent_path": "models/benedetta_case_high_agent.pt",
  "low_agent_path": "models/benedetta_case_low_agent.pt",
  "size_mb": 0.14,
  "trained_at": "2025-11-04T22:07:30.707963",
  "has_metadata": false,
  "has_history": true,
  "episodes": null,
  "metadata": null,
  "environment_config": null,
  "training_config": null,
  "reward_config": null,
  "training_history": {
    "episode_rewards": {
      "count": 500,
      "first": 169.43,
      "last": 159.80,
      "mean": 170.47,
      "min": 145.13,
      "max": 197.58
    },
    "episode_lengths": {
      "count": 500,
      "first": 26,
      "last": 25,
      "mean": 26.45,
      "min": 22,
      "max": 31
    }
  },
  "final_metrics": {
    "final_reward": 159.80,
    "avg_reward": 170.47,
    "max_reward": 197.58,
    "min_reward": 145.13,
    "final_duration": 25.0,
    "avg_duration": 26.45,
    "final_cash": -304.66,
    "avg_cash": -258.30,
    "final_invested": 16666.67,
    "avg_invested": 17636.00
  }
}
```

**Fields:**
- `name`: Model identifier
- `scenario_name`: Scenario used for training
- `high_agent_path`: Path to high-level agent file
- `low_agent_path`: Path to low-level agent file
- `size_mb`: Total model file size in MB
- `trained_at`: Training completion timestamp
- `has_metadata`: Whether metadata file exists
- `has_history`: Whether training history file exists
- `episodes`: Number of training episodes (if metadata available)
- `metadata`: Full metadata object (if available)
- `environment_config`: Environment configuration used for training
- `training_config`: Training hyperparameters used
- `reward_config`: Reward function configuration used
- `training_history`: Processed training history with statistics for each metric
- `final_metrics`: Final training metrics summary

**Error Response:** `404 Not Found`

```json
{
  "detail": "Model 'model_name' not found"
}
```

### 3. Delete Model

Delete a trained model and its associated files (model weights, metadata, history).

**Endpoint:** `DELETE /api/models/{name}`

**Parameters:**
- `name` (path): Model name to delete

**Response:** `200 OK`

```json
{
  "message": "Model 'benedetta_case' deleted successfully",
  "deleted_at": "2025-11-04T23:15:30.123456"
}
```

**Error Response:** `404 Not Found`

```json
{
  "detail": "Model 'model_name' not found"
}
```

## Usage Examples

### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8000"

# List all models
response = requests.get(f"{BASE_URL}/api/models")
models = response.json()
print(f"Found {models['total']} models")

# Get model details
model_name = "benedetta_case"
response = requests.get(f"{BASE_URL}/api/models/{model_name}")
model = response.json()
print(f"Model: {model['name']}")
print(f"Final reward: {model['final_metrics']['final_reward']}")

# Delete model
response = requests.delete(f"{BASE_URL}/api/models/{model_name}")
print(response.json()['message'])
```

### cURL

```bash
# List all models
curl http://localhost:8000/api/models

# Get model details
curl http://localhost:8000/api/models/benedetta_case

# Delete model
curl -X DELETE http://localhost:8000/api/models/benedetta_case
```

### JavaScript (fetch)

```javascript
const BASE_URL = "http://localhost:8000";

// List all models
const response = await fetch(`${BASE_URL}/api/models`);
const data = await response.json();
console.log(`Found ${data.total} models`);

// Get model details
const modelName = "benedetta_case";
const modelResponse = await fetch(`${BASE_URL}/api/models/${modelName}`);
const model = await modelResponse.json();
console.log(`Model: ${model.name}`);
console.log(`Final reward: ${model.final_metrics.final_reward}`);

// Delete model
const deleteResponse = await fetch(`${BASE_URL}/api/models/${modelName}`, {
  method: 'DELETE'
});
const result = await deleteResponse.json();
console.log(result.message);
```

## Model File Structure

Models are stored in the `models/` directory with the following files:

- `{model_name}_high_agent.pt`: High-level agent PyTorch weights
- `{model_name}_low_agent.pt`: Low-level agent PyTorch weights
- `{model_name}_metadata.json`: Training metadata (optional)
- `{model_name}_history.json`: Training history (optional)

## Training History Metrics

The training history includes the following metrics:

- `episode_rewards`: Total reward per episode
- `episode_lengths`: Duration of each episode in months
- `cash_balances`: Final cash balance per episode
- `total_invested`: Total invested amount per episode
- `high_level_losses`: High-level agent training losses
- `cumulative_wealth_growth`: Total wealth growth over time
- `cash_stability_index`: Stability of cash balance
- `sharpe_ratio`: Risk-adjusted return metric
- `goal_adherence`: How well the agent adhered to goals
- `policy_stability`: Stability of the learned policy

Each metric includes:
- `count`: Number of valid data points
- `first`: First value in the series
- `last`: Last value in the series
- `mean`: Average value
- `min`: Minimum value
- `max`: Maximum value

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful operation
- `404 Not Found`: Model not found
- `500 Internal Server Error`: Server error

Error responses include a `detail` field with a descriptive message.

## Notes

- Model names are extracted from the file names (e.g., `benedetta_case_high_agent.pt` → `benedetta_case`)
- Scenario names are inferred from model names
- NaN and Infinity values in training history are automatically filtered out
- Metadata and history files are optional; the API gracefully handles their absence
- Deleting a model removes all associated files (weights, metadata, history)
