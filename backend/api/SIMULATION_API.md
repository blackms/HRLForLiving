# Simulation API Documentation

## Overview

The Simulation API provides endpoints for running evaluations with trained HRL models on financial scenarios. It allows users to test trained models, collect performance metrics, and analyze learned strategies.

## Endpoints

### POST /api/simulation/run

Run a simulation with a trained model on a scenario.

**Request Body:**
```json
{
  "model_name": "bologna_coppia",
  "scenario_name": "bologna_coppia",
  "num_episodes": 10,
  "seed": 42
}
```

**Parameters:**
- `model_name` (string, required): Name of the trained model to use
- `scenario_name` (string, required): Name of the scenario to simulate
- `num_episodes` (integer, optional, default=10): Number of evaluation episodes
- `seed` (integer, optional): Random seed for reproducibility

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
    "duration_std": 1.2,
    "final_cash_mean": 842.5,
    "final_cash_std": 120.3,
    "final_invested_mean": 18000.0,
    "final_invested_std": 500.0,
    "final_portfolio_mean": 19178.3,
    "final_portfolio_std": 550.0,
    "total_wealth_mean": 20020.8,
    "total_wealth_std": 600.0,
    "investment_gains_mean": 1178.3,
    "investment_gains_std": 100.0,
    "avg_invest_pct": 0.333,
    "avg_save_pct": 0.333,
    "avg_consume_pct": 0.334,
    "episodes": [
      {
        "episode_id": 0,
        "duration": 27,
        "final_cash": 842.5,
        "final_invested": 18000.0,
        "final_portfolio_value": 19178.3,
        "total_wealth": 20020.8,
        "investment_gains": 1178.3,
        "months": [1, 2, 3, ..., 27],
        "cash_history": [10000, 9500, 9200, ...],
        "invested_history": [0, 500, 1000, ...],
        "portfolio_history": [0, 502, 1008, ...],
        "actions": [[0.33, 0.33, 0.34], ...]
      },
      ...
    ]
  }
}
```

**Error Responses:**
- `404 Not Found`: Model or scenario not found
- `500 Internal Server Error`: Simulation execution failed

**Example:**
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

### GET /api/simulation/results/{simulation_id}

Get results for a specific simulation.

**Path Parameters:**
- `simulation_id` (string, required): Unique simulation identifier

**Response (200 OK):**
```json
{
  "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
  "scenario_name": "bologna_coppia",
  "model_name": "bologna_coppia",
  "num_episodes": 10,
  "timestamp": "2025-11-06T12:00:00",
  "seed": 42,
  "duration_mean": 27.3,
  "duration_std": 1.2,
  "final_cash_mean": 842.5,
  "final_invested_mean": 18000.0,
  "final_portfolio_mean": 19178.3,
  "total_wealth_mean": 20020.8,
  "investment_gains_mean": 1178.3,
  "avg_invest_pct": 0.333,
  "avg_save_pct": 0.333,
  "avg_consume_pct": 0.334,
  "episodes": [...]
}
```

**Error Responses:**
- `404 Not Found`: Simulation results not found
- `500 Internal Server Error`: Failed to retrieve results

**Example:**
```bash
curl http://localhost:8000/api/simulation/results/bologna_coppia_bologna_coppia_1730901234
```

### GET /api/simulation/history

Get list of all past simulations.

**Response (200 OK):**
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
    },
    {
      "simulation_id": "milano_senior_milano_senior_1730900000",
      "scenario_name": "milano_senior",
      "model_name": "milano_senior",
      "num_episodes": 10,
      "timestamp": "2025-11-06T11:30:00",
      "total_wealth_mean": 25000.0,
      "duration_mean": 30.5
    }
  ],
  "total": 2
}
```

**Error Responses:**
- `500 Internal Server Error`: Failed to list simulations

**Example:**
```bash
curl http://localhost:8000/api/simulation/history
```

## Response Models

### SimulationResults

Complete simulation results with statistics and episode data.

**Fields:**
- `simulation_id` (string): Unique simulation identifier
- `scenario_name` (string): Name of scenario used
- `model_name` (string): Name of model used
- `num_episodes` (integer): Number of episodes run
- `timestamp` (string): Completion timestamp (ISO 8601)
- `seed` (integer, optional): Random seed used
- `duration_mean` (float): Mean episode duration in months
- `duration_std` (float): Standard deviation of duration
- `final_cash_mean` (float): Mean final cash balance
- `final_cash_std` (float): Standard deviation of final cash
- `final_invested_mean` (float): Mean final invested amount
- `final_invested_std` (float): Standard deviation of invested
- `final_portfolio_mean` (float): Mean final portfolio value
- `final_portfolio_std` (float): Standard deviation of portfolio
- `total_wealth_mean` (float): Mean total wealth (cash + portfolio)
- `total_wealth_std` (float): Standard deviation of wealth
- `investment_gains_mean` (float): Mean investment gains/losses
- `investment_gains_std` (float): Standard deviation of gains
- `avg_invest_pct` (float): Average investment percentage (0-1)
- `avg_save_pct` (float): Average save percentage (0-1)
- `avg_consume_pct` (float): Average consume percentage (0-1)
- `episodes` (array): List of episode results

### EpisodeResult

Results from a single simulation episode.

**Fields:**
- `episode_id` (integer): Episode identifier
- `duration` (integer): Episode duration in months
- `final_cash` (float): Final cash balance
- `final_invested` (float): Final invested amount
- `final_portfolio_value` (float): Final portfolio value
- `total_wealth` (float): Total wealth (cash + portfolio)
- `investment_gains` (float): Investment gains/losses
- `months` (array of integers): Month numbers [1, 2, 3, ...]
- `cash_history` (array of floats): Cash balance over time
- `invested_history` (array of floats): Invested amount over time
- `portfolio_history` (array of floats): Portfolio value over time
- `actions` (array of arrays): Actions taken [[invest%, save%, consume%], ...]

### SimulationHistoryResponse

List of past simulations with summary information.

**Fields:**
- `simulations` (array): List of simulation summaries
- `total` (integer): Total number of simulations

**Simulation Summary Fields:**
- `simulation_id` (string): Unique identifier
- `scenario_name` (string): Scenario used
- `model_name` (string): Model used
- `num_episodes` (integer): Number of episodes
- `timestamp` (string): Completion timestamp
- `total_wealth_mean` (float): Mean total wealth
- `duration_mean` (float): Mean duration

## Usage Examples

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

if response.status_code == 202:
    results = response.json()
    simulation_id = results['simulation_id']
    print(f"Simulation ID: {simulation_id}")
    print(f"Mean wealth: ${results['results']['total_wealth_mean']:.2f}")
    print(f"Mean duration: {results['results']['duration_mean']:.1f} months")
    print(f"Investment strategy: {results['results']['avg_invest_pct']:.1%} invest")
else:
    print(f"Error: {response.json()}")

# Get simulation results later
response = requests.get(
    f'http://localhost:8000/api/simulation/results/{simulation_id}'
)
results = response.json()

# Analyze episode data
for episode in results['episodes']:
    print(f"Episode {episode['episode_id']}: "
          f"{episode['duration']} months, "
          f"${episode['total_wealth']:.2f} wealth")

# List all simulations
response = requests.get('http://localhost:8000/api/simulation/history')
history = response.json()
print(f"Total simulations: {history['total']}")
for sim in history['simulations']:
    print(f"  {sim['simulation_id']}: {sim['scenario_name']} - "
          f"${sim['total_wealth_mean']:.2f}")
```

### JavaScript Client

```javascript
import axios from 'axios';

// Run simulation
const runSimulation = async () => {
  try {
    const response = await axios.post('http://localhost:8000/api/simulation/run', {
      model_name: 'bologna_coppia',
      scenario_name: 'bologna_coppia',
      num_episodes: 10,
      seed: 42
    });
    
    const { simulation_id, results } = response.data;
    console.log(`Simulation ID: ${simulation_id}`);
    console.log(`Mean wealth: $${results.total_wealth_mean.toFixed(2)}`);
    console.log(`Mean duration: ${results.duration_mean.toFixed(1)} months`);
    console.log(`Investment strategy: ${(results.avg_invest_pct * 100).toFixed(1)}% invest`);
    
    return simulation_id;
  } catch (error) {
    console.error('Simulation failed:', error.response?.data);
  }
};

// Get simulation results
const getResults = async (simulationId) => {
  try {
    const response = await axios.get(
      `http://localhost:8000/api/simulation/results/${simulationId}`
    );
    
    const results = response.data;
    
    // Analyze episodes
    results.episodes.forEach(episode => {
      console.log(`Episode ${episode.episode_id}: ` +
                  `${episode.duration} months, ` +
                  `$${episode.total_wealth.toFixed(2)} wealth`);
    });
    
    return results;
  } catch (error) {
    console.error('Failed to get results:', error.response?.data);
  }
};

// List simulation history
const listHistory = async () => {
  try {
    const response = await axios.get('http://localhost:8000/api/simulation/history');
    const { simulations, total } = response.data;
    
    console.log(`Total simulations: ${total}`);
    simulations.forEach(sim => {
      console.log(`  ${sim.simulation_id}: ${sim.scenario_name} - ` +
                  `$${sim.total_wealth_mean.toFixed(2)}`);
    });
    
    return simulations;
  } catch (error) {
    console.error('Failed to list history:', error.response?.data);
  }
};

// Complete workflow
const simulationWorkflow = async () => {
  const simulationId = await runSimulation();
  if (simulationId) {
    const results = await getResults(simulationId);
    await listHistory();
  }
};
```

## Simulation Process

1. **Load Configuration**: Read scenario YAML file
2. **Load Models**: Load trained high-level and low-level agent models
3. **Initialize Environment**: Create BudgetEnv with scenario configuration
4. **Run Episodes**: Execute evaluation episodes with deterministic policy
   - Reset environment
   - Generate initial goal from high-level agent
   - Execute episode loop:
     - Low-level agent selects action (deterministic)
     - Environment executes action
     - Record trajectory data
     - Update high-level goal every N steps
5. **Calculate Statistics**: Aggregate metrics across episodes
6. **Save Results**: Store results to JSON file
7. **Return Results**: Send results to client

## Deterministic Policy

Simulations use a deterministic policy (no exploration) to evaluate the learned strategy:
- Low-level agent: `act(state, goal, deterministic=True)`
- High-level agent: `select_goal(aggregated_state)` (always deterministic)

This ensures consistent and reproducible results for evaluation.

## Results Storage

Results are saved to `results/simulations/` directory:
- Filename: `{simulation_id}.json`
- Format: JSON with complete episode data
- Includes: configuration, statistics, trajectory data

## Performance Considerations

- **Execution Time**: ~1-2 seconds per episode
- **Memory Usage**: Loads two agent models + episode data
- **Disk I/O**: Reads scenario + models, writes results
- **Network**: Synchronous API (no WebSocket needed)

## Error Handling

All endpoints return appropriate HTTP status codes:

- `202 Accepted`: Simulation started/completed successfully
- `200 OK`: Results retrieved successfully
- `404 Not Found`: Model, scenario, or simulation not found
- `500 Internal Server Error`: Unexpected error

Error responses include:
```json
{
  "error": "NotFound",
  "message": "Model 'nonexistent' not found",
  "timestamp": "2025-11-06T12:00:00"
}
```

## Integration with Frontend

The Simulation API is designed to integrate seamlessly with the React frontend:

1. **Simulation Runner Page**: Calls POST /api/simulation/run
2. **Results Viewer Page**: Calls GET /api/simulation/results/{id}
3. **Dashboard**: Calls GET /api/simulation/history for recent simulations
4. **Comparison View**: Fetches multiple simulation results for comparison

## Related Documentation

- [Backend README](../README.md)
- [API Models Documentation](../models/API_MODELS.md)
- [File Manager Documentation](../utils/FILE_MANAGER_README.md)
- [Training API Documentation](TRAINING_API.md)
