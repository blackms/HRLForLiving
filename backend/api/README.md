# API Endpoints Documentation

This directory contains the API endpoint handlers for the HRL Finance System backend.

## Implemented Endpoints

### Scenarios API (`scenarios.py`)

Complete CRUD operations for managing financial scenarios.

**Status:** ✅ **FULLY IMPLEMENTED**

#### Endpoints

| Method | Path | Description | Status Code |
|--------|------|-------------|-------------|
| GET | `/api/scenarios` | List all scenarios | 200 |
| GET | `/api/scenarios/{name}` | Get scenario details | 200, 404 |
| POST | `/api/scenarios` | Create new scenario | 201, 400, 409 |
| PUT | `/api/scenarios/{name}` | Update scenario | 200, 400, 404, 409 |
| DELETE | `/api/scenarios/{name}` | Delete scenario | 200, 404 |
| GET | `/api/scenarios/templates` | Get preset templates | 200 |

#### Response Models

- **ScenarioSummary**: Basic scenario information with key metrics
  - `name`, `description`, `income`, `fixed_expenses`, `variable_expenses`
  - `available_monthly`, `available_pct`, `risk_tolerance`
  - `updated_at`, `size`

- **ScenarioDetail**: Complete scenario configuration
  - `name`, `description`
  - `environment` (EnvironmentConfig)
  - `training` (TrainingConfig)
  - `reward` (RewardConfig)
  - `created_at`, `updated_at`, `size`

- **ScenarioCreateResponse**: Confirmation after creation
  - `name`, `description`, `path`, `created_at`, `updated_at`, `message`

- **ScenarioUpdateResponse**: Confirmation after update
  - `name`, `description`, `path`, `updated_at`, `message`

- **ScenarioDeleteResponse**: Confirmation after deletion
  - `name`, `message`

- **TemplateResponse**: Template information
  - `name`, `display_name`, `description`
  - `environment`, `training`, `reward`

#### Error Handling

- **400 Bad Request**: Invalid input or validation error
- **404 Not Found**: Scenario doesn't exist
- **409 Conflict**: Scenario name already exists
- **500 Internal Server Error**: Unexpected server error

#### Integration

The Scenarios API integrates with:
- `backend/services/scenario_service.py` - Business logic layer
- `backend/utils/file_manager.py` - File operations with security
- `backend/models/requests.py` - Request validation (ScenarioConfig)
- `backend/models/responses.py` - Response serialization

#### Templates

Five preset scenario templates are available:

1. **Conservative** - Low-risk profile with high savings buffer
   - Risk tolerance: 0.3
   - Safety threshold: $7,500
   - Investment return: 0.4% monthly (≈5% annual)

2. **Balanced** - Moderate risk with balanced savings and investment
   - Risk tolerance: 0.5
   - Safety threshold: $6,000
   - Investment return: 0.5% monthly (≈6% annual)

3. **Aggressive** - High-risk profile focused on investment growth
   - Risk tolerance: 0.8
   - Safety threshold: $4,500
   - Investment return: 0.7% monthly (≈8.7% annual)

4. **Young Professional** - Single professional with owned home
   - Income: €2,000
   - Risk tolerance: 0.65
   - Based on Italian financial scenario

5. **Young Couple** - Dual income couple with rental
   - Income: €3,200
   - Risk tolerance: 0.55
   - Based on Italian financial scenario

#### Usage Examples

See [backend/README.md](../README.md#scenarios-api-usage) for detailed usage examples with curl commands.

### Training API (`training.py`)

Complete training orchestration with real-time WebSocket updates.

**Status:** ✅ **FULLY IMPLEMENTED**

#### Endpoints

| Method | Path | Description | Status Code |
|--------|------|-------------|-------------|
| POST | `/api/training/start` | Start model training | 202, 400, 404, 500 |
| POST | `/api/training/stop` | Stop training | 200, 400, 500 |
| GET | `/api/training/status` | Get training status | 200 |

#### WebSocket Events

**Connection:** `ws://localhost:8000/socket.io`

**Server Events:**
- `connection_established` - Connection confirmation
- `training_started` - Training initiated
- `training_progress` - Real-time progress (every episode)
- `training_completed` - Training finished
- `training_stopped` - Training stopped manually
- `training_error` - Error notification

**Client Events:**
- `connect` - Client connection
- `disconnect` - Client disconnection
- `subscribe_training` - Subscribe to updates

#### Request/Response Models

**TrainingRequest:**
- `scenario_name` (str, required): Scenario to train on
- `num_episodes` (int, default=1000): Number of episodes
- `save_interval` (int, default=100): Checkpoint interval
- `eval_episodes` (int, default=10): Evaluation episodes
- `seed` (int, optional): Random seed

**TrainingStatus:**
- `is_training` (bool): Whether training is active
- `scenario_name` (str): Current scenario
- `current_episode` (int): Current episode number
- `total_episodes` (int): Total episodes
- `start_time` (datetime): Training start time
- `latest_progress` (TrainingProgress): Latest metrics

**TrainingProgress:**
- `episode` (int): Current episode
- `total_episodes` (int): Total episodes
- `avg_reward` (float): Average reward
- `avg_duration` (float): Average duration (months)
- `avg_cash` (float): Average cash balance
- `avg_invested` (float): Average invested amount
- `stability` (float): Stability metric (0-1)
- `goal_adherence` (float): Goal adherence metric
- `elapsed_time` (float): Elapsed time (seconds)

#### Integration

The Training API integrates with:
- `backend/services/training_service.py` - Training orchestration
- `backend/websocket/training_socket.py` - WebSocket event management
- `src/training/hrl_trainer.py` - HRL training logic
- `src/agents/` - High-level and low-level agents
- `src/environment/` - BudgetEnv and RewardEngine

#### Training Process

1. Load scenario configuration from YAML
2. Create environment with EnvironmentConfig
3. Initialize high-level (Strategist) and low-level (Executor) agents
4. Execute training loop with hierarchical learning
5. Send progress updates via WebSocket every episode
6. Save checkpoints at specified intervals
7. Save final models and training history

#### Model Storage

**Final Models** (`models/` directory):
- `{scenario_name}_high_agent.pt` - High-level agent
- `{scenario_name}_low_agent.pt` - Low-level agent
- `{scenario_name}_history.json` - Training history

**Checkpoints** (`models/checkpoints/{scenario_name}/`):
- `checkpoint_episode_{N}/` - Periodic checkpoints
- Contains: high_agent.pt, low_agent.pt, metadata.json, training_history.json

#### Usage Examples

See [TRAINING_API.md](TRAINING_API.md) for detailed usage examples with Python and JavaScript clients.

## Upcoming Endpoints

### Simulation API (`simulation.py`)

Complete simulation execution with trained models.

**Status:** ✅ **FULLY IMPLEMENTED**

#### Endpoints

| Method | Path | Description | Status Code |
|--------|------|-------------|-------------|
| POST | `/api/simulation/run` | Run simulation | 202 |
| GET | `/api/simulation/results/{id}` | Get simulation results | 200, 404 |
| GET | `/api/simulation/history` | List past simulations | 200 |

#### Request/Response Models

**SimulationRequest:**
- `model_name` (str, required): Trained model name
- `scenario_name` (str, required): Scenario name
- `num_episodes` (int, default=10): Number of episodes
- `seed` (int, optional): Random seed

**SimulationResults:**
- `simulation_id` (str): Unique identifier
- `scenario_name` (str): Scenario used
- `model_name` (str): Model used
- `num_episodes` (int): Episodes run
- `timestamp` (datetime): Completion time
- `seed` (int, optional): Random seed
- Statistics: duration, cash, invested, portfolio, wealth (mean/std)
- Strategy: avg_invest_pct, avg_save_pct, avg_consume_pct
- `episodes` (list): Episode results with trajectory data

**EpisodeResult:**
- `episode_id` (int): Episode identifier
- `duration` (int): Duration in months
- Final metrics: cash, invested, portfolio_value, total_wealth, investment_gains
- Trajectory data: months, cash_history, invested_history, portfolio_history, actions

**SimulationHistoryResponse:**
- `simulations` (list): Simulation summaries
- `total` (int): Total count

#### Integration

The Simulation API integrates with:
- `backend/services/simulation_service.py` - Simulation orchestration
- `src/environment/budget_env.py` - Financial environment
- `src/agents/` - High-level and low-level agents
- `src/utils/analytics.py` - Performance metrics
- `backend/utils/file_manager.py` - Results storage

#### Simulation Process

1. Load scenario configuration from YAML
2. Load trained agent models from PyTorch files
3. Create environment with scenario configuration
4. Execute evaluation episodes with deterministic policy
5. Collect trajectory data (cash, invested, portfolio, actions)
6. Calculate aggregate statistics across episodes
7. Save results to JSON file
8. Return results to client

#### Deterministic Policy

Simulations use deterministic policy for consistent evaluation:
- Low-level agent: `act(state, goal, deterministic=True)`
- High-level agent: `select_goal(aggregated_state)`
- No exploration noise
- Reproducible results with same seed

#### Results Storage

**Location:** `results/simulations/` directory

**Filename:** `{model_name}_{scenario_name}_{timestamp}.json`

**Format:** JSON with complete episode data

**Contents:**
- Simulation metadata (ID, scenario, model, timestamp, seed)
- Aggregate statistics (mean/std for all metrics)
- Strategy learned (average action percentages)
- Complete episode data (trajectory, actions, metrics)

#### Usage Examples

See [SIMULATION_API.md](SIMULATION_API.md) for detailed usage examples with Python and JavaScript clients.

### Models API (`models.py`)

Complete model management with metadata extraction and training history.

**Status:** ✅ **FULLY IMPLEMENTED**

#### Endpoints

| Method | Path | Description | Status Code |
|--------|------|-------------|-------------|
| GET | `/api/models` | List all trained models | 200, 500 |
| GET | `/api/models/{name}` | Get model details | 200, 404, 500 |
| DELETE | `/api/models/{name}` | Delete model | 200, 404, 500 |

#### Request/Response Models

**ModelSummary:**
- `name` (str): Model identifier
- `scenario_name` (str): Scenario used for training
- `size_mb` (float): Total model file size in MB
- `trained_at` (str): Training completion timestamp
- `has_metadata` (bool): Whether metadata file exists
- `episodes` (int, optional): Number of training episodes
- `income` (float, optional): Income from environment config
- `risk_tolerance` (float, optional): Risk tolerance from environment config
- `final_reward` (float, optional): Final average reward
- `avg_reward` (float, optional): Average reward across training
- `max_reward` (float, optional): Maximum reward achieved
- `final_duration` (float, optional): Final episode duration
- `final_cash` (float, optional): Final cash balance
- `final_invested` (float, optional): Final invested amount

**ModelDetail:**
- `name` (str): Model identifier
- `scenario_name` (str): Scenario used for training
- `high_agent_path` (str): Path to high-level agent file
- `low_agent_path` (str): Path to low-level agent file
- `size_mb` (float): Total model file size in MB
- `trained_at` (str): Training completion timestamp
- `has_metadata` (bool): Whether metadata file exists
- `has_history` (bool): Whether training history file exists
- `episodes` (int, optional): Number of training episodes
- `metadata` (dict, optional): Full metadata object
- `environment_config` (dict, optional): Environment configuration
- `training_config` (dict, optional): Training hyperparameters
- `reward_config` (dict, optional): Reward function configuration
- `training_history` (dict, optional): Processed training history
- `final_metrics` (dict, optional): Final training metrics

**ModelListResponse:**
- `models` (list): List of model summaries
- `total` (int): Total count

#### Integration

The Models API integrates with:
- `backend/services/model_service.py` - Model management service
- `backend/utils/file_manager.py` - File operations
- `backend/models/responses.py` - Response serialization

#### Model File Structure

Models are stored in `models/` directory:
- `{model_name}_high_agent.pt` - High-level agent weights
- `{model_name}_low_agent.pt` - Low-level agent weights
- `{model_name}_metadata.json` - Training metadata (optional)
- `{model_name}_history.json` - Training history (optional)

#### Training History Metrics

The training history includes:
- `episode_rewards`: Total reward per episode
- `episode_lengths`: Duration in months
- `cash_balances`: Final cash balance
- `total_invested`: Total invested amount
- `cumulative_wealth_growth`: Wealth growth over time
- `cash_stability_index`: Cash stability metric
- `sharpe_ratio`: Risk-adjusted return
- `goal_adherence`: Goal following metric
- `policy_stability`: Policy consistency metric
- `high_level_losses`: High-level agent losses
- `low_level_losses`: Low-level agent losses

Each metric includes: count, first, last, mean, min, max

#### Usage Examples

See [MODELS_API.md](MODELS_API.md) for detailed usage examples with Python, cURL, and JavaScript clients.

### Reports API (`reports.py`)

🚧 **In Development**

- POST `/api/reports/generate` - Generate report
- GET `/api/reports/{id}` - Download report
- GET `/api/reports/list` - List generated reports

## Testing

To test the Scenarios API:

```bash
# Start the server
cd backend
uvicorn main:app --reload --port 8000

# In another terminal, test endpoints
curl http://localhost:8000/api/scenarios
curl http://localhost:8000/api/scenarios/templates
```

Or visit the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

```
API Layer (api/)
    ↓
Service Layer (services/)
    ↓
Utilities (utils/)
    ↓
File System (configs/, models/, results/)
```

Each API endpoint:
1. Validates request using Pydantic models
2. Calls service layer for business logic
3. Service layer uses utilities for file operations
4. Returns structured response with appropriate status code
5. Handles errors with descriptive messages
