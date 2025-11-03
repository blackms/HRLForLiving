# Personal Finance Optimization HRL System

A hierarchical reinforcement learning (HRL) system that simulates and learns to optimally allocate monthly salary among investments, savings, and discretionary spending. The system aims to maximize long-term investments while maintaining financial stability through realistic monthly economic simulation.

## Overview

The system implements a two-level hierarchical architecture:
- **High-Level Agent (Strategist)**: Defines medium-term financial strategy (6-12 months)
- **Low-Level Agent (Executor)**: Executes concrete monthly allocation actions

## Project Structure

```
.
├── src/
│   ├── __init__.py              # Main package initialization
│   ├── agents/                  # HRL agent implementations
│   │   ├── __init__.py
│   │   ├── budget_executor.py  # ✅ Low-Level Agent (PPO-based)
│   │   └── financial_strategist.py # ✅ High-Level Agent (HIRO-style)
│   ├── environment/             # Financial environment simulation
│   │   ├── __init__.py
│   │   ├── budget_env.py       # ✅ BudgetEnv implementation
│   │   └── reward_engine.py    # ✅ RewardEngine implementation
│   ├── training/                # Training orchestration
│   │   └── __init__.py
│   └── utils/                   # Configuration and utilities
│       ├── __init__.py
│       ├── config.py            # Configuration dataclasses
│       └── data_models.py       # Core data models
├── examples/                    # Usage examples
│   ├── README.md               # Examples documentation
│   ├── basic_budget_env_usage.py  # ✅ Basic BudgetEnv demo
│   └── reward_engine_usage.py  # ✅ RewardEngine demo
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_budget_env.py      # ✅ BudgetEnv tests
│   ├── test_budget_executor.py # ✅ BudgetExecutor tests
│   ├── test_financial_strategist.py # 🚧 FinancialStrategist tests
│   └── test_reward_engine.py   # ✅ RewardEngine tests
├── Requirements/                # Design documentation
│   └── HRL_Finance_System_Design.md
├── .kiro/specs/                 # Specification documents
│   └── hrl-finance-system/
│       ├── requirements.md      # System requirements
│       ├── design.md           # Detailed design
│       └── tasks.md            # Implementation tasks
└── requirements.txt            # Python dependencies

```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `gymnasium>=0.29.0` - RL environment framework
- `numpy>=1.24.0` - Numerical computing
- `stable-baselines3>=2.0.0` - RL algorithms (PPO)
- `torch>=2.0.0` - Neural network framework
- `pyyaml>=6.0` - Configuration file parsing

## Configuration

The system supports three behavioral profiles with different risk tolerances:

### Conservative Profile
- Risk tolerance: 0.3
- Safety threshold: $1,500
- Focus: Capital preservation and stability

### Balanced Profile (Default)
- Risk tolerance: 0.5
- Safety threshold: $1,000
- Focus: Balanced growth and stability

### Aggressive Profile
- Risk tolerance: 0.8
- Safety threshold: $500
- Focus: Maximum investment growth

## Core Components

### BudgetEnv - Financial Simulation Environment

The `BudgetEnv` is a custom Gymnasium environment that simulates monthly financial decisions with integrated multi-objective reward computation via `RewardEngine`.

**Key Integration Features:**
- Accepts optional `RewardConfig` parameter for customizing reward behavior
- Uses default reward configuration if none provided
- Automatically computes rewards using `RewardEngine.compute_low_level_reward()`
- Passes action, current state, and next state to reward engine for accurate reward calculation

**Usage Example:**
```python
from src.environment import BudgetEnv
from src.utils.config import EnvironmentConfig, RewardConfig

# Create environment configuration
env_config = EnvironmentConfig(
    income=3200,              # Monthly salary
    fixed_expenses=1400,      # Fixed monthly costs
    variable_expense_mean=700, # Average variable expenses
    variable_expense_std=100, # Std dev of variable expenses
    inflation=0.02,           # Annual inflation rate
    safety_threshold=1000,    # Minimum cash buffer
    max_months=60,           # Simulation duration
    initial_cash=0,          # Starting cash balance
    risk_tolerance=0.5       # Risk profile (0-1)
)

# Create reward configuration (optional - uses defaults if not provided)
reward_config = RewardConfig(
    alpha=10.0,    # Investment reward coefficient
    beta=0.1,      # Stability penalty coefficient
    gamma=5.0,     # Overspend penalty coefficient
    delta=20.0,    # Debt penalty coefficient
    lambda_=1.0,   # Wealth growth coefficient
    mu=0.5         # Stability bonus coefficient
)

# Initialize environment with custom reward configuration
env = BudgetEnv(env_config, reward_config)

# Or use default reward configuration
env = BudgetEnv(env_config)

# Reset environment
observation, info = env.reset()

# Take a step with an action [invest_ratio, save_ratio, consume_ratio]
action = [0.3, 0.5, 0.2]  # Invest 30%, save 50%, consume 20%
observation, reward, terminated, truncated, info = env.step(action)

print(f"Cash balance: ${info['cash_balance']:.2f}")
print(f"Total invested: ${info['total_invested']:.2f}")
print(f"Reward: {reward:.2f}")
print(f"Month: {info['month']}")
```

**State Space (7-dimensional):**
- `income`: Monthly salary
- `fixed_expenses`: Fixed monthly costs
- `variable_expenses`: Sampled variable costs for current month
- `cash_balance`: Current liquid funds
- `inflation`: Current inflation rate
- `risk_tolerance`: Agent's risk profile (0-1)
- `t_remaining`: Months remaining in episode

**Action Space (3-dimensional, continuous [0, 1]):**
- `invest_ratio`: Percentage to invest (automatically normalized)
- `save_ratio`: Percentage to save (automatically normalized)
- `consume_ratio`: Percentage for discretionary spending (automatically normalized)

Actions are automatically normalized to sum to 1 using softmax.

### RewardEngine - Multi-Objective Reward Computation

The `RewardEngine` is automatically integrated with `BudgetEnv` and computes rewards balancing multiple financial objectives. You can also use it standalone for custom reward calculations.

**Standalone Usage:**
```python
from src.environment import RewardEngine
from src.utils.config import RewardConfig
import numpy as np

# Create reward configuration
reward_config = RewardConfig(
    alpha=10.0,    # Investment reward coefficient
    beta=0.1,      # Stability penalty coefficient
    gamma=5.0,     # Overspend penalty coefficient
    delta=20.0,    # Debt penalty coefficient
    lambda_=1.0,   # Wealth growth coefficient
    mu=0.5         # Stability bonus coefficient
)

# Initialize reward engine
reward_engine = RewardEngine(reward_config, safety_threshold=1000)

# Compute low-level reward for a single step
action = np.array([0.3, 0.5, 0.2])  # [invest, save, consume]
state = np.array([3200, 1400, 700, 2000, 0.02, 0.5, 50])
next_state = np.array([3200, 1400, 700, 1800, 0.02, 0.5, 49])
reward = reward_engine.compute_low_level_reward(action, state, next_state)

# Compute high-level reward over a strategic period
episode_history = [...]  # List of Transition objects
high_level_reward = reward_engine.compute_high_level_reward(episode_history)
```

### BudgetExecutor - Low-Level Agent

The `BudgetExecutor` is the low-level agent that executes concrete monthly allocation decisions using PPO (Proximal Policy Optimization). It receives both the current financial state and a strategic goal vector from the high-level agent.

**Key Features:**
- 10-dimensional input (7-dimensional state + 3-dimensional goal)
- 3-dimensional continuous action output [invest, save, consume]
- Custom policy network with [128, 128] hidden layers
- Automatic action normalization to ensure sum = 1
- PPO-based learning with discount factor γ = 0.95
- Model save/load functionality

### FinancialStrategist - High-Level Agent

The `FinancialStrategist` is the high-level agent that defines medium-term financial strategy. It observes aggregated state information and generates strategic goals for the Low-Level Agent to follow.

**Key Features:**
- 5-dimensional aggregated state input (avg_cash, avg_investment_return, spending_trend, current_wealth, months_elapsed)
- 3-dimensional goal output [target_invest_ratio, safety_buffer, aggressiveness]
- Custom policy network with [64, 64] hidden layers
- State aggregation from historical observations
- HIRO-style learning with discount factor γ = 0.99
- Automatic goal constraint enforcement (sigmoid/softplus)
- Model save/load functionality

**Usage Example:**
```python
from src.agents.budget_executor import BudgetExecutor
from src.utils.config import TrainingConfig
from src.utils.data_models import Transition
import numpy as np

# Create training configuration
training_config = TrainingConfig(
    num_episodes=5000,
    gamma_low=0.95,
    gamma_high=0.99,
    high_period=6,
    batch_size=32,
    learning_rate_low=3e-4,
    learning_rate_high=1e-4
)

# Initialize executor
executor = BudgetExecutor(training_config)

# Generate action from state and goal
state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])  # Financial state
goal = np.array([0.3, 1000, 0.5])  # [target_invest_ratio, safety_buffer, aggressiveness]
action = executor.act(state, goal)

print(f"Action: invest={action[0]:.2f}, save={action[1]:.2f}, consume={action[2]:.2f}")

# Learn from experience
transitions = [
    Transition(state, goal, action, reward, next_state, done)
    for state, goal, action, reward, next_state, done in episode_data
]
metrics = executor.learn(transitions)

print(f"Training metrics: loss={metrics['loss']:.4f}, entropy={metrics['policy_entropy']:.4f}")

# Save trained model
executor.save('models/budget_executor.pth')

# Load trained model
executor.load('models/budget_executor.pth')
```

**Input Specification:**
- **State Vector (7-dimensional)**: `[income, fixed_expenses, variable_expenses, cash_balance, inflation, risk_tolerance, t_remaining]`
- **Goal Vector (3-dimensional)**: `[target_invest_ratio, safety_buffer, aggressiveness]`
- **Concatenated Input (10-dimensional)**: State + Goal

**Output Specification:**
- **Action Vector (3-dimensional)**: `[invest_ratio, save_ratio, consume_ratio]` (automatically normalized to sum = 1)

**Learning:**
The executor uses a simplified policy gradient approach with:
- Discounted returns calculation using γ_low = 0.95
- Return normalization for stable training
- Entropy bonus (0.01 coefficient) for exploration
- Adam optimizer with configurable learning rate

**Usage Example:**
```python
from src.agents.financial_strategist import FinancialStrategist
from src.utils.config import TrainingConfig
import numpy as np

# Create training configuration
training_config = TrainingConfig(
    num_episodes=5000,
    gamma_low=0.95,
    gamma_high=0.99,
    high_period=6,
    batch_size=32,
    learning_rate_low=3e-4,
    learning_rate_high=1e-4
)

# Initialize strategist
strategist = FinancialStrategist(training_config)

# Aggregate state from history
state_history = [
    np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50]),
    np.array([3200, 1400, 720, 1100, 0.02, 0.5, 49]),
    np.array([3200, 1400, 690, 1200, 0.02, 0.5, 48]),
    # ... more states
]
aggregated_state = strategist.aggregate_state(state_history)

# Generate strategic goal
goal = strategist.select_goal(aggregated_state)
print(f"Goal: target_invest={goal[0]:.2f}, safety_buffer={goal[1]:.2f}, aggressiveness={goal[2]:.2f}")

# Learn from high-level experience
high_level_transitions = [
    Transition(aggregated_state, goal, None, high_level_reward, next_aggregated_state, done)
    for aggregated_state, goal, high_level_reward, next_aggregated_state, done in episode_data
]
metrics = strategist.learn(high_level_transitions)

print(f"Training metrics: loss={metrics['loss']:.4f}, entropy={metrics['policy_entropy']:.4f}")

# Save trained model
strategist.save('models/financial_strategist.pth')

# Load trained model
strategist.load('models/financial_strategist.pth')
```

**State Aggregation:**
The strategist aggregates recent state history to compute strategic-level features:
- Average cash balance over last N months
- Average investment return (estimated from cash changes)
- Spending trend (linear fit slope of variable expenses)
- Current wealth (most recent cash balance)
- Months elapsed in the episode

**Goal Generation:**
Goals are constrained to valid ranges:
- `target_invest_ratio`: [0, 1] using sigmoid activation
- `safety_buffer`: [0, ∞) using softplus activation
- `aggressiveness`: [0, 1] using sigmoid activation

**Reward Components:**

Low-Level Reward (monthly):
- Investment reward: `α * invest_amount` (encourages investment)
- Stability penalty: `β * max(0, threshold - cash)` (penalizes low cash)
- Overspend penalty: `γ * overspend` (penalizes excessive spending)
- Debt penalty: `δ * abs(min(0, cash))` (heavily penalizes negative balance)

High-Level Reward (strategic period):
- Aggregated low-level rewards over 6-12 months
- Wealth change: `λ * Δwealth` (rewards cash balance growth)
- Stability bonus: `μ * stability_ratio * period_length` (rewards consistent positive balance)

### Environment Configuration
```python
from src.utils.config import EnvironmentConfig

config = EnvironmentConfig(
    income=3200,              # Monthly salary
    fixed_expenses=1400,      # Fixed monthly costs
    variable_expense_mean=700, # Average variable expenses
    inflation=0.02,           # Annual inflation rate
    safety_threshold=1000,    # Minimum cash buffer
    max_months=60            # Simulation duration
)
```

### Training Configuration
```python
from src.utils.config import TrainingConfig

config = TrainingConfig(
    num_episodes=5000,        # Training episodes
    gamma_low=0.95,          # Low-level discount factor
    gamma_high=0.99,         # High-level discount factor
    high_period=6,           # Strategic planning interval
    batch_size=32,           # Training batch size
    learning_rate_low=3e-4,  # Low-level learning rate
    learning_rate_high=1e-4  # High-level learning rate
)
```

### Reward Configuration
```python
from src.utils.config import RewardConfig

config = RewardConfig(
    alpha=10.0,    # Investment reward coefficient
    beta=0.1,      # Stability penalty coefficient
    gamma=5.0,     # Overspend penalty coefficient
    delta=20.0,    # Debt penalty coefficient
    lambda_=1.0,   # Wealth growth coefficient
    mu=0.5         # Stability bonus coefficient
)
```

## Development Status

### ✅ Completed
- [x] Project structure and core data models
- [x] Configuration system with behavioral profiles
- [x] Data models (Transition)
- [x] Package initialization
- [x] BudgetEnv (Gymnasium environment) - Full implementation with state management, action normalization, expense simulation, and episode termination
- [x] Reward Engine - Multi-objective reward computation for both high-level and low-level agents
- [x] RewardEngine integration with BudgetEnv - Production-ready reward computation
- [x] Low-Level Agent (Budget Executor) - PPO-based agent with policy network, action generation, and learning capabilities
- [x] High-Level Agent (Financial Strategist) - HIRO-style agent with state aggregation, goal generation, and strategic learning

### 🚧 In Progress
- [ ] Training Orchestrator
- [ ] Analytics Module

### ✅ Recently Completed
- [x] High-Level Agent (Financial Strategist) - HIRO-style agent for strategic goal generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BudgetEnv (Financial Environment)         │
│  - Simulates monthly income, expenses, inflation            │
│  - Manages cash balance and state transitions               │
└────────────┬────────────────────────────────┬───────────────┘
             │ state observation              │ aggregated state
             ▼                                ▼
    ┌────────────────────┐         ┌─────────────────────────┐
    │  Low-Level Agent   │◄────────│  High-Level Agent       │
    │  (Executor)        │  goal   │  (Strategist)           │
    │  - Monthly actions │         │  - Strategic planning   │
    └────────┬───────────┘         └─────────────────────────┘
             │ action [invest, save, consume]
             ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    Reward Engine                         │
    │  - Computes multi-objective rewards                     │
    │  - Balances growth and stability                        │
    └─────────────────────────────────────────────────────────┘
```

## Key Features

- **Hierarchical Decision-Making**: Separates strategic and tactical financial decisions
- **Realistic Simulation**: Models fixed/variable expenses, inflation, and economic uncertainty
- **Configurable Risk Profiles**: Supports conservative, balanced, and aggressive strategies
- **Multi-Objective Optimization**: Balances long-term wealth growth with short-term stability
- **Standard RL Interface**: Built on Gymnasium for easy integration with RL frameworks

## Performance Metrics

The system tracks the following metrics:
- **Cumulative Wealth Growth**: Total invested capital over simulation
- **Cash Stability Index**: Percentage of months with positive balance
- **Sharpe-like Ratio**: Return divided by standard deviation of balance
- **Goal Adherence**: Alignment between strategic goals and actual allocations
- **Policy Stability**: Consistency of actions over time

## Quick Start

Run the examples to see the system in action:

```bash
# BudgetEnv demonstration
PYTHONPATH=. python3 examples/basic_budget_env_usage.py

# RewardEngine demonstration
PYTHONPATH=. python3 examples/reward_engine_usage.py
```

These examples demonstrate:
- Creating and configuring environments and reward engines
- Taking actions and observing results
- Understanding reward components and their effects
- Running complete episodes with adaptive strategies

## Documentation

- [Requirements Document](.kiro/specs/hrl-finance-system/requirements.md) - Detailed system requirements
- [Design Document](.kiro/specs/hrl-finance-system/design.md) - Architecture and component design
- [Implementation Tasks](.kiro/specs/hrl-finance-system/tasks.md) - Development roadmap
- [HLD/LLD Document](Requirements/HRL_Finance_System_Design.md) - High and low-level design
- [Basic Usage Example](examples/basic_budget_env_usage.py) - Simple BudgetEnv demonstration
- [Changelog](CHANGELOG.md) - Version history and implementation progress

## License

This project is for research and educational purposes.

## Author

Alessio Rocchi
