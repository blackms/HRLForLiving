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
│   │   └── __init__.py
│   ├── environment/             # Financial environment simulation
│   │   ├── __init__.py
│   │   └── budget_env.py       # ✅ BudgetEnv implementation
│   ├── training/                # Training orchestration
│   │   └── __init__.py
│   └── utils/                   # Configuration and utilities
│       ├── __init__.py
│       ├── config.py            # Configuration dataclasses
│       └── data_models.py       # Core data models
├── examples/                    # Usage examples
│   ├── README.md               # Examples documentation
│   └── basic_budget_env_usage.py  # ✅ Basic BudgetEnv demo
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   └── test_budget_env.py      # ✅ BudgetEnv tests
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

The `BudgetEnv` is a custom Gymnasium environment that simulates monthly financial decisions. It's now fully implemented and ready to use.

**Usage Example:**
```python
from src.environment import BudgetEnv
from src.utils.config import EnvironmentConfig

# Create configuration
config = EnvironmentConfig(
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

# Initialize environment
env = BudgetEnv(config)

# Reset environment
observation, info = env.reset()

# Take a step with an action [invest_ratio, save_ratio, consume_ratio]
action = [0.3, 0.5, 0.2]  # Invest 30%, save 50%, consume 20%
observation, reward, terminated, truncated, info = env.step(action)

print(f"Cash balance: ${info['cash_balance']:.2f}")
print(f"Total invested: ${info['total_invested']:.2f}")
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

### 🚧 In Progress
- [ ] Reward Engine
- [ ] Low-Level Agent (Budget Executor)
- [ ] High-Level Agent (Financial Strategist)
- [ ] Training Orchestrator
- [ ] Analytics Module

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

Run the basic example to see BudgetEnv in action:

```bash
python examples/basic_budget_env_usage.py
```

This example demonstrates:
- Creating and configuring a BudgetEnv
- Taking actions and observing results
- Running a complete 12-month episode with adaptive strategy

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
