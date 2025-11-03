# Personal Finance Optimization HRL System

A hierarchical reinforcement learning (HRL) system that simulates and learns to optimally allocate monthly salary among investments, savings, and discretionary spending. The system aims to maximize long-term investments while maintaining financial stability through realistic monthly economic simulation.

> **🚀 New to the system?** Check out the [Quick Start Guide](QUICK_START.md) to get running in 5 minutes!

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
│   │   ├── __init__.py
│   │   └── hrl_trainer.py      # ✅ HRLTrainer implementation
│   └── utils/                   # Configuration and utilities
│       ├── __init__.py
│       ├── analytics.py         # ✅ AnalyticsModule implementation
│       ├── config.py            # ✅ Configuration dataclasses
│       ├── config_manager.py    # ✅ Configuration Manager
│       └── data_models.py       # ✅ Core data models
├── configs/                     # Example configuration files
│   ├── conservative.yaml        # ✅ Conservative profile config
│   ├── balanced.yaml            # ✅ Balanced profile config
│   └── aggressive.yaml          # ✅ Aggressive profile config
├── examples/                    # Usage examples
│   ├── README.md               # Examples documentation
│   ├── basic_budget_env_usage.py  # ✅ Basic BudgetEnv demo
│   ├── reward_engine_usage.py  # ✅ RewardEngine demo
│   ├── analytics_usage.py      # ✅ AnalyticsModule demo
│   └── training_with_analytics.py # ✅ Training integration demo
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── TEST_COVERAGE.md        # Test coverage summary
│   ├── test_analytics.py       # ✅ AnalyticsModule tests (18 cases)
│   ├── test_budget_env.py      # ✅ BudgetEnv tests
│   ├── test_budget_executor.py # ✅ BudgetExecutor tests
│   ├── test_config_manager.py  # ✅ ConfigurationManager tests (50+ cases)
│   ├── test_financial_strategist.py # ✅ FinancialStrategist tests
│   ├── test_hrl_trainer.py     # ✅ HRLTrainer tests (30+ cases)
│   ├── test_reward_engine.py   # ✅ RewardEngine tests
│   └── test_sanity_checks.py   # ✅ Sanity check tests (7 cases)
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
- `tensorboard>=2.14.0` - Experiment tracking and visualization

## Configuration

The system supports flexible configuration through YAML files or predefined behavioral profiles.

### Configuration Manager

The `ConfigurationManager` provides two ways to configure the system:

**1. Load from YAML file:**
```python
from src.utils.config_manager import load_config

env_config, training_config, reward_config = load_config('configs/my_config.yaml')
```

**2. Load predefined behavioral profile:**
```python
from src.utils.config_manager import load_behavioral_profile

env_config, training_config, reward_config = load_behavioral_profile('balanced')
```

### Behavioral Profiles

The system includes three predefined behavioral profiles with different risk tolerances and reward structures. Each profile is optimized for different financial goals and risk appetites.

#### Conservative Profile
**Best for:** Risk-averse individuals prioritizing financial stability and capital preservation

**Configuration:**
- Risk tolerance: 0.3 (low risk appetite)
- Safety threshold: $1,500 (higher cash buffer requirement)
- Investment reward coefficient (α): 5.0 (moderate investment incentive)
- Stability penalty coefficient (β): 0.5 (strong penalty for low cash)
- Overspend penalty coefficient (γ): 5.0
- Debt penalty coefficient (δ): 20.0

**Characteristics:**
- Maintains higher cash reserves for emergencies
- Invests conservatively, prioritizing safety over returns
- Strongly penalizes risky behavior and low cash balances
- Suitable for individuals with irregular income or high financial obligations
- Expected outcome: Lower investment returns but higher financial stability

**When to use:**
- You have unpredictable expenses or income
- You're building an emergency fund
- You have low risk tolerance
- Financial stability is more important than growth

#### Balanced Profile (Default)
**Best for:** Most users seeking a reasonable balance between growth and stability

**Configuration:**
- Risk tolerance: 0.5 (moderate risk appetite)
- Safety threshold: $1,000 (standard cash buffer)
- Investment reward coefficient (α): 10.0 (standard investment incentive)
- Stability penalty coefficient (β): 0.1 (moderate penalty for low cash)
- Overspend penalty coefficient (γ): 5.0
- Debt penalty coefficient (δ): 20.0

**Characteristics:**
- Balances investment growth with financial stability
- Maintains reasonable cash reserves
- Adapts to changing financial conditions
- Suitable for individuals with stable income and moderate expenses
- Expected outcome: Moderate investment returns with good stability

**When to use:**
- You have stable income and predictable expenses
- You want to grow wealth while maintaining safety
- You're comfortable with moderate risk
- You're starting out and unsure which profile to choose

#### Aggressive Profile
**Best for:** Risk-tolerant individuals maximizing investment growth

**Configuration:**
- Risk tolerance: 0.8 (high risk appetite)
- Safety threshold: $500 (minimal cash buffer)
- Investment reward coefficient (α): 15.0 (strong investment incentive)
- Stability penalty coefficient (β): 0.05 (weak penalty for low cash)
- Overspend penalty coefficient (γ): 5.0
- Debt penalty coefficient (δ): 20.0

**Characteristics:**
- Maximizes investment allocation
- Maintains minimal cash reserves
- Prioritizes long-term wealth accumulation over short-term stability
- Suitable for individuals with stable income, low expenses, and high risk tolerance
- Expected outcome: Higher investment returns but lower cash stability

**When to use:**
- You have very stable income and low expenses
- You have other emergency funds or safety nets
- You have high risk tolerance
- Long-term wealth growth is your primary goal
- You can handle temporary cash flow challenges

### Configuration Parameters Reference

#### Environment Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `income` | float | > 0 | 3200 | Monthly salary/income in dollars |
| `fixed_expenses` | float | ≥ 0 | 1400 | Fixed monthly costs (rent, utilities, etc.) |
| `variable_expense_mean` | float | ≥ 0 | 700 | Average variable expenses (groceries, entertainment) |
| `variable_expense_std` | float | ≥ 0 | 100 | Standard deviation of variable expenses |
| `inflation` | float | [-1, 1] | 0.02 | Monthly inflation rate (0.02 = 2% per month) |
| `safety_threshold` | float | ≥ 0 | 1000 | Minimum cash balance before penalties apply |
| `max_months` | int | > 0 | 60 | Maximum episode length in months |
| `initial_cash` | float | any | 0 | Starting cash balance |
| `risk_tolerance` | float | [0, 1] | 0.5 | Agent's risk appetite (0=conservative, 1=aggressive) |

**Tips for Environment Configuration:**
- Set `income` and `fixed_expenses` based on your target scenario
- Use `variable_expense_std` to model expense uncertainty (higher = more unpredictable)
- Adjust `safety_threshold` based on desired cash buffer (typically 1-2 months of expenses)
- Use `inflation` to model economic conditions (0.02 = 2% monthly ≈ 27% annually)
- Set `max_months` to your planning horizon (60 = 5 years)

#### Training Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `num_episodes` | int | > 0 | 5000 | Number of training episodes |
| `gamma_low` | float | [0, 1] | 0.95 | Discount factor for low-level agent (monthly decisions) |
| `gamma_high` | float | [0, 1] | 0.99 | Discount factor for high-level agent (strategic decisions) |
| `high_period` | int | > 0 | 6 | Strategic planning interval in months (6-12 recommended) |
| `batch_size` | int | > 0 | 32 | Number of transitions per policy update |
| `learning_rate_low` | float | > 0 | 3e-4 | Learning rate for low-level agent |
| `learning_rate_high` | float | > 0 | 1e-4 | Learning rate for high-level agent |

**Tips for Training Configuration:**
- Start with 5000 episodes for initial experiments, increase to 10000+ for production
- `gamma_low` (0.95) values immediate rewards more than `gamma_high` (0.99)
- `high_period` of 6 means strategic goals are updated every 6 months
- Reduce learning rates if training is unstable, increase if learning is too slow
- Larger `batch_size` provides more stable updates but requires more memory

#### Reward Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `alpha` | float | ≥ 0 | 10.0 | Investment reward coefficient (higher = more investment) |
| `beta` | float | ≥ 0 | 0.1 | Stability penalty coefficient (higher = more conservative) |
| `gamma` | float | ≥ 0 | 5.0 | Overspend penalty coefficient |
| `delta` | float | ≥ 0 | 20.0 | Debt penalty coefficient (negative balance) |
| `lambda_` | float | ≥ 0 | 1.0 | Wealth growth coefficient (high-level reward) |
| `mu` | float | ≥ 0 | 0.5 | Stability bonus coefficient (high-level reward) |

**Tips for Reward Configuration:**
- Increase `alpha` to encourage more aggressive investment behavior
- Increase `beta` to maintain higher cash reserves (more conservative)
- `delta` should be high (20.0) to strongly discourage negative balance
- Balance `lambda_` and `mu` to trade off wealth growth vs stability
- Start with default values and adjust based on observed behavior

**Reward Formula:**
```
Low-Level Reward (monthly):
r_low = α * invest_amount                    # Encourage investment
        - β * max(0, threshold - cash)       # Penalize low cash
        - γ * overspend                      # Penalize excess spending
        - δ * abs(min(0, cash))              # Penalize debt

High-Level Reward (strategic period):
r_high = Σ(r_low over period)                # Aggregate monthly rewards
         + λ * Δwealth                       # Reward wealth growth
         + μ * stability_bonus               # Reward consistent positive balance
```

### YAML Configuration Format

Create a YAML file with the following structure (see `configs/` directory for examples):

```yaml
environment:
  income: 3200                    # Monthly salary
  fixed_expenses: 1400            # Fixed monthly costs
  variable_expense_mean: 700      # Average variable expenses
  variable_expense_std: 100       # Std dev of variable expenses
  inflation: 0.02                 # Monthly inflation rate
  safety_threshold: 1000          # Minimum cash buffer
  max_months: 60                  # Simulation duration (months)
  initial_cash: 0                 # Starting cash balance
  risk_tolerance: 0.5             # Risk appetite (0-1)

training:
  num_episodes: 5000              # Training episodes
  gamma_low: 0.95                 # Low-level discount factor
  gamma_high: 0.99                # High-level discount factor
  high_period: 6                  # Strategic planning interval (months)
  batch_size: 32                  # Transitions per update
  learning_rate_low: 0.0003       # Low-level learning rate
  learning_rate_high: 0.0001      # High-level learning rate

reward:
  alpha: 10.0                     # Investment reward coefficient
  beta: 0.1                       # Stability penalty coefficient
  gamma: 5.0                      # Overspend penalty coefficient
  delta: 20.0                     # Debt penalty coefficient
  lambda_: 1.0                    # Wealth growth coefficient
  mu: 0.5                         # Stability bonus coefficient
```

**Example Configuration Files:**
- `configs/conservative.yaml` - Conservative profile with low risk tolerance
- `configs/balanced.yaml` - Balanced profile with medium risk tolerance
- `configs/aggressive.yaml` - Aggressive profile with high risk tolerance

### Configuration Validation

The Configuration Manager automatically validates all parameters:
- Income must be positive
- Expenses must be non-negative
- Inflation must be in [-1, 1]
- Discount factors (gamma) must be in [0, 1]
- Risk tolerance must be in [0, 1]
- Learning rates must be positive
- All reward coefficients must be non-negative

Invalid configurations raise a `ConfigurationError` with a descriptive message.

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

### Configuration Manager - System Configuration

The `ConfigurationManager` provides flexible configuration loading from YAML files or predefined behavioral profiles with automatic validation.

**Status:** ✅ **FULLY IMPLEMENTED** - Complete with YAML loading, behavioral profiles, and comprehensive validation

**Key Features:**
- Load configurations from YAML files
- Load predefined behavioral profiles (conservative, balanced, aggressive)
- Automatic parameter validation with descriptive error messages
- Support for all configuration types (environment, training, reward)
- Custom ConfigurationError exception for invalid configurations

**Usage Example:**
```python
from src.utils.config_manager import load_config, load_behavioral_profile, ConfigurationError

# Option 1: Load from YAML file
try:
    env_config, training_config, reward_config = load_config('configs/my_config.yaml')
except ConfigurationError as e:
    print(f"Configuration error: {e}")

# Option 2: Load predefined behavioral profile
env_config, training_config, reward_config = load_behavioral_profile('balanced')

# Use configurations
env = BudgetEnv(env_config, reward_config)
strategist = FinancialStrategist(training_config)
executor = BudgetExecutor(training_config)
```

**Behavioral Profiles:**
- **Conservative**: Low risk (0.3), high safety threshold ($1,500), lower investment rewards (α=5.0)
- **Balanced**: Medium risk (0.5), standard safety threshold ($1,000), standard rewards (α=10.0)
- **Aggressive**: High risk (0.8), low safety threshold ($500), higher investment rewards (α=15.0)

**Validation Rules:**
- Income must be positive
- Expenses must be non-negative
- Inflation must be in [-1, 1]
- Discount factors (gamma) must be in [0, 1]
- Risk tolerance must be in [0, 1]
- Learning rates must be positive
- All reward coefficients must be non-negative

### AnalyticsModule - Performance Metrics Tracking

The `AnalyticsModule` tracks and computes comprehensive performance metrics for evaluating the HRL system's financial decision-making quality.

**Status:** ✅ **FULLY TESTED** - 18 comprehensive test cases covering all functionality and edge cases

**Key Features:**
- Records step-by-step data (states, actions, rewards, goals, investments)
- Computes cumulative wealth growth (total invested capital)
- Calculates cash stability index (% months with positive balance)
- Computes Sharpe-like ratio (mean return / std balance)
- Measures goal adherence (alignment between strategic goals and actual actions)
- Tracks policy stability (variance of actions over time)
- Episode-level metric computation
- Easy reset for new episodes
- Robust edge case handling (empty data, single step, missing goals, zero variance)
- Array copying to prevent reference issues

**Usage Example:**
```python
from src.utils.analytics import AnalyticsModule
import numpy as np

# Initialize analytics module
analytics = AnalyticsModule()

# Record steps during episode
for step in episode:
    state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
    action = np.array([0.3, 0.5, 0.2])
    reward = 15.0
    goal = np.array([0.3, 1000, 0.5])
    invested_amount = 960.0  # 0.3 * 3200
    
    analytics.record_step(state, action, reward, goal, invested_amount)

# Compute metrics at episode end
metrics = analytics.compute_episode_metrics()

print(f"Cumulative wealth growth: ${metrics['cumulative_wealth_growth']:.2f}")
print(f"Cash stability index: {metrics['cash_stability_index']:.2%}")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Goal adherence: {metrics['goal_adherence']:.4f}")
print(f"Policy stability: {metrics['policy_stability']:.4f}")

# Reset for next episode
analytics.reset()
```

**Metrics Explained:**
- **Cumulative Wealth Growth**: Total amount invested over the episode, indicating long-term wealth accumulation
- **Cash Stability Index**: Percentage of months maintaining positive cash balance (0-1), higher is better
- **Sharpe-like Ratio**: Risk-adjusted return metric (mean balance / std balance), higher indicates better risk-adjusted performance
- **Goal Adherence**: Mean absolute difference between target investment ratio and actual investment action, lower indicates better goal following
- **Policy Stability**: Variance of actions over time, lower indicates more consistent decision-making

**Edge Case Handling:**
The module gracefully handles various edge cases:
- Empty data: Returns 0.0 for all metrics
- Single step: Correctly computes metrics (stability=1.0 if positive, sharpe=0.0, policy_stability=0.0)
- Missing goals: Returns 0.0 for goal_adherence
- Mismatched lengths: Uses minimum length between goals and actions
- Zero variance: Returns 0.0 for sharpe_ratio and policy_stability
- Array references: Automatically copies arrays to prevent mutation issues

### HRLTrainer - Training Orchestrator

The `HRLTrainer` coordinates the hierarchical training loop, managing interactions between the high-level and low-level agents. It implements the complete HRL training process where strategic goals are set periodically and monthly actions are executed continuously.

**Status:** ✅ **FULLY IMPLEMENTED** - Complete with AnalyticsModule integration, TensorBoard logging, and checkpointing

**Key Features:**
- Coordinates high-level (Strategist) and low-level (Executor) agent training
- Automatic AnalyticsModule integration for zero-overhead performance tracking
- Episode buffer for storing low-level transitions
- State history tracking for high-level state aggregation
- Comprehensive training metrics tracking (rewards, lengths, cash balances, investments, losses, and all 5 analytics metrics)
- Configurable high-level decision period (default: 6 months)
- Automatic policy updates for both agents
- Enhanced progress monitoring with stability and goal adherence metrics
- Deterministic evaluation mode with comprehensive summary statistics
- Optional TensorBoard logging for experiment tracking
- Checkpointing and resume functionality for long training runs
- Best model tracking based on evaluation performance
- Supports both training and evaluation modes

**Usage Example:**
```python
from src.training.hrl_trainer import HRLTrainer
from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.environment.reward_engine import RewardEngine
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig
from src.utils.logger import ExperimentLogger

# Create configurations
env_config = EnvironmentConfig(
    income=3200,
    fixed_expenses=1400,
    variable_expense_mean=700,
    variable_expense_std=100,
    inflation=0.02,
    safety_threshold=1000,
    max_months=60,
    initial_cash=0,
    risk_tolerance=0.5
)

training_config = TrainingConfig(
    num_episodes=5000,
    gamma_low=0.95,
    gamma_high=0.99,
    high_period=6,
    batch_size=32,
    learning_rate_low=3e-4,
    learning_rate_high=1e-4
)

reward_config = RewardConfig(
    alpha=10.0,
    beta=0.1,
    gamma=5.0,
    delta=20.0,
    lambda_=1.0,
    mu=0.5
)

# Initialize components
env = BudgetEnv(env_config, reward_config)
reward_engine = RewardEngine(reward_config, safety_threshold=1000)
strategist = FinancialStrategist(training_config)
executor = BudgetExecutor(training_config)

# Optional: Initialize TensorBoard logger
logger = ExperimentLogger(log_dir='runs', experiment_name='balanced_training')

# Create trainer with optional logger and configs for checkpointing
trainer = HRLTrainer(
    env, strategist, executor, reward_engine, training_config,
    logger=logger,
    env_config=env_config,
    reward_config=reward_config
)

# Option 1: Basic training
print("Starting training...")
training_history = trainer.train(num_episodes=5000)

# Option 2: Training with automatic checkpointing and best model tracking
training_history = trainer.train_with_checkpointing(
    num_episodes=5000,
    checkpoint_dir='models/checkpoints/balanced',
    save_interval=1000,      # Save checkpoint every 1000 episodes
    eval_interval=1000,      # Evaluate every 1000 episodes
    eval_episodes=10         # Use 10 episodes for evaluation
)

# Access training metrics
print(f"\nTraining Complete!")
print(f"Final average reward: {np.mean(training_history['episode_rewards'][-100:]):.2f}")
print(f"Final average cash: {np.mean(training_history['cash_balances'][-100:]):.2f}")
print(f"Final average invested: {np.mean(training_history['total_invested'][-100:]):.2f}")

# Save trained models (if not using checkpointing)
strategist.save('models/strategist.pth')
executor.save('models/executor.pth')

# Resume training from checkpoint
episode_num, history = trainer.load_checkpoint('models/checkpoints/balanced/checkpoint_episode_1000')
print(f"Resumed from episode {episode_num}")
# Continue training...
trainer.train_with_checkpointing(num_episodes=2000, checkpoint_dir='models/checkpoints/balanced')

# Evaluation
eval_metrics = trainer.evaluate(num_episodes=100)
print(f"\nEvaluation Results:")
print(f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
print(f"Mean Cash Balance: ${eval_metrics['mean_cash_balance']:.2f}")
print(f"Mean Total Invested: ${eval_metrics['mean_total_invested']:.2f}")
print(f"Mean Wealth Growth: ${eval_metrics['mean_wealth_growth']:.2f}")
print(f"Mean Cash Stability: {eval_metrics['mean_cash_stability']:.2%}")
print(f"Mean Sharpe Ratio: {eval_metrics['mean_sharpe_ratio']:.2f}")
print(f"Mean Goal Adherence: {eval_metrics['mean_goal_adherence']:.4f}")
print(f"Mean Policy Stability: {eval_metrics['mean_policy_stability']:.4f}")

# Close logger
if logger:
    logger.close()
```

**Training Process:**
1. Reset environment and initialize state history
2. Reset analytics module for new episode
3. High-level agent generates initial strategic goal
4. Low-level agent executes monthly allocation decisions following the goal
5. Record each step in analytics module (state, action, reward, goal, invested amount)
6. Store transitions in episode buffer
7. Update low-level policy when buffer reaches batch size
8. Every `high_period` steps (default: 6):
   - Compute high-level reward over the period
   - Update high-level policy
   - Generate new strategic goal
9. Handle final updates at episode termination
10. Compute episode metrics from analytics module
11. Track and print progress every 100 episodes (including stability and goal adherence)

**Training Metrics:**
The trainer tracks comprehensive metrics throughout training:
- `episode_rewards`: Cumulative reward per episode
- `episode_lengths`: Number of steps per episode
- `cash_balances`: Final cash balance per episode
- `total_invested`: Total invested capital per episode
- `low_level_losses`: Policy loss for low-level agent
- `high_level_losses`: Policy loss for high-level agent
- `cumulative_wealth_growth`: Total invested capital (from analytics)
- `cash_stability_index`: % months with positive balance (from analytics)
- `sharpe_ratio`: Risk-adjusted performance (from analytics)
- `goal_adherence`: Alignment with strategic goals (from analytics)
- `policy_stability`: Consistency of decisions (from analytics)

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

### Manual Configuration (Alternative)

You can also create configurations manually using dataclasses:

**Environment Configuration:**
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

**Training Configuration:**
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

**Reward Configuration:**
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

## Logging and Monitoring

The system includes comprehensive TensorBoard logging for experiment tracking and visualization.

### ExperimentLogger

The `ExperimentLogger` provides automatic integration with TensorBoard for tracking training progress:

**Key Features:**
- Automatic logging of training curves (rewards, losses)
- Episode metrics tracking (wealth, stability, Sharpe ratio)
- Action and goal distribution visualization
- Hyperparameter logging for reproducibility
- Real-time monitoring with TensorBoard web interface
- Zero-overhead integration with HRLTrainer

**Usage with Training Script:**
```bash
# Train with TensorBoard logging (enabled by default)
python train.py --profile balanced

# Disable logging if needed
python train.py --profile balanced --no-logging

# Custom log directory
python train.py --profile balanced --log-dir my_experiments

# View logs in TensorBoard
tensorboard --logdir=runs
# Open browser to: http://localhost:6006
```

**Manual Usage:**
```python
from src.utils.logger import ExperimentLogger
from src.training.hrl_trainer import HRLTrainer

# Initialize logger
logger = ExperimentLogger(
    log_dir='runs',
    experiment_name='my_experiment',
    enabled=True
)

# Log hyperparameters
hparams = {
    'env/income': 3200,
    'train/num_episodes': 5000,
    'reward/alpha': 10.0,
}
logger.log_hyperparameters(hparams)

# Create trainer with logger
trainer = HRLTrainer(env, high_agent, low_agent, reward_engine, config, logger=logger)

# Train (logging happens automatically)
history = trainer.train(num_episodes=5000)

# Close logger
logger.close()
```

**What Gets Logged:**
- **Training Curves**: Episode rewards, low-level losses, high-level losses
- **Episode Metrics**: Cash balance, total invested, episode length
- **Analytics Metrics**: Cumulative wealth growth, cash stability index, Sharpe ratio, goal adherence, policy stability
- **Action Distributions**: Mean and std for invest/save/consume ratios, histograms
- **Goal Distributions**: Mean and std for target_invest_ratio/safety_buffer/aggressiveness, histograms
- **Hyperparameters**: All environment, training, and reward configuration parameters

**TensorBoard Views:**
- **Scalars**: Training curves and episode metrics over time
- **Distributions**: Action and goal distributions across episodes
- **Histograms**: Detailed distribution evolution
- **Text**: Hyperparameter configuration

**Example:**
See `examples/logging_usage.py` for a complete demonstration of TensorBoard logging.

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
- [x] Training Orchestrator (HRLTrainer) - Complete training loop with policy coordination, metrics tracking, TensorBoard logging, and checkpointing
- [x] Analytics Module - Performance metrics tracking and computation with comprehensive test coverage (18 test cases)
- [x] Analytics Module integration with HRLTrainer - Automatic tracking of all 5 metrics during training and evaluation
- [x] HRLTrainer evaluation method - Deterministic evaluation with comprehensive summary statistics
- [x] Configuration Manager - YAML loading, behavioral profiles, and comprehensive validation (50+ test cases)
- [x] Main training script (train.py) - Complete CLI tool with comprehensive features including checkpointing and TensorBoard logging
- [x] Integration tests for HRLTrainer - 13 comprehensive tests covering complete training pipeline
- [x] Sanity check tests - 7 system-level validation tests for behavioral profiles and learning effectiveness
- [x] TensorBoard logging - ExperimentLogger for comprehensive experiment tracking (examples/logging_usage.py)
- [x] Checkpointing functionality - Save/load/resume with best model tracking (examples/checkpointing_usage.py, tests/test_checkpointing.py)

### 🚧 In Progress
- [ ] Evaluation script for loading and testing trained models

### ✅ Recently Completed
- [x] TensorBoard logging integration (Task 14) - ExperimentLogger with automatic tracking of training curves, episode metrics, action/goal distributions, and hyperparameters
- [x] Checkpointing and resume functionality (Task 15) - Complete implementation with save/load/resume, best model tracking, and comprehensive tests (7 test cases)
- [x] Sanity check tests (7 comprehensive tests) - System-level validation of behavioral profiles, learning effectiveness, and configuration integrity
- [x] Integration tests for HRLTrainer (13 comprehensive tests) - Complete coverage of training pipeline, component coordination, and analytics integration
- [x] Main training script (train.py) - Complete CLI tool with config/profile support, model saving, evaluation, TensorBoard logging, and checkpointing
- [x] Configuration Manager - Complete implementation with YAML loading, behavioral profiles, and validation (50+ test cases)
- [x] Analytics Module integration with HRLTrainer - Zero-overhead automatic tracking during training
- [x] HRLTrainer evaluation method - Complete with all 5 analytics metrics and summary statistics

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

The system tracks comprehensive performance metrics through the `AnalyticsModule`:

### Core Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Cumulative Wealth Growth** | Total invested capital over simulation | Higher is better - indicates long-term wealth accumulation |
| **Cash Stability Index** | Percentage of months with positive balance (0-1) | Higher is better - indicates financial stability |
| **Sharpe-like Ratio** | Mean balance / std balance | Higher is better - indicates better risk-adjusted performance |
| **Goal Adherence** | Mean absolute difference between target and actual investment | Lower is better - indicates better goal following |
| **Policy Stability** | Variance of actions over time | Lower is better - indicates more consistent decision-making |

### Using Analytics in Your Code

```python
from src.utils.analytics import AnalyticsModule

# Initialize analytics
analytics = AnalyticsModule()

# During episode execution
for step in range(episode_length):
    state, reward, done, info = env.step(action)
    analytics.record_step(
        state=state,
        action=action,
        reward=reward,
        goal=goal,  # Optional: from high-level agent
        invested_amount=info['invest_amount']  # Optional: from env info
    )

# Compute metrics at episode end
metrics = analytics.compute_episode_metrics()

# Access individual metrics
print(f"Wealth Growth: ${metrics['cumulative_wealth_growth']:.2f}")
print(f"Stability: {metrics['cash_stability_index']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Goal Adherence: {metrics['goal_adherence']:.4f}")
print(f"Policy Stability: {metrics['policy_stability']:.4f}")

# Reset for next episode
analytics.reset()
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd hrl-finance-system

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v
```

### 2. Training the HRL System

Train the system using the main training script with either YAML configuration files or predefined behavioral profiles:

**Basic Training Examples:**

```bash
# Train with a behavioral profile (recommended for quick start)
python3 train.py --profile balanced --episodes 5000

# Train with a YAML configuration file
python3 train.py --config configs/conservative.yaml

# Train with custom settings
python3 train.py --profile aggressive --episodes 10000 --output models/aggressive_run --seed 42

# Train with evaluation
python3 train.py --profile balanced --episodes 5000 --eval-episodes 20
```

**Advanced Training Examples:**

```bash
# Train with checkpointing (saves every 1000 episodes)
python3 train.py --profile balanced --episodes 10000 --save-interval 1000

# Train with custom output directory and seed for reproducibility
python3 train.py --profile aggressive --episodes 5000 --output models/run_001 --seed 42

# Train without TensorBoard logging
python3 train.py --profile balanced --no-log

# Train with custom TensorBoard log directory
python3 train.py --profile balanced --log-dir experiments/balanced_v2

# Resume training from checkpoint
python3 train.py --profile balanced --resume models/checkpoints/balanced/checkpoint_episode_5000
```

**Command-line Options:**
- `--config PATH`: Path to YAML configuration file
- `--profile {conservative,balanced,aggressive}`: Use predefined behavioral profile
- `--episodes N`: Number of training episodes (overrides config)
- `--output DIR`: Output directory for trained models (default: models/)
- `--eval-episodes N`: Number of evaluation episodes after training (default: 10)
- `--save-interval N`: Save checkpoint every N episodes (default: 1000)
- `--seed N`: Random seed for reproducibility

**Training Output:**
The training script will:
1. Load and validate configuration
2. Initialize all system components (environment, agents, trainer)
3. Execute training with progress updates every 100 episodes
4. Save trained models and training history
5. Run evaluation episodes and display performance metrics

**Saved Files:**
- `{config_name}_high_agent.pt` - Trained high-level agent (Strategist)
- `{config_name}_low_agent.pt` - Trained low-level agent (Executor)
- `{config_name}_history.json` - Complete training history with all metrics

### Evaluating Trained Models

After training, evaluate your models using the evaluation script to assess performance and generate visualizations:

```bash
# Basic evaluation with trained models
python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt

# Evaluate with specific configuration
python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --config configs/balanced.yaml

# Evaluate with custom episodes and output directory
python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --episodes 50 --output results/

# Evaluate without generating visualizations
python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --no-viz

# Evaluate with reproducible results
python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --seed 42
```

**Command-line Options:**
- `--high-agent PATH`: Path to trained high-level agent model (.pt file) [required]
- `--low-agent PATH`: Path to trained low-level agent model (.pt file) [required]
- `--config PATH`: Path to YAML configuration file (optional)
- `--profile {conservative,balanced,aggressive}`: Use predefined behavioral profile (optional)
- `--episodes N`: Number of evaluation episodes (default: 20)
- `--output DIR`: Output directory for results and visualizations (default: results/)
- `--seed N`: Random seed for reproducibility
- `--no-viz`: Skip generating visualizations

**Evaluation Output:**
The evaluation script will:
1. Load trained models from checkpoint files
2. Initialize environment with specified configuration
3. Run evaluation episodes using deterministic policies
4. Compute comprehensive performance metrics
5. Display detailed evaluation results
6. Save results to JSON file
7. Generate trajectory visualizations (unless --no-viz is specified)

**Generated Files:**
- `{config_name}_evaluation_results.json` - Comprehensive evaluation metrics and statistics
- `{config_name}_trajectory_visualization.png` - Episode trajectory plots showing:
  - Cash balance over time
  - Allocation actions over time (invest, save, consume)
  - Cumulative investment growth
  - Rewards over time
  - Goal adherence (target vs actual investment)
  - Total expenses over time
- `{config_name}_summary_statistics.png` - Statistical analysis showing:
  - Distribution of episode rewards
  - Distribution of final cash balances
  - Distribution of total investments
  - Performance metrics comparison

**Performance Metrics Reported:**
- Mean reward ± standard deviation
- Mean episode length (months survived)
- Mean final cash balance ± std
- Mean total invested ± std
- Mean cumulative wealth growth ± std
- Mean cash stability index (% positive balance months) ± std
- Mean Sharpe ratio (risk-adjusted performance) ± std
- Mean goal adherence (alignment with strategic goals) ± std
- Mean policy stability (consistency of decisions) ± std

### Running Examples

Run the examples to see individual components in action:

```bash
# BudgetEnv demonstration
PYTHONPATH=. python3 examples/basic_budget_env_usage.py

# RewardEngine demonstration
PYTHONPATH=. python3 examples/reward_engine_usage.py

# AnalyticsModule demonstration
PYTHONPATH=. python3 examples/analytics_usage.py

# Training with analytics integration
PYTHONPATH=. python3 examples/training_with_analytics.py
```

These examples demonstrate:
- Creating and configuring environments and reward engines
- Taking actions and observing results
- Understanding reward components and their effects
- Running complete episodes with adaptive strategies
- Tracking performance metrics with the AnalyticsModule
- Full training loop with automatic analytics integration

### Running Tests

Run the comprehensive test suite to verify system functionality:

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_sanity_checks.py

# Run specific test
pytest tests/test_sanity_checks.py::TestSanityChecks::test_trained_policy_outperforms_random_policy

# Run with verbose output
pytest tests/ -v

# Run only sanity checks
pytest tests/test_sanity_checks.py -v
```

**Test Categories:**
- **Unit Tests**: Test individual components in isolation (150+ tests)
- **Integration Tests**: Test complete training pipeline (13 tests in test_hrl_trainer.py)
- **Sanity Checks**: Validate system-level behavior and learning effectiveness (7 tests)

**Key Sanity Check Tests:**
- Random policy baseline validation
- Behavioral profile comparison (conservative vs balanced vs aggressive)
- Trained vs untrained policy comparison
- Profile configuration validation
- Learning effectiveness verification

## Troubleshooting

### Common Issues

#### Training doesn't converge
**Symptoms:** Rewards remain low or unstable, agent doesn't learn effective policies

**Solutions:**
1. Reduce learning rates: Try `learning_rate_low: 1e-4` and `learning_rate_high: 5e-5`
2. Increase training episodes: Use 10000+ episodes for complex scenarios
3. Adjust reward coefficients: Increase `alpha` to encourage investment, increase `beta` for stability
4. Check configuration: Ensure income > fixed_expenses + variable_expense_mean
5. Verify environment: Run sanity checks with `pytest tests/test_sanity_checks.py -v`

#### Agent goes bankrupt frequently
**Symptoms:** Episodes terminate early due to negative cash balance

**Solutions:**
1. Increase `safety_threshold` to maintain higher cash reserves
2. Increase `beta` (stability penalty) to discourage risky behavior
3. Increase `delta` (debt penalty) to strongly discourage negative balance
4. Use conservative profile: `python3 train.py --profile conservative`
5. Reduce `variable_expense_std` to decrease expense uncertainty

#### Agent doesn't invest enough
**Symptoms:** Low investment amounts, high cash balances

**Solutions:**
1. Increase `alpha` (investment reward) to encourage more investment
2. Decrease `beta` (stability penalty) to reduce cash hoarding
3. Use aggressive profile: `python3 train.py --profile aggressive`
4. Reduce `safety_threshold` to allow lower cash reserves
5. Increase `risk_tolerance` in environment configuration

#### Training is too slow
**Symptoms:** Training takes too long to complete

**Solutions:**
1. Reduce `num_episodes` for initial experiments (try 1000-2000)
2. Reduce `max_months` to shorten episodes (try 24-36 months)
3. Increase `batch_size` for faster updates (try 64 or 128)
4. Use GPU acceleration if available (PyTorch will use CUDA automatically)
5. Disable TensorBoard logging: `python3 train.py --no-log`

#### Memory issues during training
**Symptoms:** Out of memory errors, system slowdown

**Solutions:**
1. Reduce `batch_size` (try 16 or 8)
2. Reduce `max_months` to limit episode length
3. Clear episode buffer more frequently (modify HRLTrainer)
4. Use smaller neural networks (modify agent architectures)
5. Run on a machine with more RAM

### Frequently Asked Questions

**Q: How long does training take?**
A: Training 5000 episodes typically takes 10-30 minutes on a modern CPU, depending on configuration. Using GPU acceleration can reduce this to 5-10 minutes.

**Q: How do I know if my agent is learning?**
A: Monitor these indicators:
- Episode rewards should increase over time
- Cash stability index should improve (approach 1.0)
- Total invested should increase
- Episode length should increase (agent survives longer)
- Run sanity checks: `pytest tests/test_sanity_checks.py -v`

**Q: Which behavioral profile should I use?**
A: 
- **Conservative**: If you prioritize stability and have unpredictable expenses
- **Balanced**: If you want moderate growth with reasonable stability (recommended default)
- **Aggressive**: If you prioritize investment growth and have stable income

**Q: Can I customize a behavioral profile?**
A: Yes! Create a custom YAML configuration file based on one of the examples in `configs/`, then modify the parameters to suit your needs.

**Q: How do I interpret the performance metrics?**
A:
- **Cumulative Wealth Growth**: Total invested capital (higher is better)
- **Cash Stability Index**: % months with positive balance (higher is better, aim for >0.9)
- **Sharpe Ratio**: Risk-adjusted performance (higher is better, >1.0 is good)
- **Goal Adherence**: Alignment with strategic goals (lower is better, <0.1 is good)
- **Policy Stability**: Consistency of decisions (lower is better, <0.05 is good)

**Q: Can I use this for real financial planning?**
A: This system is designed for research and educational purposes. While it models realistic financial scenarios, it should not be used as the sole basis for real financial decisions. Consult with a qualified financial advisor for personal financial planning.

**Q: How do I save and load trained models?**
A: The training script automatically saves models to the output directory. Use the evaluation script to load and test them:
```bash
python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt
```

**Q: Can I modify the neural network architecture?**
A: Yes! Edit the network definitions in `src/agents/budget_executor.py` and `src/agents/financial_strategist.py`. The default architectures are:
- Low-level: [128, 128] hidden layers
- High-level: [64, 64] hidden layers

**Q: How do I visualize training progress?**
A: Use TensorBoard to monitor training in real-time:
```bash
# Start training with logging (enabled by default)
python3 train.py --profile balanced

# In another terminal, start TensorBoard
tensorboard --logdir=runs

# Open browser to http://localhost:6006
```

**Q: What if I want to change the action space?**
A: The action space is defined in `BudgetEnv` as a 3-dimensional continuous vector [invest, save, consume]. Modifying this requires changes to:
- `BudgetEnv.action_space` definition
- `BudgetEnv.step()` action processing
- `BudgetExecutor` output layer
- Reward computation logic

## Documentation

> **📚 Looking for something specific?** Check the [Documentation Index](DOCUMENTATION_INDEX.md) for a complete guide to all documentation.

### Core Documentation
- [Requirements Document](.kiro/specs/hrl-finance-system/requirements.md) - Detailed system requirements with EARS patterns
- [Design Document](.kiro/specs/hrl-finance-system/design.md) - Architecture and component design
- [Implementation Tasks](.kiro/specs/hrl-finance-system/tasks.md) - Development roadmap and task tracking
- [HLD/LLD Document](Requirements/HRL_Finance_System_Design.md) - High and low-level design specifications

### Testing Documentation
- [Test Coverage Summary](tests/TEST_COVERAGE.md) - Comprehensive test coverage overview
- [Sanity Check Tests](tests/test_sanity_checks.py) - System-level validation tests
- [Integration Tests](tests/test_hrl_trainer.py) - Complete training pipeline tests

### Examples and Tutorials
- [Examples README](examples/README.md) - Overview of all example scripts
- [Basic BudgetEnv Usage](examples/basic_budget_env_usage.py) - Simple environment demonstration
- [RewardEngine Usage](examples/reward_engine_usage.py) - Reward computation examples
- [Analytics Usage](examples/analytics_usage.py) - Performance metrics tracking
- [Training with Analytics](examples/training_with_analytics.py) - Complete training example
- [Logging Usage](examples/logging_usage.py) - TensorBoard logging demonstration
- [Checkpointing Usage](examples/checkpointing_usage.py) - Save/load/resume functionality

### Configuration Examples
- [Conservative Profile](configs/conservative.yaml) - Low risk, high stability configuration
- [Balanced Profile](configs/balanced.yaml) - Moderate risk, balanced configuration
- [Aggressive Profile](configs/aggressive.yaml) - High risk, growth-focused configuration

### Change History
- [Changelog](CHANGELOG.md) - Version history and implementation progress

## Extending the System

### Adding Custom Reward Components

You can extend the reward system by modifying `RewardEngine`:

```python
# In src/environment/reward_engine.py
class RewardEngine:
    def compute_low_level_reward(self, action, state, next_state):
        # Add custom reward component
        custom_reward = self._compute_custom_reward(action, state)
        
        # Combine with existing rewards
        base_reward = self._compute_base_reward(action, state, next_state)
        return base_reward + custom_reward
    
    def _compute_custom_reward(self, action, state):
        # Example: Reward diversification
        invest, save, consume = action
        diversification_bonus = -abs(invest - save)  # Penalize imbalance
        return 0.5 * diversification_bonus
```

### Adding Custom Metrics

Extend `AnalyticsModule` to track additional metrics:

```python
# In src/utils/analytics.py
class AnalyticsModule:
    def __init__(self):
        super().__init__()
        self.custom_metrics = []
    
    def record_step(self, state, action, reward, goal=None, invested_amount=None):
        super().record_step(state, action, reward, goal, invested_amount)
        
        # Track custom metric
        custom_value = self._compute_custom_metric(state, action)
        self.custom_metrics.append(custom_value)
    
    def compute_episode_metrics(self):
        metrics = super().compute_episode_metrics()
        
        # Add custom metric
        metrics['custom_metric'] = np.mean(self.custom_metrics)
        return metrics
```

### Creating Custom Behavioral Profiles

Add new profiles to `config_manager.py`:

```python
# In src/utils/config_manager.py
def load_behavioral_profile(profile_name: str):
    profiles = {
        'conservative': _create_conservative_config(),
        'balanced': _create_balanced_config(),
        'aggressive': _create_aggressive_config(),
        'custom': _create_custom_config(),  # Add your profile
    }
    # ... rest of implementation

def _create_custom_config():
    """Create custom behavioral profile configuration"""
    env_config = EnvironmentConfig(
        income=3200,
        fixed_expenses=1400,
        variable_expense_mean=700,
        variable_expense_std=100,
        inflation=0.02,
        safety_threshold=750,  # Custom value
        max_months=60,
        initial_cash=0,
        risk_tolerance=0.65  # Custom value
    )
    
    training_config = TrainingConfig(
        num_episodes=5000,
        gamma_low=0.95,
        gamma_high=0.99,
        high_period=8,  # Custom value
        batch_size=32,
        learning_rate_low=3e-4,
        learning_rate_high=1e-4
    )
    
    reward_config = RewardConfig(
        alpha=12.0,  # Custom value
        beta=0.15,   # Custom value
        gamma=5.0,
        delta=20.0,
        lambda_=1.0,
        mu=0.5
    )
    
    return env_config, training_config, reward_config
```

### Modifying Neural Network Architectures

Customize agent architectures for different problem complexities:

```python
# In src/agents/budget_executor.py
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Original: [128, 128]
        # Custom: Deeper network for complex scenarios
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )
```

### Adding New Environment Features

Extend `BudgetEnv` with additional financial scenarios:

```python
# In src/environment/budget_env.py
class BudgetEnv(gym.Env):
    def __init__(self, config, reward_config=None):
        super().__init__(config, reward_config)
        
        # Add new features
        self.emergency_fund = 0
        self.investment_returns = []
    
    def step(self, action):
        # Add investment returns
        if self.total_invested > 0:
            return_rate = np.random.normal(0.007, 0.02)  # ~8% annual
            investment_return = self.total_invested * return_rate
            self.cash_balance += investment_return
            self.investment_returns.append(investment_return)
        
        # Continue with normal step logic
        return super().step(action)
```

## Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues
- Use GitHub Issues to report bugs or request features
- Include system information, configuration, and error messages
- Provide minimal reproducible examples when possible

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes with clear commit messages
4. Add tests for new functionality
5. Ensure all tests pass: `pytest tests/ -v`
6. Update documentation as needed
7. Submit a pull request

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Add docstrings for classes and methods
- Keep functions focused and modular
- Write comprehensive tests for new features

### Testing Guidelines
- Write unit tests for individual components
- Add integration tests for component interactions
- Include edge case tests for robustness
- Maintain test coverage above 80%
- Run full test suite before submitting: `pytest tests/ --cov=src`

### Documentation Guidelines
- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update configuration documentation for new parameters
- Keep CHANGELOG.md up to date

## Future Enhancements

Potential areas for future development:

- **Multi-asset Investment**: Support for different investment types (stocks, bonds, real estate)
- **Tax Modeling**: Incorporate tax implications of investment decisions
- **Income Variability**: Model irregular income patterns (freelance, commission-based)
- **Life Events**: Simulate major life events (marriage, children, home purchase)
- **Debt Management**: Add support for loans, credit cards, and debt repayment strategies
- **Retirement Planning**: Long-term planning with retirement goals and pension modeling
- **Risk-Adjusted Returns**: More sophisticated investment return modeling
- **Multi-Agent Scenarios**: Household financial planning with multiple decision-makers
- **Transfer Learning**: Pre-trained models for different financial scenarios
- **Explainable AI**: Interpretability tools to understand agent decisions

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hrl_finance_system,
  author = {Rocchi, Alessio},
  title = {Personal Finance Optimization HRL System},
  year = {2024},
  description = {A hierarchical reinforcement learning system for optimal financial decision-making},
  url = {https://github.com/yourusername/hrl-finance-system}
}
```

## License

This project is for research and educational purposes.

## Author

Alessio Rocchi

## Acknowledgments

This project implements hierarchical reinforcement learning concepts from:
- HIRO: Data Efficient Hierarchical Reinforcement Learning (Nachum et al., 2018)
- Proximal Policy Optimization (Schulman et al., 2017)
- Option-Critic Architecture (Bacon et al., 2017)

Built with:
- [Gymnasium](https://gymnasium.farama.org/) - RL environment framework
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Experiment tracking
