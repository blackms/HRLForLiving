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

The system includes three predefined behavioral profiles with different risk tolerances:

#### Conservative Profile
- Risk tolerance: 0.3
- Safety threshold: $1,500
- Investment reward coefficient (α): 5.0
- Stability penalty coefficient (β): 0.5
- Focus: Capital preservation and stability

#### Balanced Profile (Default)
- Risk tolerance: 0.5
- Safety threshold: $1,000
- Investment reward coefficient (α): 10.0
- Stability penalty coefficient (β): 0.1
- Focus: Balanced growth and stability

#### Aggressive Profile
- Risk tolerance: 0.8
- Safety threshold: $500
- Investment reward coefficient (α): 15.0
- Stability penalty coefficient (β): 0.05
- Focus: Maximum investment growth

### YAML Configuration Format

Create a YAML file with the following structure (see `configs/` directory for examples):

```yaml
environment:
  income: 3200
  fixed_expenses: 1400
  variable_expense_mean: 700
  variable_expense_std: 100
  inflation: 0.02
  safety_threshold: 1000
  max_months: 60
  initial_cash: 0
  risk_tolerance: 0.5

training:
  num_episodes: 5000
  gamma_low: 0.95
  gamma_high: 0.99
  high_period: 6
  batch_size: 32
  learning_rate_low: 0.0003
  learning_rate_high: 0.0001

reward:
  alpha: 10.0    # Investment reward coefficient
  beta: 0.1      # Stability penalty coefficient
  gamma: 5.0     # Overspend penalty coefficient
  delta: 20.0    # Debt penalty coefficient
  lambda_: 1.0   # Wealth growth coefficient
  mu: 0.5        # Stability bonus coefficient
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

### Training the HRL System

Train the system using the main training script with either YAML configuration files or predefined behavioral profiles:

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

## Documentation

- [Requirements Document](.kiro/specs/hrl-finance-system/requirements.md) - Detailed system requirements
- [Design Document](.kiro/specs/hrl-finance-system/design.md) - Architecture and component design
- [Implementation Tasks](.kiro/specs/hrl-finance-system/tasks.md) - Development roadmap
- [HLD/LLD Document](Requirements/HRL_Finance_System_Design.md) - High and low-level design
- [Test Coverage Summary](tests/TEST_COVERAGE.md) - Comprehensive test coverage overview
- [Basic Usage Example](examples/basic_budget_env_usage.py) - Simple BudgetEnv demonstration
- [Changelog](CHANGELOG.md) - Version history and implementation progress

## License

This project is for research and educational purposes.

## Author

Alessio Rocchi
