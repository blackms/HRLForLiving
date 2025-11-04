# Examples

This directory contains usage examples for the Personal Finance Optimization HRL System.

## Available Examples

### 1. Basic BudgetEnv Usage (`basic_budget_env_usage.py`)

**Status:** ✅ Ready to run

Demonstrates the core functionality of the BudgetEnv:
- Environment configuration with `EnvironmentConfig`
- Resetting the environment
- Taking actions (invest, save, consume allocations)
- Observing state transitions and rewards
- Running a complete episode with adaptive strategy

**Run:**
```bash
PYTHONPATH=. python3 examples/basic_budget_env_usage.py
```

**What you'll learn:**
- How to create and configure a BudgetEnv
- Understanding the 7-dimensional state space
- How actions are normalized automatically
- How cash balance evolves over time
- How to implement simple allocation strategies

### 2. RewardEngine Usage (`reward_engine_usage.py`)

**Status:** ✅ Ready to run

Demonstrates the multi-objective reward computation system:
- Configuring reward coefficients with `RewardConfig`
- Computing low-level rewards for different scenarios
- Understanding reward components (investment, stability, overspend, debt)
- Computing high-level strategic rewards
- Aggregating rewards over multiple time steps
- Automatic reward scaling for training stability

**Run:**
```bash
PYTHONPATH=. python3 examples/reward_engine_usage.py
```

**What you'll learn:**
- How to configure and initialize a RewardEngine
- How different actions affect reward values
- Impact of cash balance on stability penalties
- How overspending and debt are penalized
- How high-level rewards aggregate low-level performance
- Understanding wealth growth and stability bonuses
- **Why rewards are automatically scaled by 1000.0 for training stability**

### 3. Analytics Module Usage (`analytics_usage.py`)

**Status:** ✅ Ready to run | **Tests:** ✅ 18 comprehensive test cases

Demonstrates the AnalyticsModule for tracking and computing performance metrics:
- Recording step-by-step data during episodes
- Computing comprehensive performance metrics
- Understanding metric interpretation
- Integration patterns for training loops
- Reset functionality for new episodes
- Edge case handling (empty data, single step, missing goals)

**Run:**
```bash
PYTHONPATH=. python3 examples/analytics_usage.py
```

**What you'll learn:**
- How to initialize and use the AnalyticsModule
- Recording states, actions, rewards, goals, and investments
- Computing 5 key performance metrics:
  - Cumulative Wealth Growth (total invested capital)
  - Cash Stability Index (% months with positive balance)
  - Sharpe-like Ratio (risk-adjusted performance)
  - Goal Adherence (alignment with strategic goals)
  - Policy Stability (consistency of decisions)
- How to interpret each metric
- Best practices for integration with training loops

### 4. Training with Analytics Integration (`training_with_analytics.py`)

**Status:** ✅ Ready to run | **Tests:** ✅ 18 comprehensive test cases

Demonstrates how the AnalyticsModule is automatically integrated with HRLTrainer:
- Automatic step recording during training
- Automatic metric computation at episode end
- Tracking learning progress over episodes
- Comparing early vs late training performance
- Zero-overhead analytics integration

**Run:**
```bash
PYTHONPATH=. python3 examples/training_with_analytics.py
```

**What you'll learn:**
- How HRLTrainer automatically integrates analytics
- No manual tracking needed in training loops
- How to access analytics-derived metrics from training history
- Analyzing learning progress using metrics
- Understanding the relationship between training and performance metrics
- Best practices for monitoring HRL training

### 5. TensorBoard Logging Usage (`logging_usage.py`)

**Status:** ✅ Ready to run

Demonstrates TensorBoard logging for experiment tracking and visualization:
- Initializing ExperimentLogger for TensorBoard
- Logging hyperparameters for experiment tracking
- Automatic logging of training curves (rewards, losses)
- Tracking episode metrics (wealth, stability, Sharpe ratio)
- Visualizing action and goal distributions
- Real-time monitoring with TensorBoard web interface

**Run:**
```bash
PYTHONPATH=. python3 examples/logging_usage.py

# Then view logs with:
tensorboard --logdir=runs
# Open browser to: http://localhost:6006
```

**What you'll learn:**
- How to initialize and configure ExperimentLogger
- Logging hyperparameters for reproducibility
- Automatic integration with HRLTrainer
- Viewing training curves in TensorBoard
- Analyzing action and goal distributions
- Comparing multiple experiments
- Best practices for experiment tracking

### 6. Checkpointing and Resume Functionality (`checkpointing_usage.py`)

**Status:** ✅ Ready to run | **Tests:** ✅ 7 comprehensive test cases

Demonstrates checkpointing and resume functionality for long training runs:
- Training with automatic checkpointing
- Saving checkpoints at regular intervals
- Evaluating and saving the best model
- Resuming training from a checkpoint
- Loading the best model for evaluation
- Checkpoint directory structure and metadata

**Run:**
```bash
PYTHONPATH=. python3 examples/checkpointing_usage.py
```

**What you'll learn:**
- How to enable automatic checkpointing during training
- Configuring save intervals and evaluation intervals
- Understanding checkpoint directory structure
- How to resume training from a saved checkpoint
- Loading and evaluating the best model
- Best practices for long training runs
- Checkpoint metadata and configuration preservation

## Coming Soon

### 7. Training with PPO (Low-Level Agent)
Example showing how to train a single PPO agent on the BudgetEnv without hierarchical structure.

### 8. Full HRL Training
Complete example with both High-Level (Strategist) and Low-Level (Executor) agents.

### 9. Behavioral Profile Comparison
Compare performance across conservative, balanced, and aggressive profiles.

### 10. Custom Reward Functions
Demonstrate how to integrate custom reward engines for different optimization objectives.

## Requirements

All examples require the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Note:** Examples must be run from the project root directory with `PYTHONPATH=.` to ensure proper module imports.

## Notes

- Examples use seed values for reproducibility
- Default configuration uses realistic monthly income/expense values
- All monetary values are in USD
