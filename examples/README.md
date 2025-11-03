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

## Coming Soon

### 3. Training with PPO (Low-Level Agent)
Example showing how to train a single PPO agent on the BudgetEnv without hierarchical structure.

### 4. Full HRL Training
Complete example with both High-Level (Strategist) and Low-Level (Executor) agents.

### 5. Behavioral Profile Comparison
Compare performance across conservative, balanced, and aggressive profiles.

### 6. Custom Reward Functions
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
