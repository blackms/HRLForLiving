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
python examples/basic_budget_env_usage.py
```

**What you'll learn:**
- How to create and configure a BudgetEnv
- Understanding the 7-dimensional state space
- How actions are normalized automatically
- How cash balance evolves over time
- How to implement simple allocation strategies

## Coming Soon

### 2. Training with PPO (Low-Level Agent)
Example showing how to train a single PPO agent on the BudgetEnv without hierarchical structure.

### 3. Full HRL Training
Complete example with both High-Level (Strategist) and Low-Level (Executor) agents.

### 4. Behavioral Profile Comparison
Compare performance across conservative, balanced, and aggressive profiles.

### 5. Custom Reward Functions
Demonstrate how to integrate custom reward engines for different optimization objectives.

## Requirements

All examples require the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Notes

- Examples use seed values for reproducibility
- Default configuration uses realistic monthly income/expense values
- All monetary values are in USD
