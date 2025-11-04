#!/usr/bin/env python3
"""Test if rewards from environment are NaN"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.utils.config_manager import load_config

# Load config
env_config, training_config, reward_config = load_config('configs/personal_eur.yaml')

# Create environment and agents
env = BudgetEnv(env_config, reward_config)
high_agent = FinancialStrategist(training_config)
low_agent = BudgetExecutor(training_config)

print("Running episode to check for NaN rewards...")

state, _ = env.reset(seed=42)
state_history = [state]

# Get initial goal
aggregated_state = high_agent.aggregate_state(state_history)
goal = high_agent.select_goal(aggregated_state)

episode_reward = 0
step = 0
done = False
nan_found = False

while not done and step < 60:
    action = low_agent.act(state, goal)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    if np.isnan(reward):
        print(f"\n!!! NaN reward at step {step} !!!")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Next state: {next_state}")
        print(f"Info: {info}")
        nan_found = True
        break
    
    episode_reward += reward
    
    if np.isnan(episode_reward):
        print(f"\n!!! NaN episode_reward at step {step} !!!")
        print(f"Reward: {reward}")
        print(f"Episode reward before: {episode_reward - reward}")
        nan_found = True
        break
    
    state = next_state
    state_history.append(state)
    done = terminated or truncated
    step += 1
    
    if (step + 1) % training_config.high_period == 0:
        aggregated_state = high_agent.aggregate_state(state_history)
        goal = high_agent.select_goal(aggregated_state)

if not nan_found:
    print(f"\n✓ No NaN found in {step} steps")
    print(f"Episode reward: {episode_reward}")
    print(f"Final cash: {env.cash_balance}")
else:
    print("\n✗ NaN detected during episode")
