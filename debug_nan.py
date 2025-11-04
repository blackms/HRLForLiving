#!/usr/bin/env python3
"""Debug script to identify NaN issues"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.utils.config_manager import load_config

# Load config
env_config, training_config, reward_config = load_config('configs/personal_eur.yaml')

print("=" * 70)
print("Configuration Check")
print("=" * 70)
print(f"Income: {env_config.income}")
print(f"Fixed Expenses: {env_config.fixed_expenses}")
print(f"Variable Expense Mean: {env_config.variable_expense_mean}")
print(f"Available after fixed: {env_config.income - env_config.fixed_expenses}")
print(f"Available after avg variable: {env_config.income - env_config.fixed_expenses - env_config.variable_expense_mean}")

# Create environment
env = BudgetEnv(env_config, reward_config)
reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)

print("\n" + "=" * 70)
print("Running Test Episode")
print("=" * 70)

state, info = env.reset(seed=42)
print(f"\nInitial state: {state}")
print(f"Initial cash: {env.cash_balance}")

done = False
step = 0
total_reward = 0

while not done and step < 12:
    # Simple balanced action
    action = np.array([0.3, 0.4, 0.3], dtype=np.float32)
    
    print(f"\n--- Step {step + 1} ---")
    print(f"Action: invest={action[0]:.2f}, save={action[1]:.2f}, consume={action[2]:.2f}")
    print(f"Cash before: {env.cash_balance:.2f}")
    
    next_state, reward, terminated, truncated, info = env.step(action)
    
    print(f"Cash after: {env.cash_balance:.2f}")
    print(f"Reward: {reward:.4f}")
    print(f"Total invested: {env.total_invested:.2f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    
    if np.isnan(reward):
        print("\n!!! NaN DETECTED IN REWARD !!!")
        print(f"State: {next_state}")
        print(f"Info: {info}")
        break
    
    if np.any(np.isnan(next_state)):
        print("\n!!! NaN DETECTED IN STATE !!!")
        print(f"State: {next_state}")
        break
    
    total_reward += reward
    done = terminated or truncated
    step += 1

print(f"\n" + "=" * 70)
print(f"Episode finished after {step} steps")
print(f"Total reward: {total_reward:.4f}")
print(f"Final cash: {env.cash_balance:.2f}")
print(f"Total invested: {env.total_invested:.2f}")
print("=" * 70)
