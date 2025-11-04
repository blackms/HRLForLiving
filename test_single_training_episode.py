#!/usr/bin/env python3
"""Test a single training episode to find NaN source"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.environment.reward_engine import RewardEngine
from src.utils.config_manager import load_config
from src.utils.analytics import AnalyticsModule

# Load config
env_config, training_config, reward_config = load_config('configs/personal_eur.yaml')

# Create components
env = BudgetEnv(env_config, reward_config)
reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
high_agent = FinancialStrategist(training_config)
low_agent = BudgetExecutor(training_config)
analytics = AnalyticsModule()

print("Running single training episode...")

# Reset
analytics.reset()
state, _ = env.reset(seed=42)
state_history = [state]

# Get initial goal
aggregated_state = high_agent.aggregate_state(state_history)
goal = high_agent.select_goal(aggregated_state)

episode_reward = 0
step = 0
done = False

print(f"Initial state: {state}")
print(f"Initial goal: {goal}")
print(f"Initial episode_reward: {episode_reward}")

while not done and step < 60:
    # Get action
    action = low_agent.act(state, goal)
    
    # Step environment
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Calculate invested amount
    invested_amount = action[0] * state[0]
    
    # Record in analytics
    analytics.record_step(state, action, reward, goal, invested_amount)
    
    # Update episode reward
    print(f"Step {step}: reward={reward:.2f}, episode_reward before={episode_reward:.2f}", end="")
    episode_reward += reward
    print(f", after={episode_reward:.2f}, isnan={np.isnan(episode_reward)}")
    
    if np.isnan(episode_reward):
        print("\n!!! NaN detected in episode_reward !!!")
        print(f"reward: {reward}")
        print(f"type(reward): {type(reward)}")
        print(f"np.isnan(reward): {np.isnan(reward)}")
        break
    
    # Update state
    state = next_state
    state_history.append(state)
    step += 1
    
    # Update goal periodically
    if (step + 1) % training_config.high_period == 0:
        aggregated_state = high_agent.aggregate_state(state_history)
        goal = high_agent.select_goal(aggregated_state)

print(f"\nEpisode completed: {step} steps")
print(f"Final episode_reward: {episode_reward}")
print(f"Is NaN: {np.isnan(episode_reward)}")

# Compute analytics metrics
print("\nComputing analytics metrics...")
metrics = analytics.compute_episode_metrics()
print(f"Metrics: {metrics}")

for key, value in metrics.items():
    if np.isnan(value):
        print(f"!!! NaN in metric '{key}': {value}")
