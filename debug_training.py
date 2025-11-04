#!/usr/bin/env python3
"""Debug training to find NaN source"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.utils.config_manager import load_config

# Load config
env_config, training_config, reward_config = load_config('configs/personal_eur.yaml')

print("Creating environment and agents...")
env = BudgetEnv(env_config, reward_config)
reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
high_agent = FinancialStrategist(training_config)
low_agent = BudgetExecutor(training_config)

print("\nRunning single episode with agent...")

state, _ = env.reset(seed=42)
done = False
step = 0
episode_reward = 0
episode_states = []
episode_actions = []
episode_rewards = []
state_history = [state]

# High-level goal (aggregate state first)
aggregated_state = high_agent.aggregate_state(state_history)
goal = high_agent.select_goal(aggregated_state)
print(f"Initial aggregated state: {aggregated_state}")
print(f"Initial goal: {goal}")

while not done and step < 60:
    # Low-level action
    action = low_agent.act(state, goal)
    
    # Check for NaN in action
    if np.any(np.isnan(action)):
        print(f"\n!!! NaN in action at step {step} !!!")
        print(f"State: {state}")
        print(f"Goal: {goal}")
        print(f"Action: {action}")
        break
    
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Check for NaN in reward or state
    if np.isnan(reward):
        print(f"\n!!! NaN in reward at step {step} !!!")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Next state: {next_state}")
        print(f"Info: {info}")
        break
    
    if np.any(np.isnan(next_state)):
        print(f"\n!!! NaN in next_state at step {step} !!!")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Next state: {next_state}")
        break
    
    episode_states.append(state)
    episode_actions.append(action)
    episode_rewards.append(reward)
    episode_reward += reward
    
    state = next_state
    state_history.append(state)
    
    # Update goal every high_period steps
    if (step + 1) % training_config.high_period == 0:
        aggregated_state = high_agent.aggregate_state(state_history)
        goal = high_agent.select_goal(aggregated_state)
    done = terminated or truncated
    step += 1
    
    if step % 10 == 0:
        print(f"Step {step}: reward={reward:.2f}, cash={env.cash_balance:.2f}, invested={env.total_invested:.2f}")

print(f"\nEpisode completed: {step} steps, total reward: {episode_reward:.2f}")
print(f"Final cash: {env.cash_balance:.2f}, Total invested: {env.total_invested:.2f}")

# Try training update
print("\n" + "=" * 70)
print("Testing training update...")
print("=" * 70)

if len(episode_rewards) > 0:
    # Compute returns
    returns = []
    G = 0
    for r in reversed(episode_rewards):
        G = r + training_config.gamma_low * G
        returns.insert(0, G)
    
    print(f"Returns computed: min={min(returns):.2f}, max={max(returns):.2f}, mean={np.mean(returns):.2f}")
    
    # Check for NaN in returns
    if np.any(np.isnan(returns)):
        print("!!! NaN in returns !!!")
    else:
        # Try low-level update
        print("\nTrying low-level agent update...")
        try:
            low_loss = low_agent.update(
                episode_states,
                episode_actions,
                episode_rewards,
                [goal] * len(episode_states)
            )
            print(f"Low-level loss: {low_loss}")
            
            if np.isnan(low_loss):
                print("!!! NaN in low-level loss !!!")
        except Exception as e:
            print(f"Error in low-level update: {e}")
        
        # Try high-level update
        print("\nTrying high-level agent update...")
        try:
            # Create dummy high-level trajectory
            high_states = episode_states[::training_config.high_period]
            high_goals = [goal] * len(high_states)
            high_rewards = [sum(episode_rewards[i:i+training_config.high_period]) 
                           for i in range(0, len(episode_rewards), training_config.high_period)]
            
            if len(high_states) > 1:
                high_loss = high_agent.update(high_states, high_goals, high_rewards)
                print(f"High-level loss: {high_loss}")
                
                if np.isnan(high_loss):
                    print("!!! NaN in high-level loss !!!")
        except Exception as e:
            print(f"Error in high-level update: {e}")

print("\n" + "=" * 70)
print("Debug complete")
print("=" * 70)
