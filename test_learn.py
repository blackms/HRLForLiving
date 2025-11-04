#!/usr/bin/env python3
"""Test the learn method directly"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agents.budget_executor import BudgetExecutor
from src.utils.config_manager import load_config
from src.utils.data_models import Transition

# Load config
env_config, training_config, reward_config = load_config('configs/personal_eur.yaml')

# Create agent
low_agent = BudgetExecutor(training_config)

# Create fake transitions
transitions = []
for i in range(32):  # batch_size
    state = np.array([3200, 2160, 400, 5000, 0.02, 0.5, 60-i], dtype=np.float32)
    goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
    action = np.array([0.3, 0.4, 0.3], dtype=np.float32)
    reward = 1000.0 + np.random.randn() * 100
    next_state = state.copy()
    next_state[3] -= 100  # Decrease cash
    done = False
    
    transition = Transition(state, goal, action, reward, next_state, done)
    transitions.append(transition)

print("Testing learn method with fake transitions...")
print(f"Number of transitions: {len(transitions)}")
print(f"Sample reward: {transitions[0].reward}")

try:
    metrics = low_agent.learn(transitions)
    print(f"\nMetrics: {metrics}")
    
    if np.isnan(metrics['loss']):
        print("!!! NaN in loss !!!")
    else:
        print("✓ No NaN in loss")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
