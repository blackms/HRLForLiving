"""
Basic usage example for BudgetEnv

This script demonstrates how to:
1. Create and configure a BudgetEnv
2. Reset the environment
3. Take actions and observe results
4. Run a simple episode
"""

import numpy as np
from src.environment import BudgetEnv
from src.utils.config import EnvironmentConfig


def main():
    print("=" * 60)
    print("BudgetEnv Basic Usage Example")
    print("=" * 60)
    
    # Create environment configuration
    config = EnvironmentConfig(
        income=3200,              # Monthly salary
        fixed_expenses=1400,      # Fixed monthly costs (rent, utilities, etc.)
        variable_expense_mean=700, # Average variable expenses (food, transport, etc.)
        variable_expense_std=100, # Standard deviation of variable expenses
        inflation=0.02,           # Annual inflation rate (2%)
        safety_threshold=1000,    # Minimum cash buffer
        max_months=60,           # Maximum simulation duration
        initial_cash=0,          # Starting cash balance
        risk_tolerance=0.5       # Risk profile (0=conservative, 1=aggressive)
    )
    
    # Initialize environment
    env = BudgetEnv(config)
    print(f"\n✓ Environment created with configuration:")
    print(f"  - Monthly income: ${config.income}")
    print(f"  - Fixed expenses: ${config.fixed_expenses}")
    print(f"  - Variable expenses (mean): ${config.variable_expense_mean}")
    print(f"  - Safety threshold: ${config.safety_threshold}")
    print(f"  - Max months: {config.max_months}")
    
    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"\n✓ Environment reset")
    print(f"  - Initial observation shape: {observation.shape}")
    print(f"  - Initial state: {observation}")
    
    # Take a single step
    print(f"\n{'=' * 60}")
    print("Taking a single step...")
    print(f"{'=' * 60}")
    
    # Action: [invest_ratio, save_ratio, consume_ratio]
    # Example: Invest 30%, save 50%, consume 20%
    action = np.array([0.3, 0.5, 0.2])
    print(f"Action (before normalization): {action}")
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nResults:")
    print(f"  - Normalized action: {info['action']}")
    print(f"  - Cash balance: ${info['cash_balance']:.2f}")
    print(f"  - Total invested: ${info['total_invested']:.2f}")
    print(f"  - Investment amount: ${info['invest_amount']:.2f}")
    print(f"  - Total expenses: ${info['total_expenses']:.2f}")
    print(f"  - Reward: {reward:.2f}")
    print(f"  - Month: {info['month']}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Truncated: {truncated}")
    
    # Run a complete episode
    print(f"\n{'=' * 60}")
    print("Running a complete episode (12 months)...")
    print(f"{'=' * 60}")
    
    observation, info = env.reset(seed=42)
    total_reward = 0
    
    for month in range(12):
        # Simple strategy: invest more as cash grows
        if observation[3] > 2000:  # cash_balance > 2000
            action = np.array([0.4, 0.4, 0.2])  # Aggressive
        elif observation[3] > 1000:  # cash_balance > 1000
            action = np.array([0.3, 0.5, 0.2])  # Balanced
        else:
            action = np.array([0.2, 0.6, 0.2])  # Conservative
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nMonth {info['month']:2d}: "
              f"Cash=${info['cash_balance']:7.2f} | "
              f"Invested=${info['total_invested']:7.2f} | "
              f"Reward={reward:6.2f}")
        
        if terminated or truncated:
            print(f"\n⚠ Episode ended at month {info['month']}")
            if terminated:
                print("  Reason: Negative cash balance")
            if truncated:
                print("  Reason: Maximum months reached")
            break
    
    print(f"\n{'=' * 60}")
    print(f"Episode Summary:")
    print(f"  - Total months: {info['month']}")
    print(f"  - Final cash balance: ${info['cash_balance']:.2f}")
    print(f"  - Total invested: ${info['total_invested']:.2f}")
    print(f"  - Total reward: {total_reward:.2f}")
    print(f"  - Average reward per month: {total_reward / info['month']:.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
