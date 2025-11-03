"""
Training with Analytics Integration Example

Demonstrates how the AnalyticsModule is integrated with HRLTrainer
to automatically track performance metrics during training.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.training.hrl_trainer import HRLTrainer
from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.environment.reward_engine import RewardEngine
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig


def main():
    """Demonstrate HRLTrainer with integrated analytics"""
    
    print("=" * 70)
    print("Training with Analytics Integration Example")
    print("=" * 70)
    
    # Configure system
    env_config = EnvironmentConfig(
        income=3200,
        fixed_expenses=1400,
        variable_expense_mean=700,
        variable_expense_std=100,
        inflation=0.02,
        safety_threshold=1000,
        max_months=24,  # Shorter episodes for demo
        initial_cash=0,
        risk_tolerance=0.5
    )
    
    training_config = TrainingConfig(
        num_episodes=10,  # Small number for demo
        gamma_low=0.95,
        gamma_high=0.99,
        high_period=6,
        batch_size=8,
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
    print("\n1. Initializing HRL system with analytics...")
    print("-" * 70)
    
    env = BudgetEnv(env_config, reward_config)
    high_agent = FinancialStrategist(training_config)
    low_agent = BudgetExecutor(training_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    
    trainer = HRLTrainer(env, high_agent, low_agent, reward_engine, training_config)
    
    print("✓ HRLTrainer initialized with AnalyticsModule")
    print(f"✓ Analytics module: {trainer.analytics}")
    
    # Run training
    print("\n2. Running training with automatic analytics tracking...")
    print("-" * 70)
    print("Note: Analytics automatically records each step and computes metrics")
    
    history = trainer.train(num_episodes=10)
    
    # Display analytics-derived metrics
    print("\n3. Analytics-derived metrics from training:")
    print("-" * 70)
    
    print("\nMetrics tracked per episode:")
    print("  • Cumulative Wealth Growth - Total invested capital")
    print("  • Cash Stability Index - % months with positive balance")
    print("  • Sharpe Ratio - Risk-adjusted performance")
    print("  • Goal Adherence - How well low-level follows high-level goals")
    print("  • Policy Stability - Consistency of decisions")
    
    # Show metrics for last 5 episodes
    print("\nLast 5 episodes metrics:")
    print("-" * 70)
    
    for i in range(max(0, len(history['episode_rewards']) - 5), len(history['episode_rewards'])):
        episode_num = i + 1
        print(f"\nEpisode {episode_num}:")
        print(f"  Reward: {history['episode_rewards'][i]:.2f}")
        print(f"  Length: {history['episode_lengths'][i]} months")
        print(f"  Final Cash: ${history['cash_balances'][i]:.2f}")
        print(f"  Total Invested: ${history['total_invested'][i]:.2f}")
        print(f"  Wealth Growth: ${history['cumulative_wealth_growth'][i]:.2f}")
        print(f"  Stability Index: {history['cash_stability_index'][i]:.2%}")
        print(f"  Sharpe Ratio: {history['sharpe_ratio'][i]:.2f}")
        print(f"  Goal Adherence: {history['goal_adherence'][i]:.4f}")
        print(f"  Policy Stability: {history['policy_stability'][i]:.4f}")
    
    # Compute aggregate statistics
    print("\n4. Aggregate statistics across all episodes:")
    print("-" * 70)
    
    avg_wealth_growth = np.mean(history['cumulative_wealth_growth'])
    avg_stability = np.mean(history['cash_stability_index'])
    avg_sharpe = np.mean(history['sharpe_ratio'])
    avg_goal_adherence = np.mean(history['goal_adherence'])
    avg_policy_stability = np.mean(history['policy_stability'])
    
    print(f"\nAverage Wealth Growth: ${avg_wealth_growth:.2f}")
    print(f"Average Stability Index: {avg_stability:.2%}")
    print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"Average Goal Adherence: {avg_goal_adherence:.4f}")
    print(f"Average Policy Stability: {avg_policy_stability:.4f}")
    
    # Show how analytics improves over training
    print("\n5. Learning progress (first vs last 3 episodes):")
    print("-" * 70)
    
    first_3_wealth = np.mean(history['cumulative_wealth_growth'][:3])
    last_3_wealth = np.mean(history['cumulative_wealth_growth'][-3:])
    
    first_3_stability = np.mean(history['cash_stability_index'][:3])
    last_3_stability = np.mean(history['cash_stability_index'][-3:])
    
    first_3_goal = np.mean(history['goal_adherence'][:3])
    last_3_goal = np.mean(history['goal_adherence'][-3:])
    
    print(f"\nWealth Growth:")
    print(f"  First 3 episodes: ${first_3_wealth:.2f}")
    print(f"  Last 3 episodes: ${last_3_wealth:.2f}")
    print(f"  Improvement: {((last_3_wealth - first_3_wealth) / (first_3_wealth + 1e-8) * 100):.1f}%")
    
    print(f"\nCash Stability:")
    print(f"  First 3 episodes: {first_3_stability:.2%}")
    print(f"  Last 3 episodes: {last_3_stability:.2%}")
    print(f"  Change: {(last_3_stability - first_3_stability) * 100:.1f} percentage points")
    
    print(f"\nGoal Adherence (lower is better):")
    print(f"  First 3 episodes: {first_3_goal:.4f}")
    print(f"  Last 3 episodes: {last_3_goal:.4f}")
    print(f"  Improvement: {((first_3_goal - last_3_goal) / (first_3_goal + 1e-8) * 100):.1f}%")
    
    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("=" * 70)
    print("""
1. Analytics is automatically integrated into HRLTrainer
2. No manual tracking needed - analytics.record_step() called automatically
3. Metrics computed at end of each episode via analytics.compute_episode_metrics()
4. All metrics stored in training_history for analysis
5. Analytics reset automatically at start of each episode

This integration provides comprehensive performance tracking without
additional code in the training loop!
    """)


if __name__ == "__main__":
    main()
