"""
Analytics Module Usage Example

Demonstrates how to use the AnalyticsModule to track and compute
performance metrics for the HRL financial system.
"""

import numpy as np
from src.utils.analytics import AnalyticsModule
from src.environment.budget_env import BudgetEnv
from src.utils.config import EnvironmentConfig, RewardConfig


def main():
    """Demonstrate AnalyticsModule usage"""
    
    print("=" * 70)
    print("Analytics Module Usage Example")
    print("=" * 70)
    
    # Initialize environment
    env_config = EnvironmentConfig(
        income=3200,
        fixed_expenses=1400,
        variable_expense_mean=700,
        variable_expense_std=100,
        inflation=0.02,
        safety_threshold=1000,
        max_months=12,  # Short episode for demo
        initial_cash=0,
        risk_tolerance=0.5
    )
    
    reward_config = RewardConfig(
        alpha=10.0,
        beta=0.1,
        gamma=5.0,
        delta=20.0,
        lambda_=1.0,
        mu=0.5
    )
    
    env = BudgetEnv(env_config, reward_config)
    
    # Initialize analytics module
    analytics = AnalyticsModule()
    
    print("\n1. Running episode with analytics tracking...")
    print("-" * 70)
    
    # Run episode
    state, _ = env.reset()
    done = False
    episode_reward = 0
    step_count = 0
    
    # Simulate a strategic goal (normally from high-level agent)
    goal = np.array([0.35, 1000, 0.6])  # [target_invest, safety_buffer, aggressiveness]
    
    while not done:
        # Simple policy: gradually increase investment ratio
        invest_ratio = min(0.2 + step_count * 0.02, 0.4)
        save_ratio = 0.5
        consume_ratio = 1.0 - invest_ratio - save_ratio
        
        action = np.array([invest_ratio, save_ratio, consume_ratio])
        
        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record step in analytics
        analytics.record_step(
            state=state,
            action=action,
            reward=reward,
            goal=goal,
            invested_amount=info['invest_amount']
        )
        
        episode_reward += reward
        step_count += 1
        
        print(f"  Step {step_count}: "
              f"Action=[{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}], "
              f"Cash=${info['cash_balance']:.2f}, "
              f"Invested=${info['invest_amount']:.2f}, "
              f"Reward={reward:.2f}")
        
        state = next_state
    
    print(f"\nEpisode completed: {step_count} steps, Total reward: {episode_reward:.2f}")
    
    # Compute metrics
    print("\n2. Computing performance metrics...")
    print("-" * 70)
    
    metrics = analytics.compute_episode_metrics()
    
    print("\nPerformance Metrics:")
    print(f"  • Cumulative Wealth Growth: ${metrics['cumulative_wealth_growth']:.2f}")
    print(f"    → Total amount invested over the episode")
    print(f"    → Higher is better (indicates wealth accumulation)")
    
    print(f"\n  • Cash Stability Index: {metrics['cash_stability_index']:.2%}")
    print(f"    → Percentage of months with positive cash balance")
    print(f"    → Higher is better (indicates financial stability)")
    
    print(f"\n  • Sharpe-like Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"    → Risk-adjusted performance (mean balance / std balance)")
    print(f"    → Higher is better (better risk-adjusted returns)")
    
    print(f"\n  • Goal Adherence: {metrics['goal_adherence']:.4f}")
    print(f"    → Mean absolute difference between target and actual investment")
    print(f"    → Lower is better (better goal following)")
    print(f"    → Target was {goal[0]:.2f}, actual varied by {metrics['goal_adherence']:.4f}")
    
    print(f"\n  • Policy Stability: {metrics['policy_stability']:.4f}")
    print(f"    → Variance of actions over time")
    print(f"    → Lower is better (more consistent decisions)")
    
    # Demonstrate reset
    print("\n3. Resetting analytics for next episode...")
    print("-" * 70)
    
    print(f"  Before reset: {len(analytics.states)} states recorded")
    analytics.reset()
    print(f"  After reset: {len(analytics.states)} states recorded")
    
    print("\n" + "=" * 70)
    print("Analytics Module demonstration complete!")
    print("=" * 70)
    
    # Show how to use in training loop
    print("\n4. Integration with training loop (pseudo-code):")
    print("-" * 70)
    print("""
    # In your training loop:
    analytics = AnalyticsModule()
    
    for episode in range(num_episodes):
        state = env.reset()
        analytics.reset()  # Clear previous episode data
        
        while not done:
            action = agent.act(state, goal)
            next_state, reward, done, info = env.step(action)
            
            # Record step
            analytics.record_step(state, action, reward, goal, info['invest_amount'])
            
            state = next_state
        
        # Compute and log metrics
        metrics = analytics.compute_episode_metrics()
        log_metrics(episode, metrics)
    """)


if __name__ == "__main__":
    main()
