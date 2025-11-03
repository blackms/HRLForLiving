"""Example demonstrating RewardEngine usage"""
import numpy as np
from src.environment import RewardEngine
from src.utils.config import RewardConfig
from src.utils.data_models import Transition


def main():
    """Demonstrate RewardEngine functionality"""
    
    print("=" * 60)
    print("RewardEngine Usage Example")
    print("=" * 60)
    
    # Create reward configuration
    reward_config = RewardConfig(
        alpha=10.0,    # Investment reward coefficient
        beta=0.1,      # Stability penalty coefficient
        gamma=5.0,     # Overspend penalty coefficient
        delta=20.0,    # Debt penalty coefficient
        lambda_=1.0,   # Wealth growth coefficient
        mu=0.5         # Stability bonus coefficient
    )
    
    # Initialize reward engine
    reward_engine = RewardEngine(reward_config, safety_threshold=1000)
    
    print("\n1. Computing Low-Level Rewards")
    print("-" * 60)
    
    # Example 1: Good investment scenario
    print("\nScenario 1: Balanced investment with stable cash")
    state = np.array([3200, 1400, 700, 2000, 0.02, 0.5, 50])
    action = np.array([0.3, 0.5, 0.2])  # 30% invest, 50% save, 20% consume
    next_state = np.array([3200, 1400, 700, 1800, 0.02, 0.5, 49])
    
    reward = reward_engine.compute_low_level_reward(action, state, next_state)
    print(f"  State: income=${state[0]}, cash=${state[3]}")
    print(f"  Action: invest={action[0]:.1%}, save={action[1]:.1%}, consume={action[2]:.1%}")
    print(f"  Next cash: ${next_state[3]}")
    print(f"  Reward: {reward:.2f}")
    
    # Example 2: Low cash scenario
    print("\nScenario 2: Investment with low cash (below safety threshold)")
    state = np.array([3200, 1400, 700, 1200, 0.02, 0.5, 50])
    action = np.array([0.3, 0.4, 0.3])
    next_state = np.array([3200, 1400, 700, 800, 0.02, 0.5, 49])
    
    reward = reward_engine.compute_low_level_reward(action, state, next_state)
    print(f"  State: income=${state[0]}, cash=${state[3]}")
    print(f"  Action: invest={action[0]:.1%}, save={action[1]:.1%}, consume={action[2]:.1%}")
    print(f"  Next cash: ${next_state[3]} (below threshold of $1000)")
    print(f"  Reward: {reward:.2f} (includes stability penalty)")
    
    # Example 3: Overspending scenario
    print("\nScenario 3: Overspending with excessive consumption")
    state = np.array([3200, 1400, 700, 3000, 0.02, 0.5, 50])
    action = np.array([0.1, 0.2, 0.7])
    next_state = np.array([3200, 1400, 700, 1500, 0.02, 0.5, 49])
    
    reward = reward_engine.compute_low_level_reward(action, state, next_state)
    print(f"  State: income=${state[0]}, cash=${state[3]}")
    print(f"  Action: invest={action[0]:.1%}, save={action[1]:.1%}, consume={action[2]:.1%}")
    print(f"  Next cash: ${next_state[3]} (large decrease)")
    print(f"  Reward: {reward:.2f} (includes overspend penalty)")
    
    # Example 4: Debt scenario
    print("\nScenario 4: Negative cash balance (debt)")
    state = np.array([3200, 1400, 700, 500, 0.02, 0.5, 50])
    action = np.array([0.6, 0.2, 0.2])
    next_state = np.array([3200, 1400, 700, -500, 0.02, 0.5, 49])
    
    reward = reward_engine.compute_low_level_reward(action, state, next_state)
    print(f"  State: income=${state[0]}, cash=${state[3]}")
    print(f"  Action: invest={action[0]:.1%}, save={action[1]:.1%}, consume={action[2]:.1%}")
    print(f"  Next cash: ${next_state[3]} (NEGATIVE - debt!)")
    print(f"  Reward: {reward:.2f} (heavy debt penalty)")
    
    # Example 5: High-level reward computation
    print("\n\n2. Computing High-Level Rewards")
    print("-" * 60)
    
    # Create a simulated episode history
    episode_history = []
    initial_cash = 1000
    
    print("\nSimulating 6-month strategic period...")
    for month in range(6):
        cash = initial_cash + month * 400  # Growing cash balance
        next_cash = initial_cash + (month + 1) * 400
        
        state = np.array([3200, 1400, 700, cash, 0.02, 0.5, 50 - month])
        goal = np.array([0.3, 1000, 0.5])
        action = np.array([0.3, 0.4, 0.3])
        low_reward = 100.0 + month * 10  # Increasing rewards
        next_state = np.array([3200, 1400, 700, next_cash, 0.02, 0.5, 49 - month])
        done = False
        
        transition = Transition(state, goal, action, low_reward, next_state, done)
        episode_history.append(transition)
        
        print(f"  Month {month + 1}: cash=${cash} -> ${next_cash}, reward={low_reward:.2f}")
    
    high_reward = reward_engine.compute_high_level_reward(episode_history)
    
    print(f"\nHigh-Level Reward Components:")
    print(f"  - Aggregated low-level rewards: {sum(t.reward for t in episode_history):.2f}")
    print(f"  - Wealth change: ${episode_history[-1].next_state[3] - episode_history[0].state[3]:.2f}")
    print(f"  - Stability bonus: All months positive balance")
    print(f"  - Total high-level reward: {high_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
