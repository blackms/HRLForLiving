"""
Example: Using Checkpointing and Resume Functionality

This example demonstrates how to:
1. Train with automatic checkpointing
2. Resume training from a checkpoint
3. Load the best model for evaluation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.training.hrl_trainer import HRLTrainer
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig


def example_train_with_checkpointing():
    """Example: Train with automatic checkpointing"""
    print("=" * 70)
    print("Example 1: Training with Automatic Checkpointing")
    print("=" * 70)
    
    # Create configurations
    env_config = EnvironmentConfig(
        income=3200,
        fixed_expenses=1400,
        variable_expense_mean=700,
        variable_expense_std=100,
        inflation=0.02,
        safety_threshold=1000,
        max_months=60,
        initial_cash=0,
        risk_tolerance=0.5
    )
    
    training_config = TrainingConfig(
        num_episodes=100,  # Small for demo
        gamma_low=0.95,
        gamma_high=0.99,
        high_period=6,
        batch_size=32,
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
    
    # Initialize system
    env = BudgetEnv(env_config, reward_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    high_agent = FinancialStrategist(training_config)
    low_agent = BudgetExecutor(training_config)
    
    trainer = HRLTrainer(
        env, high_agent, low_agent, reward_engine, training_config,
        env_config=env_config, reward_config=reward_config
    )
    
    # Train with checkpointing
    print("\nTraining with checkpointing...")
    history = trainer.train_with_checkpointing(
        num_episodes=100,
        checkpoint_dir='examples/checkpoints/demo',
        save_interval=25,  # Save every 25 episodes
        eval_interval=25,  # Evaluate every 25 episodes
        eval_episodes=5    # Use 5 episodes for evaluation
    )
    
    print(f"\nTraining complete!")
    print(f"Total episodes: {len(history['episode_rewards'])}")
    print(f"Best checkpoint: {trainer.best_checkpoint_path}")
    print(f"Best evaluation score: {trainer.best_eval_score:.2f}")


def example_resume_training():
    """Example: Resume training from a checkpoint"""
    print("\n" + "=" * 70)
    print("Example 2: Resuming Training from Checkpoint")
    print("=" * 70)
    
    # Create configurations (same as before)
    env_config = EnvironmentConfig(
        income=3200,
        fixed_expenses=1400,
        variable_expense_mean=700,
        variable_expense_std=100,
        inflation=0.02,
        safety_threshold=1000,
        max_months=60,
        initial_cash=0,
        risk_tolerance=0.5
    )
    
    training_config = TrainingConfig(
        num_episodes=50,  # Additional episodes
        gamma_low=0.95,
        gamma_high=0.99,
        high_period=6,
        batch_size=32,
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
    
    # Initialize system
    env = BudgetEnv(env_config, reward_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    high_agent = FinancialStrategist(training_config)
    low_agent = BudgetExecutor(training_config)
    
    trainer = HRLTrainer(
        env, high_agent, low_agent, reward_engine, training_config,
        env_config=env_config, reward_config=reward_config
    )
    
    # Load checkpoint
    checkpoint_path = 'examples/checkpoints/demo/checkpoint_episode_100'
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    try:
        episode, history = trainer.load_checkpoint(checkpoint_path)
        print(f"✓ Checkpoint loaded successfully")
        print(f"  Resuming from episode: {episode}")
        print(f"  Previous training episodes: {len(history['episode_rewards'])}")
        
        # Continue training
        print("\nContinuing training for 50 more episodes...")
        trainer.train_with_checkpointing(
            num_episodes=50,
            checkpoint_dir='examples/checkpoints/demo',
            save_interval=25,
            eval_interval=25,
            eval_episodes=5
        )
        
        print(f"\nTraining complete!")
        print(f"Total episodes: {len(trainer.training_history['episode_rewards'])}")
        
    except FileNotFoundError as e:
        print(f"Checkpoint not found: {e}")
        print("Please run example_train_with_checkpointing() first")


def example_load_best_model():
    """Example: Load and evaluate the best model"""
    print("\n" + "=" * 70)
    print("Example 3: Loading and Evaluating Best Model")
    print("=" * 70)
    
    # Create configurations
    env_config = EnvironmentConfig(
        income=3200,
        fixed_expenses=1400,
        variable_expense_mean=700,
        variable_expense_std=100,
        inflation=0.02,
        safety_threshold=1000,
        max_months=60,
        initial_cash=0,
        risk_tolerance=0.5
    )
    
    training_config = TrainingConfig(
        num_episodes=100,
        gamma_low=0.95,
        gamma_high=0.99,
        high_period=6,
        batch_size=32,
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
    
    # Initialize system
    env = BudgetEnv(env_config, reward_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    high_agent = FinancialStrategist(training_config)
    low_agent = BudgetExecutor(training_config)
    
    trainer = HRLTrainer(
        env, high_agent, low_agent, reward_engine, training_config,
        env_config=env_config, reward_config=reward_config
    )
    
    # Load best checkpoint
    best_checkpoint_path = 'examples/checkpoints/demo/checkpoint_best'
    print(f"\nLoading best model from: {best_checkpoint_path}")
    
    try:
        episode, history = trainer.load_checkpoint(best_checkpoint_path)
        print(f"✓ Best model loaded successfully")
        print(f"  Model from episode: {episode}")
        
        # Evaluate the best model
        print("\nEvaluating best model over 20 episodes...")
        eval_results = trainer.evaluate(num_episodes=20)
        
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Mean Cash Balance: ${eval_results['mean_cash_balance']:.2f}")
        print(f"  Mean Total Invested: ${eval_results['mean_total_invested']:.2f}")
        print(f"  Mean Wealth Growth: ${eval_results['mean_wealth_growth']:.2f}")
        print(f"  Mean Cash Stability: {eval_results['mean_cash_stability']:.2%}")
        print(f"  Mean Sharpe Ratio: {eval_results['mean_sharpe_ratio']:.2f}")
        print(f"  Mean Goal Adherence: {eval_results['mean_goal_adherence']:.4f}")
        print(f"  Mean Policy Stability: {eval_results['mean_policy_stability']:.4f}")
        
    except FileNotFoundError as e:
        print(f"Best checkpoint not found: {e}")
        print("Please run example_train_with_checkpointing() first")


if __name__ == "__main__":
    # Run examples
    print("Checkpointing Usage Examples")
    print("=" * 70)
    
    # Example 1: Train with checkpointing
    example_train_with_checkpointing()
    
    # Example 2: Resume training (uncomment to run)
    # example_resume_training()
    
    # Example 3: Load and evaluate best model (uncomment to run)
    # example_load_best_model()
    
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)
    print("\nCheckpoints saved to: examples/checkpoints/demo/")
    print("\nTo resume training or load the best model, uncomment the")
    print("corresponding example function calls in this script.")
