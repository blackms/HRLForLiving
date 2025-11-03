"""
Example: Using TensorBoard Logging with HRL Finance System

This example demonstrates how to use the ExperimentLogger to track
training progress with TensorBoard.

Usage:
    python examples/logging_usage.py
    
    # Then view logs with:
    tensorboard --logdir=runs
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.training.hrl_trainer import HRLTrainer
from src.utils.config_manager import load_behavioral_profile
from src.utils.logger import ExperimentLogger


def main():
    """Demonstrate TensorBoard logging with HRL training"""
    
    print("=" * 70)
    print("TensorBoard Logging Example")
    print("=" * 70)
    
    # Load configuration
    print("\nLoading balanced behavioral profile...")
    env_config, training_config, reward_config = load_behavioral_profile('balanced')
    
    # Override for quick demo
    training_config.num_episodes = 100
    
    # Initialize logger
    print("\nInitializing TensorBoard logger...")
    logger = ExperimentLogger(
        log_dir='runs',
        experiment_name='logging_demo',
        enabled=True
    )
    
    # Log hyperparameters
    hparams = {
        'env/income': env_config.income,
        'env/risk_tolerance': env_config.risk_tolerance,
        'train/num_episodes': training_config.num_episodes,
        'train/learning_rate_low': training_config.learning_rate_low,
        'train/learning_rate_high': training_config.learning_rate_high,
        'reward/alpha': reward_config.alpha,
        'reward/beta': reward_config.beta,
    }
    logger.log_hyperparameters(hparams)
    print("✓ Hyperparameters logged")
    
    # Initialize system
    print("\nInitializing HRL system...")
    env = BudgetEnv(env_config, reward_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    high_agent = FinancialStrategist(training_config)
    low_agent = BudgetExecutor(training_config)
    trainer = HRLTrainer(env, high_agent, low_agent, reward_engine, training_config, logger=logger)
    print("✓ System initialized")
    
    # Train with logging
    print(f"\nTraining for {training_config.num_episodes} episodes with TensorBoard logging...")
    print("(This will log training curves, episode metrics, action distributions, etc.)")
    print()
    
    history = trainer.train(num_episodes=training_config.num_episodes)
    
    # Close logger
    logger.close()
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nTensorBoard logs saved to: {logger.log_dir}")
    print("\nTo view the logs, run:")
    print("  tensorboard --logdir=runs")
    print("\nThen open your browser to: http://localhost:6006")
    print("\nYou can view:")
    print("  - Training curves (rewards, losses)")
    print("  - Episode metrics (wealth, stability, Sharpe ratio)")
    print("  - Action distributions (invest, save, consume)")
    print("  - Goal distributions (target_invest_ratio, safety_buffer, aggressiveness)")
    print("  - Analytics metrics (goal adherence, policy stability)")


if __name__ == "__main__":
    main()
