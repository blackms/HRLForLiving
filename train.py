#!/usr/bin/env python3
"""
Main Training Script for HRL Finance System

This script trains the hierarchical reinforcement learning system for personal
finance optimization. It supports both YAML configuration files and predefined
behavioral profiles (conservative, balanced, aggressive).

Usage:
    # Train with YAML config
    python train.py --config configs/balanced.yaml
    
    # Train with behavioral profile
    python train.py --profile balanced
    
    # Train with custom episodes
    python train.py --profile aggressive --episodes 10000
    
    # Specify output directory for models
    python train.py --config configs/balanced.yaml --output models/
"""

import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.training.hrl_trainer import HRLTrainer
from src.utils.config_manager import load_config, load_behavioral_profile, ConfigurationError
from src.utils.logger import ExperimentLogger


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train HRL Finance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --config configs/balanced.yaml
  python train.py --profile balanced
  python train.py --profile aggressive --episodes 10000
  python train.py --config configs/balanced.yaml --output models/
        """
    )
    
    # Configuration source (mutually exclusive)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )
    config_group.add_argument(
        '--profile',
        type=str,
        choices=['conservative', 'balanced', 'aggressive'],
        help='Behavioral profile to use (conservative, balanced, or aggressive)'
    )
    
    # Training parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Number of training episodes (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for trained models (default: models/)'
    )
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes after training (default: 10)'
    )
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1000,
        help='Save model checkpoint every N episodes (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='runs',
        help='Directory for TensorBoard logs (default: runs/)'
    )
    parser.add_argument(
        '--no-logging',
        action='store_true',
        help='Disable TensorBoard logging'
    )
    
    return parser.parse_args()


def load_configuration(args):
    """
    Load configuration from file or profile
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Tuple of (env_config, training_config, reward_config, config_name)
    """
    try:
        if args.config:
            print(f"Loading configuration from: {args.config}")
            env_config, training_config, reward_config = load_config(args.config)
            config_name = Path(args.config).stem
        else:
            print(f"Loading behavioral profile: {args.profile}")
            env_config, training_config, reward_config = load_behavioral_profile(args.profile)
            config_name = args.profile
        
        # Override episodes if specified
        if args.episodes is not None:
            print(f"Overriding num_episodes: {training_config.num_episodes} -> {args.episodes}")
            training_config.num_episodes = args.episodes
        
        return env_config, training_config, reward_config, config_name
    
    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)


def print_configuration(env_config, training_config, reward_config):
    """Print configuration summary"""
    print("\n" + "=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    
    print("\nEnvironment:")
    print(f"  Income: ${env_config.income:.2f}")
    print(f"  Fixed Expenses: ${env_config.fixed_expenses:.2f}")
    print(f"  Variable Expenses: ${env_config.variable_expense_mean:.2f} ± ${env_config.variable_expense_std:.2f}")
    print(f"  Inflation: {env_config.inflation:.2%}")
    print(f"  Safety Threshold: ${env_config.safety_threshold:.2f}")
    print(f"  Max Months: {env_config.max_months}")
    print(f"  Initial Cash: ${env_config.initial_cash:.2f}")
    print(f"  Risk Tolerance: {env_config.risk_tolerance:.2f}")
    
    print("\nTraining:")
    print(f"  Episodes: {training_config.num_episodes}")
    print(f"  Gamma Low: {training_config.gamma_low}")
    print(f"  Gamma High: {training_config.gamma_high}")
    print(f"  High Period: {training_config.high_period} months")
    print(f"  Batch Size: {training_config.batch_size}")
    print(f"  Learning Rate Low: {training_config.learning_rate_low}")
    print(f"  Learning Rate High: {training_config.learning_rate_high}")
    
    print("\nReward Coefficients:")
    print(f"  Alpha (investment): {reward_config.alpha}")
    print(f"  Beta (stability): {reward_config.beta}")
    print(f"  Gamma (overspend): {reward_config.gamma}")
    print(f"  Delta (debt): {reward_config.delta}")
    print(f"  Lambda (wealth growth): {reward_config.lambda_}")
    print(f"  Mu (stability bonus): {reward_config.mu}")


def initialize_system(env_config, training_config, reward_config, logger=None, seed=None):
    """
    Initialize all system components
    
    Args:
        env_config: Environment configuration
        training_config: Training configuration
        reward_config: Reward configuration
        logger: Optional ExperimentLogger for TensorBoard logging
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (env, high_agent, low_agent, reward_engine, trainer)
    """
    print("\n" + "=" * 70)
    print("Initializing HRL System")
    print("=" * 70)
    
    # Set random seed if provided
    if seed is not None:
        print(f"\nSetting random seed: {seed}")
        np.random.seed(seed)
    
    # Initialize environment
    print("\n1. Creating BudgetEnv...")
    env = BudgetEnv(env_config, reward_config)
    print("   ✓ BudgetEnv initialized")
    
    # Initialize reward engine
    print("\n2. Creating RewardEngine...")
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    print("   ✓ RewardEngine initialized")
    
    # Initialize agents
    print("\n3. Creating agents...")
    high_agent = FinancialStrategist(training_config)
    print("   ✓ FinancialStrategist (high-level agent) initialized")
    
    low_agent = BudgetExecutor(training_config)
    print("   ✓ BudgetExecutor (low-level agent) initialized")
    
    # Initialize trainer
    print("\n4. Creating HRLTrainer...")
    trainer = HRLTrainer(env, high_agent, low_agent, reward_engine, training_config, logger=logger)
    print("   ✓ HRLTrainer initialized with AnalyticsModule")
    
    return env, high_agent, low_agent, reward_engine, trainer


def train_system(trainer, training_config, output_dir, config_name, save_interval):
    """
    Execute training loop with periodic checkpointing
    
    Args:
        trainer: HRLTrainer instance
        training_config: Training configuration
        output_dir: Directory to save models
        config_name: Name of configuration for file naming
        save_interval: Save checkpoint every N episodes
        
    Returns:
        Training history dictionary
    """
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    print(f"\nTraining for {training_config.num_episodes} episodes...")
    print(f"Checkpoints will be saved every {save_interval} episodes to: {output_dir}/")
    print("\nProgress:")
    print("-" * 70)
    
    # Run training
    history = trainer.train(num_episodes=training_config.num_episodes)
    
    print("\n" + "-" * 70)
    print("Training completed!")
    
    return history


def save_models(trainer, output_dir, config_name):
    """
    Save trained models and training history
    
    Args:
        trainer: HRLTrainer instance with trained models
        output_dir: Directory to save models
        config_name: Name of configuration for file naming
    """
    print("\n" + "=" * 70)
    print("Saving Models")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save high-level agent
    high_agent_path = output_path / f"{config_name}_high_agent.pt"
    trainer.high_agent.save(str(high_agent_path))
    print(f"\n✓ High-level agent saved to: {high_agent_path}")
    
    # Save low-level agent
    low_agent_path = output_path / f"{config_name}_low_agent.pt"
    trainer.low_agent.save(str(low_agent_path))
    print(f"✓ Low-level agent saved to: {low_agent_path}")
    
    # Save training history
    history_path = output_path / f"{config_name}_history.json"
    
    # Convert numpy arrays to lists for JSON serialization
    history_json = {}
    for key, value in trainer.training_history.items():
        if isinstance(value, list):
            history_json[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        else:
            history_json[key] = value
    
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    print(f"✓ Training history saved to: {history_path}")


def print_training_summary(history):
    """Print summary of training results"""
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    
    # Calculate statistics over last 100 episodes
    last_n = min(100, len(history['episode_rewards']))
    
    avg_reward = np.mean(history['episode_rewards'][-last_n:])
    avg_length = np.mean(history['episode_lengths'][-last_n:])
    avg_cash = np.mean(history['cash_balances'][-last_n:])
    avg_invested = np.mean(history['total_invested'][-last_n:])
    avg_wealth_growth = np.mean(history['cumulative_wealth_growth'][-last_n:])
    avg_stability = np.mean(history['cash_stability_index'][-last_n:])
    avg_sharpe = np.mean(history['sharpe_ratio'][-last_n:])
    avg_goal_adherence = np.mean(history['goal_adherence'][-last_n:])
    avg_policy_stability = np.mean(history['policy_stability'][-last_n:])
    
    print(f"\nLast {last_n} episodes:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Average Episode Length: {avg_length:.1f} months")
    print(f"  Average Final Cash: ${avg_cash:.2f}")
    print(f"  Average Total Invested: ${avg_invested:.2f}")
    print(f"  Average Wealth Growth: ${avg_wealth_growth:.2f}")
    print(f"  Average Stability Index: {avg_stability:.2%}")
    print(f"  Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"  Average Goal Adherence: {avg_goal_adherence:.4f}")
    print(f"  Average Policy Stability: {avg_policy_stability:.4f}")


def evaluate_system(trainer, num_episodes):
    """
    Evaluate trained system
    
    Args:
        trainer: HRLTrainer instance with trained models
        num_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation results dictionary
    """
    print("\n" + "=" * 70)
    print("Evaluating Trained System")
    print("=" * 70)
    print(f"\nRunning {num_episodes} evaluation episodes...")
    
    eval_results = trainer.evaluate(num_episodes=num_episodes)
    
    return eval_results


def print_evaluation_results(eval_results):
    """Print evaluation results"""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    print(f"\nPerformance Metrics (mean ± std):")
    print(f"  Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Final Cash Balance: ${eval_results['mean_cash_balance']:.2f}")
    print(f"  Total Invested: ${eval_results['mean_total_invested']:.2f}")
    print(f"  Wealth Growth: ${eval_results['mean_wealth_growth']:.2f}")
    print(f"  Cash Stability Index: {eval_results['mean_cash_stability']:.2%}")
    print(f"  Sharpe Ratio: {eval_results['mean_sharpe_ratio']:.2f}")
    print(f"  Goal Adherence: {eval_results['mean_goal_adherence']:.4f}")
    print(f"  Policy Stability: {eval_results['mean_policy_stability']:.4f}")


def main():
    """Main training script"""
    # Parse arguments
    args = parse_arguments()
    
    print("=" * 70)
    print("HRL Finance System - Training Script")
    print("=" * 70)
    
    # Load configuration
    env_config, training_config, reward_config, config_name = load_configuration(args)
    
    # Print configuration
    print_configuration(env_config, training_config, reward_config)
    
    # Initialize logger
    logger = None
    if not args.no_logging:
        experiment_name = f"{config_name}_{training_config.num_episodes}ep"
        if args.seed is not None:
            experiment_name += f"_seed{args.seed}"
        logger = ExperimentLogger(log_dir=args.log_dir, experiment_name=experiment_name)
        
        # Log hyperparameters
        hparams = {
            # Environment
            'env/income': env_config.income,
            'env/fixed_expenses': env_config.fixed_expenses,
            'env/variable_expense_mean': env_config.variable_expense_mean,
            'env/variable_expense_std': env_config.variable_expense_std,
            'env/inflation': env_config.inflation,
            'env/safety_threshold': env_config.safety_threshold,
            'env/max_months': env_config.max_months,
            'env/initial_cash': env_config.initial_cash,
            'env/risk_tolerance': env_config.risk_tolerance,
            # Training
            'train/num_episodes': training_config.num_episodes,
            'train/gamma_low': training_config.gamma_low,
            'train/gamma_high': training_config.gamma_high,
            'train/high_period': training_config.high_period,
            'train/batch_size': training_config.batch_size,
            'train/learning_rate_low': training_config.learning_rate_low,
            'train/learning_rate_high': training_config.learning_rate_high,
            # Reward
            'reward/alpha': reward_config.alpha,
            'reward/beta': reward_config.beta,
            'reward/gamma': reward_config.gamma,
            'reward/delta': reward_config.delta,
            'reward/lambda': reward_config.lambda_,
            'reward/mu': reward_config.mu,
        }
        if args.seed is not None:
            hparams['seed'] = args.seed
        
        logger.log_hyperparameters(hparams)
    
    # Initialize system
    env, high_agent, low_agent, reward_engine, trainer = initialize_system(
        env_config, training_config, reward_config, logger=logger, seed=args.seed
    )
    
    # Train system
    history = train_system(
        trainer, training_config, args.output, config_name, args.save_interval
    )
    
    # Print training summary
    print_training_summary(history)
    
    # Save models
    save_models(trainer, args.output, config_name)
    
    # Evaluate system
    if args.eval_episodes > 0:
        eval_results = evaluate_system(trainer, args.eval_episodes)
        print_evaluation_results(eval_results)
    
    # Close logger
    if logger is not None:
        logger.close()
        print(f"\nTensorBoard logs saved to: {logger.log_dir}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModels saved to: {args.output}/")
    print(f"  - {config_name}_high_agent.pt")
    print(f"  - {config_name}_low_agent.pt")
    print(f"  - {config_name}_history.json")
    if logger is not None:
        print(f"\nTo view training logs, run:")
        print(f"  tensorboard --logdir={args.log_dir}")
    print("\nYou can now use these models for evaluation or further training.")


if __name__ == "__main__":
    main()
