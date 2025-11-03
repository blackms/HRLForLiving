#!/usr/bin/env python3
"""
Evaluation Script for HRL Finance System

This script loads trained models and evaluates their performance on the financial
optimization task. It runs evaluation episodes, computes performance metrics, and
generates visualizations of episode trajectories.

Usage:
    # Evaluate with trained models
    python evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt
    
    # Evaluate with specific configuration
    python evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --config configs/balanced.yaml
    
    # Evaluate with custom episodes and output
    python evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --episodes 50 --output results/
"""

import argparse
import sys
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.utils.config_manager import load_config, load_behavioral_profile, ConfigurationError
from src.utils.analytics import AnalyticsModule


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained HRL Finance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt
  python evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --config configs/balanced.yaml
  python evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --episodes 50 --output results/
        """
    )
    
    # Model paths (required)
    parser.add_argument(
        '--high-agent',
        type=str,
        required=True,
        help='Path to trained high-level agent model (.pt file)'
    )
    parser.add_argument(
        '--low-agent',
        type=str,
        required=True,
        help='Path to trained low-level agent model (.pt file)'
    )
    
    # Configuration source (mutually exclusive, optional)
    config_group = parser.add_mutually_exclusive_group()
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
    
    # Evaluation parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes (default: 20)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results and visualizations (default: results/)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip generating visualizations'
    )
    
    return parser.parse_args()


def load_configuration(args):
    """
    Load configuration from file, profile, or use default
    
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
        elif args.profile:
            print(f"Loading behavioral profile: {args.profile}")
            env_config, training_config, reward_config = load_behavioral_profile(args.profile)
            config_name = args.profile
        else:
            print("No configuration specified, using balanced profile")
            env_config, training_config, reward_config = load_behavioral_profile('balanced')
            config_name = 'balanced'
        
        return env_config, training_config, reward_config, config_name
    
    except ConfigurationError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)


def load_trained_models(high_agent_path: str, low_agent_path: str, training_config):
    """
    Load trained agent models from checkpoint files
    
    Args:
        high_agent_path: Path to high-level agent model
        low_agent_path: Path to low-level agent model
        training_config: Training configuration
        
    Returns:
        Tuple of (high_agent, low_agent)
    """
    print("\n" + "=" * 70)
    print("Loading Trained Models")
    print("=" * 70)
    
    # Initialize agents
    print(f"\n1. Loading high-level agent from: {high_agent_path}")
    high_agent = FinancialStrategist(training_config)
    
    if not os.path.exists(high_agent_path):
        print(f"   ERROR: Model file not found: {high_agent_path}", file=sys.stderr)
        sys.exit(1)
    
    high_agent.load(high_agent_path)
    print("   ✓ High-level agent loaded successfully")
    
    print(f"\n2. Loading low-level agent from: {low_agent_path}")
    low_agent = BudgetExecutor(training_config)
    
    if not os.path.exists(low_agent_path):
        print(f"   ERROR: Model file not found: {low_agent_path}", file=sys.stderr)
        sys.exit(1)
    
    low_agent.load(low_agent_path)
    print("   ✓ Low-level agent loaded successfully")
    
    return high_agent, low_agent


def run_evaluation_episode(
    env: BudgetEnv,
    high_agent: FinancialStrategist,
    low_agent: BudgetExecutor,
    reward_engine: RewardEngine,
    high_period: int
) -> Dict:
    """
    Run a single evaluation episode and collect detailed trajectory data
    
    Args:
        env: BudgetEnv instance
        high_agent: Trained FinancialStrategist
        low_agent: Trained BudgetExecutor
        reward_engine: RewardEngine instance
        high_period: High-level decision period
        
    Returns:
        Dictionary containing episode results and trajectory data
    """
    # Reset environment
    state, _ = env.reset()
    
    # Initialize state history
    state_history = [state]
    
    # Generate initial goal
    aggregated_state = high_agent.aggregate_state(state_history)
    goal = high_agent.select_goal(aggregated_state)
    
    # Episode tracking
    episode_reward = 0
    episode_length = 0
    steps_since_high_update = 0
    
    # Trajectory data
    trajectory = {
        'states': [],
        'actions': [],
        'goals': [],
        'rewards': [],
        'cash_balances': [],
        'invested_amounts': [],
        'total_expenses': []
    }
    
    # Execute episode (deterministic policy)
    done = False
    while not done:
        # Low-level agent generates action (deterministic)
        action = low_agent.act(state, goal, deterministic=True)
        
        # Execute action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track metrics
        episode_reward += reward
        episode_length += 1
        steps_since_high_update += 1
        
        # Store trajectory data
        trajectory['states'].append(state.copy())
        trajectory['actions'].append(action.copy())
        trajectory['goals'].append(goal.copy())
        trajectory['rewards'].append(reward)
        trajectory['cash_balances'].append(info['cash_balance'])
        trajectory['invested_amounts'].append(info['invest_amount'])
        trajectory['total_expenses'].append(info['total_expenses'])
        
        # Update state history
        state_history.append(next_state)
        
        # High-level re-planning every high_period steps
        if steps_since_high_update >= high_period and not done:
            aggregated_state = high_agent.aggregate_state(state_history)
            goal = high_agent.select_goal(aggregated_state)
            steps_since_high_update = 0
        
        # Move to next state
        state = next_state
    
    # Compute episode metrics using AnalyticsModule
    analytics = AnalyticsModule()
    for i in range(len(trajectory['states'])):
        analytics.record_step(
            state=trajectory['states'][i],
            action=trajectory['actions'][i],
            reward=trajectory['rewards'][i],
            goal=trajectory['goals'][i],
            invested_amount=trajectory['invested_amounts'][i]
        )
    
    episode_metrics = analytics.compute_episode_metrics()
    
    # Compile results
    results = {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'final_cash': info['cash_balance'],
        'total_invested': info['total_invested'],
        'cumulative_wealth_growth': episode_metrics['cumulative_wealth_growth'],
        'cash_stability_index': episode_metrics['cash_stability_index'],
        'sharpe_ratio': episode_metrics['sharpe_ratio'],
        'goal_adherence': episode_metrics['goal_adherence'],
        'policy_stability': episode_metrics['policy_stability'],
        'trajectory': trajectory
    }
    
    return results


def evaluate_models(
    env: BudgetEnv,
    high_agent: FinancialStrategist,
    low_agent: BudgetExecutor,
    reward_engine: RewardEngine,
    num_episodes: int,
    high_period: int
) -> Dict:
    """
    Evaluate trained models over multiple episodes
    
    Args:
        env: BudgetEnv instance
        high_agent: Trained FinancialStrategist
        low_agent: Trained BudgetExecutor
        reward_engine: RewardEngine instance
        num_episodes: Number of evaluation episodes
        high_period: High-level decision period
        
    Returns:
        Dictionary containing evaluation results
    """
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)
    print(f"\nEvaluating over {num_episodes} episodes...")
    
    all_results = []
    
    for episode in range(num_episodes):
        results = run_evaluation_episode(env, high_agent, low_agent, reward_engine, high_period)
        all_results.append(results)
        
        # Print progress
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"  Episode {episode + 1}/{num_episodes} - "
                  f"Reward: {results['episode_reward']:.2f}, "
                  f"Cash: ${results['final_cash']:.2f}, "
                  f"Invested: ${results['total_invested']:.2f}")
    
    # Aggregate statistics
    eval_summary = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean([r['episode_reward'] for r in all_results]),
        'std_reward': np.std([r['episode_reward'] for r in all_results]),
        'mean_episode_length': np.mean([r['episode_length'] for r in all_results]),
        'mean_final_cash': np.mean([r['final_cash'] for r in all_results]),
        'std_final_cash': np.std([r['final_cash'] for r in all_results]),
        'mean_total_invested': np.mean([r['total_invested'] for r in all_results]),
        'std_total_invested': np.std([r['total_invested'] for r in all_results]),
        'mean_wealth_growth': np.mean([r['cumulative_wealth_growth'] for r in all_results]),
        'std_wealth_growth': np.std([r['cumulative_wealth_growth'] for r in all_results]),
        'mean_cash_stability': np.mean([r['cash_stability_index'] for r in all_results]),
        'std_cash_stability': np.std([r['cash_stability_index'] for r in all_results]),
        'mean_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in all_results]),
        'std_sharpe_ratio': np.std([r['sharpe_ratio'] for r in all_results]),
        'mean_goal_adherence': np.mean([r['goal_adherence'] for r in all_results]),
        'std_goal_adherence': np.std([r['goal_adherence'] for r in all_results]),
        'mean_policy_stability': np.mean([r['policy_stability'] for r in all_results]),
        'std_policy_stability': np.std([r['policy_stability'] for r in all_results]),
        'all_episodes': all_results
    }
    
    return eval_summary


def print_evaluation_results(eval_summary: Dict):
    """Print evaluation results in a formatted manner"""
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    
    print(f"\nEvaluated over {eval_summary['num_episodes']} episodes")
    
    print("\n" + "-" * 70)
    print("Performance Metrics (mean ± std)")
    print("-" * 70)
    
    print(f"\nReward:")
    print(f"  {eval_summary['mean_reward']:.2f} ± {eval_summary['std_reward']:.2f}")
    
    print(f"\nEpisode Length:")
    print(f"  {eval_summary['mean_episode_length']:.1f} months")
    
    print(f"\nFinal Cash Balance:")
    print(f"  ${eval_summary['mean_final_cash']:.2f} ± ${eval_summary['std_final_cash']:.2f}")
    
    print(f"\nTotal Invested:")
    print(f"  ${eval_summary['mean_total_invested']:.2f} ± ${eval_summary['std_total_invested']:.2f}")
    
    print(f"\nCumulative Wealth Growth:")
    print(f"  ${eval_summary['mean_wealth_growth']:.2f} ± ${eval_summary['std_wealth_growth']:.2f}")
    
    print(f"\nCash Stability Index:")
    print(f"  {eval_summary['mean_cash_stability']:.2%} ± {eval_summary['std_cash_stability']:.2%}")
    
    print(f"\nSharpe Ratio:")
    print(f"  {eval_summary['mean_sharpe_ratio']:.2f} ± {eval_summary['std_sharpe_ratio']:.2f}")
    
    print(f"\nGoal Adherence:")
    print(f"  {eval_summary['mean_goal_adherence']:.4f} ± {eval_summary['std_goal_adherence']:.4f}")
    
    print(f"\nPolicy Stability:")
    print(f"  {eval_summary['mean_policy_stability']:.4f} ± {eval_summary['std_policy_stability']:.4f}")


def save_results(eval_summary: Dict, output_dir: str, config_name: str):
    """
    Save evaluation results to JSON file
    
    Args:
        eval_summary: Evaluation summary dictionary
        output_dir: Output directory
        config_name: Configuration name for file naming
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for JSON serialization (remove trajectory data for compactness)
    results_json = {
        'num_episodes': eval_summary['num_episodes'],
        'mean_reward': float(eval_summary['mean_reward']),
        'std_reward': float(eval_summary['std_reward']),
        'mean_episode_length': float(eval_summary['mean_episode_length']),
        'mean_final_cash': float(eval_summary['mean_final_cash']),
        'std_final_cash': float(eval_summary['std_final_cash']),
        'mean_total_invested': float(eval_summary['mean_total_invested']),
        'std_total_invested': float(eval_summary['std_total_invested']),
        'mean_wealth_growth': float(eval_summary['mean_wealth_growth']),
        'std_wealth_growth': float(eval_summary['std_wealth_growth']),
        'mean_cash_stability': float(eval_summary['mean_cash_stability']),
        'std_cash_stability': float(eval_summary['std_cash_stability']),
        'mean_sharpe_ratio': float(eval_summary['mean_sharpe_ratio']),
        'std_sharpe_ratio': float(eval_summary['std_sharpe_ratio']),
        'mean_goal_adherence': float(eval_summary['mean_goal_adherence']),
        'std_goal_adherence': float(eval_summary['std_goal_adherence']),
        'mean_policy_stability': float(eval_summary['mean_policy_stability']),
        'std_policy_stability': float(eval_summary['std_policy_stability']),
        'episode_rewards': [float(ep['episode_reward']) for ep in eval_summary['all_episodes']],
        'episode_lengths': [int(ep['episode_length']) for ep in eval_summary['all_episodes']],
        'final_cash_balances': [float(ep['final_cash']) for ep in eval_summary['all_episodes']],
        'total_invested': [float(ep['total_invested']) for ep in eval_summary['all_episodes']]
    }
    
    results_path = output_path / f"{config_name}_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to: {results_path}")


def generate_visualizations(eval_summary: Dict, output_dir: str, config_name: str):
    """
    Generate visualizations of episode trajectories
    
    Args:
        eval_summary: Evaluation summary dictionary
        output_dir: Output directory
        config_name: Configuration name for file naming
    """
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Select a representative episode (median reward)
    episode_rewards = [ep['episode_reward'] for ep in eval_summary['all_episodes']]
    median_idx = np.argsort(episode_rewards)[len(episode_rewards) // 2]
    representative_episode = eval_summary['all_episodes'][median_idx]
    trajectory = representative_episode['trajectory']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Episode Trajectory Analysis - {config_name.title()}', fontsize=16, fontweight='bold')
    
    months = np.arange(1, len(trajectory['cash_balances']) + 1)
    
    # 1. Cash Balance Over Time
    ax = axes[0, 0]
    ax.plot(months, trajectory['cash_balances'], 'b-', linewidth=2, label='Cash Balance')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Line')
    ax.set_xlabel('Month')
    ax.set_ylabel('Cash Balance ($)')
    ax.set_title('Cash Balance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Allocation Actions Over Time
    ax = axes[0, 1]
    actions_array = np.array(trajectory['actions'])
    ax.plot(months, actions_array[:, 0], 'g-', linewidth=2, label='Invest')
    ax.plot(months, actions_array[:, 1], 'b-', linewidth=2, label='Save')
    ax.plot(months, actions_array[:, 2], 'r-', linewidth=2, label='Consume')
    ax.set_xlabel('Month')
    ax.set_ylabel('Allocation Ratio')
    ax.set_title('Allocation Actions Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 3. Cumulative Investment
    ax = axes[1, 0]
    cumulative_invested = np.cumsum(trajectory['invested_amounts'])
    ax.plot(months, cumulative_invested, 'g-', linewidth=2)
    ax.fill_between(months, 0, cumulative_invested, alpha=0.3, color='g')
    ax.set_xlabel('Month')
    ax.set_ylabel('Cumulative Investment ($)')
    ax.set_title('Cumulative Investment Over Time')
    ax.grid(True, alpha=0.3)
    
    # 4. Rewards Over Time
    ax = axes[1, 1]
    ax.plot(months, trajectory['rewards'], 'purple', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards Over Time')
    ax.grid(True, alpha=0.3)
    
    # 5. Goal vs Actual Investment Ratio
    ax = axes[2, 0]
    goals_array = np.array(trajectory['goals'])
    ax.plot(months, goals_array[:, 0], 'b--', linewidth=2, label='Target (Goal)', alpha=0.7)
    ax.plot(months, actions_array[:, 0], 'g-', linewidth=2, label='Actual (Action)')
    ax.set_xlabel('Month')
    ax.set_ylabel('Investment Ratio')
    ax.set_title('Goal Adherence: Target vs Actual Investment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # 6. Expenses Over Time
    ax = axes[2, 1]
    ax.plot(months, trajectory['total_expenses'], 'r-', linewidth=2, label='Total Expenses')
    ax.set_xlabel('Month')
    ax.set_ylabel('Expenses ($)')
    ax.set_title('Total Expenses Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    viz_path = output_path / f"{config_name}_trajectory_visualization.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Trajectory visualization saved to: {viz_path}")
    
    # Create summary statistics visualization
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle(f'Evaluation Summary Statistics - {config_name.title()}', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards Distribution
    ax = axes2[0, 0]
    ax.hist(episode_rewards, bins=15, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(eval_summary['mean_reward'], color='r', linestyle='--', linewidth=2, label=f"Mean: {eval_summary['mean_reward']:.2f}")
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final Cash Balance Distribution
    ax = axes2[0, 1]
    final_cash = [ep['final_cash'] for ep in eval_summary['all_episodes']]
    ax.hist(final_cash, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(eval_summary['mean_final_cash'], color='r', linestyle='--', linewidth=2, label=f"Mean: ${eval_summary['mean_final_cash']:.2f}")
    ax.set_xlabel('Final Cash Balance ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Final Cash Balance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Total Invested Distribution
    ax = axes2[1, 0]
    total_invested = [ep['total_invested'] for ep in eval_summary['all_episodes']]
    ax.hist(total_invested, bins=15, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(eval_summary['mean_total_invested'], color='r', linestyle='--', linewidth=2, label=f"Mean: ${eval_summary['mean_total_invested']:.2f}")
    ax.set_xlabel('Total Invested ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Total Investment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Performance Metrics Comparison
    ax = axes2[1, 1]
    metrics = ['Stability\nIndex', 'Sharpe\nRatio', 'Goal\nAdherence', 'Policy\nStability']
    values = [
        eval_summary['mean_cash_stability'],
        eval_summary['mean_sharpe_ratio'] / 10,  # Scale for visibility
        1 - eval_summary['mean_goal_adherence'],  # Invert for better = higher
        1 - eval_summary['mean_policy_stability']  # Invert for better = higher
    ]
    colors = ['green', 'blue', 'orange', 'purple']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Performance Metrics Overview')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    summary_viz_path = output_path / f"{config_name}_summary_statistics.png"
    plt.savefig(summary_viz_path, dpi=300, bbox_inches='tight')
    print(f"✓ Summary statistics visualization saved to: {summary_viz_path}")
    
    plt.close('all')


def main():
    """Main evaluation script"""
    # Parse arguments
    args = parse_arguments()
    
    print("=" * 70)
    print("HRL Finance System - Evaluation Script")
    print("=" * 70)
    
    # Load configuration
    env_config, training_config, reward_config, config_name = load_configuration(args)
    
    # Set random seed if provided
    if args.seed is not None:
        print(f"\nSetting random seed: {args.seed}")
        np.random.seed(args.seed)
    
    # Load trained models
    high_agent, low_agent = load_trained_models(
        args.high_agent, args.low_agent, training_config
    )
    
    # Initialize environment and reward engine
    print("\n" + "=" * 70)
    print("Initializing Environment")
    print("=" * 70)
    print("\nCreating BudgetEnv and RewardEngine...")
    env = BudgetEnv(env_config, reward_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    print("✓ Environment initialized")
    
    # Run evaluation
    eval_summary = evaluate_models(
        env, high_agent, low_agent, reward_engine,
        args.episodes, training_config.high_period
    )
    
    # Print results
    print_evaluation_results(eval_summary)
    
    # Save results
    save_results(eval_summary, args.output, config_name)
    
    # Generate visualizations
    if not args.no_viz:
        generate_visualizations(eval_summary, args.output, config_name)
    else:
        print("\nSkipping visualizations (--no-viz flag set)")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output}/")
    print(f"  - {config_name}_evaluation_results.json")
    if not args.no_viz:
        print(f"  - {config_name}_trajectory_visualization.png")
        print(f"  - {config_name}_summary_statistics.png")


if __name__ == "__main__":
    main()
