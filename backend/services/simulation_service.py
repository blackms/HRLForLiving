"""Simulation service for running evaluations with trained models"""
import os
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig
from src.utils.analytics import AnalyticsModule
from backend.utils.file_manager import (
    read_yaml_config,
    save_json_results,
    list_json_results,
    read_json_results,
    ensure_directories
)


class SimulationService:
    """Service for running simulations with trained models"""
    
    def __init__(self):
        """Initialize the simulation service"""
        ensure_directories()
    
    async def run_simulation(
        self,
        model_name: str,
        scenario_name: str,
        num_episodes: int = 10,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run simulation with a trained model on a scenario
        
        Args:
            model_name: Name of the trained model to use
            scenario_name: Name of the scenario to simulate
            num_episodes: Number of simulation episodes
            seed: Random seed for reproducibility
            
        Returns:
            dict: Simulation results with statistics and episode data
            
        Raises:
            FileNotFoundError: If model or scenario not found
            ValueError: If model files are invalid
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Load scenario configuration
        try:
            scenario_config = read_yaml_config(scenario_name, 'scenarios')
        except FileNotFoundError:
            raise FileNotFoundError(f"Scenario '{scenario_name}' not found")
        
        # Parse configuration
        env_config = EnvironmentConfig(**scenario_config['environment'])
        training_config = TrainingConfig(**scenario_config.get('training', {}))
        reward_config = RewardConfig(**scenario_config.get('reward', {}))
        
        # Load trained models
        high_agent, low_agent = self._load_models(model_name, training_config)
        
        # Create environment and reward engine
        env = BudgetEnv(env_config, reward_config)
        reward_engine = RewardEngine(reward_config)
        
        # Run evaluation episodes
        print(f"Running simulation: {model_name} on {scenario_name} ({num_episodes} episodes)")
        all_episodes = []
        
        for episode_idx in range(num_episodes):
            episode_result = self._run_episode(
                env=env,
                high_agent=high_agent,
                low_agent=low_agent,
                reward_engine=reward_engine,
                high_period=training_config.high_period,
                episode_id=episode_idx
            )
            all_episodes.append(episode_result)
            
            if (episode_idx + 1) % 5 == 0 or episode_idx == 0:
                print(f"  Episode {episode_idx + 1}/{num_episodes} - "
                      f"Duration: {episode_result['duration']} months, "
                      f"Wealth: ${episode_result['total_wealth']:.2f}")
        
        # Calculate aggregate statistics
        statistics = self._calculate_statistics(all_episodes)
        
        # Generate unique simulation ID
        simulation_id = f"{model_name}_{scenario_name}_{int(time.time())}"
        
        # Compile results
        results = {
            'simulation_id': simulation_id,
            'scenario_name': scenario_name,
            'model_name': model_name,
            'num_episodes': num_episodes,
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
            **statistics,
            'episodes': all_episodes
        }
        
        # Save results to JSON
        save_json_results(simulation_id, results, 'simulations')
        print(f"Simulation complete! Results saved: {simulation_id}")
        
        return results
    
    def _load_models(
        self,
        model_name: str,
        training_config: TrainingConfig
    ) -> tuple:
        """
        Load trained agent models
        
        Args:
            model_name: Name of the model
            training_config: Training configuration
            
        Returns:
            tuple: (high_agent, low_agent)
            
        Raises:
            FileNotFoundError: If model files not found
        """
        high_agent_path = os.path.join('models', f"{model_name}_high_agent.pt")
        low_agent_path = os.path.join('models', f"{model_name}_low_agent.pt")
        
        if not os.path.exists(high_agent_path):
            raise FileNotFoundError(f"High-level agent model not found: {high_agent_path}")
        
        if not os.path.exists(low_agent_path):
            raise FileNotFoundError(f"Low-level agent model not found: {low_agent_path}")
        
        # Initialize agents with dummy dimensions (will be set by load)
        high_agent = FinancialStrategist(
            state_dim=10,  # Will be overridden by loaded model
            goal_dim=3,
            gamma=training_config.gamma_high,
            lr=training_config.learning_rate_high
        )
        
        low_agent = BudgetExecutor(
            state_dim=10,  # Will be overridden by loaded model
            goal_dim=3,
            action_dim=3,
            gamma=training_config.gamma_low,
            lr=training_config.learning_rate_low
        )
        
        # Load model weights
        high_agent.load(high_agent_path)
        low_agent.load(low_agent_path)
        
        return high_agent, low_agent
    
    def _run_episode(
        self,
        env: BudgetEnv,
        high_agent: FinancialStrategist,
        low_agent: BudgetExecutor,
        reward_engine: RewardEngine,
        high_period: int,
        episode_id: int
    ) -> Dict[str, Any]:
        """
        Run a single evaluation episode
        
        Args:
            env: BudgetEnv instance
            high_agent: Trained FinancialStrategist
            low_agent: Trained BudgetExecutor
            reward_engine: RewardEngine instance
            high_period: High-level decision period
            episode_id: Episode identifier
            
        Returns:
            dict: Episode results with trajectory data
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
        months = []
        cash_history = []
        invested_history = []
        portfolio_history = []
        actions = []
        
        # Analytics
        analytics = AnalyticsModule()
        
        # Execute episode (deterministic policy)
        done = False
        while not done:
            # Low-level agent generates action (deterministic)
            action = low_agent.act(state, goal, deterministic=True)
            
            # Execute action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Calculate invested amount
            invested_amount = action[0] * state[0]
            
            # Record step in analytics
            analytics.record_step(
                state=state,
                action=action,
                reward=reward,
                goal=goal,
                invested_amount=invested_amount
            )
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            steps_since_high_update += 1
            
            # Store trajectory data
            months.append(episode_length)
            cash_history.append(float(info['cash_balance']))
            invested_history.append(float(info.get('total_invested', 0)))
            portfolio_history.append(float(info.get('portfolio_value', 0)))
            actions.append([float(action[0]), float(action[1]), float(action[2])])
            
            # Update state history
            state_history.append(next_state)
            
            # High-level re-planning every high_period steps
            if steps_since_high_update >= high_period and not done:
                aggregated_state = high_agent.aggregate_state(state_history)
                goal = high_agent.select_goal(aggregated_state)
                steps_since_high_update = 0
            
            # Move to next state
            state = next_state
        
        # Compute episode metrics
        episode_metrics = analytics.compute_episode_metrics()
        
        # Calculate investment gains
        total_invested = invested_history[-1] if invested_history else 0
        portfolio_value = portfolio_history[-1] if portfolio_history else 0
        investment_gains = portfolio_value - total_invested
        
        # Compile episode results
        episode_result = {
            'episode_id': episode_id,
            'duration': episode_length,
            'final_cash': float(info['cash_balance']),
            'final_invested': total_invested,
            'final_portfolio_value': portfolio_value,
            'total_wealth': float(info['cash_balance']) + portfolio_value,
            'investment_gains': investment_gains,
            'months': months,
            'cash_history': cash_history,
            'invested_history': invested_history,
            'portfolio_history': portfolio_history,
            'actions': actions
        }
        
        return episode_result
    
    def _calculate_statistics(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate statistics from multiple episodes
        
        Args:
            episodes: List of episode results
            
        Returns:
            dict: Aggregate statistics
        """
        # Extract metrics
        durations = [ep['duration'] for ep in episodes]
        final_cash = [ep['final_cash'] for ep in episodes]
        final_invested = [ep['final_invested'] for ep in episodes]
        final_portfolio = [ep['final_portfolio_value'] for ep in episodes]
        total_wealth = [ep['total_wealth'] for ep in episodes]
        investment_gains = [ep['investment_gains'] for ep in episodes]
        
        # Calculate action distribution (average across all episodes and timesteps)
        all_actions = []
        for ep in episodes:
            all_actions.extend(ep['actions'])
        
        actions_array = np.array(all_actions)
        avg_invest_pct = float(np.mean(actions_array[:, 0]))
        avg_save_pct = float(np.mean(actions_array[:, 1]))
        avg_consume_pct = float(np.mean(actions_array[:, 2]))
        
        # Compile statistics
        statistics = {
            'duration_mean': float(np.mean(durations)),
            'duration_std': float(np.std(durations)),
            'final_cash_mean': float(np.mean(final_cash)),
            'final_cash_std': float(np.std(final_cash)),
            'final_invested_mean': float(np.mean(final_invested)),
            'final_invested_std': float(np.std(final_invested)),
            'final_portfolio_mean': float(np.mean(final_portfolio)),
            'final_portfolio_std': float(np.std(final_portfolio)),
            'total_wealth_mean': float(np.mean(total_wealth)),
            'total_wealth_std': float(np.std(total_wealth)),
            'investment_gains_mean': float(np.mean(investment_gains)),
            'investment_gains_std': float(np.std(investment_gains)),
            'avg_invest_pct': avg_invest_pct,
            'avg_save_pct': avg_save_pct,
            'avg_consume_pct': avg_consume_pct
        }
        
        return statistics
    
    def get_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """
        Get results for a specific simulation
        
        Args:
            simulation_id: Unique simulation identifier
            
        Returns:
            dict: Simulation results
            
        Raises:
            FileNotFoundError: If simulation results not found
        """
        try:
            results = read_json_results(simulation_id, 'simulations')
            return results
        except FileNotFoundError:
            raise FileNotFoundError(f"Simulation results not found: {simulation_id}")
    
    def list_simulations(self) -> List[Dict[str, Any]]:
        """
        List all available simulation results
        
        Returns:
            list: List of simulation summaries
        """
        simulation_files = list_json_results('simulations')
        
        simulations = []
        for file_info in simulation_files:
            # Read the simulation file to get metadata
            try:
                results = read_json_results(file_info['name'].replace('.json', ''), 'simulations')
                summary = {
                    'simulation_id': results.get('simulation_id', file_info['name'].replace('.json', '')),
                    'scenario_name': results.get('scenario_name', 'unknown'),
                    'model_name': results.get('model_name', 'unknown'),
                    'num_episodes': results.get('num_episodes', 0),
                    'timestamp': results.get('timestamp', file_info['modified']),
                    'total_wealth_mean': results.get('total_wealth_mean', 0),
                    'duration_mean': results.get('duration_mean', 0)
                }
                simulations.append(summary)
            except Exception as e:
                print(f"Warning: Could not read simulation file {file_info['name']}: {e}")
                continue
        
        # Sort by timestamp (most recent first)
        simulations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return simulations


# Global simulation service instance
simulation_service = SimulationService()
