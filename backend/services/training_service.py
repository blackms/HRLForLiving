"""Training service for HRL model training orchestration"""
import os
import asyncio
import time
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import yaml

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.training.hrl_trainer import HRLTrainer
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig
from backend.utils.file_manager import (
    read_yaml_config,
    save_model,
    ensure_directories
)


class TrainingService:
    """Service for managing HRL model training"""
    
    def __init__(self):
        """Initialize the training service"""
        self._training_task: Optional[asyncio.Task] = None
        self._trainer: Optional[HRLTrainer] = None
        self._stop_requested: bool = False
        self._training_status: Dict[str, Any] = {
            'is_training': False,
            'scenario_name': None,
            'current_episode': 0,
            'total_episodes': 0,
            'start_time': None,
            'latest_progress': None
        }
        self._progress_callback: Optional[Callable] = None
        
        # Ensure directories exist
        ensure_directories()
    
    def set_progress_callback(self, callback: Callable):
        """
        Set callback function for progress updates
        
        Args:
            callback: Async function to call with progress updates
        """
        self._progress_callback = callback
    
    async def start_training(
        self,
        scenario_name: str,
        num_episodes: int = 1000,
        save_interval: int = 100,
        eval_episodes: int = 10,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Start training a model on a scenario
        
        Args:
            scenario_name: Name of the scenario to train on
            num_episodes: Number of training episodes
            save_interval: Save checkpoint every N episodes
            eval_episodes: Number of evaluation episodes
            seed: Random seed for reproducibility
            
        Returns:
            dict: Training start confirmation with status
            
        Raises:
            ValueError: If training is already in progress
            FileNotFoundError: If scenario config not found
        """
        # Check if training is already in progress
        if self._training_status['is_training']:
            raise ValueError(
                f"Training already in progress for scenario: "
                f"{self._training_status['scenario_name']}"
            )
        
        # Load scenario configuration
        try:
            scenario_config = read_yaml_config(scenario_name, 'scenarios')
        except FileNotFoundError:
            raise FileNotFoundError(f"Scenario '{scenario_name}' not found")
        
        # Update training status
        self._training_status = {
            'is_training': True,
            'scenario_name': scenario_name,
            'current_episode': 0,
            'total_episodes': num_episodes,
            'start_time': datetime.now(),
            'latest_progress': None
        }
        self._stop_requested = False
        
        # Start training in background task
        self._training_task = asyncio.create_task(
            self._run_training(
                scenario_config=scenario_config,
                scenario_name=scenario_name,
                num_episodes=num_episodes,
                save_interval=save_interval,
                eval_episodes=eval_episodes,
                seed=seed
            )
        )
        
        return {
            'status': 'started',
            'scenario_name': scenario_name,
            'num_episodes': num_episodes,
            'start_time': self._training_status['start_time'].isoformat()
        }
    
    async def _run_training(
        self,
        scenario_config: Dict[str, Any],
        scenario_name: str,
        num_episodes: int,
        save_interval: int,
        eval_episodes: int,
        seed: Optional[int]
    ):
        """
        Internal method to run training loop
        
        Args:
            scenario_config: Scenario configuration dictionary
            scenario_name: Name of the scenario
            num_episodes: Number of training episodes
            save_interval: Save checkpoint every N episodes
            eval_episodes: Number of evaluation episodes
            seed: Random seed for reproducibility
        """
        try:
            # Parse configuration
            env_config = EnvironmentConfig(**scenario_config['environment'])
            training_config = TrainingConfig(**scenario_config.get('training', {}))
            reward_config = RewardConfig(**scenario_config.get('reward', {}))
            
            # Override num_episodes from request
            training_config.num_episodes = num_episodes
            
            # Create environment
            env = BudgetEnv(env_config, reward_config)
            
            # Create reward engine
            reward_engine = RewardEngine(reward_config)
            
            # Create agents
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            goal_dim = 3  # [invest_ratio, save_ratio, consume_ratio]
            
            high_agent = FinancialStrategist(
                state_dim=state_dim,
                goal_dim=goal_dim,
                gamma=training_config.gamma_high,
                lr=training_config.learning_rate_high
            )
            
            low_agent = BudgetExecutor(
                state_dim=state_dim,
                goal_dim=goal_dim,
                action_dim=action_dim,
                gamma=training_config.gamma_low,
                lr=training_config.learning_rate_low
            )
            
            # Create trainer
            self._trainer = HRLTrainer(
                env=env,
                high_agent=high_agent,
                low_agent=low_agent,
                reward_engine=reward_engine,
                config=training_config,
                logger=None,  # No TensorBoard logging for API
                env_config=env_config,
                reward_config=reward_config
            )
            
            # Training loop with progress updates
            start_time = time.time()
            
            for episode in range(num_episodes):
                # Check if stop was requested
                if self._stop_requested:
                    print(f"Training stopped at episode {episode}")
                    break
                
                # Run one episode
                await self._run_episode(episode, start_time)
                
                # Save checkpoint at intervals
                if (episode + 1) % save_interval == 0:
                    checkpoint_dir = os.path.join('models', 'checkpoints', scenario_name)
                    self._trainer.save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        episode=episode + 1,
                        is_best=False
                    )
                    print(f"Checkpoint saved at episode {episode + 1}")
            
            # Save final model
            model_name = f"{scenario_name}"
            high_agent_path = os.path.join('models', f"{model_name}_high_agent.pt")
            low_agent_path = os.path.join('models', f"{model_name}_low_agent.pt")
            
            high_agent.save(high_agent_path)
            low_agent.save(low_agent_path)
            
            # Save training history
            history_path = os.path.join('models', f"{model_name}_history.json")
            import json
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_serializable = {}
                for key, value in self._trainer.training_history.items():
                    if isinstance(value, list):
                        history_serializable[key] = [
                            float(v) if hasattr(v, 'item') else v 
                            for v in value
                        ]
                    else:
                        history_serializable[key] = value
                json.dump(history_serializable, f, indent=2)
            
            print(f"Training complete! Models saved: {model_name}")
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Update status
            self._training_status['is_training'] = False
            self._trainer = None
    
    async def _run_episode(self, episode: int, start_time: float):
        """
        Run a single training episode with progress updates
        
        Args:
            episode: Current episode number
            start_time: Training start time (for elapsed time calculation)
        """
        # Reset analytics
        self._trainer.analytics.reset()
        
        # Reset episode tracking
        self._trainer.episode_actions = []
        self._trainer.episode_goals = []
        
        # Reset environment
        state, _ = self._trainer.env.reset()
        
        # Initialize state history
        self._trainer.state_history = [state]
        
        # Generate initial goal
        aggregated_state = self._trainer.high_agent.aggregate_state(
            self._trainer.state_history
        )
        goal = self._trainer.high_agent.select_goal(aggregated_state)
        
        # Episode tracking
        episode_reward = 0
        episode_length = 0
        self._trainer.episode_buffer = []
        
        # High-level tracking
        steps_since_high_update = 0
        high_level_transitions = []
        
        # Execute episode
        done = False
        while not done:
            # Low-level action
            action = self._trainer.low_agent.act(state, goal)
            
            # Track actions and goals
            self._trainer.episode_actions.append(action.copy())
            self._trainer.episode_goals.append(goal.copy())
            
            # Execute action
            next_state, reward, terminated, truncated, info = self._trainer.env.step(action)
            done = terminated or truncated
            
            # Calculate invested amount
            invested_amount = action[0] * state[0]
            
            # Record step in analytics
            self._trainer.analytics.record_step(
                state=state,
                action=action,
                reward=reward,
                goal=goal,
                invested_amount=invested_amount
            )
            
            # Store transition
            from src.utils.data_models import Transition
            transition = Transition(
                state=state,
                goal=goal,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            self._trainer.episode_buffer.append(transition)
            
            # Update state history
            self._trainer.state_history.append(next_state)
            
            # Track metrics
            episode_reward += reward
            episode_length += 1
            steps_since_high_update += 1
            
            # Update low-level policy
            if len(self._trainer.episode_buffer) >= self._trainer.config.batch_size:
                low_metrics = self._trainer.low_agent.learn(
                    self._trainer.episode_buffer[-self._trainer.config.batch_size:]
                )
                self._trainer.training_history['low_level_losses'].append(low_metrics['loss'])
            
            # High-level re-planning
            if steps_since_high_update >= self._trainer.config.high_period and not done:
                # Compute high-level reward
                period_transitions = self._trainer.episode_buffer[-steps_since_high_update:]
                high_level_reward = self._trainer.reward_engine.compute_high_level_reward(
                    period_transitions
                )
                
                # Create high-level transition
                high_transition = Transition(
                    state=aggregated_state,
                    goal=goal,
                    action=goal,
                    reward=high_level_reward,
                    next_state=self._trainer.high_agent.aggregate_state(
                        self._trainer.state_history
                    ),
                    done=False
                )
                high_level_transitions.append(high_transition)
                
                # Update high-level policy
                if len(high_level_transitions) > 0:
                    high_metrics = self._trainer.high_agent.learn(high_level_transitions)
                    self._trainer.training_history['high_level_losses'].append(
                        high_metrics['loss']
                    )
                
                # Generate new goal
                aggregated_state = self._trainer.high_agent.aggregate_state(
                    self._trainer.state_history
                )
                goal = self._trainer.high_agent.select_goal(aggregated_state)
                
                # Reset counter
                steps_since_high_update = 0
            
            # Move to next state
            state = next_state
        
        # Episode termination: final high-level update
        if len(self._trainer.episode_buffer) > 0:
            # Compute final high-level reward
            period_transitions = (
                self._trainer.episode_buffer[-steps_since_high_update:] 
                if steps_since_high_update > 0 
                else self._trainer.episode_buffer
            )
            high_level_reward = self._trainer.reward_engine.compute_high_level_reward(
                period_transitions
            )
            
            # Create final high-level transition
            final_aggregated_state = self._trainer.high_agent.aggregate_state(
                self._trainer.state_history
            )
            high_transition = Transition(
                state=aggregated_state,
                goal=goal,
                action=goal,
                reward=high_level_reward,
                next_state=final_aggregated_state,
                done=True
            )
            high_level_transitions.append(high_transition)
            
            # Final high-level update
            if len(high_level_transitions) > 0:
                high_metrics = self._trainer.high_agent.learn(high_level_transitions)
                self._trainer.training_history['high_level_losses'].append(
                    high_metrics['loss']
                )
            
            # Final low-level update
            if len(self._trainer.episode_buffer) >= self._trainer.config.batch_size:
                low_metrics = self._trainer.low_agent.learn(
                    self._trainer.episode_buffer[-self._trainer.config.batch_size:]
                )
                self._trainer.training_history['low_level_losses'].append(low_metrics['loss'])
        
        # Compute episode metrics
        episode_metrics = self._trainer.analytics.compute_episode_metrics()
        
        # Store episode metrics
        self._trainer.training_history['episode_rewards'].append(episode_reward)
        self._trainer.training_history['episode_lengths'].append(episode_length)
        self._trainer.training_history['cash_balances'].append(info.get('cash_balance', 0))
        self._trainer.training_history['total_invested'].append(info.get('total_invested', 0))
        self._trainer.training_history['cumulative_wealth_growth'].append(
            episode_metrics['cumulative_wealth_growth']
        )
        self._trainer.training_history['cash_stability_index'].append(
            episode_metrics['cash_stability_index']
        )
        self._trainer.training_history['sharpe_ratio'].append(episode_metrics['sharpe_ratio'])
        self._trainer.training_history['goal_adherence'].append(
            episode_metrics['goal_adherence']
        )
        self._trainer.training_history['policy_stability'].append(
            episode_metrics['policy_stability']
        )
        
        # Update training status and send progress update
        elapsed_time = time.time() - start_time
        
        # Calculate recent averages (last 10 episodes or available)
        recent_window = min(10, len(self._trainer.training_history['episode_rewards']))
        avg_reward = float(
            sum(self._trainer.training_history['episode_rewards'][-recent_window:]) / recent_window
        )
        avg_duration = float(
            sum(self._trainer.training_history['episode_lengths'][-recent_window:]) / recent_window
        )
        avg_cash = float(
            sum(self._trainer.training_history['cash_balances'][-recent_window:]) / recent_window
        )
        avg_invested = float(
            sum(self._trainer.training_history['total_invested'][-recent_window:]) / recent_window
        )
        avg_stability = float(
            sum(self._trainer.training_history['cash_stability_index'][-recent_window:]) / recent_window
        )
        avg_goal_adherence = float(
            sum(self._trainer.training_history['goal_adherence'][-recent_window:]) / recent_window
        )
        
        progress = {
            'episode': episode + 1,
            'total_episodes': self._training_status['total_episodes'],
            'avg_reward': avg_reward,
            'avg_duration': avg_duration,
            'avg_cash': avg_cash,
            'avg_invested': avg_invested,
            'stability': avg_stability,
            'goal_adherence': avg_goal_adherence,
            'elapsed_time': elapsed_time
        }
        
        self._training_status['current_episode'] = episode + 1
        self._training_status['latest_progress'] = progress
        
        # Send progress update via callback
        if self._progress_callback:
            await self._progress_callback(progress)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{self._training_status['total_episodes']} - "
                f"Reward: {avg_reward:.2f}, Duration: {avg_duration:.1f}, "
                f"Stability: {avg_stability:.2%}"
            )
    
    async def stop_training(self) -> Dict[str, Any]:
        """
        Stop the current training process gracefully
        
        Returns:
            dict: Stop confirmation with final status
            
        Raises:
            ValueError: If no training is in progress
        """
        if not self._training_status['is_training']:
            raise ValueError("No training in progress")
        
        # Request stop
        self._stop_requested = True
        
        # Wait for training task to complete
        if self._training_task:
            await self._training_task
        
        return {
            'status': 'stopped',
            'scenario_name': self._training_status['scenario_name'],
            'episodes_completed': self._training_status['current_episode'],
            'total_episodes': self._training_status['total_episodes']
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current training status
        
        Returns:
            dict: Current training status
        """
        return {
            'is_training': self._training_status['is_training'],
            'scenario_name': self._training_status['scenario_name'],
            'current_episode': self._training_status['current_episode'],
            'total_episodes': self._training_status['total_episodes'],
            'start_time': (
                self._training_status['start_time'].isoformat() 
                if self._training_status['start_time'] 
                else None
            ),
            'latest_progress': self._training_status['latest_progress']
        }


# Global training service instance
training_service = TrainingService()
