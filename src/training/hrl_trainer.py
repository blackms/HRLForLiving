"""Training Orchestrator for Hierarchical Reinforcement Learning"""
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple
from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.environment.reward_engine import RewardEngine
from src.utils.config import TrainingConfig, EnvironmentConfig, RewardConfig
from src.utils.data_models import Transition
from src.utils.analytics import AnalyticsModule
from src.utils.logger import ExperimentLogger


class HRLTrainer:
    """
    Training Orchestrator that coordinates the HRL training loop.
    
    The HRLTrainer manages the interaction between the high-level agent (Strategist)
    and low-level agent (Executor), coordinating policy updates and managing the
    training process. It implements the hierarchical training loop where:
    - The high-level agent sets strategic goals every N steps
    - The low-level agent executes monthly allocation decisions
    - Both agents learn from their respective experiences
    
    Uses PPO for low-level training and HIRO/Option-Critic for high-level training.
    """
    
    def __init__(
        self,
        env: BudgetEnv,
        high_agent: FinancialStrategist,
        low_agent: BudgetExecutor,
        reward_engine: RewardEngine,
        config: TrainingConfig,
        logger: Optional[ExperimentLogger] = None,
        env_config: Optional[EnvironmentConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        """
        Initialize the HRLTrainer with all necessary components.
        
        Args:
            env: BudgetEnv instance for financial simulation
            high_agent: FinancialStrategist for strategic goal generation
            low_agent: BudgetExecutor for monthly allocation decisions
            reward_engine: RewardEngine for reward computation
            config: TrainingConfig containing hyperparameters
            logger: Optional ExperimentLogger for TensorBoard logging
            env_config: Optional EnvironmentConfig for checkpointing
            reward_config: Optional RewardConfig for checkpointing
        """
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.reward_engine = reward_engine
        self.config = config
        self.logger = logger
        self.env_config = env_config
        self.reward_config = reward_config
        
        # Episode buffer for storing transitions
        self.episode_buffer: List[Transition] = []
        
        # State history for high-level agent aggregation
        self.state_history: List[np.ndarray] = []
        
        # Analytics module for tracking performance metrics
        self.analytics = AnalyticsModule()
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'cash_balances': [],
            'total_invested': [],
            'low_level_losses': [],
            'high_level_losses': [],
            'cumulative_wealth_growth': [],
            'cash_stability_index': [],
            'sharpe_ratio': [],
            'goal_adherence': [],
            'policy_stability': []
        }
        
        # Episode-level tracking for logging
        self.episode_actions: List[np.ndarray] = []
        self.episode_goals: List[np.ndarray] = []
        
        # Checkpointing state
        self.current_episode = 0
        self.best_eval_score = -np.inf
        self.best_checkpoint_path = None
    
    def train(self, num_episodes: int) -> Dict:
        """
        Execute the main HRL training loop.
        
        For each episode:
        1. Reset environment and get initial state
        2. Generate initial goal from high-level agent
        3. Execute monthly steps with low-level agent
        4. Store transitions in episode buffer
        5. Update policies according to HRL schedule
        
        Args:
            num_episodes: Number of training episodes to run
            
        Returns:
            dict: Training history with all collected metrics
        """
        for episode in range(num_episodes):
            # Reset analytics for new episode
            self.analytics.reset()
            
            # Reset episode-level tracking for logging
            self.episode_actions = []
            self.episode_goals = []
            
            # Reset environment and get initial state
            state, _ = self.env.reset()
            
            # Initialize state history for high-level agent
            self.state_history = [state]
            
            # Generate initial goal from high-level agent
            aggregated_state = self.high_agent.aggregate_state(self.state_history)
            goal = self.high_agent.select_goal(aggregated_state)
            
            # Episode tracking
            episode_reward = 0
            episode_length = 0
            self.episode_buffer = []
            
            # Track when to update high-level policy
            steps_since_high_update = 0
            high_level_transitions = []
            
            # Execute episode
            done = False
            while not done:
                # Low-level agent generates action based on state and goal
                action = self.low_agent.act(state, goal)
                
                # Track actions and goals for logging
                self.episode_actions.append(action.copy())
                self.episode_goals.append(goal.copy())
                
                # Execute action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Calculate invested amount for analytics
                invested_amount = action[0] * state[0]  # invest_ratio * income
                
                # Record step in analytics module
                self.analytics.record_step(
                    state=state,
                    action=action,
                    reward=reward,
                    goal=goal,
                    invested_amount=invested_amount
                )
                
                # Store transition in episode buffer
                transition = Transition(
                    state=state,
                    goal=goal,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                self.episode_buffer.append(transition)
                
                # Update state history for high-level aggregation
                self.state_history.append(next_state)
                
                # Track episode metrics
                episode_reward += reward
                episode_length += 1
                steps_since_high_update += 1
                
                # Update low-level policy when buffer reaches batch size
                if len(self.episode_buffer) >= self.config.batch_size:
                    low_metrics = self.low_agent.learn(self.episode_buffer[-self.config.batch_size:])
                    self.training_history['low_level_losses'].append(low_metrics['loss'])
                
                # High-level re-planning every high_period steps
                if steps_since_high_update >= self.config.high_period and not done:
                    # Compute high-level reward over the period
                    period_transitions = self.episode_buffer[-steps_since_high_update:]
                    high_level_reward = self.reward_engine.compute_high_level_reward(period_transitions)
                    
                    # Create high-level transition
                    high_transition = Transition(
                        state=aggregated_state,
                        goal=goal,
                        action=goal,  # For high-level, action is the goal itself
                        reward=high_level_reward,
                        next_state=self.high_agent.aggregate_state(self.state_history),
                        done=False
                    )
                    high_level_transitions.append(high_transition)
                    
                    # Update high-level policy
                    if len(high_level_transitions) > 0:
                        high_metrics = self.high_agent.learn(high_level_transitions)
                        self.training_history['high_level_losses'].append(high_metrics['loss'])
                    
                    # Generate new goal
                    aggregated_state = self.high_agent.aggregate_state(self.state_history)
                    goal = self.high_agent.select_goal(aggregated_state)
                    
                    # Reset counter
                    steps_since_high_update = 0
                
                # Move to next state
                state = next_state
            
            # Episode termination: handle final high-level update
            if len(self.episode_buffer) > 0:
                # Compute final high-level reward
                period_transitions = self.episode_buffer[-steps_since_high_update:] if steps_since_high_update > 0 else self.episode_buffer
                high_level_reward = self.reward_engine.compute_high_level_reward(period_transitions)
                
                # Create final high-level transition
                final_aggregated_state = self.high_agent.aggregate_state(self.state_history)
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
                    high_metrics = self.high_agent.learn(high_level_transitions)
                    self.training_history['high_level_losses'].append(high_metrics['loss'])
                
                # Final low-level update with remaining transitions
                if len(self.episode_buffer) >= self.config.batch_size:
                    low_metrics = self.low_agent.learn(self.episode_buffer[-self.config.batch_size:])
                    self.training_history['low_level_losses'].append(low_metrics['loss'])
            
            # Compute episode metrics from analytics module
            episode_metrics = self.analytics.compute_episode_metrics()
            
            # Store episode metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['cash_balances'].append(info.get('cash_balance', 0))
            self.training_history['total_invested'].append(info.get('total_invested', 0))
            self.training_history['cumulative_wealth_growth'].append(episode_metrics['cumulative_wealth_growth'])
            self.training_history['cash_stability_index'].append(episode_metrics['cash_stability_index'])
            self.training_history['sharpe_ratio'].append(episode_metrics['sharpe_ratio'])
            self.training_history['goal_adherence'].append(episode_metrics['goal_adherence'])
            self.training_history['policy_stability'].append(episode_metrics['policy_stability'])
            
            # Log to TensorBoard
            if self.logger is not None:
                # Log episode metrics
                self.logger.log_episode_metrics(episode, {
                    'reward': episode_reward,
                    'length': episode_length,
                    'cash_balance': info.get('cash_balance', 0),
                    'total_invested': info.get('total_invested', 0)
                })
                
                # Log analytics metrics
                self.logger.log_analytics_metrics(episode, episode_metrics)
                
                # Log action and goal distributions
                if len(self.episode_actions) > 0:
                    self.logger.log_action_distribution(episode, np.array(self.episode_actions))
                if len(self.episode_goals) > 0:
                    self.logger.log_goal_distribution(episode, np.array(self.episode_goals))
                
                # Log training losses
                if len(self.training_history['low_level_losses']) > 0:
                    self.logger.log_training_curves(episode, {
                        'low_level_loss': self.training_history['low_level_losses'][-1]
                    })
                if len(self.training_history['high_level_losses']) > 0:
                    self.logger.log_training_curves(episode, {
                        'high_level_loss': self.training_history['high_level_losses'][-1]
                    })
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                avg_cash = np.mean(self.training_history['cash_balances'][-100:])
                avg_invested = np.mean(self.training_history['total_invested'][-100:])
                avg_stability = np.mean(self.training_history['cash_stability_index'][-100:])
                avg_goal_adherence = np.mean(self.training_history['goal_adherence'][-100:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Cash: {avg_cash:.2f}, "
                      f"Avg Invested: {avg_invested:.2f}, "
                      f"Stability: {avg_stability:.2%}, "
                      f"Goal Adherence: {avg_goal_adherence:.4f}")
        
        return self.training_history
    
    def save_checkpoint(
        self,
        checkpoint_dir: str,
        episode: int,
        is_best: bool = False,
        prefix: str = "checkpoint"
    ) -> str:
        """
        Save a training checkpoint including models, configuration, and training state.
        
        Saves:
        - High-level agent model
        - Low-level agent model
        - Training configuration
        - Environment configuration
        - Reward configuration
        - Training history
        - Current episode number
        
        Args:
            checkpoint_dir: Directory to save checkpoint files
            episode: Current episode number
            is_best: Whether this is the best model so far
            prefix: Prefix for checkpoint filenames (default: "checkpoint")
            
        Returns:
            str: Path to the checkpoint directory
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint subdirectory with episode number
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_best")
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_episode_{episode}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save high-level agent
        high_agent_path = os.path.join(checkpoint_path, "high_agent.pt")
        self.high_agent.save(high_agent_path)
        
        # Save low-level agent
        low_agent_path = os.path.join(checkpoint_path, "low_agent.pt")
        self.low_agent.save(low_agent_path)
        
        # Save configurations and training state
        # Use episode parameter as the current episode for consistency
        metadata = {
            'episode': episode,
            'current_episode': episode,
            'best_eval_score': float(self.best_eval_score),
            'training_config': {
                'num_episodes': self.config.num_episodes,
                'gamma_low': self.config.gamma_low,
                'gamma_high': self.config.gamma_high,
                'high_period': self.config.high_period,
                'batch_size': self.config.batch_size,
                'learning_rate_low': self.config.learning_rate_low,
                'learning_rate_high': self.config.learning_rate_high
            }
        }
        
        # Add environment config if available
        if self.env_config is not None:
            metadata['environment_config'] = {
                'income': self.env_config.income,
                'fixed_expenses': self.env_config.fixed_expenses,
                'variable_expense_mean': self.env_config.variable_expense_mean,
                'variable_expense_std': self.env_config.variable_expense_std,
                'inflation': self.env_config.inflation,
                'safety_threshold': self.env_config.safety_threshold,
                'max_months': self.env_config.max_months,
                'initial_cash': self.env_config.initial_cash,
                'risk_tolerance': self.env_config.risk_tolerance
            }
        
        # Add reward config if available
        if self.reward_config is not None:
            metadata['reward_config'] = {
                'alpha': self.reward_config.alpha,
                'beta': self.reward_config.beta,
                'gamma': self.reward_config.gamma,
                'delta': self.reward_config.delta,
                'lambda_': self.reward_config.lambda_,
                'mu': self.reward_config.mu
            }
        
        # Save metadata
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save training history (convert numpy arrays to lists for JSON serialization)
        history_serializable = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            else:
                history_serializable[key] = value
        
        history_path = os.path.join(checkpoint_path, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Tuple[int, Dict]:
        """
        Load a training checkpoint to resume training.
        
        Loads:
        - High-level agent model
        - Low-level agent model
        - Training history
        - Current episode number
        - Best evaluation score
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            
        Returns:
            tuple: (episode_number, training_history)
        """
        # Load high-level agent
        high_agent_path = os.path.join(checkpoint_path, "high_agent.pt")
        if os.path.exists(high_agent_path):
            self.high_agent.load(high_agent_path)
        else:
            raise FileNotFoundError(f"High-level agent model not found at {high_agent_path}")
        
        # Load low-level agent
        low_agent_path = os.path.join(checkpoint_path, "low_agent.pt")
        if os.path.exists(low_agent_path):
            self.low_agent.load(low_agent_path)
        else:
            raise FileNotFoundError(f"Low-level agent model not found at {low_agent_path}")
        
        # Load metadata
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        episode_number = 0
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            episode_number = metadata.get('episode', 0)
            self.current_episode = metadata.get('current_episode', episode_number)
            self.best_eval_score = metadata.get('best_eval_score', -np.inf)
        else:
            print(f"Warning: Metadata not found at {metadata_path}")
            self.current_episode = 0
            self.best_eval_score = -np.inf
        
        # Load training history
        history_path = os.path.join(checkpoint_path, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        else:
            print(f"Warning: Training history not found at {history_path}")
            self.training_history = {
                'episode_rewards': [],
                'episode_lengths': [],
                'cash_balances': [],
                'total_invested': [],
                'low_level_losses': [],
                'high_level_losses': [],
                'cumulative_wealth_growth': [],
                'cash_stability_index': [],
                'sharpe_ratio': [],
                'goal_adherence': [],
                'policy_stability': []
            }
        
        return episode_number, self.training_history
    
    def train_with_checkpointing(
        self,
        num_episodes: int,
        checkpoint_dir: str,
        save_interval: int = 1000,
        eval_interval: int = 1000,
        eval_episodes: int = 10
    ) -> Dict:
        """
        Execute training loop with automatic checkpointing and best model tracking.
        
        Saves checkpoints every N episodes and keeps the best model based on
        evaluation performance. The best model is determined by the mean reward
        during evaluation episodes.
        
        Args:
            num_episodes: Number of training episodes to run
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N episodes (default: 1000)
            eval_interval: Evaluate and potentially save best model every N episodes (default: 1000)
            eval_episodes: Number of episodes to run for evaluation (default: 10)
            
        Returns:
            dict: Training history with all collected metrics
        """
        print(f"Starting training with checkpointing...")
        print(f"Checkpoints will be saved to: {checkpoint_dir}")
        print(f"Save interval: {save_interval} episodes")
        print(f"Evaluation interval: {eval_interval} episodes")
        
        for episode in range(self.current_episode, self.current_episode + num_episodes):
            # Reset analytics for new episode
            self.analytics.reset()
            
            # Reset episode-level tracking for logging
            self.episode_actions = []
            self.episode_goals = []
            
            # Reset environment and get initial state
            state, _ = self.env.reset()
            
            # Initialize state history for high-level agent
            self.state_history = [state]
            
            # Generate initial goal from high-level agent
            aggregated_state = self.high_agent.aggregate_state(self.state_history)
            goal = self.high_agent.select_goal(aggregated_state)
            
            # Episode tracking
            episode_reward = 0
            episode_length = 0
            self.episode_buffer = []
            
            # Track when to update high-level policy
            steps_since_high_update = 0
            high_level_transitions = []
            
            # Execute episode
            done = False
            while not done:
                # Low-level agent generates action based on state and goal
                action = self.low_agent.act(state, goal)
                
                # Track actions and goals for logging
                self.episode_actions.append(action.copy())
                self.episode_goals.append(goal.copy())
                
                # Execute action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Calculate invested amount for analytics
                invested_amount = action[0] * state[0]  # invest_ratio * income
                
                # Record step in analytics module
                self.analytics.record_step(
                    state=state,
                    action=action,
                    reward=reward,
                    goal=goal,
                    invested_amount=invested_amount
                )
                
                # Store transition in episode buffer
                transition = Transition(
                    state=state,
                    goal=goal,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                self.episode_buffer.append(transition)
                
                # Update state history for high-level aggregation
                self.state_history.append(next_state)
                
                # Track episode metrics
                episode_reward += reward
                episode_length += 1
                steps_since_high_update += 1
                
                # Update low-level policy when buffer reaches batch size
                if len(self.episode_buffer) >= self.config.batch_size:
                    low_metrics = self.low_agent.learn(self.episode_buffer[-self.config.batch_size:])
                    self.training_history['low_level_losses'].append(low_metrics['loss'])
                
                # High-level re-planning every high_period steps
                if steps_since_high_update >= self.config.high_period and not done:
                    # Compute high-level reward over the period
                    period_transitions = self.episode_buffer[-steps_since_high_update:]
                    high_level_reward = self.reward_engine.compute_high_level_reward(period_transitions)
                    
                    # Create high-level transition
                    high_transition = Transition(
                        state=aggregated_state,
                        goal=goal,
                        action=goal,  # For high-level, action is the goal itself
                        reward=high_level_reward,
                        next_state=self.high_agent.aggregate_state(self.state_history),
                        done=False
                    )
                    high_level_transitions.append(high_transition)
                    
                    # Update high-level policy
                    if len(high_level_transitions) > 0:
                        high_metrics = self.high_agent.learn(high_level_transitions)
                        self.training_history['high_level_losses'].append(high_metrics['loss'])
                    
                    # Generate new goal
                    aggregated_state = self.high_agent.aggregate_state(self.state_history)
                    goal = self.high_agent.select_goal(aggregated_state)
                    
                    # Reset counter
                    steps_since_high_update = 0
                
                # Move to next state
                state = next_state
            
            # Episode termination: handle final high-level update
            if len(self.episode_buffer) > 0:
                # Compute final high-level reward
                period_transitions = self.episode_buffer[-steps_since_high_update:] if steps_since_high_update > 0 else self.episode_buffer
                high_level_reward = self.reward_engine.compute_high_level_reward(period_transitions)
                
                # Create final high-level transition
                final_aggregated_state = self.high_agent.aggregate_state(self.state_history)
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
                    high_metrics = self.high_agent.learn(high_level_transitions)
                    self.training_history['high_level_losses'].append(high_metrics['loss'])
                
                # Final low-level update with remaining transitions
                if len(self.episode_buffer) >= self.config.batch_size:
                    low_metrics = self.low_agent.learn(self.episode_buffer[-self.config.batch_size:])
                    self.training_history['low_level_losses'].append(low_metrics['loss'])
            
            # Compute episode metrics from analytics module
            episode_metrics = self.analytics.compute_episode_metrics()
            
            # Store episode metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['cash_balances'].append(info.get('cash_balance', 0))
            self.training_history['total_invested'].append(info.get('total_invested', 0))
            self.training_history['cumulative_wealth_growth'].append(episode_metrics['cumulative_wealth_growth'])
            self.training_history['cash_stability_index'].append(episode_metrics['cash_stability_index'])
            self.training_history['sharpe_ratio'].append(episode_metrics['sharpe_ratio'])
            self.training_history['goal_adherence'].append(episode_metrics['goal_adherence'])
            self.training_history['policy_stability'].append(episode_metrics['policy_stability'])
            
            # Update current episode
            self.current_episode = episode + 1
            
            # Log to TensorBoard
            if self.logger is not None:
                # Log episode metrics
                self.logger.log_episode_metrics(episode, {
                    'reward': episode_reward,
                    'length': episode_length,
                    'cash_balance': info.get('cash_balance', 0),
                    'total_invested': info.get('total_invested', 0)
                })
                
                # Log analytics metrics
                self.logger.log_analytics_metrics(episode, episode_metrics)
                
                # Log action and goal distributions
                if len(self.episode_actions) > 0:
                    self.logger.log_action_distribution(episode, np.array(self.episode_actions))
                if len(self.episode_goals) > 0:
                    self.logger.log_goal_distribution(episode, np.array(self.episode_goals))
                
                # Log training losses
                if len(self.training_history['low_level_losses']) > 0:
                    self.logger.log_training_curves(episode, {
                        'low_level_loss': self.training_history['low_level_losses'][-1]
                    })
                if len(self.training_history['high_level_losses']) > 0:
                    self.logger.log_training_curves(episode, {
                        'high_level_loss': self.training_history['high_level_losses'][-1]
                    })
            
            # Save checkpoint at regular intervals
            if (episode + 1) % save_interval == 0:
                checkpoint_path = self.save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    episode=episode + 1,
                    is_best=False
                )
                print(f"Checkpoint saved at episode {episode + 1}: {checkpoint_path}")
            
            # Evaluate and potentially save best model
            if (episode + 1) % eval_interval == 0:
                print(f"\nEvaluating at episode {episode + 1}...")
                eval_results = self.evaluate(eval_episodes)
                eval_score = eval_results['mean_reward']
                
                print(f"Evaluation score: {eval_score:.2f} (best: {self.best_eval_score:.2f})")
                
                # Save best model if this is the best so far (and not NaN)
                if not np.isnan(eval_score) and eval_score > self.best_eval_score:
                    self.best_eval_score = eval_score
                    best_checkpoint_path = self.save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        episode=episode + 1,
                        is_best=True
                    )
                    self.best_checkpoint_path = best_checkpoint_path
                    print(f"New best model saved! Score: {eval_score:.2f}")
                    print(f"Best checkpoint: {best_checkpoint_path}")
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                avg_cash = np.mean(self.training_history['cash_balances'][-100:])
                avg_invested = np.mean(self.training_history['total_invested'][-100:])
                avg_stability = np.mean(self.training_history['cash_stability_index'][-100:])
                avg_goal_adherence = np.mean(self.training_history['goal_adherence'][-100:])
                print(f"Episode {episode + 1}/{self.current_episode + num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Cash: {avg_cash:.2f}, "
                      f"Avg Invested: {avg_invested:.2f}, "
                      f"Stability: {avg_stability:.2%}, "
                      f"Goal Adherence: {avg_goal_adherence:.4f}")
        
        # Save final checkpoint
        final_checkpoint_path = self.save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            episode=self.current_episode,
            is_best=False,
            prefix="checkpoint_final"
        )
        print(f"\nTraining complete! Final checkpoint saved: {final_checkpoint_path}")
        if self.best_checkpoint_path:
            print(f"Best model checkpoint: {self.best_checkpoint_path}")
            print(f"Best evaluation score: {self.best_eval_score:.2f}")
        
        return self.training_history
    
    def evaluate(self, num_episodes: int) -> Dict:
        """
        Run evaluation episodes without learning.
        
        Executes episodes using the current policies in deterministic mode
        to assess performance. Collects metrics including:
        - Cumulative wealth growth (total invested)
        - Cash stability index (% months with positive balance)
        - Sharpe-like ratio (mean return / std balance)
        - Goal adherence (mean absolute difference)
        - Policy stability (variance of actions)
        
        Args:
            num_episodes: Number of evaluation episodes to run
            
        Returns:
            dict: Evaluation results with performance metrics
        """
        eval_results = {
            'episode_rewards': [],
            'episode_lengths': [],
            'cash_balances': [],
            'total_invested': [],
            'cumulative_wealth_growth': [],
            'cash_stability_index': [],
            'sharpe_ratio': [],
            'goal_adherence': [],
            'policy_stability': []
        }
        
        for episode in range(num_episodes):
            # Reset environment
            state, _ = self.env.reset()
            
            # Initialize state history
            state_history = [state]
            
            # Generate initial goal
            aggregated_state = self.high_agent.aggregate_state(state_history)
            goal = self.high_agent.select_goal(aggregated_state)
            
            # Episode tracking
            episode_reward = 0
            episode_length = 0
            steps_since_high_update = 0
            
            # Metrics tracking
            cash_history = []
            action_history = []
            goal_adherence_values = []
            positive_balance_months = 0
            
            # Execute episode (deterministic policy)
            done = False
            while not done:
                # Low-level agent generates action (deterministic)
                action = self.low_agent.act(state, goal, deterministic=True)
                
                # Execute action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Track metrics
                episode_reward += reward
                episode_length += 1
                steps_since_high_update += 1
                
                cash_balance = info.get('cash_balance', 0)
                cash_history.append(cash_balance)
                action_history.append(action)
                
                # Track positive balance months
                if cash_balance > 0:
                    positive_balance_months += 1
                
                # Track goal adherence (difference between target and actual investment)
                target_invest_ratio = goal[0]
                actual_invest_ratio = action[0]
                goal_adherence_values.append(abs(target_invest_ratio - actual_invest_ratio))
                
                # Update state history
                state_history.append(next_state)
                
                # High-level re-planning every high_period steps
                if steps_since_high_update >= self.config.high_period and not done:
                    aggregated_state = self.high_agent.aggregate_state(state_history)
                    goal = self.high_agent.select_goal(aggregated_state)
                    steps_since_high_update = 0
                
                # Move to next state
                state = next_state
            
            # Compute episode metrics
            final_cash = info.get('cash_balance', 0)
            total_invested = info.get('total_invested', 0)
            
            # 1. Cumulative wealth growth
            cumulative_wealth = total_invested
            
            # 2. Cash stability index (% months with positive balance)
            cash_stability = positive_balance_months / episode_length if episode_length > 0 else 0
            
            # 3. Sharpe-like ratio (mean return / std balance)
            if len(cash_history) > 1:
                cash_array = np.array(cash_history)
                mean_cash = np.mean(cash_array)
                std_cash = np.std(cash_array)
                sharpe_ratio = mean_cash / (std_cash + 1e-8)
            else:
                sharpe_ratio = 0
            
            # 4. Goal adherence (mean absolute difference)
            goal_adherence = np.mean(goal_adherence_values) if goal_adherence_values else 0
            
            # 5. Policy stability (variance of actions)
            if len(action_history) > 1:
                action_array = np.array(action_history)
                policy_stability = np.mean(np.var(action_array, axis=0))
            else:
                policy_stability = 0
            
            # Store episode results
            eval_results['episode_rewards'].append(episode_reward)
            eval_results['episode_lengths'].append(episode_length)
            eval_results['cash_balances'].append(final_cash)
            eval_results['total_invested'].append(total_invested)
            eval_results['cumulative_wealth_growth'].append(cumulative_wealth)
            eval_results['cash_stability_index'].append(cash_stability)
            eval_results['sharpe_ratio'].append(sharpe_ratio)
            eval_results['goal_adherence'].append(goal_adherence)
            eval_results['policy_stability'].append(policy_stability)
        
        # Compute aggregate statistics
        eval_summary = {
            'mean_reward': np.mean(eval_results['episode_rewards']),
            'std_reward': np.std(eval_results['episode_rewards']),
            'mean_cash_balance': np.mean(eval_results['cash_balances']),
            'mean_total_invested': np.mean(eval_results['total_invested']),
            'mean_wealth_growth': np.mean(eval_results['cumulative_wealth_growth']),
            'mean_cash_stability': np.mean(eval_results['cash_stability_index']),
            'mean_sharpe_ratio': np.mean(eval_results['sharpe_ratio']),
            'mean_goal_adherence': np.mean(eval_results['goal_adherence']),
            'mean_policy_stability': np.mean(eval_results['policy_stability']),
            'detailed_results': eval_results
        }
        
        return eval_summary
