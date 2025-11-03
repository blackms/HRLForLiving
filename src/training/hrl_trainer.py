"""Training Orchestrator for Hierarchical Reinforcement Learning"""
import numpy as np
from typing import List, Dict
from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.environment.reward_engine import RewardEngine
from src.utils.config import TrainingConfig
from src.utils.data_models import Transition


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
        config: TrainingConfig
    ):
        """
        Initialize the HRLTrainer with all necessary components.
        
        Args:
            env: BudgetEnv instance for financial simulation
            high_agent: FinancialStrategist for strategic goal generation
            low_agent: BudgetExecutor for monthly allocation decisions
            reward_engine: RewardEngine for reward computation
            config: TrainingConfig containing hyperparameters
        """
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.reward_engine = reward_engine
        self.config = config
        
        # Episode buffer for storing transitions
        self.episode_buffer: List[Transition] = []
        
        # State history for high-level agent aggregation
        self.state_history: List[np.ndarray] = []
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'cash_balances': [],
            'total_invested': [],
            'low_level_losses': [],
            'high_level_losses': []
        }
    
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
                
                # Execute action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
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
            
            # Store episode metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['cash_balances'].append(info.get('cash_balance', 0))
            self.training_history['total_invested'].append(info.get('total_invested', 0))
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                avg_cash = np.mean(self.training_history['cash_balances'][-100:])
                avg_invested = np.mean(self.training_history['total_invested'][-100:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Cash: {avg_cash:.2f}, "
                      f"Avg Invested: {avg_invested:.2f}")
        
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
