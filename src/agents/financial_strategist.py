"""High-Level Agent (Financial Strategist) for strategic goal generation"""
import numpy as np
from typing import List, Dict
import torch
import torch.nn as nn
from src.utils.config import TrainingConfig
from src.utils.data_models import Transition


class StrategistNetwork(nn.Module):
    """Neural network policy for the FinancialStrategist"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(StrategistNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Initialize weights with Xavier/Glorot initialization
        # Literature: "Proper initialization prevents gradient vanishing/explosion" (Glorot & Bengio, 2010)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.01)  # Small gain for stability
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        # Add input normalization layer
        # Clip inputs to prevent extreme values
        x = torch.clamp(x, -10.0, 10.0)
        return self.network(x)


class FinancialStrategist:
    """
    High-Level Agent that defines medium-term financial strategy.
    
    The FinancialStrategist observes aggregated state information (average cash,
    investment returns, spending trends) and generates strategic goals for the
    Low-Level Agent to follow. Goals include target investment ratio, safety buffer,
    and aggressiveness level.
    
    Uses HIRO or Option-Critic algorithm for hierarchical learning.
    """
    
    def __init__(self, config: TrainingConfig, aggregated_state_dim: int = 5):
        """
        Initialize the FinancialStrategist with neural network policy.
        
        Args:
            config: TrainingConfig containing hyperparameters
            aggregated_state_dim: Dimension of aggregated state vector (default: 5)
        """
        self.config = config
        self.aggregated_state_dim = aggregated_state_dim
        self.goal_dim = 3  # [target_invest_ratio, safety_buffer, aggressiveness]
        
        # Initialize policy network
        self.policy_network = StrategistNetwork(self.aggregated_state_dim, self.goal_dim)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate_high
        )
        
        # Training metrics
        self.training_metrics = {
            'loss': [],
            'policy_entropy': []
        }

    def aggregate_state(self, history: List[np.ndarray]) -> np.ndarray:
        """
        Compute macro features from state history.
        
        Aggregates recent state history to compute strategic-level features:
        - Average cash balance over last N months
        - Average investment return (estimated from cash changes)
        - Spending trend (change in expenses over time)
        - Current wealth (current cash balance)
        - Months elapsed
        
        Args:
            history: List of state vectors from recent months
                    Each state: [income, fixed, variable, cash, inflation, risk, t_remaining]
        
        Returns:
            np.ndarray: 5-dimensional aggregated state
                       [avg_cash, avg_investment_return, spending_trend, current_wealth, months_elapsed]
        """
        if not history:
            # Return default aggregated state if no history
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Convert history to numpy array for easier processing
        history_array = np.array(history, dtype=np.float32)
        
        # Extract relevant features from state vectors
        # State format: [income, fixed, variable, cash, inflation, risk, t_remaining]
        cash_balances = history_array[:, 3]  # Index 3 is cash balance
        variable_expenses = history_array[:, 2]  # Index 2 is variable expenses
        
        # 1. Average cash over last N months
        avg_cash = np.mean(cash_balances)
        
        # 2. Average investment return (estimated from cash balance changes)
        if len(cash_balances) > 1:
            cash_changes = np.diff(cash_balances)
            avg_investment_return = np.mean(cash_changes)
        else:
            avg_investment_return = 0.0
        
        # 3. Spending trend (change in variable expenses over time)
        if len(variable_expenses) > 1:
            # Calculate linear trend using simple slope
            x = np.arange(len(variable_expenses))
            spending_trend = np.polyfit(x, variable_expenses, 1)[0]  # Slope of linear fit
        else:
            spending_trend = 0.0
        
        # 4. Current wealth (most recent cash balance)
        current_wealth = cash_balances[-1]
        
        # 5. Months elapsed (inferred from t_remaining)
        if len(history_array) > 0:
            # Assuming max_months is known or can be inferred
            t_remaining_first = history_array[0, 6]
            t_remaining_current = history_array[-1, 6]
            months_elapsed = t_remaining_first - t_remaining_current
        else:
            months_elapsed = 0.0
        
        # Construct aggregated state with normalization
        # Literature: "State normalization is critical for hierarchical RL" (Nachum et al., 2018 - HIRO)
        aggregated_state = np.array([
            avg_cash / 10000.0,                    # Normalize to ~0.5-1.0
            avg_investment_return / 1000.0,        # Normalize to ~-0.5 to 0.5
            spending_trend / 100.0,                # Normalize to ~-0.1 to 0.1
            current_wealth / 10000.0,              # Normalize to ~0.5-1.0
            months_elapsed / 120.0                 # Normalize to [0, 1]
        ], dtype=np.float32)
        
        # Safety check for NaN/Inf in aggregated state
        if np.any(np.isnan(aggregated_state)) or np.any(np.isinf(aggregated_state)):
            print(f"WARNING: Invalid aggregated state! Returning default.")
            aggregated_state = np.array([0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        
        return aggregated_state

    def select_goal(self, state: np.ndarray) -> np.ndarray:
        """
        Generate goal vector from aggregated state.
        
        Uses the policy network to generate a strategic goal vector that guides
        the Low-Level Agent's allocation decisions. The goal includes:
        - target_invest_ratio: Desired investment percentage [0, 1]
        - safety_buffer: Minimum cash reserve target [0, inf)
        - aggressiveness: Risk appetite parameter [0, 1]
        
        Args:
            state: Aggregated state vector (5-dimensional)
                  [avg_cash, avg_investment_return, spending_trend, current_wealth, months_elapsed]
        
        Returns:
            np.ndarray: 3-dimensional goal vector
                       [target_invest_ratio, safety_buffer, aggressiveness]
        """
        # Ensure state is numpy array
        state = np.array(state, dtype=np.float32)
        
        # Validate dimension
        if state.shape[0] != self.aggregated_state_dim:
            raise ValueError(
                f"Expected aggregated state dimension {self.aggregated_state_dim}, "
                f"got {state.shape[0]}"
            )
        
        # Convert to torch tensor
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            goal_raw = self.policy_network(state_tensor).squeeze(0).numpy()
        
        # Check for NaN in goal (can happen with untrained network)
        if np.any(np.isnan(goal_raw)):
            print(f"WARNING: NaN in goal from network! Using default goal.")
            goal_raw = np.array([0.3, 1000.0, 0.5], dtype=np.float32)
        
        # Apply constraints to ensure values are within valid ranges
        # target_invest_ratio: [0, 1] using sigmoid
        goal_raw_clipped = np.clip(goal_raw, -20, 20)  # Clip to prevent overflow
        target_invest_ratio = 1.0 / (1.0 + np.exp(-goal_raw_clipped[0]))
        
        # safety_buffer: [0, inf) using softplus (ensures positive)
        # Use numerically stable softplus: softplus(x) = log(1 + exp(x)) ≈ x for large x
        if goal_raw_clipped[1] > 20:
            safety_buffer = goal_raw_clipped[1]
        else:
            safety_buffer = np.log(1.0 + np.exp(goal_raw_clipped[1]))
        
        # aggressiveness: [0, 1] using sigmoid
        aggressiveness = 1.0 / (1.0 + np.exp(-goal_raw_clipped[2]))
        
        # Construct goal vector
        goal = np.array([
            target_invest_ratio,
            safety_buffer,
            aggressiveness
        ], dtype=np.float32)
        
        return goal

    def learn(self, transitions: List[Transition]) -> Dict[str, float]:
        """
        Update high-level policy using collected transitions.
        
        Applies a simplified HIRO-style algorithm with discount factor γ_high = 0.99
        to update the strategic policy based on long-term rewards. The high-level
        agent learns to generate goals that maximize cumulative wealth growth and
        financial stability.
        
        Args:
            transitions: List of Transition objects containing high-level experiences
                        (aggregated_state, goal, action, reward, next_aggregated_state, done)
        
        Returns:
            dict: Training metrics with keys 'loss' and 'policy_entropy'
        """
        if not transitions:
            return {'loss': 0.0, 'policy_entropy': 0.0}
        
        # Extract data from transitions
        states = []
        goals = []
        rewards = []
        next_states = []
        dones = []
        
        for transition in transitions:
            states.append(transition.state)
            goals.append(transition.goal)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        goals = np.array(goals, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        # Calculate returns using discount factor γ_high = 0.99
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.config.gamma_high * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize returns for stable training (with better numerical stability)
        if len(returns) > 1:
            returns_std = returns.std()
            if returns_std > 1e-6:  # Only normalize if there's meaningful variance
                returns = (returns - returns.mean()) / returns_std
            else:
                returns = returns - returns.mean()  # Just center if no variance
        
        # Convert to torch tensors
        states_tensor = torch.FloatTensor(states)
        goals_tensor = torch.FloatTensor(goals)
        returns_tensor = torch.FloatTensor(returns)
        
        # Perform policy update
        self.optimizer.zero_grad()
        
        # Get goal predictions from policy
        goal_predictions = self.policy_network(states_tensor)
        
        # Calculate policy loss using mean squared error between predicted and actual goals
        # weighted by returns (this is a simplified approach)
        # In full HIRO, we would use intrinsic rewards and goal-conditioned value functions
        mse_loss = torch.mean((goal_predictions - goals_tensor) ** 2, dim=1)
        
        # Clip returns to prevent extreme gradients
        # Literature: "Return clipping improves stability" (Mnih et al., 2016 - A3C)
        returns_tensor_clipped = torch.clamp(returns_tensor, -10.0, 10.0)
        
        policy_loss = (mse_loss * returns_tensor_clipped).mean()
        
        # Calculate a pseudo-entropy for exploration (variance of predictions)
        # Use safer variance calculation
        if len(goal_predictions) > 1:
            goal_variance = torch.var(goal_predictions, dim=0, unbiased=False).mean()
            # Clip variance to prevent NaN
            goal_variance = torch.clamp(goal_variance, 0.0, 100.0)
        else:
            goal_variance = torch.tensor(0.0)
        
        entropy = goal_variance  # Use variance as a proxy for entropy
        
        # Total loss (policy loss - entropy bonus for exploration)
        # Reduce entropy coefficient for stability
        loss = policy_loss - 0.001 * entropy  # Reduced from 0.01
        
        # Safety check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss detected! Skipping update.")
            return {'loss': 0.0, 'policy_entropy': 0.0, 'n_updates': 0}
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping for stability (more aggressive for high-level)
        # Literature: "Hierarchical policies need stronger regularization" (Nachum et al., 2018)
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.1)
        
        # Check for NaN gradients
        has_nan_grad = False
        for param in self.policy_network.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print(f"WARNING: NaN gradients detected! Skipping update.")
            self.optimizer.zero_grad()
            return {'loss': 0.0, 'policy_entropy': 0.0, 'n_updates': 0}
        
        self.optimizer.step()
        
        # Store metrics
        loss_value = loss.item()
        entropy_value = entropy.item()
        
        self.training_metrics['loss'].append(loss_value)
        self.training_metrics['policy_entropy'].append(entropy_value)
        
        return {
            'loss': loss_value,
            'policy_entropy': entropy_value,
            'n_updates': 1
        }
    
    def save(self, path: str):
        """
        Save the policy model to disk.
        
        Args:
            path: File path to save the model
        """
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """
        Load the policy model from disk.
        
        Args:
            path: File path to load the model from
        """
        checkpoint = torch.load(path, weights_only=False)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
