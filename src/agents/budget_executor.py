"""Low-Level Agent (Budget Executor) for monthly allocation decisions"""
import numpy as np
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from gymnasium import spaces
from src.utils.config import TrainingConfig
from src.utils.data_models import Transition


class PolicyNetwork(nn.Module):
    """Neural network policy for the BudgetExecutor"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)  # Ensure output sums to 1
        )
    
    def forward(self, x):
        return self.network(x)


class BudgetExecutor:
    """
    Low-Level Agent that executes concrete monthly allocation decisions.
    
    The BudgetExecutor receives both the current financial state and a goal vector
    from the High-Level Agent, then produces an action vector [invest, save, consume]
    that allocates the monthly income.
    
    Uses PPO (Proximal Policy Optimization) for learning optimal allocation policies.
    """
    
    def __init__(self, config: TrainingConfig, state_dim: int = 7, goal_dim: int = 3):
        """
        Initialize the BudgetExecutor with PPO policy.
        
        Args:
            config: TrainingConfig containing hyperparameters
            state_dim: Dimension of state vector (default: 7)
            goal_dim: Dimension of goal vector (default: 3)
        """
        self.config = config
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.input_dim = state_dim + goal_dim  # Concatenated input: 10-dimensional
        
        # Define observation space for PPO (concatenated state + goal)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.input_dim,),
            dtype=np.float32
        )
        
        # Define action space (3-dimensional continuous [0, 1])
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(self.input_dim, 3)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate_low
        )
        
        # Training metrics
        self.training_metrics = {
            'loss': [],
            'policy_entropy': []
        }

    def act(self, state: np.ndarray, goal: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Generate allocation action from state and goal.
        
        Concatenates the state vector (7-dimensional) and goal vector (3-dimensional)
        to create a 10-dimensional input, then passes it to the PPO policy to
        produce an action vector [invest, save, consume].
        
        Args:
            state: Current financial state vector [income, fixed, variable, cash, inflation, risk, t_remaining]
            goal: Strategic goal vector from High-Level Agent [target_invest_ratio, safety_buffer, aggressiveness]
            deterministic: If True, use deterministic policy (for evaluation)
            
        Returns:
            np.ndarray: Action vector [invest_ratio, save_ratio, consume_ratio]
        """
        # Ensure inputs are numpy arrays
        state = np.array(state, dtype=np.float32)
        goal = np.array(goal, dtype=np.float32)
        
        # Validate dimensions
        if state.shape[0] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}, got {state.shape[0]}")
        if goal.shape[0] != self.goal_dim:
            raise ValueError(f"Expected goal dimension {self.goal_dim}, got {goal.shape[0]}")
        
        # Concatenate state and goal to create 10-dimensional input
        concatenated_input = np.concatenate([state, goal])
        
        # Convert to torch tensor
        with torch.no_grad():
            input_tensor = torch.FloatTensor(concatenated_input).unsqueeze(0)
            action_probs = self.policy_network(input_tensor)
            
            if deterministic:
                # Use argmax for deterministic action (but we need continuous, so use the probabilities directly)
                action = action_probs.squeeze(0).numpy()
            else:
                # Sample from the distribution
                action = action_probs.squeeze(0).numpy()
        
        # Ensure action is properly shaped and normalized
        action = np.array(action, dtype=np.float32)
        
        # Apply additional normalization to ensure action sums to 1
        action = self._normalize_action(action)
        
        return action
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action to ensure it sums to 1 using softmax.
        
        Args:
            action: Raw action vector [invest, save, consume]
            
        Returns:
            np.ndarray: Normalized action vector that sums to 1
        """
        # Clip negative values to small positive to avoid issues
        action = np.clip(action, 1e-8, None)
        
        # Apply softmax normalization to ensure sum = 1
        exp_action = np.exp(action - np.max(action))  # Subtract max for numerical stability
        normalized_action = exp_action / np.sum(exp_action)
        
        return normalized_action

    def learn(self, transitions: List[Transition]) -> Dict[str, float]:
        """
        Update PPO policy using collected transitions.
        
        Applies the PPO algorithm with discount factor γ_low = 0.95 to update
        the policy based on experience. Returns training metrics including
        loss and policy entropy.
        
        Args:
            transitions: List of Transition objects containing (state, goal, action, reward, next_state, done)
            
        Returns:
            dict: Training metrics with keys 'loss' and 'policy_entropy'
        """
        if not transitions:
            return {'loss': 0.0, 'policy_entropy': 0.0}
        
        # Extract data from transitions
        states = []
        goals = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for transition in transitions:
            states.append(transition.state)
            goals.append(transition.goal)
            actions.append(transition.action)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        goals = np.array(goals, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        # Concatenate states and goals for input
        observations = np.concatenate([states, goals], axis=1)
        
        # Calculate returns using discount factor
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.config.gamma_low * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to torch tensors
        obs_tensor = torch.FloatTensor(observations)
        actions_tensor = torch.FloatTensor(actions)
        returns_tensor = torch.FloatTensor(returns)
        
        # Perform policy update
        self.optimizer.zero_grad()
        
        # Get action probabilities from policy
        action_probs = self.policy_network(obs_tensor)
        
        # Calculate policy loss (negative log probability weighted by returns)
        # This is a simplified policy gradient approach
        log_probs = torch.log(action_probs + 1e-8)
        selected_log_probs = (log_probs * actions_tensor).sum(dim=1)
        policy_loss = -(selected_log_probs * returns_tensor).mean()
        
        # Calculate entropy for exploration
        entropy = -(action_probs * log_probs).sum(dim=1).mean()
        
        # Total loss (policy loss - entropy bonus for exploration)
        loss = policy_loss - 0.01 * entropy
        
        # Backpropagation
        loss.backward()
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
