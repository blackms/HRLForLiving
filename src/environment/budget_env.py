"""BudgetEnv - Gymnasium environment for personal finance simulation"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.utils.config import EnvironmentConfig, RewardConfig
from src.environment.reward_engine import RewardEngine


class BudgetEnv(gym.Env):
    """
    Custom Gymnasium environment that simulates monthly financial decisions.
    
    The agent must allocate monthly income among investments, savings, and consumption
    while maintaining financial stability and avoiding negative cash balance.
    
    State Space (7-dimensional):
        - income: Monthly salary
        - fixed_expenses: Fixed monthly costs
        - variable_expenses: Sampled variable costs
        - cash_balance: Current liquid funds
        - inflation: Current inflation rate
        - risk_tolerance: Agent's risk profile
        - t_remaining: Months remaining in episode
    
    Action Space (3-dimensional, continuous [0, 1], sum=1):
        - invest_ratio: Percentage to invest
        - save_ratio: Percentage to save
        - consume_ratio: Percentage for discretionary spending
    """
    
    metadata = {'render_modes': []}
    
    def __init__(self, config: EnvironmentConfig, reward_config: RewardConfig = None):
        """
        Initialize the BudgetEnv with configuration parameters.
        
        Args:
            config: EnvironmentConfig containing simulation parameters
            reward_config: RewardConfig for reward computation (optional, uses default if None)
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        
        # Initialize RewardEngine
        if reward_config is None:
            reward_config = RewardConfig()
        self.reward_engine = RewardEngine(reward_config, safety_threshold=config.safety_threshold)
        
        # Define observation space (7-dimensional continuous)
        # [income, fixed_expenses, variable_expenses, cash_balance, inflation, risk_tolerance, t_remaining]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -np.inf, 0, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, 1, 1, config.max_months], dtype=np.float32),
            shape=(7,),
            dtype=np.float32
        )
        
        # Define action space (3-dimensional continuous [0, 1])
        # [invest_ratio, save_ratio, consume_ratio]
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.cash_balance = config.initial_cash
        self.current_month = 0
        self.total_invested = 0
        self.investment_value = 0  # Track investment portfolio value with returns
        
        # Store environment parameters
        self.income = config.income
        self.fixed_expenses = config.fixed_expenses
        self.inflation = config.inflation
        self.risk_tolerance = config.risk_tolerance
        self.max_months = config.max_months
        
        # Investment return parameters
        self.investment_return_mean = config.investment_return_mean
        self.investment_return_std = config.investment_return_std
        self.investment_return_type = config.investment_return_type
        
        # Current variable expense (sampled each step)
        self.current_variable_expense = 0

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            observation: Initial state observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        # Set numpy random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset state variables
        self.cash_balance = self.config.initial_cash
        self.current_month = 0
        self.total_invested = 0
        self.investment_value = 0  # Reset portfolio value
        
        # Reset environment parameters from config
        self.income = self.config.income
        self.fixed_expenses = self.config.fixed_expenses
        self.inflation = self.config.inflation
        
        # Sample initial variable expense
        self.current_variable_expense = self._sample_variable_expense()
        
        # Get initial state observation
        observation = self._get_state()
        info = {}
        
        return observation, info
    
    def _get_state(self):
        """
        Construct the current state observation vector.
        
        Returns:
            np.ndarray: 7-dimensional state vector (original values, not normalized)
        """
        t_remaining = self.max_months - self.current_month
        
        # Return original values - normalization will be done in the neural networks if needed
        state = np.array([
            self.income,
            self.fixed_expenses,
            self.current_variable_expense,
            self.cash_balance,
            self.inflation,
            self.risk_tolerance,
            t_remaining
        ], dtype=np.float32)
        
        return state

    def _normalize_action(self, action):
        """
        Normalize action to ensure it sums to 1 using softmax.
        Handles negative or invalid action values.
        
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

    def _sample_variable_expense(self):
        """
        Sample variable expenses from a normal distribution.
        
        Returns:
            float: Sampled variable expense amount
        """
        expense = np.random.normal(
            self.config.variable_expense_mean,
            self.config.variable_expense_std
        )
        # Ensure non-negative expense
        return max(0, expense)
    
    def _apply_inflation(self):
        """
        Apply inflation adjustments to expenses.
        Called at each step to simulate economic changes.
        """
        self.fixed_expenses *= (1 + self.inflation)
        # Variable expense mean is adjusted through config, but we apply to current sample
        self.current_variable_expense *= (1 + self.inflation)
    
    def _apply_investment_returns(self):
        """
        Apply investment returns to the portfolio.
        
        Returns can be:
        - "none": No returns (0%)
        - "fixed": Fixed monthly return (investment_return_mean)
        - "stochastic": Normally distributed returns (mean, std)
        
        Returns:
            float: Monthly return amount in EUR
        """
        if self.investment_return_type == "none" or self.investment_value == 0:
            return 0.0
        
        if self.investment_return_type == "fixed":
            monthly_return_rate = self.investment_return_mean
        elif self.investment_return_type == "stochastic":
            # Sample from normal distribution
            monthly_return_rate = np.random.normal(
                self.investment_return_mean,
                self.investment_return_std
            )
        else:
            raise ValueError(f"Unknown investment_return_type: {self.investment_return_type}")
        
        # Calculate return amount
        return_amount = self.investment_value * monthly_return_rate
        
        # Update investment value
        self.investment_value += return_amount
        
        return return_amount

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action: Action vector [invest_ratio, save_ratio, consume_ratio]
            
        Returns:
            observation: Next state observation
            reward: Reward for this step
            terminated: Whether episode has ended
            truncated: Whether episode was truncated (max steps)
            info: Additional information dictionary
        """
        # Store current state before action
        current_state = self._get_state()
        
        # Normalize action to ensure it sums to 1
        action = self._normalize_action(action)
        invest_ratio, save_ratio, consume_ratio = action
        
        # Calculate allocation amounts based on income
        invest_amount = invest_ratio * self.income
        save_amount = save_ratio * self.income
        consume_amount = consume_ratio * self.income
        
        # Sample new variable expense for this month
        self.current_variable_expense = self._sample_variable_expense()
        
        # Apply inflation to expenses
        self._apply_inflation()
        
        # Apply investment returns BEFORE new investment
        investment_return = self._apply_investment_returns()
        
        # Calculate total expenses
        total_expenses = self.fixed_expenses + self.current_variable_expense
        
        # Update cash balance
        # Cash increases by income, decreases by expenses and investment
        # Savings stay in cash, consumption is part of expenses (discretionary)
        self.cash_balance = (
            self.cash_balance 
            + self.income 
            - self.fixed_expenses 
            - self.current_variable_expense 
            - invest_amount
        )
        
        # Track total invested (principal only)
        self.total_invested += invest_amount
        
        # Add new investment to portfolio value
        self.investment_value += invest_amount
        
        # Increment month counter
        self.current_month += 1
        
        # Get next state after action
        next_state = self._get_state()
        
        # Calculate reward using RewardEngine
        reward = self.reward_engine.compute_low_level_reward(action, current_state, next_state)
        
        # Check termination conditions
        terminated = self.cash_balance < 0  # Negative cash balance
        truncated = self.current_month >= self.max_months  # Max months reached
        
        # Additional info
        info = {
            'cash_balance': self.cash_balance,
            'total_invested': self.total_invested,  # Principal invested
            'investment_value': self.investment_value,  # Portfolio value with returns
            'investment_return': investment_return,  # This month's return
            'month': self.current_month,
            'action': action,
            'invest_amount': invest_amount,
            'total_expenses': total_expenses
        }
        
        return next_state, reward, terminated, truncated, info

