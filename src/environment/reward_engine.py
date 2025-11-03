"""Reward Engine for computing multi-objective rewards"""
import numpy as np
from typing import List
from src.utils.config import RewardConfig
from src.utils.data_models import Transition


class RewardEngine:
    """
    Computes multi-objective rewards for both high-level and low-level agents.
    
    The reward function balances multiple objectives:
    - Encouraging investment for long-term wealth growth
    - Maintaining financial stability (positive cash balance)
    - Penalizing overspending and debt
    - Rewarding consistent positive balance over time
    """
    
    def __init__(self, config: RewardConfig, safety_threshold: float = 1000):
        """
        Initialize the RewardEngine with reward coefficients.
        
        Args:
            config: RewardConfig containing coefficients (α, β, γ, δ, λ, μ)
            safety_threshold: Minimum cash balance to avoid stability penalty
        """
        self.config = config
        self.safety_threshold = safety_threshold
        
        # Store coefficients for easy access
        self.alpha = config.alpha      # Investment reward coefficient
        self.beta = config.beta        # Stability penalty coefficient
        self.gamma = config.gamma      # Overspend penalty coefficient
        self.delta = config.delta      # Debt penalty coefficient
        self.lambda_ = config.lambda_  # Wealth growth coefficient
        self.mu = config.mu            # Stability bonus coefficient

    def compute_low_level_reward(
        self, 
        action: np.ndarray, 
        state: np.ndarray, 
        next_state: np.ndarray
    ) -> float:
        """
        Compute immediate monthly reward for the low-level agent.
        
        The reward function combines multiple objectives:
        - Investment reward: α * invest_amount
        - Stability penalty: β * max(0, threshold - cash)
        - Overspend penalty: γ * overspend
        - Debt penalty: δ * abs(min(0, cash))
        
        Args:
            action: Action vector [invest_ratio, save_ratio, consume_ratio]
            state: Current state vector [income, fixed, variable, cash, inflation, risk, t_remaining]
            next_state: Next state vector after action
            
        Returns:
            float: Combined reward value
        """
        # Extract relevant values from state
        income = state[0]
        current_cash = state[3]
        
        # Extract next state cash balance
        next_cash = next_state[3]
        
        # Calculate investment amount from action
        invest_ratio = action[0]
        invest_amount = invest_ratio * income
        
        # 1. Investment reward: Encourage investment
        investment_reward = self.alpha * invest_amount
        
        # 2. Stability penalty: Penalize low cash balance
        stability_penalty = 0
        if next_cash < self.safety_threshold:
            stability_penalty = self.beta * (self.safety_threshold - next_cash)
        
        # 3. Overspend penalty: Penalize when expenses exceed available funds
        # Overspend occurs when cash decreases more than expected from normal operations
        overspend = 0
        cash_change = next_cash - current_cash
        # If cash decreased significantly beyond investment, it's overspending
        expected_decrease = invest_amount
        if cash_change < -expected_decrease:
            overspend = abs(cash_change + expected_decrease)
        overspend_penalty = self.gamma * overspend
        
        # 4. Debt penalty: Heavily penalize negative cash balance
        debt_penalty = 0
        if next_cash < 0:
            debt_penalty = self.delta * abs(next_cash)
        
        # Combine all components
        reward = (
            investment_reward 
            - stability_penalty 
            - overspend_penalty 
            - debt_penalty
        )
        
        return reward

    def compute_high_level_reward(self, episode_history: List[Transition]) -> float:
        """
        Compute strategic reward for the high-level agent over a period.
        
        The high-level reward aggregates low-level rewards and adds:
        - Wealth change term: λ * Δwealth
        - Stability bonus: μ * stability_bonus for consistent positive balance
        
        Args:
            episode_history: List of Transition objects over the strategic period
            
        Returns:
            float: Aggregated high-level reward
        """
        if not episode_history:
            return 0.0
        
        # 1. Aggregate low-level rewards over the period
        total_low_level_reward = sum(transition.reward for transition in episode_history)
        
        # 2. Calculate wealth change (Δwealth)
        # Wealth is measured by cash balance change over the period
        initial_state = episode_history[0].state
        final_state = episode_history[-1].next_state
        
        initial_cash = initial_state[3]
        final_cash = final_state[3]
        wealth_change = final_cash - initial_cash
        
        # Wealth growth reward
        wealth_reward = self.lambda_ * wealth_change
        
        # 3. Calculate stability bonus
        # Count months with positive balance
        positive_balance_count = 0
        for transition in episode_history:
            next_cash = transition.next_state[3]
            if next_cash > 0:
                positive_balance_count += 1
        
        # Stability bonus: reward consistent positive balance
        stability_ratio = positive_balance_count / len(episode_history)
        stability_bonus = self.mu * stability_ratio * len(episode_history)
        
        # Combine all components
        high_level_reward = (
            total_low_level_reward 
            + wealth_reward 
            + stability_bonus
        )
        
        return high_level_reward
