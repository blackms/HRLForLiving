"""Analytics module for tracking and computing performance metrics"""
import numpy as np
from typing import List, Dict, Optional


class AnalyticsModule:
    """
    Tracks and computes performance metrics for HRL financial system.
    
    Metrics computed:
    - Cumulative wealth growth (total invested)
    - Cash stability index (% months with positive balance)
    - Sharpe-like ratio (mean return / std balance)
    - Goal adherence (mean absolute difference)
    - Policy stability (variance of actions)
    """
    
    def __init__(self):
        """Initialize metric trackers"""
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.cash_balances: List[float] = []
        self.goals: List[np.ndarray] = []
        self.invested_amounts: List[float] = []

    def record_step(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float,
        goal: Optional[np.ndarray] = None,
        invested_amount: Optional[float] = None
    ):
        """
        Record data from a single step.
        
        Args:
            state: Current state observation
            action: Action taken [invest, save, consume]
            reward: Reward received
            goal: Optional goal vector from high-level agent
            invested_amount: Optional amount invested this step
        """
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(float(reward))
        
        # Extract cash balance from state (index 3)
        if len(state) >= 4:
            self.cash_balances.append(float(state[3]))
        
        if goal is not None:
            self.goals.append(goal.copy())
        
        if invested_amount is not None:
            self.invested_amounts.append(float(invested_amount))

    def compute_episode_metrics(self) -> Dict[str, float]:
        """
        Compute all performance metrics for the completed episode.
        
        Returns:
            Dictionary containing:
            - cumulative_wealth_growth: Total invested capital
            - cash_stability_index: Percentage of months with positive balance
            - sharpe_ratio: Mean return / std balance
            - goal_adherence: Mean absolute difference between target and actual
            - policy_stability: Variance of actions over time
        """
        metrics = {}
        
        # Cumulative wealth growth (total invested)
        if self.invested_amounts:
            metrics['cumulative_wealth_growth'] = sum(self.invested_amounts)
        else:
            metrics['cumulative_wealth_growth'] = 0.0
        
        # Cash stability index (% months with positive balance)
        if self.cash_balances:
            positive_months = sum(1 for cash in self.cash_balances if cash > 0)
            metrics['cash_stability_index'] = positive_months / len(self.cash_balances)
        else:
            metrics['cash_stability_index'] = 0.0
        
        # Sharpe-like ratio (mean return / std balance)
        if len(self.cash_balances) > 1:
            mean_balance = np.mean(self.cash_balances)
            std_balance = np.std(self.cash_balances)
            if std_balance > 0:
                metrics['sharpe_ratio'] = mean_balance / std_balance
            else:
                metrics['sharpe_ratio'] = 0.0
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Goal adherence (mean absolute difference between target and actual)
        if self.goals and self.actions:
            # Compare target_invest_ratio (goal[0]) with actual invest action (action[0])
            min_len = min(len(self.goals), len(self.actions))
            if min_len > 0:
                differences = []
                for i in range(min_len):
                    target_invest = self.goals[i][0]  # target_invest_ratio
                    actual_invest = self.actions[i][0]  # invest action
                    differences.append(abs(target_invest - actual_invest))
                metrics['goal_adherence'] = np.mean(differences)
            else:
                metrics['goal_adherence'] = 0.0
        else:
            metrics['goal_adherence'] = 0.0
        
        # Policy stability (variance of actions over time)
        if len(self.actions) > 1:
            actions_array = np.array(self.actions)
            # Compute variance across all action dimensions
            metrics['policy_stability'] = np.mean(np.var(actions_array, axis=0))
        else:
            metrics['policy_stability'] = 0.0
        
        return metrics

    def reset(self):
        """Clear all episode data for a new episode"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.cash_balances.clear()
        self.goals.clear()
        self.invested_amounts.clear()
