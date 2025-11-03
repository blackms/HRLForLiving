"""Configuration dataclasses and behavioral profiles"""
from dataclasses import dataclass
from enum import Enum


@dataclass
class EnvironmentConfig:
    """Configuration for the financial environment simulation"""
    income: float = 3200
    fixed_expenses: float = 1400
    variable_expense_mean: float = 700
    variable_expense_std: float = 100
    inflation: float = 0.02
    safety_threshold: float = 1000
    max_months: int = 60
    initial_cash: float = 0
    risk_tolerance: float = 0.5


@dataclass
class TrainingConfig:
    """Configuration for training the HRL system"""
    num_episodes: int = 5000
    gamma_low: float = 0.95
    gamma_high: float = 0.99
    high_period: int = 6
    batch_size: int = 32
    learning_rate_low: float = 3e-4
    learning_rate_high: float = 1e-4


@dataclass
class RewardConfig:
    """Configuration for reward computation"""
    alpha: float = 10.0    # Investment reward coefficient
    beta: float = 0.1      # Stability penalty coefficient
    gamma: float = 5.0     # Overspend penalty coefficient
    delta: float = 20.0    # Debt penalty coefficient
    lambda_: float = 1.0   # Wealth growth coefficient
    mu: float = 0.5        # Stability bonus coefficient


class BehavioralProfile(Enum):
    """Predefined risk profiles for different financial strategies"""
    CONSERVATIVE = {
        'risk_tolerance': 0.3,
        'safety_threshold': 1500,
        'reward_config': {
            'alpha': 5.0,
            'beta': 0.5,
            'gamma': 5.0,
            'delta': 20.0,
            'lambda_': 1.0,
            'mu': 0.5
        }
    }
    BALANCED = {
        'risk_tolerance': 0.5,
        'safety_threshold': 1000,
        'reward_config': {
            'alpha': 10.0,
            'beta': 0.1,
            'gamma': 5.0,
            'delta': 20.0,
            'lambda_': 1.0,
            'mu': 0.5
        }
    }
    AGGRESSIVE = {
        'risk_tolerance': 0.8,
        'safety_threshold': 500,
        'reward_config': {
            'alpha': 15.0,
            'beta': 0.05,
            'gamma': 5.0,
            'delta': 20.0,
            'lambda_': 1.0,
            'mu': 0.5
        }
    }
