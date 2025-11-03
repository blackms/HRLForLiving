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
