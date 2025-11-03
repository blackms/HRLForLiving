"""Unit tests for BudgetEnv"""
import pytest
import numpy as np
from src.environment.budget_env import BudgetEnv
from src.utils.config import EnvironmentConfig


class TestBudgetEnv:
    """Test suite for BudgetEnv class"""
    
    @pytest.fixture
    def default_config(self):
        """Create default environment configuration"""
        return EnvironmentConfig(
            income=3200,
            fixed_expenses=1400,
            variable_expense_mean=700,
            variable_expense_std=100,
            inflation=0.02,
            safety_threshold=1000,
            max_months=60,
            initial_cash=0,
            risk_tolerance=0.5
        )
    
    @pytest.fixture
    def env(self, default_config):
        """Create BudgetEnv instance"""
        return BudgetEnv(default_config)
    
    def test_initialization(self, env, default_config):
        """Test environment initialization"""
        assert env.config == default_config
        assert env.cash_balance == default_config.initial_cash
        assert env.current_month == 0
        assert env.total_invested == 0
        assert env.observation_space.shape == (7,)
        assert env.action_space.shape == (3,)
    
    def test_reset(self, env):
        """Test reset functionality"""
        # Modify state
        env.cash_balance = 5000
        env.current_month = 10
        env.total_invested = 2000
        
        # Reset environment
        observation, info = env.reset()
        
        # Check state is reset
        assert env.cash_balance == 0
        assert env.current_month == 0
        assert env.total_invested == 0
        assert observation.shape == (7,)
        assert isinstance(info, dict)
    
    def test_reset_with_seed(self, env):
        """Test reset with seed for reproducibility"""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        # Variable expense should be the same with same seed
        assert obs1[2] == obs2[2]  # variable_expense
    
    def test_get_state(self, env):
        """Test state observation construction"""
        env.reset()
        state = env._get_state()
        
        assert state.shape == (7,)
        assert np.isclose(state[0], env.income)
        assert np.isclose(state[1], env.fixed_expenses)
        assert np.isclose(state[3], env.cash_balance)
        assert np.isclose(state[4], env.inflation)
        assert np.isclose(state[5], env.risk_tolerance)
        assert state[6] == env.max_months - env.current_month
    
    def test_action_normalization_valid(self, env):
        """Test action normalization with valid inputs"""
        action = np.array([0.5, 0.3, 0.2])
        normalized = env._normalize_action(action)
        
        assert normalized.shape == (3,)
        assert np.isclose(np.sum(normalized), 1.0)
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
    
    def test_action_normalization_negative(self, env):
        """Test action normalization handles negative values"""
        action = np.array([-0.5, 0.3, 0.2])
        normalized = env._normalize_action(action)
        
        assert normalized.shape == (3,)
        assert np.isclose(np.sum(normalized), 1.0)
        assert np.all(normalized >= 0)
    
    def test_action_normalization_unnormalized(self, env):
        """Test action normalization with unnormalized input"""
        action = np.array([2.0, 3.0, 1.0])
        normalized = env._normalize_action(action)
        
        assert normalized.shape == (3,)
        assert np.isclose(np.sum(normalized), 1.0)
        assert np.all(normalized >= 0)
    
    def test_variable_expense_sampling(self, env):
        """Test variable expense sampling distribution"""
        env.reset(seed=42)
        samples = [env._sample_variable_expense() for _ in range(1000)]
        
        # Check samples are non-negative
        assert all(s >= 0 for s in samples)
        
        # Check mean is approximately correct (within reasonable range)
        mean = np.mean(samples)
        assert 600 < mean < 800  # Should be around 700
    
    def test_step_basic(self, env):
        """Test basic step execution"""
        env.reset()
        action = np.array([0.3, 0.4, 0.3])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (7,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        assert env.current_month == 1
    
    def test_step_cash_balance_update(self, env):
        """Test cash balance updates correctly"""
        env.reset()
        initial_cash = env.cash_balance
        action = np.array([0.2, 0.5, 0.3])  # 20% invest, 50% save, 30% consume
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Cash should increase by income minus expenses and investment
        expected_invest = 0.2 * env.income
        assert info['invest_amount'] > 0
        assert env.total_invested > 0
    
    def test_termination_negative_cash(self, env):
        """Test episode terminates on negative cash balance"""
        config = EnvironmentConfig(
            income=1000,
            fixed_expenses=2000,  # Expenses exceed income
            variable_expense_mean=500,
            variable_expense_std=50,
            max_months=60
        )
        env = BudgetEnv(config)
        env.reset()
        
        action = np.array([0.5, 0.3, 0.2])  # High investment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should terminate due to negative cash
        assert terminated or env.cash_balance < 0
    
    def test_truncation_max_months(self, env):
        """Test episode truncates at max months"""
        config = EnvironmentConfig(max_months=5)
        env = BudgetEnv(config)
        env.reset()
        
        action = np.array([0.1, 0.5, 0.4])
        
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(action)
            if i < 4:
                assert not truncated
            else:
                assert truncated
    
    def test_inflation_application(self, env):
        """Test inflation is applied to expenses"""
        env.reset()
        initial_fixed = env.fixed_expenses
        initial_variable = env.current_variable_expense
        
        action = np.array([0.3, 0.4, 0.3])
        env.step(action)
        
        # Expenses should increase due to inflation
        assert env.fixed_expenses > initial_fixed
    
    def test_info_dict_contents(self, env):
        """Test info dictionary contains expected keys"""
        env.reset()
        action = np.array([0.3, 0.4, 0.3])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert 'cash_balance' in info
        assert 'total_invested' in info
        assert 'month' in info
        assert 'action' in info
        assert 'invest_amount' in info
        assert 'total_expenses' in info
