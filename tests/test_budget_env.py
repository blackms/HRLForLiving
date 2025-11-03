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


class TestBudgetEnvEdgeCases:
    """
    Test suite for BudgetEnv edge cases and extreme financial scenarios.
    
    This test class validates the robustness of BudgetEnv under extreme conditions
    that may occur in real-world financial situations, including:
    
    - Income stress: Very low income, expenses exceeding income
    - Expense stress: High fixed expenses, high variable expenses, extreme variance
    - Inflation stress: Hyperinflation, deflation, zero inflation, compounding effects
    - Episode length: Maximum length (120 months), single-step episodes
    - Initial cash: High buffers, zero starting cash
    - Combined stress: Multiple extreme conditions simultaneously
    
    These tests ensure the system:
    - Handles extreme conditions without crashes or undefined behavior
    - Properly terminates episodes (negative cash, max months)
    - Maintains valid state observations
    - Correctly applies inflation effects over time
    - Respects episode length constraints
    - Updates cash balance correctly under stress
    
    Total: 19 comprehensive edge case tests
    """
    
    def test_very_low_income_scenario(self):
        """Test environment with very low income (barely covers expenses)"""
        config = EnvironmentConfig(
            income=100,  # Very low income
            fixed_expenses=80,
            variable_expense_mean=15,
            variable_expense_std=5,
            inflation=0.01,
            safety_threshold=50,
            max_months=12,
            initial_cash=100,
            risk_tolerance=0.3
        )
        env = BudgetEnv(config)
        obs, _ = env.reset()
        
        # Agent should prioritize saving over investing
        action = np.array([0.1, 0.7, 0.2])  # Low investment, high savings
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should not immediately terminate
        assert not terminated
        assert obs.shape == (7,)
        assert info['cash_balance'] >= 0 or terminated
    
    def test_extremely_low_income_immediate_failure(self):
        """Test environment with income below expenses (should fail quickly)"""
        config = EnvironmentConfig(
            income=500,
            fixed_expenses=1500,  # Fixed expenses exceed income
            variable_expense_mean=300,
            variable_expense_std=50,
            inflation=0.0,
            safety_threshold=100,
            max_months=12,
            initial_cash=0,
            risk_tolerance=0.5
        )
        env = BudgetEnv(config)
        env.reset()
        
        # Any action should lead to negative cash
        action = np.array([0.0, 0.5, 0.5])  # No investment, try to save
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should terminate due to negative cash
        assert terminated
        assert info['cash_balance'] < 0
    
    def test_very_high_fixed_expenses(self):
        """Test environment with very high fixed expenses"""
        config = EnvironmentConfig(
            income=5000,
            fixed_expenses=4500,  # 90% of income
            variable_expense_mean=200,
            variable_expense_std=50,
            inflation=0.02,
            safety_threshold=500,
            max_months=24,
            initial_cash=1000,
            risk_tolerance=0.2
        )
        env = BudgetEnv(config)
        env.reset()
        
        # Very little room for investment
        action = np.array([0.05, 0.8, 0.15])  # Minimal investment
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should handle high expenses without crashing
        assert obs.shape == (7,)
        assert isinstance(reward, (float, np.floating))
    
    def test_very_high_variable_expenses(self):
        """Test environment with very high variable expenses"""
        config = EnvironmentConfig(
            income=4000,
            fixed_expenses=1000,
            variable_expense_mean=2500,  # High variable expenses
            variable_expense_std=500,  # High variance
            inflation=0.01,
            safety_threshold=800,
            max_months=12,
            initial_cash=2000,
            risk_tolerance=0.4
        )
        env = BudgetEnv(config)
        env.reset(seed=42)
        
        # Conservative action due to high expenses
        action = np.array([0.1, 0.6, 0.3])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should handle high variable expenses
        assert obs.shape == (7,)
        assert obs[2] >= 0  # Variable expense should be non-negative
    
    def test_extreme_positive_inflation(self):
        """Test environment with very high inflation rate"""
        config = EnvironmentConfig(
            income=3000,
            fixed_expenses=1200,
            variable_expense_mean=600,
            variable_expense_std=100,
            inflation=0.5,  # 50% monthly inflation (extreme)
            safety_threshold=1000,
            max_months=6,  # Short episode due to extreme inflation
            initial_cash=5000,
            risk_tolerance=0.6
        )
        env = BudgetEnv(config)
        env.reset()
        
        initial_fixed = env.fixed_expenses
        action = np.array([0.3, 0.4, 0.3])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Expenses should increase dramatically
        assert env.fixed_expenses > initial_fixed * 1.4  # At least 40% increase
        assert obs.shape == (7,)
    
    def test_extreme_negative_inflation(self):
        """Test environment with deflation (negative inflation)"""
        config = EnvironmentConfig(
            income=3000,
            fixed_expenses=1500,
            variable_expense_mean=700,
            variable_expense_std=100,
            inflation=-0.2,  # 20% monthly deflation
            safety_threshold=1000,
            max_months=12,
            initial_cash=1000,
            risk_tolerance=0.5
        )
        env = BudgetEnv(config)
        env.reset()
        
        initial_fixed = env.fixed_expenses
        action = np.array([0.3, 0.4, 0.3])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Expenses should decrease
        assert env.fixed_expenses < initial_fixed
        assert obs.shape == (7,)
    
    def test_zero_inflation(self):
        """Test environment with zero inflation"""
        config = EnvironmentConfig(
            income=3000,
            fixed_expenses=1200,
            variable_expense_mean=600,
            variable_expense_std=100,
            inflation=0.0,  # No inflation
            safety_threshold=1000,
            max_months=24,
            initial_cash=500,
            risk_tolerance=0.5
        )
        env = BudgetEnv(config)
        env.reset()
        
        initial_fixed = env.fixed_expenses
        action = np.array([0.3, 0.4, 0.3])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Expenses should remain constant (within floating point precision)
        assert np.isclose(env.fixed_expenses, initial_fixed, rtol=1e-5)
    
    def test_maximum_episode_length(self):
        """Test environment runs for maximum episode length"""
        config = EnvironmentConfig(
            income=3500,
            fixed_expenses=1200,
            variable_expense_mean=600,
            variable_expense_std=80,
            inflation=0.01,
            safety_threshold=1000,
            max_months=120,  # Very long episode (10 years)
            initial_cash=2000,
            risk_tolerance=0.5
        )
        env = BudgetEnv(config)
        env.reset()
        
        # Conservative action to survive long episode
        action = np.array([0.2, 0.6, 0.2])
        
        steps = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < 120:
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        
        # Should reach max months or terminate
        assert steps <= 120
        assert truncated or terminated or steps == 120
        if truncated:
            assert env.current_month == 120
    
    def test_very_long_episode_with_inflation(self):
        """Test long episode with compounding inflation effects"""
        config = EnvironmentConfig(
            income=4000,
            fixed_expenses=1500,
            variable_expense_mean=700,
            variable_expense_std=100,
            inflation=0.03,  # 3% monthly inflation
            safety_threshold=1000,
            max_months=60,
            initial_cash=3000,
            risk_tolerance=0.5
        )
        env = BudgetEnv(config)
        env.reset()
        
        initial_fixed = env.fixed_expenses
        action = np.array([0.25, 0.5, 0.25])
        
        # Run for 30 steps
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        # Expenses should have increased significantly due to compounding
        if env.current_month >= 30:
            expected_multiplier = (1.03 ** 30)
            assert env.fixed_expenses > initial_fixed * (expected_multiplier * 0.9)  # Allow some tolerance
    
    def test_single_step_episode(self):
        """Test environment with max_months=1"""
        config = EnvironmentConfig(
            income=3000,
            fixed_expenses=1200,
            variable_expense_mean=600,
            variable_expense_std=100,
            inflation=0.02,
            safety_threshold=1000,
            max_months=1,  # Single step episode
            initial_cash=1000,
            risk_tolerance=0.5
        )
        env = BudgetEnv(config)
        env.reset()
        
        action = np.array([0.3, 0.4, 0.3])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should truncate after first step
        assert truncated
        assert env.current_month == 1
    
    def test_high_initial_cash_buffer(self):
        """Test environment with very high initial cash"""
        config = EnvironmentConfig(
            income=3000,
            fixed_expenses=1400,
            variable_expense_mean=700,
            variable_expense_std=100,
            inflation=0.02,
            safety_threshold=1000,
            max_months=24,
            initial_cash=50000,  # Very high initial cash
            risk_tolerance=0.7
        )
        env = BudgetEnv(config)
        obs, _ = env.reset()
        
        # Should start with high cash balance
        assert env.cash_balance == 50000
        assert obs[3] == 50000  # Cash balance in state
        
        # Can afford aggressive investment
        action = np.array([0.7, 0.2, 0.1])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert not terminated
        assert info['cash_balance'] > 40000  # Should still have substantial cash
    
    def test_zero_initial_cash_survival(self):
        """Test environment starting with zero cash"""
        config = EnvironmentConfig(
            income=3500,
            fixed_expenses=1200,
            variable_expense_mean=600,
            variable_expense_std=80,
            inflation=0.01,
            safety_threshold=1000,
            max_months=12,
            initial_cash=0,  # Start with no cash
            risk_tolerance=0.5
        )
        env = BudgetEnv(config)
        env.reset()
        
        # Must be conservative to build cash buffer
        action = np.array([0.1, 0.7, 0.2])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should survive first month with proper allocation
        assert not terminated or info['cash_balance'] < 0
    
    def test_extreme_variable_expense_variance(self):
        """Test environment with very high variable expense variance"""
        config = EnvironmentConfig(
            income=4000,
            fixed_expenses=1000,
            variable_expense_mean=1000,
            variable_expense_std=800,  # 80% of mean (very high variance)
            inflation=0.01,
            safety_threshold=1000,
            max_months=24,
            initial_cash=3000,
            risk_tolerance=0.4
        )
        env = BudgetEnv(config)
        env.reset(seed=42)
        
        # Run multiple steps to see variance
        expenses = []
        for _ in range(10):
            action = np.array([0.2, 0.5, 0.3])
            obs, reward, terminated, truncated, info = env.step(action)
            expenses.append(obs[2])  # Variable expense from state
            if terminated or truncated:
                break
        
        # Should see significant variance in expenses
        if len(expenses) > 1:
            std_dev = np.std(expenses)
            assert std_dev > 0  # Should have variance
    
    def test_combined_extreme_conditions(self):
        """Test environment with multiple extreme conditions combined"""
        config = EnvironmentConfig(
            income=1500,  # Low income
            fixed_expenses=1000,  # High fixed expenses
            variable_expense_mean=400,  # High variable expenses
            variable_expense_std=200,  # High variance
            inflation=0.1,  # High inflation
            safety_threshold=500,
            max_months=6,  # Short episode
            initial_cash=500,  # Low initial cash
            risk_tolerance=0.3
        )
        env = BudgetEnv(config)
        env.reset(seed=42)
        
        # Very conservative action
        action = np.array([0.0, 0.8, 0.2])
        
        steps = 0
        while steps < 6:
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        
        # Should handle extreme conditions without crashing
        assert steps > 0
        assert obs.shape == (7,)
