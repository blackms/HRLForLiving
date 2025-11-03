"""Unit tests for RewardEngine"""
import pytest
import numpy as np
from src.environment.reward_engine import RewardEngine
from src.utils.config import RewardConfig
from src.utils.data_models import Transition


class TestRewardEngine:
    """Test suite for RewardEngine class"""
    
    @pytest.fixture
    def default_config(self):
        """Create default reward configuration"""
        return RewardConfig(
            alpha=10.0,
            beta=0.1,
            gamma=5.0,
            delta=20.0,
            lambda_=1.0,
            mu=0.5
        )
    
    @pytest.fixture
    def reward_engine(self, default_config):
        """Create RewardEngine instance"""
        return RewardEngine(default_config, safety_threshold=1000)
    
    def test_initialization(self, reward_engine, default_config):
        """Test reward engine initialization"""
        assert reward_engine.config == default_config
        assert reward_engine.safety_threshold == 1000
        assert reward_engine.alpha == 10.0
        assert reward_engine.beta == 0.1
        assert reward_engine.gamma == 5.0
        assert reward_engine.delta == 20.0
        assert reward_engine.lambda_ == 1.0
        assert reward_engine.mu == 0.5
    
    def test_low_level_reward_high_investment(self, reward_engine):
        """Test low-level reward with high investment scenario"""
        # State: [income, fixed, variable, cash, inflation, risk, t_remaining]
        state = np.array([3200, 1400, 700, 2000, 0.02, 0.5, 50])
        # Action: [invest, save, consume] - high investment
        action = np.array([0.5, 0.3, 0.2])
        # Next state with reduced cash due to investment
        next_state = np.array([3200, 1400, 700, 1500, 0.02, 0.5, 49])
        
        reward = reward_engine.compute_low_level_reward(action, state, next_state)
        
        # Should have positive reward from investment
        # Investment amount = 0.5 * 3200 = 1600
        # Investment reward = 10.0 * 1600 = 16000
        expected_investment_reward = 10.0 * 1600
        assert reward > 0
        assert reward >= expected_investment_reward * 0.9  # Allow for small penalties
    
    def test_low_level_reward_low_cash(self, reward_engine):
        """Test low-level reward with low cash balance"""
        # State with decent cash
        state = np.array([3200, 1400, 700, 1200, 0.02, 0.5, 50])
        # Action with moderate investment
        action = np.array([0.3, 0.4, 0.3])
        # Next state with cash below safety threshold
        next_state = np.array([3200, 1400, 700, 800, 0.02, 0.5, 49])
        
        reward = reward_engine.compute_low_level_reward(action, state, next_state)
        
        # Should have stability penalty
        # Stability penalty = 0.1 * (1000 - 800) = 20
        expected_penalty = 0.1 * (1000 - 800)
        # Investment reward = 10.0 * (0.3 * 3200) = 9600
        expected_investment = 10.0 * (0.3 * 3200)
        
        # Reward should be positive but reduced by penalty
        assert reward < expected_investment
        assert reward > 0  # Still positive due to high investment reward
    
    def test_low_level_reward_overspend(self, reward_engine):
        """Test low-level reward with overspending scenario"""
        # State with good cash
        state = np.array([3200, 1400, 700, 3000, 0.02, 0.5, 50])
        # Action with low investment, high consumption
        action = np.array([0.1, 0.2, 0.7])
        # Next state with significant cash decrease beyond investment
        next_state = np.array([3200, 1400, 700, 1500, 0.02, 0.5, 49])
        
        reward = reward_engine.compute_low_level_reward(action, state, next_state)
        
        # Investment amount = 0.1 * 3200 = 320
        # Cash change = 1500 - 3000 = -1500
        # Expected decrease from investment = 320
        # Overspend = abs(-1500 + 320) = 1180
        # Overspend penalty = 5.0 * 1180 = 5900
        
        # Reward should be negative or very low due to overspend penalty
        assert reward < 5000  # Much less than investment reward alone
    
    def test_low_level_reward_debt(self, reward_engine):
        """Test low-level reward with negative cash (debt)"""
        # State with low cash
        state = np.array([3200, 1400, 700, 500, 0.02, 0.5, 50])
        # Action with high investment
        action = np.array([0.6, 0.2, 0.2])
        # Next state with negative cash
        next_state = np.array([3200, 1400, 700, -500, 0.02, 0.5, 49])
        
        reward = reward_engine.compute_low_level_reward(action, state, next_state)
        
        # Debt penalty = 20.0 * abs(-500) = 10000
        # Investment reward = 10.0 * (0.6 * 3200) = 19200
        # Even with high investment, debt penalty should significantly reduce reward
        # Calculate reward without debt penalty for comparison
        state_no_debt = state.copy()
        next_state_no_debt = next_state.copy()
        next_state_no_debt[3] = 1500  # Positive cash
        reward_no_debt = reward_engine.compute_low_level_reward(action, state_no_debt, next_state_no_debt)
        
        # Reward with debt should be much lower than without debt
        assert reward < reward_no_debt - 5000  # At least 5000 penalty applied
    
    def test_low_level_reward_stable_scenario(self, reward_engine):
        """Test low-level reward with stable, balanced scenario"""
        # State with good cash above threshold
        state = np.array([3200, 1400, 700, 2500, 0.02, 0.5, 50])
        # Balanced action
        action = np.array([0.3, 0.4, 0.3])
        # Next state maintaining good cash
        next_state = np.array([3200, 1400, 700, 2200, 0.02, 0.5, 49])
        
        reward = reward_engine.compute_low_level_reward(action, state, next_state)
        
        # Should have positive reward with minimal penalties
        # Investment reward = 10.0 * (0.3 * 3200) = 9600
        assert reward > 9000  # Close to full investment reward
    
    def test_high_level_reward_aggregation(self, reward_engine):
        """Test high-level reward aggregates low-level rewards"""
        # Create episode history with multiple transitions
        episode_history = []
        for i in range(6):
            state = np.array([3200, 1400, 700, 2000 + i*100, 0.02, 0.5, 50-i])
            goal = np.array([0.3, 1000, 0.5])
            action = np.array([0.3, 0.4, 0.3])
            reward = 100.0  # Arbitrary low-level reward
            next_state = np.array([3200, 1400, 700, 2100 + i*100, 0.02, 0.5, 49-i])
            done = False
            
            transition = Transition(state, goal, action, reward, next_state, done)
            episode_history.append(transition)
        
        high_reward = reward_engine.compute_high_level_reward(episode_history)
        
        # Should aggregate 6 * 100 = 600 from low-level rewards
        # Plus wealth change and stability bonus
        assert high_reward > 600  # At least the sum of low-level rewards
    
    def test_high_level_reward_wealth_growth(self, reward_engine):
        """Test high-level reward includes wealth change term"""
        # Create episode with significant wealth growth
        episode_history = []
        initial_cash = 1000
        final_cash = 3000
        
        for i in range(6):
            cash = initial_cash + (final_cash - initial_cash) * i / 5
            next_cash = initial_cash + (final_cash - initial_cash) * (i + 1) / 5
            
            state = np.array([3200, 1400, 700, cash, 0.02, 0.5, 50-i])
            goal = np.array([0.3, 1000, 0.5])
            action = np.array([0.3, 0.4, 0.3])
            reward = 50.0
            next_state = np.array([3200, 1400, 700, next_cash, 0.02, 0.5, 49-i])
            done = False
            
            transition = Transition(state, goal, action, reward, next_state, done)
            episode_history.append(transition)
        
        high_reward = reward_engine.compute_high_level_reward(episode_history)
        
        # Wealth change = 3000 - 1000 = 2000
        # Wealth reward = 1.0 * 2000 = 2000
        # Total should include this wealth reward
        assert high_reward > 2000  # Should include wealth change component
    
    def test_high_level_reward_stability_bonus(self, reward_engine):
        """Test high-level reward includes stability bonus"""
        # Create episode with all positive balances
        episode_history = []
        for i in range(6):
            state = np.array([3200, 1400, 700, 2000, 0.02, 0.5, 50-i])
            goal = np.array([0.3, 1000, 0.5])
            action = np.array([0.3, 0.4, 0.3])
            reward = 100.0
            next_state = np.array([3200, 1400, 700, 2100, 0.02, 0.5, 49-i])
            done = False
            
            transition = Transition(state, goal, action, reward, next_state, done)
            episode_history.append(transition)
        
        high_reward = reward_engine.compute_high_level_reward(episode_history)
        
        # Stability bonus = 0.5 * (6/6) * 6 = 3.0
        # Should be included in total reward
        assert high_reward > 600  # Low-level rewards + wealth + stability
    
    def test_high_level_reward_empty_history(self, reward_engine):
        """Test high-level reward with empty history"""
        episode_history = []
        
        high_reward = reward_engine.compute_high_level_reward(episode_history)
        
        assert high_reward == 0.0
    
    def test_reward_coefficient_effects(self, reward_engine):
        """Test that different coefficients affect rewards"""
        state = np.array([3200, 1400, 700, 2000, 0.02, 0.5, 50])
        action = np.array([0.5, 0.3, 0.2])
        next_state = np.array([3200, 1400, 700, 1500, 0.02, 0.5, 49])
        
        # Test with default coefficients
        reward1 = reward_engine.compute_low_level_reward(action, state, next_state)
        
        # Create engine with different alpha (investment coefficient)
        config2 = RewardConfig(alpha=20.0, beta=0.1, gamma=5.0, delta=20.0, lambda_=1.0, mu=0.5)
        engine2 = RewardEngine(config2, safety_threshold=1000)
        reward2 = engine2.compute_low_level_reward(action, state, next_state)
        
        # Higher alpha should give higher reward
        assert reward2 > reward1
    
    def test_low_level_reward_no_penalties(self, reward_engine):
        """Test low-level reward when no penalties apply"""
        # Perfect scenario: good cash, no overspend, no debt
        state = np.array([3200, 1400, 700, 5000, 0.02, 0.5, 50])
        action = np.array([0.2, 0.5, 0.3])
        next_state = np.array([3200, 1400, 700, 4500, 0.02, 0.5, 49])
        
        reward = reward_engine.compute_low_level_reward(action, state, next_state)
        
        # Should be close to pure investment reward
        # Investment = 0.2 * 3200 = 640
        # Investment reward = 10.0 * 640 = 6400
        expected = 10.0 * 640
        assert reward >= expected * 0.95  # Allow small margin
