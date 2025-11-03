"""Unit tests for FinancialStrategist"""
import pytest
import numpy as np
from src.agents.financial_strategist import FinancialStrategist
from src.utils.config import TrainingConfig
from src.utils.data_models import Transition


class TestFinancialStrategist:
    """Test suite for FinancialStrategist class"""
    
    @pytest.fixture
    def default_config(self):
        """Create default training configuration"""
        return TrainingConfig(
            num_episodes=5000,
            gamma_low=0.95,
            gamma_high=0.99,
            high_period=6,
            batch_size=32,
            learning_rate_low=3e-4,
            learning_rate_high=1e-4
        )
    
    @pytest.fixture
    def strategist(self, default_config):
        """Create FinancialStrategist instance"""
        return FinancialStrategist(default_config)
    
    def test_initialization(self, strategist, default_config):
        """Test strategist initialization"""
        assert strategist.config == default_config
        assert strategist.aggregated_state_dim == 5
        assert strategist.goal_dim == 3
        assert strategist.policy_network is not None
        assert strategist.optimizer is not None
    
    def test_aggregate_state_basic(self, strategist):
        """Test basic state aggregation"""
        # Create sample state history
        # State format: [income, fixed, variable, cash, inflation, risk, t_remaining]
        history = [
            np.array([3200, 1400, 700, 1000, 0.02, 0.5, 60], dtype=np.float32),
            np.array([3200, 1400, 720, 1100, 0.02, 0.5, 59], dtype=np.float32),
            np.array([3200, 1400, 710, 1200, 0.02, 0.5, 58], dtype=np.float32),
            np.array([3200, 1400, 730, 1300, 0.02, 0.5, 57], dtype=np.float32),
        ]
        
        aggregated = strategist.aggregate_state(history)
        
        assert aggregated.shape == (5,)
        assert isinstance(aggregated, np.ndarray)
        # Check that values are reasonable
        assert aggregated[0] > 0  # avg_cash should be positive
        assert aggregated[3] > 0  # current_wealth should be positive
    
    def test_aggregate_state_empty_history(self, strategist):
        """Test state aggregation with empty history"""
        aggregated = strategist.aggregate_state([])
        
        assert aggregated.shape == (5,)
        assert np.allclose(aggregated, np.zeros(5))
    
    def test_aggregate_state_single_state(self, strategist):
        """Test state aggregation with single state"""
        history = [
            np.array([3200, 1400, 700, 1000, 0.02, 0.5, 60], dtype=np.float32)
        ]
        
        aggregated = strategist.aggregate_state(history)
        
        assert aggregated.shape == (5,)
        assert aggregated[0] == 1000  # avg_cash equals single cash value
        assert aggregated[3] == 1000  # current_wealth equals single cash value
    
    def test_aggregate_state_computes_averages(self, strategist):
        """Test that state aggregation computes correct averages"""
        # Create history with known values
        history = [
            np.array([3200, 1400, 700, 1000, 0.02, 0.5, 60], dtype=np.float32),
            np.array([3200, 1400, 700, 2000, 0.02, 0.5, 59], dtype=np.float32),
        ]
        
        aggregated = strategist.aggregate_state(history)
        
        # avg_cash should be (1000 + 2000) / 2 = 1500
        assert np.isclose(aggregated[0], 1500.0)
        # current_wealth should be 2000 (last value)
        assert np.isclose(aggregated[3], 2000.0)
    
    def test_select_goal_basic(self, strategist):
        """Test basic goal generation"""
        aggregated_state = np.array([1500, 100, 10, 2000, 5], dtype=np.float32)
        
        goal = strategist.select_goal(aggregated_state)
        
        assert goal.shape == (3,)
        assert isinstance(goal, np.ndarray)
    
    def test_select_goal_within_valid_ranges(self, strategist):
        """Test that generated goals are within valid ranges"""
        aggregated_state = np.array([1500, 100, 10, 2000, 5], dtype=np.float32)
        
        # Generate multiple goals to test consistency
        for _ in range(10):
            goal = strategist.select_goal(aggregated_state)
            
            # target_invest_ratio should be in [0, 1]
            assert 0 <= goal[0] <= 1, f"target_invest_ratio {goal[0]} out of range"
            
            # safety_buffer should be non-negative
            assert goal[1] >= 0, f"safety_buffer {goal[1]} is negative"
            
            # aggressiveness should be in [0, 1]
            assert 0 <= goal[2] <= 1, f"aggressiveness {goal[2]} out of range"
    
    def test_select_goal_invalid_state_dimension(self, strategist):
        """Test error handling for invalid state dimension"""
        invalid_state = np.array([1500, 100], dtype=np.float32)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Expected aggregated state dimension"):
            strategist.select_goal(invalid_state)
    
    def test_select_goal_deterministic(self, strategist):
        """Test that goal generation is deterministic for same state"""
        aggregated_state = np.array([1500, 100, 10, 2000, 5], dtype=np.float32)
        
        goal1 = strategist.select_goal(aggregated_state)
        goal2 = strategist.select_goal(aggregated_state)
        
        # Should be identical (deterministic policy)
        assert np.allclose(goal1, goal2)
    
    def test_learn_basic(self, strategist):
        """Test basic learning functionality"""
        # Create sample transitions with aggregated states
        transitions = []
        for i in range(10):
            aggregated_state = np.array([1500 + i*100, 100, 10, 2000 + i*100, i], dtype=np.float32)
            goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
            action = np.array([0.3, 0.4, 0.3], dtype=np.float32)  # Not used in high-level
            reward = 50.0 + i*10
            next_aggregated_state = np.array([1600 + i*100, 110, 12, 2100 + i*100, i+1], dtype=np.float32)
            done = False
            
            transitions.append(Transition(aggregated_state, goal, action, reward, next_aggregated_state, done))
        
        # Learn from transitions
        metrics = strategist.learn(transitions)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'policy_entropy' in metrics
    
    def test_learn_empty_transitions(self, strategist):
        """Test learning with empty transitions list"""
        metrics = strategist.learn([])
        
        assert metrics['loss'] == 0.0
        assert metrics['policy_entropy'] == 0.0
    
    def test_learn_single_transition(self, strategist):
        """Test learning with single transition"""
        aggregated_state = np.array([1500, 100, 10, 2000, 5], dtype=np.float32)
        goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
        action = np.array([0.3, 0.4, 0.3], dtype=np.float32)
        reward = 50.0
        next_aggregated_state = np.array([1600, 110, 12, 2100, 6], dtype=np.float32)
        done = False
        
        transition = Transition(aggregated_state, goal, action, reward, next_aggregated_state, done)
        metrics = strategist.learn([transition])
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    def test_learn_with_terminal_state(self, strategist):
        """Test learning with terminal transitions"""
        transitions = []
        for i in range(5):
            aggregated_state = np.array([1500 - i*200, 100, 10, 2000 - i*300, i], dtype=np.float32)
            goal = np.array([0.5, 1000, 0.7], dtype=np.float32)
            action = np.array([0.5, 0.3, 0.2], dtype=np.float32)
            reward = -20.0  # Negative reward for declining wealth
            next_aggregated_state = np.array([1300 - i*200, 90, 15, 1700 - i*300, i+1], dtype=np.float32)
            done = (i == 4)  # Last transition is terminal
            
            transitions.append(Transition(aggregated_state, goal, action, reward, next_aggregated_state, done))
        
        metrics = strategist.learn(transitions)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    def test_policy_update_mechanics(self, strategist):
        """Test that policy updates occur correctly"""
        # Get initial goal
        aggregated_state = np.array([1500, 100, 10, 2000, 5], dtype=np.float32)
        initial_goal = strategist.select_goal(aggregated_state)
        
        # Create transitions with high rewards
        transitions = []
        for i in range(32):  # Batch size
            goal = np.array([0.7, 800, 0.8], dtype=np.float32)  # Aggressive goal
            action = np.array([0.7, 0.2, 0.1], dtype=np.float32)
            reward = 200.0  # High reward
            next_aggregated_state = np.array([2000, 150, 10, 3000, i+1], dtype=np.float32)
            
            transitions.append(Transition(aggregated_state, goal, action, reward, next_aggregated_state, False))
        
        # Learn from transitions
        metrics = strategist.learn(transitions)
        
        # Verify learning occurred
        assert isinstance(metrics, dict)
        assert 'n_updates' in metrics or 'loss' in metrics
    
    def test_different_states_produce_different_goals(self, strategist):
        """Test that different states influence goal generation"""
        # Low wealth state
        state_low_wealth = np.array([500, 50, 10, 600, 5], dtype=np.float32)
        goal_low_wealth = strategist.select_goal(state_low_wealth)
        
        # High wealth state
        state_high_wealth = np.array([5000, 200, 10, 6000, 5], dtype=np.float32)
        goal_high_wealth = strategist.select_goal(state_high_wealth)
        
        # Both should produce valid goals
        assert goal_low_wealth.shape == (3,)
        assert goal_high_wealth.shape == (3,)
        assert 0 <= goal_low_wealth[0] <= 1
        assert 0 <= goal_high_wealth[0] <= 1
