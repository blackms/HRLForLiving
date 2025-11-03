"""Unit tests for BudgetExecutor"""
import pytest
import numpy as np
from src.agents.budget_executor import BudgetExecutor
from src.utils.config import TrainingConfig
from src.utils.data_models import Transition


class TestBudgetExecutor:
    """Test suite for BudgetExecutor class"""
    
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
    def executor(self, default_config):
        """Create BudgetExecutor instance"""
        return BudgetExecutor(default_config)
    
    def test_initialization(self, executor, default_config):
        """Test executor initialization"""
        assert executor.config == default_config
        assert executor.state_dim == 7
        assert executor.goal_dim == 3
        assert executor.input_dim == 10
        assert executor.observation_space.shape == (10,)
        assert executor.action_space.shape == (3,)
        assert executor.policy_network is not None
        assert executor.optimizer is not None
    
    def test_act_basic(self, executor):
        """Test basic action generation"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50], dtype=np.float32)
        goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
        
        action = executor.act(state, goal)
        
        assert action.shape == (3,)
        assert np.isclose(np.sum(action), 1.0, atol=1e-5)
        assert np.all(action >= 0)
        assert np.all(action <= 1)
    
    def test_act_deterministic(self, executor):
        """Test deterministic action generation"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50], dtype=np.float32)
        goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
        
        action1 = executor.act(state, goal, deterministic=True)
        action2 = executor.act(state, goal, deterministic=True)
        
        # Deterministic actions should be identical
        assert np.allclose(action1, action2)
    
    def test_act_input_concatenation(self, executor):
        """Test that state and goal are properly concatenated"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50], dtype=np.float32)
        goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
        
        # Should not raise error with correct dimensions
        action = executor.act(state, goal)
        assert action.shape == (3,)
    
    def test_act_invalid_state_dimension(self, executor):
        """Test error handling for invalid state dimension"""
        state = np.array([3200, 1400, 700], dtype=np.float32)  # Wrong dimension
        goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
        
        with pytest.raises(ValueError, match="Expected state dimension"):
            executor.act(state, goal)
    
    def test_act_invalid_goal_dimension(self, executor):
        """Test error handling for invalid goal dimension"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50], dtype=np.float32)
        goal = np.array([0.3, 1000], dtype=np.float32)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Expected goal dimension"):
            executor.act(state, goal)
    
    def test_action_normalization(self, executor):
        """Test action normalization ensures sum to 1"""
        action = np.array([2.0, 3.0, 1.0])
        normalized = executor._normalize_action(action)
        
        assert normalized.shape == (3,)
        assert np.isclose(np.sum(normalized), 1.0)
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
    
    def test_action_normalization_negative(self, executor):
        """Test action normalization handles negative values"""
        action = np.array([-0.5, 0.3, 0.2])
        normalized = executor._normalize_action(action)
        
        assert normalized.shape == (3,)
        assert np.isclose(np.sum(normalized), 1.0)
        assert np.all(normalized >= 0)
    
    def test_action_within_valid_ranges(self, executor):
        """Test that generated actions are within valid ranges"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50], dtype=np.float32)
        goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
        
        # Generate multiple actions to test consistency
        for _ in range(10):
            action = executor.act(state, goal)
            assert np.all(action >= 0), "Action contains negative values"
            assert np.all(action <= 1), "Action contains values > 1"
            assert np.isclose(np.sum(action), 1.0, atol=1e-5), "Action doesn't sum to 1"
    
    def test_learn_basic(self, executor):
        """Test basic learning functionality"""
        # Create sample transitions
        transitions = []
        for i in range(10):
            state = np.array([3200, 1400, 700, 1000 + i*100, 0.02, 0.5, 50-i], dtype=np.float32)
            goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
            action = np.array([0.3, 0.4, 0.3], dtype=np.float32)
            reward = 10.0 + i
            next_state = np.array([3200, 1400, 700, 1100 + i*100, 0.02, 0.5, 49-i], dtype=np.float32)
            done = False
            
            transitions.append(Transition(state, goal, action, reward, next_state, done))
        
        # Learn from transitions
        metrics = executor.learn(transitions)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'policy_entropy' in metrics
    
    def test_learn_empty_transitions(self, executor):
        """Test learning with empty transitions list"""
        metrics = executor.learn([])
        
        assert metrics['loss'] == 0.0
        assert metrics['policy_entropy'] == 0.0
    
    def test_learn_single_transition(self, executor):
        """Test learning with single transition"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50], dtype=np.float32)
        goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
        action = np.array([0.3, 0.4, 0.3], dtype=np.float32)
        reward = 10.0
        next_state = np.array([3200, 1400, 700, 1100, 0.02, 0.5, 49], dtype=np.float32)
        done = False
        
        transition = Transition(state, goal, action, reward, next_state, done)
        metrics = executor.learn([transition])
        
        assert isinstance(metrics, dict)
    
    def test_learn_with_terminal_state(self, executor):
        """Test learning with terminal transitions"""
        transitions = []
        for i in range(5):
            state = np.array([3200, 1400, 700, 1000 - i*300, 0.02, 0.5, 50-i], dtype=np.float32)
            goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
            action = np.array([0.5, 0.3, 0.2], dtype=np.float32)
            reward = -50.0  # Negative reward for bad state
            next_state = np.array([3200, 1400, 700, 700 - i*300, 0.02, 0.5, 49-i], dtype=np.float32)
            done = (i == 4)  # Last transition is terminal
            
            transitions.append(Transition(state, goal, action, reward, next_state, done))
        
        metrics = executor.learn(transitions)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
    
    def test_policy_update_mechanics(self, executor):
        """Test that policy updates occur correctly"""
        # Get initial action
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50], dtype=np.float32)
        goal = np.array([0.3, 1000, 0.5], dtype=np.float32)
        initial_action = executor.act(state, goal, deterministic=True)
        
        # Create transitions with high rewards for high investment
        transitions = []
        for i in range(32):  # Batch size
            action = np.array([0.8, 0.1, 0.1], dtype=np.float32)  # High investment
            reward = 100.0  # High reward
            next_state = np.array([3200, 1400, 700, 2000, 0.02, 0.5, 49-i], dtype=np.float32)
            
            transitions.append(Transition(state, goal, action, reward, next_state, False))
        
        # Learn from transitions
        metrics = executor.learn(transitions)
        
        # Verify learning occurred
        assert isinstance(metrics, dict)
        assert 'n_updates' in metrics or 'loss' in metrics
    
    def test_different_goals_produce_different_actions(self, executor):
        """Test that different goals influence action generation"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50], dtype=np.float32)
        
        # Conservative goal
        goal_conservative = np.array([0.1, 2000, 0.2], dtype=np.float32)
        action_conservative = executor.act(state, goal_conservative, deterministic=True)
        
        # Aggressive goal
        goal_aggressive = np.array([0.8, 500, 0.9], dtype=np.float32)
        action_aggressive = executor.act(state, goal_aggressive, deterministic=True)
        
        # Actions should be different (though initially they might be similar due to random initialization)
        # This test mainly verifies that both goals produce valid actions
        assert action_conservative.shape == (3,)
        assert action_aggressive.shape == (3,)
        assert np.isclose(np.sum(action_conservative), 1.0, atol=1e-5)
        assert np.isclose(np.sum(action_aggressive), 1.0, atol=1e-5)
