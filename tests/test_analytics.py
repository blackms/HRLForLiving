"""Unit tests for AnalyticsModule"""
import pytest
import numpy as np
from src.utils.analytics import AnalyticsModule


class TestAnalyticsModule:
    """Test suite for AnalyticsModule class"""
    
    @pytest.fixture
    def analytics(self):
        """Create AnalyticsModule instance"""
        return AnalyticsModule()
    
    def test_initialization(self, analytics):
        """Test analytics module initialization"""
        assert len(analytics.states) == 0
        assert len(analytics.actions) == 0
        assert len(analytics.rewards) == 0
        assert len(analytics.cash_balances) == 0
        assert len(analytics.goals) == 0
        assert len(analytics.invested_amounts) == 0
    
    def test_record_step_basic(self, analytics):
        """Test basic step recording"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        reward = 10.5
        
        analytics.record_step(state, action, reward)
        
        assert len(analytics.states) == 1
        assert len(analytics.actions) == 1
        assert len(analytics.rewards) == 1
        assert len(analytics.cash_balances) == 1
        assert np.array_equal(analytics.states[0], state)
        assert np.array_equal(analytics.actions[0], action)
        assert analytics.rewards[0] == reward
        assert analytics.cash_balances[0] == 1000  # state[3]
    
    def test_record_step_with_goal(self, analytics):
        """Test step recording with goal"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        reward = 10.5
        goal = np.array([0.35, 1200, 0.6])
        
        analytics.record_step(state, action, reward, goal=goal)
        
        assert len(analytics.goals) == 1
        assert np.array_equal(analytics.goals[0], goal)
    
    def test_record_step_with_invested_amount(self, analytics):
        """Test step recording with invested amount"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        reward = 10.5
        invested = 960.0
        
        analytics.record_step(state, action, reward, invested_amount=invested)
        
        assert len(analytics.invested_amounts) == 1
        assert analytics.invested_amounts[0] == invested
    
    def test_record_multiple_steps(self, analytics):
        """Test recording multiple steps"""
        for i in range(5):
            state = np.array([3200, 1400, 700, 1000 + i * 100, 0.02, 0.5, 50 - i])
            action = np.array([0.3, 0.4, 0.3])
            reward = 10.0 + i
            
            analytics.record_step(state, action, reward)
        
        assert len(analytics.states) == 5
        assert len(analytics.actions) == 5
        assert len(analytics.rewards) == 5
        assert len(analytics.cash_balances) == 5
    
    def test_compute_metrics_empty(self, analytics):
        """Test metric computation with no data"""
        metrics = analytics.compute_episode_metrics()
        
        assert metrics['cumulative_wealth_growth'] == 0.0
        assert metrics['cash_stability_index'] == 0.0
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['goal_adherence'] == 0.0
        assert metrics['policy_stability'] == 0.0
    
    def test_compute_metrics_cumulative_wealth(self, analytics):
        """Test cumulative wealth growth calculation"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        
        analytics.record_step(state, action, 10.0, invested_amount=500)
        analytics.record_step(state, action, 10.0, invested_amount=600)
        analytics.record_step(state, action, 10.0, invested_amount=700)
        
        metrics = analytics.compute_episode_metrics()
        
        assert metrics['cumulative_wealth_growth'] == 1800.0
    
    def test_compute_metrics_cash_stability(self, analytics):
        """Test cash stability index calculation"""
        # 3 positive, 2 negative cash balances
        cash_values = [1000, 500, -100, 200, -50]
        
        for i, cash in enumerate(cash_values):
            state = np.array([3200, 1400, 700, cash, 0.02, 0.5, 50 - i])
            action = np.array([0.3, 0.4, 0.3])
            analytics.record_step(state, action, 10.0)
        
        metrics = analytics.compute_episode_metrics()
        
        # 3 out of 5 months positive = 0.6
        assert metrics['cash_stability_index'] == 0.6
    
    def test_compute_metrics_sharpe_ratio(self, analytics):
        """Test Sharpe-like ratio calculation"""
        cash_values = [1000, 1100, 1200, 1300, 1400]
        
        for i, cash in enumerate(cash_values):
            state = np.array([3200, 1400, 700, cash, 0.02, 0.5, 50 - i])
            action = np.array([0.3, 0.4, 0.3])
            analytics.record_step(state, action, 10.0)
        
        metrics = analytics.compute_episode_metrics()
        
        mean_balance = np.mean(cash_values)
        std_balance = np.std(cash_values)
        expected_sharpe = mean_balance / std_balance
        
        assert np.isclose(metrics['sharpe_ratio'], expected_sharpe)
    
    def test_compute_metrics_sharpe_ratio_zero_std(self, analytics):
        """Test Sharpe ratio with zero standard deviation"""
        # All same cash balance
        for i in range(5):
            state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50 - i])
            action = np.array([0.3, 0.4, 0.3])
            analytics.record_step(state, action, 10.0)
        
        metrics = analytics.compute_episode_metrics()
        
        # Should return 0 when std is 0
        assert metrics['sharpe_ratio'] == 0.0
    
    def test_compute_metrics_goal_adherence(self, analytics):
        """Test goal adherence calculation"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        
        # Goal target invest: 0.4, actual invest: 0.3 -> diff = 0.1
        goal1 = np.array([0.4, 1200, 0.6])
        action1 = np.array([0.3, 0.4, 0.3])
        analytics.record_step(state, action1, 10.0, goal=goal1)
        
        # Goal target invest: 0.5, actual invest: 0.4 -> diff = 0.1
        goal2 = np.array([0.5, 1200, 0.6])
        action2 = np.array([0.4, 0.3, 0.3])
        analytics.record_step(state, action2, 10.0, goal=goal2)
        
        metrics = analytics.compute_episode_metrics()
        
        # Mean of [0.1, 0.1] = 0.1
        assert np.isclose(metrics['goal_adherence'], 0.1)
    
    def test_compute_metrics_policy_stability(self, analytics):
        """Test policy stability calculation"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        
        # Varying actions
        actions = [
            np.array([0.3, 0.4, 0.3]),
            np.array([0.35, 0.35, 0.3]),
            np.array([0.25, 0.45, 0.3]),
            np.array([0.3, 0.4, 0.3]),
        ]
        
        for action in actions:
            analytics.record_step(state, action, 10.0)
        
        metrics = analytics.compute_episode_metrics()
        
        # Should compute variance across actions
        actions_array = np.array(actions)
        expected_variance = np.mean(np.var(actions_array, axis=0))
        
        assert np.isclose(metrics['policy_stability'], expected_variance)
    
    def test_reset_functionality(self, analytics):
        """Test reset clears all data"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        goal = np.array([0.35, 1200, 0.6])
        
        # Record some data
        analytics.record_step(state, action, 10.0, goal=goal, invested_amount=500)
        analytics.record_step(state, action, 10.0, goal=goal, invested_amount=600)
        
        assert len(analytics.states) == 2
        assert len(analytics.actions) == 2
        
        # Reset
        analytics.reset()
        
        # All lists should be empty
        assert len(analytics.states) == 0
        assert len(analytics.actions) == 0
        assert len(analytics.rewards) == 0
        assert len(analytics.cash_balances) == 0
        assert len(analytics.goals) == 0
        assert len(analytics.invested_amounts) == 0
    
    def test_metrics_after_reset(self, analytics):
        """Test metrics computation after reset"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        
        # Record, compute, reset
        analytics.record_step(state, action, 10.0, invested_amount=500)
        metrics1 = analytics.compute_episode_metrics()
        analytics.reset()
        
        # Metrics should be zero after reset
        metrics2 = analytics.compute_episode_metrics()
        
        assert metrics1['cumulative_wealth_growth'] == 500.0
        assert metrics2['cumulative_wealth_growth'] == 0.0
    
    def test_compute_metrics_single_step(self, analytics):
        """Test metric computation with single step (edge case)"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        goal = np.array([0.35, 1200, 0.6])
        
        analytics.record_step(state, action, 10.0, goal=goal, invested_amount=500)
        
        metrics = analytics.compute_episode_metrics()
        
        # Cumulative wealth should work with single step
        assert metrics['cumulative_wealth_growth'] == 500.0
        
        # Cash stability should be 1.0 (100% positive)
        assert metrics['cash_stability_index'] == 1.0
        
        # Sharpe ratio should be 0 with single data point
        assert metrics['sharpe_ratio'] == 0.0
        
        # Goal adherence should work with single step
        assert np.isclose(metrics['goal_adherence'], abs(0.35 - 0.3))
        
        # Policy stability should be 0 with single action
        assert metrics['policy_stability'] == 0.0
    
    def test_compute_metrics_single_step_negative_cash(self, analytics):
        """Test single step with negative cash balance"""
        state = np.array([3200, 1400, 700, -500, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        
        analytics.record_step(state, action, -20.0)
        
        metrics = analytics.compute_episode_metrics()
        
        # Cash stability should be 0.0 (0% positive)
        assert metrics['cash_stability_index'] == 0.0
    
    def test_goal_adherence_without_goals(self, analytics):
        """Test goal adherence when no goals are recorded"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        
        # Record without goals
        analytics.record_step(state, action, 10.0)
        analytics.record_step(state, action, 10.0)
        
        metrics = analytics.compute_episode_metrics()
        
        # Should return 0.0 when no goals
        assert metrics['goal_adherence'] == 0.0
    
    def test_goal_adherence_mismatched_lengths(self, analytics):
        """Test goal adherence with different numbers of goals and actions"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        goal = np.array([0.35, 1200, 0.6])
        
        # Record 3 actions but only 2 goals
        analytics.record_step(state, action, 10.0, goal=goal)
        analytics.record_step(state, action, 10.0, goal=goal)
        analytics.record_step(state, action, 10.0)  # No goal
        
        metrics = analytics.compute_episode_metrics()
        
        # Should compute based on minimum length (2)
        assert np.isclose(metrics['goal_adherence'], abs(0.35 - 0.3))
    
    def test_record_step_copies_arrays(self, analytics):
        """Test that record_step copies arrays to prevent reference issues"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        goal = np.array([0.35, 1200, 0.6])
        
        analytics.record_step(state, action, 10.0, goal=goal)
        
        # Modify original arrays
        state[3] = 2000
        action[0] = 0.9
        goal[0] = 0.9
        
        # Recorded values should not change
        assert analytics.states[0][3] == 1000
        assert analytics.actions[0][0] == 0.3
        assert analytics.goals[0][0] == 0.35
    
    def test_cumulative_wealth_without_invested_amounts(self, analytics):
        """Test cumulative wealth when no invested amounts recorded"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        
        # Record without invested amounts
        analytics.record_step(state, action, 10.0)
        analytics.record_step(state, action, 10.0)
        
        metrics = analytics.compute_episode_metrics()
        
        # Should return 0.0 when no invested amounts
        assert metrics['cumulative_wealth_growth'] == 0.0
    
    def test_policy_stability_identical_actions(self, analytics):
        """Test policy stability with identical actions (zero variance)"""
        state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
        action = np.array([0.3, 0.4, 0.3])
        
        # Record same action multiple times
        for _ in range(5):
            analytics.record_step(state, action, 10.0)
        
        metrics = analytics.compute_episode_metrics()
        
        # Variance should be 0 for identical actions
        assert metrics['policy_stability'] == 0.0
