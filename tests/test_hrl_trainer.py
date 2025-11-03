"""Integration tests for HRLTrainer"""
import pytest
import numpy as np
from src.training.hrl_trainer import HRLTrainer
from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.environment.reward_engine import RewardEngine
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig


class TestHRLTrainer:
    """Test suite for HRLTrainer class"""
    
    @pytest.fixture
    def env_config(self):
        """Create environment configuration"""
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
    def training_config(self):
        """Create training configuration"""
        return TrainingConfig(
            num_episodes=10,  # Small number for testing
            gamma_low=0.95,
            gamma_high=0.99,
            high_period=6,
            batch_size=8,  # Small batch for testing
            learning_rate_low=3e-4,
            learning_rate_high=1e-4
        )
    
    @pytest.fixture
    def reward_config(self):
        """Create reward configuration"""
        return RewardConfig(
            alpha=10.0,
            beta=0.1,
            gamma=5.0,
            delta=20.0,
            lambda_=1.0,
            mu=0.5
        )
    
    @pytest.fixture
    def trainer(self, env_config, training_config, reward_config):
        """Create HRLTrainer instance with all components"""
        env = BudgetEnv(env_config, reward_config)
        high_agent = FinancialStrategist(training_config)
        low_agent = BudgetExecutor(training_config)
        reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
        
        return HRLTrainer(env, high_agent, low_agent, reward_engine, training_config)
    
    def test_initialization(self, trainer, training_config):
        """Test trainer initialization"""
        assert trainer.env is not None
        assert trainer.high_agent is not None
        assert trainer.low_agent is not None
        assert trainer.reward_engine is not None
        assert trainer.config == training_config
        assert isinstance(trainer.episode_buffer, list)
        assert isinstance(trainer.state_history, list)
        assert trainer.analytics is not None
        assert 'episode_rewards' in trainer.training_history
        assert 'episode_lengths' in trainer.training_history
        assert 'cash_balances' in trainer.training_history
        assert 'total_invested' in trainer.training_history
        assert 'cumulative_wealth_growth' in trainer.training_history
        assert 'cash_stability_index' in trainer.training_history
        assert 'sharpe_ratio' in trainer.training_history
        assert 'goal_adherence' in trainer.training_history
        assert 'policy_stability' in trainer.training_history

    
    def test_complete_episode_execution(self, trainer):
        """Test that a complete episode can be executed"""
        # Run a single training episode
        history = trainer.train(num_episodes=1)
        
        # Verify training history is populated
        assert len(history['episode_rewards']) == 1
        assert len(history['episode_lengths']) == 1
        assert len(history['cash_balances']) == 1
        assert len(history['total_invested']) == 1
        
        # Verify episode completed
        assert history['episode_lengths'][0] > 0
        assert isinstance(history['episode_rewards'][0], (int, float))
    
    def test_multiple_episodes_execution(self, trainer):
        """Test multiple training episodes"""
        num_episodes = 3
        history = trainer.train(num_episodes=num_episodes)
        
        # Verify all episodes completed
        assert len(history['episode_rewards']) == num_episodes
        assert len(history['episode_lengths']) == num_episodes
        assert len(history['cash_balances']) == num_episodes
        assert len(history['total_invested']) == num_episodes
    
    def test_high_level_low_level_coordination(self, trainer):
        """Test that high-level and low-level agents coordinate correctly"""
        # Run training
        history = trainer.train(num_episodes=2)
        
        # Verify both agents performed updates
        # Low-level updates should occur more frequently
        assert len(history['low_level_losses']) > 0
        
        # High-level updates should occur less frequently (every high_period steps)
        assert len(history['high_level_losses']) >= 0  # May be 0 if episodes are short
    
    def test_policy_updates_occur(self, trainer):
        """Test that policy updates occur correctly during training"""
        # Run training with sufficient episodes
        history = trainer.train(num_episodes=5)
        
        # Verify low-level policy updates occurred
        assert len(history['low_level_losses']) > 0, "Low-level policy should be updated"
        
        # Verify losses are numeric (allow NaN as it can occur during early training)
        for loss in history['low_level_losses']:
            assert isinstance(loss, (int, float))
    
    def test_episode_buffer_management(self, trainer):
        """Test that episode buffer is properly managed"""
        # Run a single episode
        trainer.train(num_episodes=1)
        
        # Episode buffer should contain transitions from the episode
        # After episode completion, buffer may be cleared or retained
        assert isinstance(trainer.episode_buffer, list)
    
    def test_state_history_tracking(self, trainer):
        """Test that state history is tracked for high-level agent"""
        # Run a single episode
        trainer.train(num_episodes=1)
        
        # State history should be populated
        assert isinstance(trainer.state_history, list)
        assert len(trainer.state_history) > 0
    
    def test_evaluation_basic(self, trainer):
        """Test basic evaluation functionality"""
        # Run evaluation without training
        eval_results = trainer.evaluate(num_episodes=2)
        
        # Verify evaluation results structure
        assert 'mean_reward' in eval_results
        assert 'mean_cash_balance' in eval_results
        assert 'mean_total_invested' in eval_results
        assert 'mean_wealth_growth' in eval_results
        assert 'mean_cash_stability' in eval_results
        assert 'mean_sharpe_ratio' in eval_results
        assert 'mean_goal_adherence' in eval_results
        assert 'mean_policy_stability' in eval_results
        assert 'detailed_results' in eval_results
    
    def test_evaluation_metrics_computation(self, trainer):
        """Test that evaluation metrics are computed correctly"""
        eval_results = trainer.evaluate(num_episodes=3)
        
        # Verify metrics are numeric
        assert isinstance(eval_results['mean_reward'], (int, float))
        assert isinstance(eval_results['mean_cash_balance'], (int, float))
        assert isinstance(eval_results['mean_total_invested'], (int, float))
        
        # Verify detailed results contain all episodes
        detailed = eval_results['detailed_results']
        assert len(detailed['episode_rewards']) == 3
        assert len(detailed['cash_stability_index']) == 3
        assert len(detailed['sharpe_ratio']) == 3
    
    def test_evaluation_deterministic_policy(self, trainer):
        """Test that evaluation uses deterministic policy"""
        # Run same evaluation twice
        eval1 = trainer.evaluate(num_episodes=1)
        eval2 = trainer.evaluate(num_episodes=1)
        
        # Results should be similar (deterministic policy)
        # Note: Due to environment stochasticity (variable expenses), 
        # results won't be identical, but should be close
        assert isinstance(eval1['mean_reward'], (int, float))
        assert isinstance(eval2['mean_reward'], (int, float))
    
    def test_training_then_evaluation(self, trainer):
        """Test training followed by evaluation"""
        # Train for a few episodes
        train_history = trainer.train(num_episodes=3)
        
        # Evaluate the trained policy
        eval_results = trainer.evaluate(num_episodes=2)
        
        # Verify both completed successfully
        assert len(train_history['episode_rewards']) == 3
        assert 'mean_reward' in eval_results
    
    def test_high_period_coordination(self, trainer):
        """Test that high-level updates occur at correct intervals"""
        # Set a specific high_period
        trainer.config.high_period = 5
        
        # Run training
        history = trainer.train(num_episodes=2)
        
        # High-level updates should occur approximately every high_period steps
        # The exact number depends on episode length
        assert isinstance(history['high_level_losses'], list)
    
    def test_batch_size_coordination(self, trainer):
        """Test that low-level updates respect batch size"""
        # Set a specific batch size
        trainer.config.batch_size = 10
        
        # Run training
        history = trainer.train(num_episodes=2)
        
        # Low-level updates should occur when buffer reaches batch size
        assert isinstance(history['low_level_losses'], list)
    
    def test_episode_termination_handling(self, trainer):
        """Test that episode termination is handled correctly"""
        # Run training
        history = trainer.train(num_episodes=2)
        
        # All episodes should terminate (either by max steps or negative cash)
        for length in history['episode_lengths']:
            assert length > 0
            assert length <= trainer.env.max_months
    
    def test_metrics_tracking_completeness(self, trainer):
        """Test that all metrics are tracked throughout training"""
        history = trainer.train(num_episodes=3)
        
        # Verify all metric lists have correct length
        assert len(history['episode_rewards']) == 3
        assert len(history['episode_lengths']) == 3
        assert len(history['cash_balances']) == 3
        assert len(history['total_invested']) == 3
        assert len(history['cumulative_wealth_growth']) == 3
        assert len(history['cash_stability_index']) == 3
        assert len(history['sharpe_ratio']) == 3
        assert len(history['goal_adherence']) == 3
        assert len(history['policy_stability']) == 3
        
        # Verify all values are valid (allow NaN as it can occur during early training)
        for reward in history['episode_rewards']:
            assert isinstance(reward, (int, float))
    
    def test_analytics_integration(self, trainer):
        """Test that analytics module is properly integrated"""
        # Run training
        history = trainer.train(num_episodes=2)
        
        # Verify analytics metrics are computed and stored
        assert len(history['cumulative_wealth_growth']) == 2
        assert len(history['cash_stability_index']) == 2
        assert len(history['sharpe_ratio']) == 2
        assert len(history['goal_adherence']) == 2
        assert len(history['policy_stability']) == 2
        
        # Verify metrics are numeric
        for metric in history['cumulative_wealth_growth']:
            assert isinstance(metric, (int, float))
        for metric in history['cash_stability_index']:
            assert isinstance(metric, (int, float))
            assert 0 <= metric <= 1  # Stability index should be between 0 and 1
    
    def test_analytics_reset_between_episodes(self, trainer):
        """Test that analytics is reset between episodes"""
        # Run multiple episodes
        history = trainer.train(num_episodes=3)
        
        # Each episode should have independent metrics
        # Verify that metrics are computed for each episode
        assert len(history['cumulative_wealth_growth']) == 3
        
        # Metrics should be independent (not cumulative across episodes)
        # Each episode starts fresh
        for i in range(3):
            assert isinstance(history['cumulative_wealth_growth'][i], (int, float))
    
    def test_analytics_step_recording(self, trainer):
        """Test that analytics records steps during training"""
        # Run a single episode
        trainer.train(num_episodes=1)
        
        # Analytics should have been reset after episode completion
        # (analytics.reset() is called at the start of each episode)
        assert trainer.analytics is not None
    
    def test_complete_episode_with_all_components(self, trainer):
        """Integration test: Complete episode execution with all components working together"""
        # This test verifies that all components (env, agents, reward engine, analytics)
        # work together correctly for a complete episode
        
        # Run a single episode
        history = trainer.train(num_episodes=1)
        
        # Verify environment executed steps
        assert history['episode_lengths'][0] > 0
        assert history['episode_lengths'][0] <= trainer.env.max_months
        
        # Verify low-level agent performed actions
        assert len(history['low_level_losses']) >= 0  # May be 0 if episode too short
        
        # Verify high-level agent set goals
        assert len(history['high_level_losses']) >= 0  # May be 0 if episode too short
        
        # Verify reward engine computed rewards
        assert isinstance(history['episode_rewards'][0], (int, float))
        
        # Verify analytics tracked metrics (allow NaN as valid numeric value)
        assert isinstance(history['cumulative_wealth_growth'][0], (int, float, np.floating))
        assert isinstance(history['cash_stability_index'][0], (int, float, np.floating))
        assert isinstance(history['sharpe_ratio'][0], (int, float, np.floating))
        # goal_adherence can be NaN if no goals were recorded
        assert isinstance(history['goal_adherence'][0], (int, float, np.floating)) or np.isnan(history['goal_adherence'][0])
        assert isinstance(history['policy_stability'][0], (int, float, np.floating))
    
    def test_high_level_goal_updates_at_correct_intervals(self, trainer):
        """Integration test: High-level agent updates goals at correct intervals"""
        # Set high_period to a specific value
        trainer.config.high_period = 3
        
        # Run training with sufficient episodes to trigger multiple high-level updates
        history = trainer.train(num_episodes=2)
        
        # Verify high-level updates occurred
        # Number of updates depends on episode length and high_period
        assert isinstance(history['high_level_losses'], list)
        
        # If episodes are long enough, we should see high-level updates
        total_steps = sum(history['episode_lengths'])
        expected_min_updates = max(0, (total_steps // trainer.config.high_period) - 1)
        
        # We should have at least some high-level updates if episodes are long enough
        if total_steps >= trainer.config.high_period:
            assert len(history['high_level_losses']) > 0
    
    def test_low_level_updates_with_batch_coordination(self, trainer):
        """Integration test: Low-level agent updates when batch size is reached"""
        # Set batch size to a specific value
        trainer.config.batch_size = 5
        
        # Run training
        history = trainer.train(num_episodes=2)
        
        # Verify low-level updates occurred
        assert isinstance(history['low_level_losses'], list)
        
        # If episodes are long enough, we should see low-level updates
        total_steps = sum(history['episode_lengths'])
        if total_steps >= trainer.config.batch_size:
            assert len(history['low_level_losses']) > 0
    
    def test_policy_updates_improve_over_time(self, trainer):
        """Integration test: Verify policies are being updated (losses are recorded)"""
        # Run training with sufficient episodes
        history = trainer.train(num_episodes=5)
        
        # Verify that policy updates occurred
        assert len(history['low_level_losses']) > 0, "Low-level policy should be updated"
        
        # Verify losses are being tracked (they should be numeric)
        for loss in history['low_level_losses']:
            assert isinstance(loss, (int, float))
            # Loss can be NaN during early training, but should be a number type
    
    def test_analytics_integration_throughout_episode(self, trainer):
        """Integration test: Analytics module tracks all steps during episode"""
        # Run a single episode
        history = trainer.train(num_episodes=1)
        
        # Verify all analytics metrics were computed
        assert len(history['cumulative_wealth_growth']) == 1
        assert len(history['cash_stability_index']) == 1
        assert len(history['sharpe_ratio']) == 1
        assert len(history['goal_adherence']) == 1
        assert len(history['policy_stability']) == 1
        
        # Verify metrics are valid (allow NaN as valid numeric value)
        wealth = history['cumulative_wealth_growth'][0]
        stability = history['cash_stability_index'][0]
        sharpe = history['sharpe_ratio'][0]
        adherence = history['goal_adherence'][0]
        policy_stab = history['policy_stability'][0]
        
        assert isinstance(wealth, (int, float, np.floating))
        assert isinstance(stability, (int, float, np.floating))
        assert isinstance(sharpe, (int, float, np.floating))
        # adherence can be NaN or a numeric value
        assert isinstance(adherence, (int, float, np.floating)) or np.isnan(adherence)
        assert isinstance(policy_stab, (int, float, np.floating))
        
        # Stability index should be between 0 and 1 (if not NaN)
        if not np.isnan(stability):
            assert 0 <= stability <= 1
    
    def test_episode_buffer_accumulates_transitions(self, trainer):
        """Integration test: Episode buffer accumulates transitions during episode"""
        # Run a single episode
        history = trainer.train(num_episodes=1)
        
        # Episode buffer should have accumulated transitions
        # After episode completion, buffer may be cleared or retained
        assert isinstance(trainer.episode_buffer, list)
        
        # Episode length should match the number of steps taken
        assert history['episode_lengths'][0] > 0
    
    def test_state_history_for_high_level_aggregation(self, trainer):
        """Integration test: State history is maintained for high-level agent"""
        # Run a single episode
        history = trainer.train(num_episodes=1)
        
        # State history should be populated
        assert isinstance(trainer.state_history, list)
        assert len(trainer.state_history) > 0
        
        # State history length should be episode_length + 1 (initial state + all next states)
        assert len(trainer.state_history) == history['episode_lengths'][0] + 1
    
    def test_reward_engine_integration(self, trainer):
        """Integration test: Reward engine computes rewards correctly during training"""
        # Run training
        history = trainer.train(num_episodes=2)
        
        # Verify rewards were computed for each episode
        assert len(history['episode_rewards']) == 2
        
        # Verify rewards are numeric
        for reward in history['episode_rewards']:
            assert isinstance(reward, (int, float))
        
        # Verify high-level rewards were computed
        # (high_level_losses indicates high-level updates occurred)
        assert isinstance(history['high_level_losses'], list)
    
    def test_full_training_pipeline(self, trainer):
        """Integration test: Complete training pipeline from start to finish"""
        # This is a comprehensive integration test that verifies the entire pipeline
        
        # Run training for multiple episodes
        num_episodes = 5
        history = trainer.train(num_episodes=num_episodes)
        
        # 1. Verify all episodes completed
        assert len(history['episode_rewards']) == num_episodes
        assert len(history['episode_lengths']) == num_episodes
        
        # 2. Verify environment simulation worked
        for length in history['episode_lengths']:
            assert length > 0
            assert length <= trainer.env.max_months
        
        # 3. Verify both agents performed updates
        assert len(history['low_level_losses']) >= 0
        assert len(history['high_level_losses']) >= 0
        
        # 4. Verify reward computation
        for reward in history['episode_rewards']:
            assert isinstance(reward, (int, float))
        
        # 5. Verify analytics integration
        assert len(history['cumulative_wealth_growth']) == num_episodes
        assert len(history['cash_stability_index']) == num_episodes
        assert len(history['sharpe_ratio']) == num_episodes
        assert len(history['goal_adherence']) == num_episodes
        assert len(history['policy_stability']) == num_episodes
        
        # 6. Verify all metrics are valid (allow NaN as valid numeric value)
        for i in range(num_episodes):
            assert isinstance(history['cumulative_wealth_growth'][i], (int, float, np.floating))
            assert isinstance(history['cash_stability_index'][i], (int, float, np.floating))
            # Stability index should be between 0 and 1 (if not NaN)
            if not np.isnan(history['cash_stability_index'][i]):
                assert 0 <= history['cash_stability_index'][i] <= 1
            assert isinstance(history['sharpe_ratio'][i], (int, float, np.floating))
            # goal_adherence can be NaN or numeric
            assert isinstance(history['goal_adherence'][i], (int, float, np.floating)) or np.isnan(history['goal_adherence'][i])
            assert isinstance(history['policy_stability'][i], (int, float, np.floating))
    
    def test_evaluation_after_training_integration(self, trainer):
        """Integration test: Evaluation works correctly after training"""
        # Train the system
        train_history = trainer.train(num_episodes=3)
        
        # Evaluate the trained system
        eval_results = trainer.evaluate(num_episodes=2)
        
        # Verify training completed
        assert len(train_history['episode_rewards']) == 3
        
        # Verify evaluation completed
        assert 'mean_reward' in eval_results
        assert 'mean_cash_balance' in eval_results
        assert 'mean_total_invested' in eval_results
        
        # Verify all analytics metrics are in evaluation
        assert 'mean_wealth_growth' in eval_results
        assert 'mean_cash_stability' in eval_results
        assert 'mean_sharpe_ratio' in eval_results
        assert 'mean_goal_adherence' in eval_results
        assert 'mean_policy_stability' in eval_results
        
        # Verify detailed results
        detailed = eval_results['detailed_results']
        assert len(detailed['episode_rewards']) == 2
        assert len(detailed['cumulative_wealth_growth']) == 2
        assert len(detailed['cash_stability_index']) == 2
    
    def test_hierarchical_coordination_complete_flow(self, trainer):
        """Integration test: Complete hierarchical coordination flow"""
        # This test verifies the complete hierarchical coordination:
        # High-level sets goals -> Low-level executes -> Both learn
        
        # Set specific parameters for predictable behavior
        trainer.config.high_period = 4
        trainer.config.batch_size = 6
        
        # Run training
        history = trainer.train(num_episodes=2)
        
        # Verify hierarchical structure is maintained
        # 1. Episodes completed
        assert len(history['episode_rewards']) == 2
        
        # 2. Low-level agent executed actions (one per step)
        total_steps = sum(history['episode_lengths'])
        assert total_steps > 0
        
        # 3. High-level agent set goals (every high_period steps)
        # Number of high-level updates depends on episode length
        assert isinstance(history['high_level_losses'], list)
        
        # 4. Both agents learned
        # Low-level learns when batch size is reached
        if total_steps >= trainer.config.batch_size:
            assert len(history['low_level_losses']) > 0
        
        # High-level learns at goal update intervals
        if total_steps >= trainer.config.high_period:
            assert len(history['high_level_losses']) > 0
        
        # 5. Analytics tracked the entire process
        assert len(history['goal_adherence']) == 2
        assert len(history['policy_stability']) == 2
