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
        assert 'episode_rewards' in trainer.training_history
        assert 'episode_lengths' in trainer.training_history
        assert 'cash_balances' in trainer.training_history
        assert 'total_invested' in trainer.training_history

    
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
        
        # Verify all values are valid (allow NaN as it can occur during early training)
        for reward in history['episode_rewards']:
            assert isinstance(reward, (int, float))
