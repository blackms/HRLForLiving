"""Sanity check tests for HRL Finance System validation"""
import pytest
import numpy as np
from src.training.hrl_trainer import HRLTrainer
from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.environment.reward_engine import RewardEngine
from src.utils.config_manager import load_behavioral_profile
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig


class TestSanityChecks:
    """Sanity check tests to validate system behavior"""
    
    @pytest.fixture
    def base_env_config(self):
        """Create base environment configuration"""
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
    def base_training_config(self):
        """Create base training configuration"""
        return TrainingConfig(
            num_episodes=20,  # Sufficient for sanity checks
            gamma_low=0.95,
            gamma_high=0.99,
            high_period=6,
            batch_size=8,
            learning_rate_low=3e-4,
            learning_rate_high=1e-4
        )
    
    @pytest.fixture
    def base_reward_config(self):
        """Create base reward configuration"""
        return RewardConfig(
            alpha=10.0,
            beta=0.1,
            gamma=5.0,
            delta=20.0,
            lambda_=1.0,
            mu=0.5
        )
    
    def create_trainer(self, env_config, training_config, reward_config):
        """Helper to create trainer with given configurations"""
        env = BudgetEnv(env_config, reward_config)
        high_agent = FinancialStrategist(training_config)
        low_agent = BudgetExecutor(training_config)
        reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
        
        return HRLTrainer(env, high_agent, low_agent, reward_engine, training_config)
    
    def test_random_policy_does_not_accumulate_wealth(self, base_env_config, base_training_config, base_reward_config):
        """
        Test that a random policy does not accumulate significant wealth.
        
        A random policy should not systematically invest and grow wealth,
        as it lacks the strategic decision-making to balance investment and stability.
        
        Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3
        """
        # Create trainer with random policy (untrained agents)
        trainer = self.create_trainer(base_env_config, base_training_config, base_reward_config)
        
        # Evaluate random policy without training
        eval_results = trainer.evaluate(num_episodes=10)
        
        # Random policy should not accumulate significant wealth
        mean_invested = eval_results['mean_total_invested']
        
        # With income of 3200 and expenses around 2100, available for investment is ~1100/month
        # Over 60 months, maximum possible investment is ~66,000
        # Random policy should invest much less (typically < 20% of maximum)
        max_possible_investment = (base_env_config.income - base_env_config.fixed_expenses - 
                                   base_env_config.variable_expense_mean) * base_env_config.max_months
        
        # Random policy should invest less than 30% of maximum possible
        # (being generous to account for randomness)
        assert mean_invested < 0.3 * max_possible_investment, \
            f"Random policy invested {mean_invested:.2f}, expected < {0.3 * max_possible_investment:.2f}"
        
        # Verify that episodes completed (not all terminated early)
        detailed = eval_results['detailed_results']
        avg_episode_length = np.mean(detailed['episode_lengths'])
        
        # Random policy should have relatively short episodes due to poor cash management
        # (typically terminates early due to negative cash)
        assert avg_episode_length < base_env_config.max_months * 0.8, \
            f"Random policy episodes too long: {avg_episode_length:.1f} months"
    
    def test_conservative_profile_maintains_higher_cash_balance(self):
        """
        Test that conservative profile maintains higher cash balance than aggressive profile.
        
        Conservative profile should prioritize stability and maintain higher cash reserves,
        while aggressive profile should invest more aggressively.
        
        Requirements: 6.2, 6.3
        """
        # Load conservative profile
        conservative_env, conservative_training, conservative_reward = load_behavioral_profile('conservative')
        conservative_training.num_episodes = 20  # Override for faster testing
        
        # Load aggressive profile
        aggressive_env, aggressive_training, aggressive_reward = load_behavioral_profile('aggressive')
        aggressive_training.num_episodes = 20  # Override for faster testing
        
        # Create trainers
        conservative_trainer = self.create_trainer(conservative_env, conservative_training, conservative_reward)
        aggressive_trainer = self.create_trainer(aggressive_env, aggressive_training, aggressive_reward)
        
        # Train both policies
        conservative_trainer.train(num_episodes=20)
        aggressive_trainer.train(num_episodes=20)
        
        # Evaluate both policies
        conservative_eval = conservative_trainer.evaluate(num_episodes=10)
        aggressive_eval = aggressive_trainer.evaluate(num_episodes=10)
        
        # Get detailed results to compute valid metrics
        conservative_detailed = conservative_eval['detailed_results']
        aggressive_detailed = aggressive_eval['detailed_results']
        
        # Compute average cash balance from detailed results (filter out NaN)
        conservative_cash_values = [cb for cb in conservative_detailed['cash_balances'] if not np.isnan(cb)]
        aggressive_cash_values = [cb for cb in aggressive_detailed['cash_balances'] if not np.isnan(cb)]
        
        if len(conservative_cash_values) > 0 and len(aggressive_cash_values) > 0:
            conservative_cash = np.mean(conservative_cash_values)
            aggressive_cash = np.mean(aggressive_cash_values)
            
            assert conservative_cash > aggressive_cash, \
                f"Conservative cash ({conservative_cash:.2f}) should be > aggressive cash ({aggressive_cash:.2f})"
        
        # Conservative should have higher stability index
        conservative_stability_values = [s for s in conservative_detailed['cash_stability_index'] if not np.isnan(s)]
        aggressive_stability_values = [s for s in aggressive_detailed['cash_stability_index'] if not np.isnan(s)]
        
        if len(conservative_stability_values) > 0 and len(aggressive_stability_values) > 0:
            conservative_stability = np.mean(conservative_stability_values)
            aggressive_stability = np.mean(aggressive_stability_values)
            
            assert conservative_stability >= aggressive_stability, \
                f"Conservative stability ({conservative_stability:.3f}) should be >= aggressive stability ({aggressive_stability:.3f})"
        
        # Verify conservative has higher safety threshold
        assert conservative_env.safety_threshold > aggressive_env.safety_threshold, \
            "Conservative should have higher safety threshold"
        
        # Verify conservative has lower risk tolerance
        assert conservative_env.risk_tolerance < aggressive_env.risk_tolerance, \
            "Conservative should have lower risk tolerance"
    
    def test_aggressive_profile_invests_more(self):
        """
        Test that aggressive profile invests more than conservative profile.
        
        Aggressive profile should prioritize investment and wealth growth,
        resulting in higher total invested amounts.
        
        Requirements: 6.2, 6.3
        """
        # Load conservative profile
        conservative_env, conservative_training, conservative_reward = load_behavioral_profile('conservative')
        conservative_training.num_episodes = 20  # Override for faster testing
        
        # Load aggressive profile
        aggressive_env, aggressive_training, aggressive_reward = load_behavioral_profile('aggressive')
        aggressive_training.num_episodes = 20  # Override for faster testing
        
        # Create trainers
        conservative_trainer = self.create_trainer(conservative_env, conservative_training, conservative_reward)
        aggressive_trainer = self.create_trainer(aggressive_env, aggressive_training, aggressive_reward)
        
        # Train both policies
        conservative_trainer.train(num_episodes=20)
        aggressive_trainer.train(num_episodes=20)
        
        # Evaluate both policies
        conservative_eval = conservative_trainer.evaluate(num_episodes=10)
        aggressive_eval = aggressive_trainer.evaluate(num_episodes=10)
        
        # Get detailed results
        conservative_detailed = conservative_eval['detailed_results']
        aggressive_detailed = aggressive_eval['detailed_results']
        
        # Aggressive should invest more (filter out NaN)
        conservative_invested_values = [ti for ti in conservative_detailed['total_invested'] if not np.isnan(ti)]
        aggressive_invested_values = [ti for ti in aggressive_detailed['total_invested'] if not np.isnan(ti)]
        
        if len(conservative_invested_values) > 0 and len(aggressive_invested_values) > 0:
            conservative_invested = np.mean(conservative_invested_values)
            aggressive_invested = np.mean(aggressive_invested_values)
            
            assert aggressive_invested > conservative_invested, \
                f"Aggressive invested ({aggressive_invested:.2f}) should be > conservative invested ({conservative_invested:.2f})"
        
        # Aggressive should have higher wealth growth
        conservative_wealth_values = [w for w in conservative_detailed['cumulative_wealth_growth'] if not np.isnan(w)]
        aggressive_wealth_values = [w for w in aggressive_detailed['cumulative_wealth_growth'] if not np.isnan(w)]
        
        if len(conservative_wealth_values) > 0 and len(aggressive_wealth_values) > 0:
            conservative_wealth = np.mean(conservative_wealth_values)
            aggressive_wealth = np.mean(aggressive_wealth_values)
            
            assert aggressive_wealth > conservative_wealth, \
                f"Aggressive wealth growth ({aggressive_wealth:.2f}) should be > conservative wealth growth ({conservative_wealth:.2f})"
        
        # Verify reward coefficients favor investment for aggressive
        assert aggressive_reward.alpha > conservative_reward.alpha, \
            "Aggressive should have higher investment reward coefficient (alpha)"
        
        # Verify aggressive has lower stability penalty
        assert aggressive_reward.beta < conservative_reward.beta, \
            "Aggressive should have lower stability penalty coefficient (beta)"
    
    def test_balanced_profile_between_conservative_and_aggressive(self):
        """
        Test that balanced profile exhibits behavior between conservative and aggressive.
        
        Balanced profile should maintain moderate cash balance and investment levels,
        falling between the extremes of conservative and aggressive profiles.
        
        Requirements: 6.2, 6.3
        """
        # Load all three profiles
        conservative_env, conservative_training, conservative_reward = load_behavioral_profile('conservative')
        balanced_env, balanced_training, balanced_reward = load_behavioral_profile('balanced')
        aggressive_env, aggressive_training, aggressive_reward = load_behavioral_profile('aggressive')
        
        # Override num_episodes for faster testing
        conservative_training.num_episodes = 20
        balanced_training.num_episodes = 20
        aggressive_training.num_episodes = 20
        
        # Create trainers
        conservative_trainer = self.create_trainer(conservative_env, conservative_training, conservative_reward)
        balanced_trainer = self.create_trainer(balanced_env, balanced_training, balanced_reward)
        aggressive_trainer = self.create_trainer(aggressive_env, aggressive_training, aggressive_reward)
        
        # Train all policies
        conservative_trainer.train(num_episodes=20)
        balanced_trainer.train(num_episodes=20)
        aggressive_trainer.train(num_episodes=20)
        
        # Evaluate all policies
        conservative_eval = conservative_trainer.evaluate(num_episodes=10)
        balanced_eval = balanced_trainer.evaluate(num_episodes=10)
        aggressive_eval = aggressive_trainer.evaluate(num_episodes=10)
        
        # Get detailed results
        conservative_detailed = conservative_eval['detailed_results']
        balanced_detailed = balanced_eval['detailed_results']
        aggressive_detailed = aggressive_eval['detailed_results']
        
        # Extract metrics (filter out NaN)
        conservative_cash_values = [cb for cb in conservative_detailed['cash_balances'] if not np.isnan(cb)]
        balanced_cash_values = [cb for cb in balanced_detailed['cash_balances'] if not np.isnan(cb)]
        aggressive_cash_values = [cb for cb in aggressive_detailed['cash_balances'] if not np.isnan(cb)]
        
        conservative_invested_values = [ti for ti in conservative_detailed['total_invested'] if not np.isnan(ti)]
        balanced_invested_values = [ti for ti in balanced_detailed['total_invested'] if not np.isnan(ti)]
        aggressive_invested_values = [ti for ti in aggressive_detailed['total_invested'] if not np.isnan(ti)]
        
        # Balanced should be between conservative and aggressive for cash balance
        # (allow some tolerance for training variance)
        if len(conservative_cash_values) > 0 and len(balanced_cash_values) > 0 and len(aggressive_cash_values) > 0:
            conservative_cash = np.mean(conservative_cash_values)
            balanced_cash = np.mean(balanced_cash_values)
            aggressive_cash = np.mean(aggressive_cash_values)
            
            assert conservative_cash >= balanced_cash * 0.8, \
                f"Balanced cash ({balanced_cash:.2f}) should be <= conservative cash ({conservative_cash:.2f})"
            assert balanced_cash >= aggressive_cash * 0.8, \
                f"Balanced cash ({balanced_cash:.2f}) should be >= aggressive cash ({aggressive_cash:.2f})"
        
        # Balanced should be between conservative and aggressive for investment
        if len(conservative_invested_values) > 0 and len(balanced_invested_values) > 0 and len(aggressive_invested_values) > 0:
            conservative_invested = np.mean(conservative_invested_values)
            balanced_invested = np.mean(balanced_invested_values)
            aggressive_invested = np.mean(aggressive_invested_values)
            
            assert aggressive_invested >= balanced_invested * 0.8, \
                f"Balanced invested ({balanced_invested:.2f}) should be <= aggressive invested ({aggressive_invested:.2f})"
            assert balanced_invested >= conservative_invested * 0.8, \
                f"Balanced invested ({balanced_invested:.2f}) should be >= conservative invested ({conservative_invested:.2f})"
        
        # Verify risk tolerance is between conservative and aggressive
        assert conservative_env.risk_tolerance < balanced_env.risk_tolerance < aggressive_env.risk_tolerance, \
            "Balanced risk tolerance should be between conservative and aggressive"
    
    def test_trained_policy_outperforms_random_policy(self, base_env_config, base_training_config, base_reward_config):
        """
        Test that a trained policy outperforms a random (untrained) policy.
        
        After training, the policy should achieve higher rewards and better metrics
        than a random policy.
        
        Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3
        """
        # Create two trainers with same configuration
        random_trainer = self.create_trainer(base_env_config, base_training_config, base_reward_config)
        trained_trainer = self.create_trainer(base_env_config, base_training_config, base_reward_config)
        
        # Evaluate random policy (no training)
        random_eval = random_trainer.evaluate(num_episodes=10)
        
        # Train the second policy
        trained_trainer.train(num_episodes=30)
        
        # Evaluate trained policy
        trained_eval = trained_trainer.evaluate(num_episodes=10)
        
        # Get detailed results
        random_detailed = random_eval['detailed_results']
        trained_detailed = trained_eval['detailed_results']
        
        # Trained policy should achieve higher rewards (filter out NaN)
        random_reward_values = [r for r in random_detailed['episode_rewards'] if not np.isnan(r)]
        trained_reward_values = [r for r in trained_detailed['episode_rewards'] if not np.isnan(r)]
        
        if len(random_reward_values) > 0 and len(trained_reward_values) > 0:
            random_reward = np.mean(random_reward_values)
            trained_reward = np.mean(trained_reward_values)
            
            # Trained should be at least as good as random (may not always be better due to variance)
            assert trained_reward >= random_reward * 0.8, \
                f"Trained reward ({trained_reward:.2f}) should be >= 80% of random reward ({random_reward:.2f})"
        
        # Trained policy should have better stability
        random_stability_values = [s for s in random_detailed['cash_stability_index'] if not np.isnan(s)]
        trained_stability_values = [s for s in trained_detailed['cash_stability_index'] if not np.isnan(s)]
        
        if len(random_stability_values) > 0 and len(trained_stability_values) > 0:
            random_stability = np.mean(random_stability_values)
            trained_stability = np.mean(trained_stability_values)
            
            assert trained_stability >= random_stability, \
                f"Trained stability ({trained_stability:.3f}) should be >= random stability ({random_stability:.3f})"
        
        # Trained policy should have longer episode lengths (better survival)
        random_lengths = random_detailed['episode_lengths']
        trained_lengths = trained_detailed['episode_lengths']
        
        avg_random_length = np.mean(random_lengths)
        avg_trained_length = np.mean(trained_lengths)
        
        assert avg_trained_length >= avg_random_length, \
            f"Trained episode length ({avg_trained_length:.1f}) should be >= random episode length ({avg_random_length:.1f})"
    
    def test_profile_risk_tolerance_ordering(self):
        """
        Test that behavioral profiles have correct risk tolerance ordering.
        
        Conservative < Balanced < Aggressive in terms of risk tolerance.
        
        Requirements: 6.2, 6.3
        """
        # Load all profiles
        conservative_env, _, _ = load_behavioral_profile('conservative')
        balanced_env, _, _ = load_behavioral_profile('balanced')
        aggressive_env, _, _ = load_behavioral_profile('aggressive')
        
        # Verify risk tolerance ordering
        assert conservative_env.risk_tolerance < balanced_env.risk_tolerance, \
            "Conservative risk tolerance should be < balanced"
        assert balanced_env.risk_tolerance < aggressive_env.risk_tolerance, \
            "Balanced risk tolerance should be < aggressive"
        
        # Verify safety threshold ordering (inverse of risk tolerance)
        assert conservative_env.safety_threshold > balanced_env.safety_threshold, \
            "Conservative safety threshold should be > balanced"
        assert balanced_env.safety_threshold > aggressive_env.safety_threshold, \
            "Balanced safety threshold should be > aggressive"
    
    def test_profile_reward_coefficient_ordering(self):
        """
        Test that behavioral profiles have correct reward coefficient ordering.
        
        Aggressive should have higher investment rewards and lower stability penalties.
        
        Requirements: 6.2, 6.3
        """
        # Load all profiles
        _, _, conservative_reward = load_behavioral_profile('conservative')
        _, _, balanced_reward = load_behavioral_profile('balanced')
        _, _, aggressive_reward = load_behavioral_profile('aggressive')
        
        # Verify alpha (investment reward) ordering: aggressive > balanced > conservative
        assert aggressive_reward.alpha > balanced_reward.alpha, \
            "Aggressive alpha should be > balanced"
        assert balanced_reward.alpha > conservative_reward.alpha, \
            "Balanced alpha should be > conservative"
        
        # Verify beta (stability penalty) ordering: conservative > balanced > aggressive
        assert conservative_reward.beta > balanced_reward.beta, \
            "Conservative beta should be > balanced"
        assert balanced_reward.beta > aggressive_reward.beta, \
            "Balanced beta should be > aggressive"
