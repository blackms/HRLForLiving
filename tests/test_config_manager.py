"""Tests for configuration manager"""
import pytest
import tempfile
import yaml
from pathlib import Path
from src.utils.config_manager import (
    load_config,
    load_behavioral_profile,
    ConfigurationError
)
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig


class TestLoadConfig:
    """Tests for load_config function"""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file"""
        config_data = {
            'environment': {
                'income': 4000,
                'fixed_expenses': 1500,
                'variable_expense_mean': 800,
                'variable_expense_std': 150,
                'inflation': 0.03,
                'safety_threshold': 1200,
                'max_months': 48,
                'initial_cash': 500,
                'risk_tolerance': 0.6
            },
            'training': {
                'num_episodes': 3000,
                'gamma_low': 0.9,
                'gamma_high': 0.98,
                'high_period': 8,
                'batch_size': 64,
                'learning_rate_low': 0.0005,
                'learning_rate_high': 0.0002
            },
            'reward': {
                'alpha': 12.0,
                'beta': 0.15,
                'gamma': 6.0,
                'delta': 25.0,
                'lambda_': 1.5,
                'mu': 0.6
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            env_config, training_config, reward_config = load_config(temp_path)
            
            assert env_config.income == 4000
            assert env_config.fixed_expenses == 1500
            assert env_config.risk_tolerance == 0.6
            
            assert training_config.num_episodes == 3000
            assert training_config.gamma_low == 0.9
            
            assert reward_config.alpha == 12.0
            assert reward_config.beta == 0.15
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_with_defaults(self):
        """Test loading config with partial data uses defaults"""
        config_data = {
            'environment': {
                'income': 3500
            },
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            env_config, training_config, reward_config = load_config(temp_path)
            
            assert env_config.income == 3500
            assert env_config.fixed_expenses == 1400  # default
            assert training_config.num_episodes == 5000  # default
            assert reward_config.alpha == 10.0  # default
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist"""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config("nonexistent_file.yaml")
    
    def test_load_config_empty_file(self):
        """Test error when config file is empty"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Configuration file is empty"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_invalid_yaml(self):
        """Test error when YAML is malformed"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestLoadBehavioralProfile:
    """Tests for load_behavioral_profile function"""
    
    def test_load_conservative_profile(self):
        """Test loading conservative behavioral profile"""
        env_config, training_config, reward_config = load_behavioral_profile("conservative")
        
        assert env_config.risk_tolerance == 0.3
        assert env_config.safety_threshold == 1500
        assert reward_config.alpha == 5.0
        assert reward_config.beta == 0.5
    
    def test_load_balanced_profile(self):
        """Test loading balanced behavioral profile"""
        env_config, training_config, reward_config = load_behavioral_profile("balanced")
        
        assert env_config.risk_tolerance == 0.5
        assert env_config.safety_threshold == 1000
        assert reward_config.alpha == 10.0
        assert reward_config.beta == 0.1
    
    def test_load_aggressive_profile(self):
        """Test loading aggressive behavioral profile"""
        env_config, training_config, reward_config = load_behavioral_profile("aggressive")
        
        assert env_config.risk_tolerance == 0.8
        assert env_config.safety_threshold == 500
        assert reward_config.alpha == 15.0
        assert reward_config.beta == 0.05
    
    def test_load_profile_case_insensitive(self):
        """Test profile loading is case insensitive"""
        env1, _, _ = load_behavioral_profile("Conservative")
        env2, _, _ = load_behavioral_profile("CONSERVATIVE")
        env3, _, _ = load_behavioral_profile("conservative")
        
        assert env1.risk_tolerance == env2.risk_tolerance == env3.risk_tolerance
    
    def test_load_invalid_profile(self):
        """Test error when profile name is invalid"""
        with pytest.raises(ConfigurationError, match="Unknown behavioral profile"):
            load_behavioral_profile("invalid_profile")


class TestConfigurationValidation:
    """Tests for configuration validation"""
    
    # Environment validation tests
    def test_invalid_income(self):
        """Test validation fails for non-positive income"""
        config_data = {
            'environment': {'income': -100},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="income must be positive"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_zero_income(self):
        """Test validation fails for zero income"""
        config_data = {
            'environment': {'income': 0},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="income must be positive"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_fixed_expenses(self):
        """Test validation fails for negative fixed expenses"""
        config_data = {
            'environment': {'fixed_expenses': -500},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="fixed_expenses must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_variable_expense_mean(self):
        """Test validation fails for negative variable expense mean"""
        config_data = {
            'environment': {'variable_expense_mean': -200},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="variable_expense_mean must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_variable_expense_std(self):
        """Test validation fails for negative variable expense std"""
        config_data = {
            'environment': {'variable_expense_std': -50},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="variable_expense_std must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_inflation_below_range(self):
        """Test validation fails for inflation below -1"""
        config_data = {
            'environment': {'inflation': -1.5},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="inflation must be in"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_inflation_above_range(self):
        """Test validation fails for inflation above 1"""
        config_data = {
            'environment': {'inflation': 1.5},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="inflation must be in"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_inflation_boundary_values(self):
        """Test inflation boundary values are accepted"""
        config_data = {
            'environment': {'inflation': -1.0},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            env_config, _, _ = load_config(temp_path)
            assert env_config.inflation == -1.0
        finally:
            Path(temp_path).unlink()
        
        config_data['environment']['inflation'] = 1.0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            env_config, _, _ = load_config(temp_path)
            assert env_config.inflation == 1.0
        finally:
            Path(temp_path).unlink()
    
    def test_negative_safety_threshold(self):
        """Test validation fails for negative safety threshold"""
        config_data = {
            'environment': {'safety_threshold': -100},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="safety_threshold must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_zero_max_months(self):
        """Test validation fails for zero max_months"""
        config_data = {
            'environment': {'max_months': 0},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="max_months must be positive"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_initial_cash(self):
        """Test validation fails for negative initial cash"""
        config_data = {
            'environment': {'initial_cash': -500},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="initial_cash must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_risk_tolerance(self):
        """Test validation fails for risk_tolerance out of range"""
        config_data = {
            'environment': {'risk_tolerance': 1.5},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="risk_tolerance must be in"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_risk_tolerance_below_range(self):
        """Test validation fails for risk_tolerance below 0"""
        config_data = {
            'environment': {'risk_tolerance': -0.1},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="risk_tolerance must be in"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_risk_tolerance_boundary_values(self):
        """Test risk_tolerance boundary values are accepted"""
        config_data = {
            'environment': {'risk_tolerance': 0.0},
            'training': {},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            env_config, _, _ = load_config(temp_path)
            assert env_config.risk_tolerance == 0.0
        finally:
            Path(temp_path).unlink()
        
        config_data['environment']['risk_tolerance'] = 1.0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            env_config, _, _ = load_config(temp_path)
            assert env_config.risk_tolerance == 1.0
        finally:
            Path(temp_path).unlink()
    
    # Training validation tests
    def test_zero_num_episodes(self):
        """Test validation fails for zero num_episodes"""
        config_data = {
            'environment': {},
            'training': {'num_episodes': 0},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="num_episodes must be positive"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_gamma_low(self):
        """Test validation fails for gamma_low out of range"""
        config_data = {
            'environment': {},
            'training': {'gamma_low': 1.5},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="gamma_low must be in"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_gamma_low_below_range(self):
        """Test validation fails for gamma_low below 0"""
        config_data = {
            'environment': {},
            'training': {'gamma_low': -0.1},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="gamma_low must be in"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_gamma_low_boundary_values(self):
        """Test gamma_low boundary values are accepted"""
        config_data = {
            'environment': {},
            'training': {'gamma_low': 0.0},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            _, training_config, _ = load_config(temp_path)
            assert training_config.gamma_low == 0.0
        finally:
            Path(temp_path).unlink()
        
        config_data['training']['gamma_low'] = 1.0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            _, training_config, _ = load_config(temp_path)
            assert training_config.gamma_low == 1.0
        finally:
            Path(temp_path).unlink()
    
    def test_invalid_gamma_high(self):
        """Test validation fails for gamma_high out of range"""
        config_data = {
            'environment': {},
            'training': {'gamma_high': 1.5},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="gamma_high must be in"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_gamma_high_boundary_values(self):
        """Test gamma_high boundary values are accepted"""
        config_data = {
            'environment': {},
            'training': {'gamma_high': 0.0},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            _, training_config, _ = load_config(temp_path)
            assert training_config.gamma_high == 0.0
        finally:
            Path(temp_path).unlink()
        
        config_data['training']['gamma_high'] = 1.0
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            _, training_config, _ = load_config(temp_path)
            assert training_config.gamma_high == 1.0
        finally:
            Path(temp_path).unlink()
    
    def test_zero_high_period(self):
        """Test validation fails for zero high_period"""
        config_data = {
            'environment': {},
            'training': {'high_period': 0},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="high_period must be positive"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_zero_batch_size(self):
        """Test validation fails for zero batch_size"""
        config_data = {
            'environment': {},
            'training': {'batch_size': 0},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="batch_size must be positive"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_zero_learning_rate_low(self):
        """Test validation fails for zero learning_rate_low"""
        config_data = {
            'environment': {},
            'training': {'learning_rate_low': 0.0},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="learning_rate_low must be positive"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_learning_rate_high(self):
        """Test validation fails for negative learning_rate_high"""
        config_data = {
            'environment': {},
            'training': {'learning_rate_high': -0.001},
            'reward': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="learning_rate_high must be positive"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    # Reward validation tests
    def test_invalid_reward_coefficient(self):
        """Test validation fails for negative reward coefficient"""
        config_data = {
            'environment': {},
            'training': {},
            'reward': {'alpha': -5.0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="alpha must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_beta(self):
        """Test validation fails for negative beta"""
        config_data = {
            'environment': {},
            'training': {},
            'reward': {'beta': -0.5}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="beta must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_gamma_reward(self):
        """Test validation fails for negative gamma reward coefficient"""
        config_data = {
            'environment': {},
            'training': {},
            'reward': {'gamma': -2.0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="gamma must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_delta(self):
        """Test validation fails for negative delta"""
        config_data = {
            'environment': {},
            'training': {},
            'reward': {'delta': -10.0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="delta must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_lambda(self):
        """Test validation fails for negative lambda_"""
        config_data = {
            'environment': {},
            'training': {},
            'reward': {'lambda_': -1.0}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="lambda_ must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_negative_mu(self):
        """Test validation fails for negative mu"""
        config_data = {
            'environment': {},
            'training': {},
            'reward': {'mu': -0.5}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ConfigurationError, match="mu must be non-negative"):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_zero_reward_coefficients_accepted(self):
        """Test that zero values are accepted for reward coefficients"""
        config_data = {
            'environment': {},
            'training': {},
            'reward': {
                'alpha': 0.0,
                'beta': 0.0,
                'gamma': 0.0,
                'delta': 0.0,
                'lambda_': 0.0,
                'mu': 0.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            _, _, reward_config = load_config(temp_path)
            assert reward_config.alpha == 0.0
            assert reward_config.beta == 0.0
            assert reward_config.gamma == 0.0
            assert reward_config.delta == 0.0
            assert reward_config.lambda_ == 0.0
            assert reward_config.mu == 0.0
        finally:
            Path(temp_path).unlink()


class TestConfigurationOverrides:
    """Tests for configuration overrides"""
    
    def test_profile_with_custom_overrides(self):
        """Test that profile can be loaded and then overridden"""
        # Load base profile
        env_config, training_config, reward_config = load_behavioral_profile("conservative")
        
        # Verify base values
        assert env_config.risk_tolerance == 0.3
        assert env_config.income == 3200  # default
        
        # Now load custom config that could override
        config_data = {
            'environment': {
                'income': 5000,
                'risk_tolerance': 0.3  # same as conservative
            },
            'training': {},
            'reward': {
                'alpha': 5.0  # same as conservative
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            env_config2, _, reward_config2 = load_config(temp_path)
            
            # Custom income should be applied
            assert env_config2.income == 5000
            # Risk tolerance matches conservative
            assert env_config2.risk_tolerance == 0.3
            # Reward coefficient matches conservative
            assert reward_config2.alpha == 5.0
        finally:
            Path(temp_path).unlink()
