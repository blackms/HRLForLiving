"""Configuration manager for loading and validating system configurations"""
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple
from .config import EnvironmentConfig, TrainingConfig, RewardConfig, BehavioralProfile


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters"""
    pass


def load_config(yaml_path: str) -> Tuple[EnvironmentConfig, TrainingConfig, RewardConfig]:
    """
    Load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        Tuple of (EnvironmentConfig, TrainingConfig, RewardConfig)
        
    Raises:
        ConfigurationError: If file not found or configuration is invalid
        
    Example YAML structure:
        environment:
            income: 3200
            fixed_expenses: 1400
            variable_expense_mean: 700
            variable_expense_std: 100
            inflation: 0.02
            safety_threshold: 1000
            max_months: 60
            initial_cash: 0
            risk_tolerance: 0.5
        training:
            num_episodes: 5000
            gamma_low: 0.95
            gamma_high: 0.99
            high_period: 6
            batch_size: 32
            learning_rate_low: 0.0003
            learning_rate_high: 0.0001
        reward:
            alpha: 10.0
            beta: 0.1
            gamma: 5.0
            delta: 20.0
            lambda_: 1.0
            mu: 0.5
    """
    path = Path(yaml_path)
    
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {yaml_path}")
    
    try:
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file: {e}")
    
    if config_dict is None:
        raise ConfigurationError("Configuration file is empty")
    
    # Extract configuration sections
    env_config_dict = config_dict.get('environment', {})
    training_config_dict = config_dict.get('training', {})
    reward_config_dict = config_dict.get('reward', {})
    
    # Create configuration instances
    try:
        env_config = EnvironmentConfig(**env_config_dict)
        training_config = TrainingConfig(**training_config_dict)
        reward_config = RewardConfig(**reward_config_dict)
    except TypeError as e:
        raise ConfigurationError(f"Invalid configuration parameters: {e}")
    
    # Validate configurations
    _validate_environment_config(env_config)
    _validate_training_config(training_config)
    _validate_reward_config(reward_config)
    
    return env_config, training_config, reward_config


def load_behavioral_profile(profile_name: str) -> Tuple[EnvironmentConfig, TrainingConfig, RewardConfig]:
    """
    Load predefined behavioral profile configuration.
    
    Args:
        profile_name: Name of behavioral profile ("conservative", "balanced", or "aggressive")
        
    Returns:
        Tuple of (EnvironmentConfig, TrainingConfig, RewardConfig) with profile-specific settings
        
    Raises:
        ConfigurationError: If profile_name is not recognized
        
    Profiles:
        - conservative: Low risk tolerance, high safety threshold, lower investment rewards
        - balanced: Medium risk tolerance, standard safety threshold, standard rewards
        - aggressive: High risk tolerance, low safety threshold, higher investment rewards
    """
    profile_name_lower = profile_name.lower()
    
    # Map profile names to enum
    profile_map = {
        'conservative': BehavioralProfile.CONSERVATIVE,
        'balanced': BehavioralProfile.BALANCED,
        'aggressive': BehavioralProfile.AGGRESSIVE
    }
    
    if profile_name_lower not in profile_map:
        raise ConfigurationError(
            f"Unknown behavioral profile: {profile_name}. "
            f"Valid options are: {', '.join(profile_map.keys())}"
        )
    
    profile = profile_map[profile_name_lower]
    profile_data = profile.value
    
    # Create environment config with profile-specific settings
    env_config = EnvironmentConfig(
        risk_tolerance=profile_data['risk_tolerance'],
        safety_threshold=profile_data['safety_threshold']
    )
    
    # Create reward config with profile-specific coefficients
    reward_data = profile_data['reward_config']
    reward_config = RewardConfig(
        alpha=reward_data['alpha'],
        beta=reward_data['beta'],
        gamma=reward_data['gamma'],
        delta=reward_data['delta'],
        lambda_=reward_data['lambda_'],
        mu=reward_data['mu']
    )
    
    # Use default training config
    training_config = TrainingConfig()
    
    return env_config, training_config, reward_config


def _validate_environment_config(config: EnvironmentConfig) -> None:
    """
    Validate environment configuration parameters.
    
    Args:
        config: EnvironmentConfig instance to validate
        
    Raises:
        ConfigurationError: If any parameter is invalid
    """
    if config.income <= 0:
        raise ConfigurationError(f"income must be positive, got {config.income}")
    
    if config.fixed_expenses < 0:
        raise ConfigurationError(f"fixed_expenses must be non-negative, got {config.fixed_expenses}")
    
    if config.variable_expense_mean < 0:
        raise ConfigurationError(f"variable_expense_mean must be non-negative, got {config.variable_expense_mean}")
    
    if config.variable_expense_std < 0:
        raise ConfigurationError(f"variable_expense_std must be non-negative, got {config.variable_expense_std}")
    
    if config.inflation < -1 or config.inflation > 1:
        raise ConfigurationError(f"inflation must be in [-1, 1], got {config.inflation}")
    
    if config.safety_threshold < 0:
        raise ConfigurationError(f"safety_threshold must be non-negative, got {config.safety_threshold}")
    
    if config.max_months <= 0:
        raise ConfigurationError(f"max_months must be positive, got {config.max_months}")
    
    if config.initial_cash < 0:
        raise ConfigurationError(f"initial_cash must be non-negative, got {config.initial_cash}")
    
    if config.risk_tolerance < 0 or config.risk_tolerance > 1:
        raise ConfigurationError(f"risk_tolerance must be in [0, 1], got {config.risk_tolerance}")


def _validate_training_config(config: TrainingConfig) -> None:
    """
    Validate training configuration parameters.
    
    Args:
        config: TrainingConfig instance to validate
        
    Raises:
        ConfigurationError: If any parameter is invalid
    """
    if config.num_episodes <= 0:
        raise ConfigurationError(f"num_episodes must be positive, got {config.num_episodes}")
    
    if config.gamma_low < 0 or config.gamma_low > 1:
        raise ConfigurationError(f"gamma_low must be in [0, 1], got {config.gamma_low}")
    
    if config.gamma_high < 0 or config.gamma_high > 1:
        raise ConfigurationError(f"gamma_high must be in [0, 1], got {config.gamma_high}")
    
    if config.high_period <= 0:
        raise ConfigurationError(f"high_period must be positive, got {config.high_period}")
    
    if config.batch_size <= 0:
        raise ConfigurationError(f"batch_size must be positive, got {config.batch_size}")
    
    if config.learning_rate_low <= 0:
        raise ConfigurationError(f"learning_rate_low must be positive, got {config.learning_rate_low}")
    
    if config.learning_rate_high <= 0:
        raise ConfigurationError(f"learning_rate_high must be positive, got {config.learning_rate_high}")


def _validate_reward_config(config: RewardConfig) -> None:
    """
    Validate reward configuration parameters.
    
    Args:
        config: RewardConfig instance to validate
        
    Raises:
        ConfigurationError: If any parameter is invalid
    """
    # All reward coefficients should be non-negative
    if config.alpha < 0:
        raise ConfigurationError(f"alpha must be non-negative, got {config.alpha}")
    
    if config.beta < 0:
        raise ConfigurationError(f"beta must be non-negative, got {config.beta}")
    
    if config.gamma < 0:
        raise ConfigurationError(f"gamma must be non-negative, got {config.gamma}")
    
    if config.delta < 0:
        raise ConfigurationError(f"delta must be non-negative, got {config.delta}")
    
    if config.lambda_ < 0:
        raise ConfigurationError(f"lambda_ must be non-negative, got {config.lambda_}")
    
    if config.mu < 0:
        raise ConfigurationError(f"mu must be non-negative, got {config.mu}")
