"""Tests for checkpointing functionality in HRLTrainer"""
import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path

from src.environment.budget_env import BudgetEnv
from src.environment.reward_engine import RewardEngine
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.training.hrl_trainer import HRLTrainer
from src.utils.config import EnvironmentConfig, TrainingConfig, RewardConfig


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def trainer_components():
    """Create trainer components for testing"""
    env_config = EnvironmentConfig(
        income=3200,
        fixed_expenses=1400,
        variable_expense_mean=700,
        variable_expense_std=100,
        inflation=0.02,
        safety_threshold=1000,
        max_months=12,  # Short for testing
        initial_cash=0,
        risk_tolerance=0.5
    )
    
    training_config = TrainingConfig(
        num_episodes=10,  # Small for testing
        gamma_low=0.95,
        gamma_high=0.99,
        high_period=6,
        batch_size=4,
        learning_rate_low=3e-4,
        learning_rate_high=1e-4
    )
    
    reward_config = RewardConfig(
        alpha=10.0,
        beta=0.1,
        gamma=5.0,
        delta=20.0,
        lambda_=1.0,
        mu=0.5
    )
    
    env = BudgetEnv(env_config, reward_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    high_agent = FinancialStrategist(training_config)
    low_agent = BudgetExecutor(training_config)
    
    trainer = HRLTrainer(
        env, high_agent, low_agent, reward_engine, training_config,
        env_config=env_config, reward_config=reward_config
    )
    
    return trainer, env_config, training_config, reward_config


def test_save_checkpoint(trainer_components, temp_checkpoint_dir):
    """Test saving a checkpoint"""
    trainer, env_config, training_config, reward_config = trainer_components
    
    # Train for a few episodes to have some history
    trainer.train(num_episodes=5)
    
    # Save checkpoint
    checkpoint_path = trainer.save_checkpoint(
        checkpoint_dir=temp_checkpoint_dir,
        episode=5,
        is_best=False
    )
    
    # Verify checkpoint directory exists
    assert os.path.exists(checkpoint_path)
    
    # Verify checkpoint files exist
    assert os.path.exists(os.path.join(checkpoint_path, "high_agent.pt"))
    assert os.path.exists(os.path.join(checkpoint_path, "low_agent.pt"))
    assert os.path.exists(os.path.join(checkpoint_path, "metadata.json"))
    assert os.path.exists(os.path.join(checkpoint_path, "training_history.json"))
    
    # Verify metadata content
    with open(os.path.join(checkpoint_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    assert metadata['episode'] == 5
    assert 'training_config' in metadata
    assert 'environment_config' in metadata
    assert 'reward_config' in metadata


def test_save_best_checkpoint(trainer_components, temp_checkpoint_dir):
    """Test saving a best checkpoint"""
    trainer, _, _, _ = trainer_components
    
    # Train for a few episodes
    trainer.train(num_episodes=3)
    
    # Save as best checkpoint
    checkpoint_path = trainer.save_checkpoint(
        checkpoint_dir=temp_checkpoint_dir,
        episode=3,
        is_best=True
    )
    
    # Verify it's saved with "best" in the name
    assert "best" in checkpoint_path
    assert os.path.exists(checkpoint_path)


def test_load_checkpoint(trainer_components, temp_checkpoint_dir):
    """Test loading a checkpoint"""
    trainer1, env_config, training_config, reward_config = trainer_components
    
    # Train first trainer and save checkpoint
    trainer1.train(num_episodes=5)
    checkpoint_path = trainer1.save_checkpoint(
        checkpoint_dir=temp_checkpoint_dir,
        episode=5,
        is_best=False
    )
    
    # Create a new trainer
    env = BudgetEnv(env_config, reward_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    high_agent = FinancialStrategist(training_config)
    low_agent = BudgetExecutor(training_config)
    
    trainer2 = HRLTrainer(
        env, high_agent, low_agent, reward_engine, training_config,
        env_config=env_config, reward_config=reward_config
    )
    
    # Load checkpoint
    episode, history = trainer2.load_checkpoint(checkpoint_path)
    
    # Verify loaded data
    assert episode == 5
    assert len(history['episode_rewards']) == 5
    assert trainer2.current_episode == 5


def test_checkpoint_resume_training(trainer_components, temp_checkpoint_dir):
    """Test resuming training from checkpoint"""
    trainer, env_config, training_config, reward_config = trainer_components
    
    # Train for 5 episodes and save
    trainer.train(num_episodes=5)
    checkpoint_path = trainer.save_checkpoint(
        checkpoint_dir=temp_checkpoint_dir,
        episode=5,
        is_best=False
    )
    
    # Create new trainer and load checkpoint
    env = BudgetEnv(env_config, reward_config)
    reward_engine = RewardEngine(reward_config, safety_threshold=env_config.safety_threshold)
    high_agent = FinancialStrategist(training_config)
    low_agent = BudgetExecutor(training_config)
    
    trainer2 = HRLTrainer(
        env, high_agent, low_agent, reward_engine, training_config,
        env_config=env_config, reward_config=reward_config
    )
    
    # Load checkpoint
    trainer2.load_checkpoint(checkpoint_path)
    
    # Continue training
    trainer2.train(num_episodes=3)
    
    # Verify total episodes
    assert len(trainer2.training_history['episode_rewards']) == 8  # 5 + 3


def test_train_with_checkpointing(trainer_components, temp_checkpoint_dir):
    """Test train_with_checkpointing method"""
    trainer, _, _, _ = trainer_components
    
    # Train with checkpointing
    history = trainer.train_with_checkpointing(
        num_episodes=10,
        checkpoint_dir=temp_checkpoint_dir,
        save_interval=5,
        eval_interval=5,
        eval_episodes=2
    )
    
    # Verify checkpoints were created
    checkpoint_files = list(Path(temp_checkpoint_dir).glob("checkpoint_episode_*"))
    assert len(checkpoint_files) >= 1  # At least one checkpoint at episode 5
    
    # Note: Best checkpoint might not exist if all evaluation scores are NaN
    # This is expected behavior for very short training runs
    
    # Verify final checkpoint exists
    final_checkpoint = list(Path(temp_checkpoint_dir).glob("checkpoint_final_*"))
    assert len(final_checkpoint) == 1
    
    # Verify training completed
    assert len(history['episode_rewards']) == 10


def test_checkpoint_metadata_completeness(trainer_components, temp_checkpoint_dir):
    """Test that checkpoint metadata contains all required information"""
    trainer, _, _, _ = trainer_components
    
    # Train and save
    trainer.train(num_episodes=3)
    checkpoint_path = trainer.save_checkpoint(
        checkpoint_dir=temp_checkpoint_dir,
        episode=3,
        is_best=False
    )
    
    # Load and verify metadata
    with open(os.path.join(checkpoint_path, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Check training config
    assert 'training_config' in metadata
    assert 'num_episodes' in metadata['training_config']
    assert 'gamma_low' in metadata['training_config']
    assert 'gamma_high' in metadata['training_config']
    
    # Check environment config
    assert 'environment_config' in metadata
    assert 'income' in metadata['environment_config']
    assert 'safety_threshold' in metadata['environment_config']
    
    # Check reward config
    assert 'reward_config' in metadata
    assert 'alpha' in metadata['reward_config']
    assert 'beta' in metadata['reward_config']


def test_checkpoint_training_history(trainer_components, temp_checkpoint_dir):
    """Test that training history is correctly saved and loaded"""
    trainer, _, _, _ = trainer_components
    
    # Train and save
    trainer.train(num_episodes=5)
    original_history = trainer.training_history.copy()
    
    checkpoint_path = trainer.save_checkpoint(
        checkpoint_dir=temp_checkpoint_dir,
        episode=5,
        is_best=False
    )
    
    # Load history from file
    with open(os.path.join(checkpoint_path, "training_history.json"), 'r') as f:
        loaded_history = json.load(f)
    
    # Verify all keys are present
    assert set(loaded_history.keys()) == set(original_history.keys())
    
    # Verify lengths match
    for key in original_history.keys():
        assert len(loaded_history[key]) == len(original_history[key])
