"""Pytest configuration and fixtures for backend tests"""
import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def temp_configs_dir():
    """Create a temporary configs directory for testing"""
    temp_dir = tempfile.mkdtemp()
    configs_dir = Path(temp_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Store original configs path
    original_path = Path("configs")
    
    yield configs_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_models_dir():
    """Create a temporary models directory for testing"""
    temp_dir = tempfile.mkdtemp()
    models_dir = Path(temp_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    yield models_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_results_dir():
    """Create a temporary results directory for testing"""
    temp_dir = tempfile.mkdtemp()
    results_dir = Path(temp_dir) / "results" / "simulations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    yield results_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_scenario_config():
    """Sample scenario configuration for testing"""
    return {
        "name": "test_scenario",
        "description": "Test scenario for unit tests",
        "environment": {
            "income": 2000.0,
            "fixed_expenses": 800.0,
            "variable_expense_mean": 400.0,
            "variable_expense_std": 100.0,
            "inflation": 0.02,
            "safety_threshold": 3000.0,
            "max_months": 120,
            "initial_cash": 5000.0,
            "risk_tolerance": 0.5,
            "investment_return_mean": 0.005,
            "investment_return_std": 0.02,
            "investment_return_type": "stochastic"
        },
        "training": {
            "num_episodes": 100,
            "save_interval": 50,
            "eval_episodes": 5
        },
        "reward": {
            "wealth_weight": 1.0,
            "stability_weight": 0.5,
            "goal_weight": 0.3
        }
    }


@pytest.fixture
def sample_training_request():
    """Sample training request for testing"""
    return {
        "scenario_name": "test_scenario",
        "num_episodes": 10,
        "save_interval": 5,
        "eval_episodes": 2,
        "seed": 42
    }


@pytest.fixture
def sample_simulation_request():
    """Sample simulation request for testing"""
    return {
        "model_name": "test_model",
        "scenario_name": "test_scenario",
        "num_episodes": 5,
        "seed": 42
    }
