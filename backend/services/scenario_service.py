"""Scenario service layer for managing financial scenarios

This module provides business logic for:
- Creating and validating scenarios
- Reading scenario configurations
- Listing available scenarios
- Updating and deleting scenarios
- Managing scenario templates
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from backend.models.requests import ScenarioConfig, EnvironmentConfig, TrainingConfig, RewardConfig
from backend.utils.file_manager import (
    read_yaml_config,
    write_yaml_config,
    delete_yaml_config,
    list_yaml_configs,
    SCENARIOS_DIR,
    BASE_DIR
)


class ScenarioService:
    """Service for managing financial scenarios"""
    
    @staticmethod
    def create_scenario(scenario: ScenarioConfig) -> Dict[str, Any]:
        """Create a new scenario
        
        Args:
            scenario: ScenarioConfig with all configuration data
            
        Returns:
            Dictionary with created scenario information
            
        Raises:
            ValueError: If scenario name already exists or validation fails
        """
        # Check if scenario already exists
        try:
            existing = ScenarioService.get_scenario(scenario.name)
            if existing:
                raise ValueError(f"Scenario '{scenario.name}' already exists")
        except FileNotFoundError:
            # Good - scenario doesn't exist yet
            pass
        
        # Convert Pydantic model to dictionary for YAML storage
        scenario_dict = {
            'environment': scenario.environment.model_dump(),
            'training': scenario.training.model_dump(),
            'reward': scenario.reward.model_dump(by_alias=True)  # Use alias for lambda_
        }
        
        # Add metadata
        metadata = {
            'name': scenario.name,
            'description': scenario.description,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        # Write to file
        file_path = write_yaml_config(scenario.name, scenario_dict, scenarios=True)
        
        return {
            'name': scenario.name,
            'description': scenario.description,
            'path': str(file_path.relative_to(BASE_DIR)),
            'created_at': metadata['created_at'],
            'updated_at': metadata['updated_at']
        }
    
    @staticmethod
    def get_scenario(name: str) -> Dict[str, Any]:
        """Get scenario configuration by name
        
        Args:
            name: Scenario name
            
        Returns:
            Dictionary with complete scenario configuration
            
        Raises:
            FileNotFoundError: If scenario doesn't exist
            ValueError: If scenario file is invalid
        """
        # Read YAML file
        config = read_yaml_config(name, scenarios=True)
        
        # Validate structure
        if 'environment' not in config:
            raise ValueError(f"Scenario '{name}' missing 'environment' section")
        
        # Add defaults for optional sections
        if 'training' not in config:
            config['training'] = TrainingConfig().model_dump()
        if 'reward' not in config:
            config['reward'] = RewardConfig().model_dump(by_alias=True)
        
        # Validate using Pydantic models
        try:
            env_config = EnvironmentConfig(**config['environment'])
            training_config = TrainingConfig(**config['training'])
            reward_config = RewardConfig(**config['reward'])
        except Exception as e:
            raise ValueError(f"Invalid scenario configuration: {str(e)}")
        
        # Get file metadata
        file_path = SCENARIOS_DIR / f"{name}.yaml"
        if not file_path.exists():
            file_path = SCENARIOS_DIR / f"{name}.yml"
        
        stat = file_path.stat()
        
        return {
            'name': name,
            'description': config.get('description'),
            'environment': env_config.model_dump(),
            'training': training_config.model_dump(),
            'reward': reward_config.model_dump(by_alias=True),
            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'updated_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'size': stat.st_size
        }
    
    @staticmethod
    def list_scenarios() -> List[Dict[str, Any]]:
        """List all available scenarios
        
        Returns:
            List of scenario summaries with basic information
        """
        configs = list_yaml_configs(scenarios=True)
        
        scenarios = []
        for config_info in configs:
            try:
                # Read basic info from file
                config = read_yaml_config(config_info['name'], scenarios=True)
                
                # Extract key metrics
                env = config.get('environment', {})
                income = env.get('income', 0)
                fixed = env.get('fixed_expenses', 0)
                variable = env.get('variable_expense_mean', 0)
                available = income - fixed - variable
                available_pct = (available / income * 100) if income > 0 else 0
                
                scenarios.append({
                    'name': config_info['name'],
                    'description': config.get('description', ''),
                    'income': income,
                    'fixed_expenses': fixed,
                    'variable_expenses': variable,
                    'available_monthly': available,
                    'available_pct': round(available_pct, 1),
                    'risk_tolerance': env.get('risk_tolerance', 0.5),
                    'updated_at': config_info['modified'].isoformat(),
                    'size': config_info['size']
                })
            except Exception:
                # Skip invalid scenarios
                continue
        
        return scenarios
    
    @staticmethod
    def update_scenario(name: str, scenario: ScenarioConfig) -> Dict[str, Any]:
        """Update an existing scenario
        
        Args:
            name: Current scenario name
            scenario: Updated ScenarioConfig
            
        Returns:
            Dictionary with updated scenario information
            
        Raises:
            FileNotFoundError: If scenario doesn't exist
            ValueError: If validation fails or trying to rename to existing name
        """
        # Check if scenario exists
        existing = ScenarioService.get_scenario(name)
        
        # If renaming, check new name doesn't exist
        if scenario.name != name:
            try:
                ScenarioService.get_scenario(scenario.name)
                raise ValueError(f"Scenario '{scenario.name}' already exists")
            except FileNotFoundError:
                # Good - new name doesn't exist
                pass
            
            # Delete old file
            delete_yaml_config(name, scenarios=True)
        
        # Convert Pydantic model to dictionary
        scenario_dict = {
            'environment': scenario.environment.model_dump(),
            'training': scenario.training.model_dump(),
            'reward': scenario.reward.model_dump(by_alias=True)
        }
        
        # Add description if provided
        if scenario.description:
            scenario_dict['description'] = scenario.description
        
        # Write to file (with new name if renamed)
        file_path = write_yaml_config(scenario.name, scenario_dict, scenarios=True)
        
        return {
            'name': scenario.name,
            'description': scenario.description,
            'path': str(file_path.relative_to(BASE_DIR)),
            'updated_at': datetime.now().isoformat()
        }
    
    @staticmethod
    def delete_scenario(name: str) -> bool:
        """Delete a scenario
        
        Args:
            name: Scenario name to delete
            
        Returns:
            True if deleted, False if didn't exist
        """
        return delete_yaml_config(name, scenarios=True)
    
    @staticmethod
    def get_templates() -> List[Dict[str, Any]]:
        """Get preset scenario templates
        
        Returns:
            List of template configurations
        """
        templates = [
            {
                'name': 'conservative',
                'display_name': 'Conservative Profile',
                'description': 'Low-risk profile with high savings buffer',
                'environment': {
                    'income': 2500,
                    'fixed_expenses': 1200,
                    'variable_expense_mean': 600,
                    'variable_expense_std': 100,
                    'inflation': 0.02,
                    'safety_threshold': 7500,
                    'max_months': 120,
                    'initial_cash': 10000,
                    'risk_tolerance': 0.3,
                    'investment_return_mean': 0.004,
                    'investment_return_std': 0.015,
                    'investment_return_type': 'stochastic'
                },
                'training': TrainingConfig().model_dump(),
                'reward': RewardConfig().model_dump(by_alias=True)
            },
            {
                'name': 'balanced',
                'display_name': 'Balanced Profile',
                'description': 'Moderate risk with balanced savings and investment',
                'environment': {
                    'income': 3000,
                    'fixed_expenses': 1400,
                    'variable_expense_mean': 700,
                    'variable_expense_std': 120,
                    'inflation': 0.02,
                    'safety_threshold': 6000,
                    'max_months': 120,
                    'initial_cash': 8000,
                    'risk_tolerance': 0.5,
                    'investment_return_mean': 0.005,
                    'investment_return_std': 0.02,
                    'investment_return_type': 'stochastic'
                },
                'training': TrainingConfig().model_dump(),
                'reward': RewardConfig().model_dump(by_alias=True)
            },
            {
                'name': 'aggressive',
                'display_name': 'Aggressive Profile',
                'description': 'High-risk profile focused on investment growth',
                'environment': {
                    'income': 3500,
                    'fixed_expenses': 1500,
                    'variable_expense_mean': 800,
                    'variable_expense_std': 150,
                    'inflation': 0.02,
                    'safety_threshold': 4500,
                    'max_months': 120,
                    'initial_cash': 5000,
                    'risk_tolerance': 0.8,
                    'investment_return_mean': 0.007,
                    'investment_return_std': 0.03,
                    'investment_return_type': 'stochastic'
                },
                'training': TrainingConfig().model_dump(),
                'reward': RewardConfig().model_dump(by_alias=True)
            },
            {
                'name': 'young_professional',
                'display_name': 'Young Professional',
                'description': 'Single professional with owned home, moderate expenses',
                'environment': {
                    'income': 2000,
                    'fixed_expenses': 770,
                    'variable_expense_mean': 500,
                    'variable_expense_std': 120,
                    'inflation': 0.02,
                    'safety_threshold': 5000,
                    'max_months': 120,
                    'initial_cash': 5000,
                    'risk_tolerance': 0.65,
                    'investment_return_mean': 0.005,
                    'investment_return_std': 0.02,
                    'investment_return_type': 'stochastic'
                },
                'training': TrainingConfig().model_dump(),
                'reward': RewardConfig().model_dump(by_alias=True)
            },
            {
                'name': 'young_couple',
                'display_name': 'Young Couple',
                'description': 'Dual income couple with rental, moderate-high risk',
                'environment': {
                    'income': 3200,
                    'fixed_expenses': 1800,
                    'variable_expense_mean': 800,
                    'variable_expense_std': 150,
                    'inflation': 0.02,
                    'safety_threshold': 3500,
                    'max_months': 120,
                    'initial_cash': 10000,
                    'risk_tolerance': 0.55,
                    'investment_return_mean': 0.005,
                    'investment_return_std': 0.02,
                    'investment_return_type': 'stochastic'
                },
                'training': TrainingConfig().model_dump(),
                'reward': RewardConfig().model_dump(by_alias=True)
            }
        ]
        
        return templates
