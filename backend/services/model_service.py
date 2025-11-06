"""Model service layer for managing trained models

This module provides business logic for:
- Listing trained models with metadata
- Getting model details and training history
- Deleting models
- Extracting metadata from training history files
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json

from backend.utils.file_manager import (
    list_pytorch_models,
    delete_pytorch_model,
    MODELS_DIR,
    BASE_DIR
)


class ModelService:
    """Service for managing trained models"""
    
    @staticmethod
    def list_models() -> List[Dict[str, Any]]:
        """List all trained models with metadata
        
        Returns:
            List of model summaries with training information
        """
        models_info = list_pytorch_models()
        
        models = []
        for model_info in models_info:
            try:
                # Try to load metadata and history
                metadata = ModelService._load_metadata(model_info['name'])
                history = ModelService._load_history(model_info['name'])
                
                # Extract key metrics
                model_summary = {
                    'name': model_info['name'],
                    'scenario_name': ModelService._extract_scenario_name(model_info['name']),
                    'size_mb': model_info['size_mb'],
                    'trained_at': model_info['modified'].isoformat(),
                    'has_metadata': model_info['has_metadata']
                }
                
                # Add metadata if available
                if metadata:
                    model_summary['episodes'] = metadata.get('episode', metadata.get('current_episode', 0))
                    
                    # Extract environment config
                    env_config = metadata.get('environment_config', {})
                    model_summary['income'] = env_config.get('income')
                    model_summary['risk_tolerance'] = env_config.get('risk_tolerance')
                
                # Add history metrics if available
                if history:
                    final_metrics = ModelService._extract_final_metrics(history)
                    model_summary.update(final_metrics)
                
                models.append(model_summary)
            except Exception:
                # If we can't load metadata, just include basic info
                models.append({
                    'name': model_info['name'],
                    'size_mb': model_info['size_mb'],
                    'trained_at': model_info['modified'].isoformat(),
                    'has_metadata': model_info['has_metadata'],
                    'scenario_name': ModelService._extract_scenario_name(model_info['name'])
                })
        
        return models
    
    @staticmethod
    def get_model(name: str) -> Dict[str, Any]:
        """Get detailed model information
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with complete model information
            
        Raises:
            FileNotFoundError: If model doesn't exist
        """
        # Check if model files exist
        high_path = MODELS_DIR / f"{name}_high_agent.pt"
        low_path = MODELS_DIR / f"{name}_low_agent.pt"
        
        if not high_path.exists() or not low_path.exists():
            raise FileNotFoundError(f"Model '{name}' not found")
        
        # Get file stats
        high_stat = high_path.stat()
        low_stat = low_path.stat()
        total_size = (high_stat.st_size + low_stat.st_size) / (1024 * 1024)
        
        modified = max(
            datetime.fromtimestamp(high_stat.st_mtime),
            datetime.fromtimestamp(low_stat.st_mtime)
        )
        
        # Load metadata and history
        metadata = ModelService._load_metadata(name)
        history = ModelService._load_history(name)
        
        model_detail = {
            'name': name,
            'scenario_name': ModelService._extract_scenario_name(name),
            'high_agent_path': str(high_path.relative_to(BASE_DIR)),
            'low_agent_path': str(low_path.relative_to(BASE_DIR)),
            'size_mb': round(total_size, 2),
            'trained_at': modified.isoformat(),
            'has_metadata': metadata is not None,
            'has_history': history is not None
        }
        
        # Add metadata if available
        if metadata:
            model_detail['metadata'] = metadata
            model_detail['episodes'] = metadata.get('episode', metadata.get('current_episode', 0))
            
            # Extract configs
            model_detail['environment_config'] = metadata.get('environment_config', {})
            model_detail['training_config'] = metadata.get('training_config', {})
            model_detail['reward_config'] = metadata.get('reward_config', {})
        
        # Add history metrics if available
        if history:
            model_detail['training_history'] = ModelService._process_history(history)
            final_metrics = ModelService._extract_final_metrics(history)
            model_detail['final_metrics'] = final_metrics
        
        return model_detail
    
    @staticmethod
    def delete_model(name: str) -> bool:
        """Delete a model
        
        Args:
            name: Model name to delete
            
        Returns:
            True if deleted, False if didn't exist
        """
        return delete_pytorch_model(name)
    
    @staticmethod
    def _load_metadata(model_name: str) -> Optional[Dict[str, Any]]:
        """Load model metadata from JSON file
        
        Args:
            model_name: Name of the model
            
        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    @staticmethod
    def _load_history(model_name: str) -> Optional[Dict[str, Any]]:
        """Load training history from JSON file
        
        Args:
            model_name: Name of the model
            
        Returns:
            History dictionary or None if not found
        """
        history_path = MODELS_DIR / f"{model_name}_history.json"
        
        if not history_path.exists():
            return None
        
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    @staticmethod
    def _extract_scenario_name(model_name: str) -> str:
        """Extract scenario name from model name
        
        Args:
            model_name: Full model name
            
        Returns:
            Extracted scenario name
        """
        # Model names typically follow pattern: scenario_name or scenario_name_variant
        # Remove common suffixes
        name = model_name.replace('_agent', '').replace('_high', '').replace('_low', '')
        return name
    
    @staticmethod
    def _extract_final_metrics(history: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final metrics from training history
        
        Args:
            history: Training history dictionary
            
        Returns:
            Dictionary with final metrics
        """
        import math
        
        def is_valid_number(value):
            """Check if value is a valid finite number"""
            if value is None:
                return False
            try:
                return math.isfinite(float(value))
            except (ValueError, TypeError):
                return False
        
        metrics = {}
        
        # Get episode rewards (filter out NaN and Infinity values)
        episode_rewards = history.get('episode_rewards', [])
        valid_rewards = [r for r in episode_rewards if is_valid_number(r)]
        
        if valid_rewards:
            metrics['final_reward'] = valid_rewards[-1]
            metrics['avg_reward'] = sum(valid_rewards) / len(valid_rewards)
            metrics['max_reward'] = max(valid_rewards)
            metrics['min_reward'] = min(valid_rewards)
        
        # Get episode lengths
        episode_lengths = history.get('episode_lengths', [])
        valid_lengths = [l for l in episode_lengths if is_valid_number(l)]
        
        if valid_lengths:
            metrics['final_duration'] = valid_lengths[-1]
            metrics['avg_duration'] = sum(valid_lengths) / len(valid_lengths)
        
        # Get cash balances
        cash_balances = history.get('cash_balances', [])
        valid_cash = [c for c in cash_balances if is_valid_number(c)]
        
        if valid_cash:
            metrics['final_cash'] = valid_cash[-1]
            metrics['avg_cash'] = sum(valid_cash) / len(valid_cash)
        
        # Get invested amounts
        total_invested = history.get('total_invested', [])
        valid_invested = [i for i in total_invested if is_valid_number(i)]
        
        if valid_invested:
            metrics['final_invested'] = valid_invested[-1]
            metrics['avg_invested'] = sum(valid_invested) / len(valid_invested)
        
        return metrics
    
    @staticmethod
    def _process_history(history: Dict[str, Any]) -> Dict[str, Any]:
        """Process training history for API response
        
        Args:
            history: Raw training history
            
        Returns:
            Processed history with summary statistics
        """
        import math
        
        def is_valid_number(value):
            """Check if value is a valid finite number"""
            if value is None:
                return False
            try:
                return math.isfinite(float(value))
            except (ValueError, TypeError):
                return False
        
        processed = {}
        
        # Process each metric array
        for key, values in history.items():
            if isinstance(values, list):
                # Filter out NaN and Infinity values
                valid_values = [v for v in values if is_valid_number(v)]
                
                if valid_values:
                    processed[key] = {
                        'count': len(valid_values),
                        'first': valid_values[0] if valid_values else None,
                        'last': valid_values[-1] if valid_values else None,
                        'mean': sum(valid_values) / len(valid_values) if valid_values else None,
                        'min': min(valid_values) if valid_values else None,
                        'max': max(valid_values) if valid_values else None
                    }
        
        return processed
