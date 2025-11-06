"""File management utilities for HRL Finance System

This module provides functions for:
- Reading/writing YAML configuration files
- Listing scenarios and models from file system
- Saving/loading PyTorch models
- JSON file operations for results storage
- Path validation and sanitization
"""
import os
import json
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re


# Base directories
BASE_DIR = Path(__file__).parent.parent.parent  # Project root
CONFIGS_DIR = BASE_DIR / "configs"
SCENARIOS_DIR = CONFIGS_DIR / "scenarios"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks
    
    Args:
        filename: Raw filename input
        
    Returns:
        Sanitized filename safe for file system operations
        
    Raises:
        ValueError: If filename is invalid or contains dangerous patterns
    """
    if not filename:
        raise ValueError("Filename cannot be empty")
    
    # Remove any path separators
    filename = os.path.basename(filename)
    
    # Remove or replace dangerous characters
    # Allow alphanumeric, underscore, hyphen, and dot
    sanitized = re.sub(r'[^\w\-.]', '_', filename)
    
    # Prevent hidden files
    if sanitized.startswith('.'):
        sanitized = '_' + sanitized[1:]
    
    # Prevent empty result
    if not sanitized or sanitized == '_':
        raise ValueError(f"Invalid filename: {filename}")
    
    return sanitized


def validate_path(path: Path, base_dir: Path) -> Path:
    """Validate that a path is within the allowed base directory
    
    Args:
        path: Path to validate
        base_dir: Base directory that path must be within
        
    Returns:
        Resolved absolute path
        
    Raises:
        ValueError: If path is outside base directory
    """
    try:
        resolved_path = path.resolve()
        resolved_base = base_dir.resolve()
        
        # Check if path is within base directory
        resolved_path.relative_to(resolved_base)
        return resolved_path
    except (ValueError, RuntimeError):
        raise ValueError(f"Path {path} is outside allowed directory {base_dir}")


# YAML Configuration Functions

def read_yaml_config(filename: str, scenarios: bool = False) -> Dict[str, Any]:
    """Read YAML configuration file
    
    Args:
        filename: Name of the YAML file (without path)
        scenarios: If True, look in scenarios subdirectory
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML is invalid or path is unsafe
    """
    sanitized = sanitize_filename(filename)
    
    # Ensure .yaml extension
    if not sanitized.endswith('.yaml') and not sanitized.endswith('.yml'):
        sanitized += '.yaml'
    
    # Determine directory
    base_dir = SCENARIOS_DIR if scenarios else CONFIGS_DIR
    file_path = base_dir / sanitized
    
    # Validate path
    file_path = validate_path(file_path, CONFIGS_DIR)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {filename}: {str(e)}")


def write_yaml_config(filename: str, config: Dict[str, Any], scenarios: bool = False) -> Path:
    """Write YAML configuration file
    
    Args:
        filename: Name of the YAML file (without path)
        config: Configuration dictionary to write
        scenarios: If True, write to scenarios subdirectory
        
    Returns:
        Path to written file
        
    Raises:
        ValueError: If filename is invalid or path is unsafe
    """
    sanitized = sanitize_filename(filename)
    
    # Ensure .yaml extension
    if not sanitized.endswith('.yaml') and not sanitized.endswith('.yml'):
        sanitized += '.yaml'
    
    # Determine directory
    base_dir = SCENARIOS_DIR if scenarios else CONFIGS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = base_dir / sanitized
    
    # Validate path
    file_path = validate_path(file_path, CONFIGS_DIR)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        return file_path
    except Exception as e:
        raise ValueError(f"Failed to write YAML file {filename}: {str(e)}")


def delete_yaml_config(filename: str, scenarios: bool = False) -> bool:
    """Delete YAML configuration file
    
    Args:
        filename: Name of the YAML file (without path)
        scenarios: If True, delete from scenarios subdirectory
        
    Returns:
        True if file was deleted, False if it didn't exist
        
    Raises:
        ValueError: If filename is invalid or path is unsafe
    """
    sanitized = sanitize_filename(filename)
    
    # Ensure .yaml extension
    if not sanitized.endswith('.yaml') and not sanitized.endswith('.yml'):
        sanitized += '.yaml'
    
    # Determine directory
    base_dir = SCENARIOS_DIR if scenarios else CONFIGS_DIR
    file_path = base_dir / sanitized
    
    # Validate path
    try:
        file_path = validate_path(file_path, CONFIGS_DIR)
    except ValueError:
        return False
    
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def list_yaml_configs(scenarios: bool = False) -> List[Dict[str, Any]]:
    """List all YAML configuration files
    
    Args:
        scenarios: If True, list from scenarios subdirectory
        
    Returns:
        List of dictionaries with file information:
        - name: filename without extension
        - path: relative path from base
        - size: file size in bytes
        - modified: last modification timestamp
    """
    base_dir = SCENARIOS_DIR if scenarios else CONFIGS_DIR
    
    if not base_dir.exists():
        return []
    
    configs = []
    for file_path in base_dir.glob('*.yaml'):
        try:
            # Validate path
            validate_path(file_path, CONFIGS_DIR)
            
            stat = file_path.stat()
            configs.append({
                'name': file_path.stem,
                'filename': file_path.name,
                'path': str(file_path.relative_to(BASE_DIR)),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
        except (ValueError, OSError):
            # Skip invalid or inaccessible files
            continue
    
    # Sort by modification time (newest first)
    configs.sort(key=lambda x: x['modified'], reverse=True)
    return configs


# PyTorch Model Functions

def save_pytorch_model(model_name: str, high_agent: Any, low_agent: Any, 
                       metadata: Optional[Dict[str, Any]] = None) -> Tuple[Path, Path]:
    """Save PyTorch models for high and low level agents
    
    Args:
        model_name: Name for the model (used as prefix)
        high_agent: High-level agent with state_dict()
        low_agent: Low-level agent with state_dict()
        metadata: Optional metadata to save alongside models
        
    Returns:
        Tuple of (high_agent_path, low_agent_path)
        
    Raises:
        ValueError: If model_name is invalid
    """
    sanitized = sanitize_filename(model_name)
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    high_path = MODELS_DIR / f"{sanitized}_high_agent.pt"
    low_path = MODELS_DIR / f"{sanitized}_low_agent.pt"
    
    # Validate paths
    high_path = validate_path(high_path, MODELS_DIR)
    low_path = validate_path(low_path, MODELS_DIR)
    
    # Save models
    torch.save(high_agent.state_dict(), high_path)
    torch.save(low_agent.state_dict(), low_path)
    
    # Save metadata if provided
    if metadata:
        metadata_path = MODELS_DIR / f"{sanitized}_metadata.json"
        metadata_path = validate_path(metadata_path, MODELS_DIR)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    return high_path, low_path


def load_pytorch_model(model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load PyTorch model state dictionaries
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (high_agent_state_dict, low_agent_state_dict)
        
    Raises:
        FileNotFoundError: If model files don't exist
        ValueError: If model_name is invalid
    """
    sanitized = sanitize_filename(model_name)
    
    # Define file paths
    high_path = MODELS_DIR / f"{sanitized}_high_agent.pt"
    low_path = MODELS_DIR / f"{sanitized}_low_agent.pt"
    
    # Validate paths
    high_path = validate_path(high_path, MODELS_DIR)
    low_path = validate_path(low_path, MODELS_DIR)
    
    if not high_path.exists():
        raise FileNotFoundError(f"High-level agent model not found: {model_name}")
    if not low_path.exists():
        raise FileNotFoundError(f"Low-level agent model not found: {model_name}")
    
    # Load models
    high_state = torch.load(high_path, map_location='cpu')
    low_state = torch.load(low_path, map_location='cpu')
    
    return high_state, low_state


def delete_pytorch_model(model_name: str) -> bool:
    """Delete PyTorch model files
    
    Args:
        model_name: Name of the model to delete
        
    Returns:
        True if models were deleted, False if they didn't exist
        
    Raises:
        ValueError: If model_name is invalid
    """
    sanitized = sanitize_filename(model_name)
    
    # Define file paths
    high_path = MODELS_DIR / f"{sanitized}_high_agent.pt"
    low_path = MODELS_DIR / f"{sanitized}_low_agent.pt"
    metadata_path = MODELS_DIR / f"{sanitized}_metadata.json"
    history_path = MODELS_DIR / f"{sanitized}_history.json"
    
    deleted = False
    
    # Delete each file if it exists
    for path in [high_path, low_path, metadata_path, history_path]:
        try:
            validated = validate_path(path, MODELS_DIR)
            if validated.exists():
                validated.unlink()
                deleted = True
        except (ValueError, OSError):
            continue
    
    return deleted


def list_pytorch_models() -> List[Dict[str, Any]]:
    """List all PyTorch models in the models directory
    
    Returns:
        List of dictionaries with model information:
        - name: model name (without suffix)
        - high_agent_path: path to high-level agent
        - low_agent_path: path to low-level agent
        - size_mb: total size in MB
        - modified: last modification timestamp
        - has_metadata: whether metadata file exists
    """
    if not MODELS_DIR.exists():
        return []
    
    # Find all high-level agent files
    models = {}
    for high_path in MODELS_DIR.glob('*_high_agent.pt'):
        try:
            # Validate path
            validate_path(high_path, MODELS_DIR)
            
            # Extract model name
            model_name = high_path.stem.replace('_high_agent', '')
            
            # Check for corresponding low-level agent
            low_path = MODELS_DIR / f"{model_name}_low_agent.pt"
            if not low_path.exists():
                continue
            
            # Get file stats
            high_stat = high_path.stat()
            low_stat = low_path.stat()
            total_size = (high_stat.st_size + low_stat.st_size) / (1024 * 1024)  # MB
            
            # Check for metadata
            metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
            has_metadata = metadata_path.exists()
            
            # Use most recent modification time
            modified = max(
                datetime.fromtimestamp(high_stat.st_mtime),
                datetime.fromtimestamp(low_stat.st_mtime)
            )
            
            models[model_name] = {
                'name': model_name,
                'high_agent_path': str(high_path.relative_to(BASE_DIR)),
                'low_agent_path': str(low_path.relative_to(BASE_DIR)),
                'size_mb': round(total_size, 2),
                'modified': modified,
                'has_metadata': has_metadata
            }
        except (ValueError, OSError):
            # Skip invalid or inaccessible files
            continue
    
    # Convert to list and sort by modification time (newest first)
    model_list = list(models.values())
    model_list.sort(key=lambda x: x['modified'], reverse=True)
    return model_list


# JSON Results Functions

def save_json_results(filename: str, data: Dict[str, Any], subdir: Optional[str] = None) -> Path:
    """Save results data to JSON file
    
    Args:
        filename: Name of the JSON file (without path)
        data: Data dictionary to save
        subdir: Optional subdirectory within results folder
        
    Returns:
        Path to written file
        
    Raises:
        ValueError: If filename is invalid or path is unsafe
    """
    sanitized = sanitize_filename(filename)
    
    # Ensure .json extension
    if not sanitized.endswith('.json'):
        sanitized += '.json'
    
    # Determine directory
    if subdir:
        sanitized_subdir = sanitize_filename(subdir)
        base_dir = RESULTS_DIR / sanitized_subdir
    else:
        base_dir = RESULTS_DIR
    
    base_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = base_dir / sanitized
    
    # Validate path
    file_path = validate_path(file_path, RESULTS_DIR)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return file_path
    except Exception as e:
        raise ValueError(f"Failed to write JSON file {filename}: {str(e)}")


def read_json_results(filename: str, subdir: Optional[str] = None) -> Dict[str, Any]:
    """Read results data from JSON file
    
    Args:
        filename: Name of the JSON file (without path)
        subdir: Optional subdirectory within results folder
        
    Returns:
        Dictionary containing results data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or path is unsafe
    """
    sanitized = sanitize_filename(filename)
    
    # Ensure .json extension
    if not sanitized.endswith('.json'):
        sanitized += '.json'
    
    # Determine directory
    if subdir:
        sanitized_subdir = sanitize_filename(subdir)
        base_dir = RESULTS_DIR / sanitized_subdir
    else:
        base_dir = RESULTS_DIR
    
    file_path = base_dir / sanitized
    
    # Validate path
    file_path = validate_path(file_path, RESULTS_DIR)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Results file not found: {filename}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filename}: {str(e)}")


def list_json_results(subdir: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all JSON result files
    
    Args:
        subdir: Optional subdirectory within results folder
        
    Returns:
        List of dictionaries with file information:
        - name: filename without extension
        - path: relative path from base
        - size: file size in bytes
        - modified: last modification timestamp
    """
    # Determine directory
    if subdir:
        sanitized_subdir = sanitize_filename(subdir)
        base_dir = RESULTS_DIR / sanitized_subdir
    else:
        base_dir = RESULTS_DIR
    
    if not base_dir.exists():
        return []
    
    results = []
    for file_path in base_dir.glob('*.json'):
        try:
            # Validate path
            validate_path(file_path, RESULTS_DIR)
            
            stat = file_path.stat()
            results.append({
                'name': file_path.stem,
                'filename': file_path.name,
                'path': str(file_path.relative_to(BASE_DIR)),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            })
        except (ValueError, OSError):
            # Skip invalid or inaccessible files
            continue
    
    # Sort by modification time (newest first)
    results.sort(key=lambda x: x['modified'], reverse=True)
    return results


# Utility Functions

def ensure_directories() -> None:
    """Ensure all required directories exist"""
    for directory in [CONFIGS_DIR, SCENARIOS_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    if not file_path.exists():
        return 0.0
    return file_path.stat().st_size / (1024 * 1024)
