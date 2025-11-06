# File Manager Utilities

This module provides comprehensive file management utilities for the HRL Finance System backend with built-in security features to prevent path traversal attacks and ensure safe file operations.

## Features

### YAML Configuration Management
- `read_yaml_config(filename, scenarios=False)` - Read YAML configuration files
  - Automatically adds `.yaml` extension if missing
  - Validates path to prevent directory traversal
  - Returns empty dict for empty YAML files
  - Raises `FileNotFoundError` if file doesn't exist
  - Raises `ValueError` for invalid YAML syntax

- `write_yaml_config(filename, config, scenarios=False)` - Write YAML configuration files
  - Automatically creates directories if they don't exist
  - Automatically adds `.yaml` extension if missing
  - Uses `safe_dump` for security
  - Preserves key order (no sorting)
  - Returns `Path` object to written file

- `delete_yaml_config(filename, scenarios=False)` - Delete YAML configuration files
  - Returns `True` if file was deleted, `False` if it didn't exist
  - Safely handles invalid paths

- `list_yaml_configs(scenarios=False)` - List all YAML configuration files
  - Returns list of dicts with: `name`, `filename`, `path`, `size`, `modified`
  - Sorted by modification time (newest first)
  - Skips invalid or inaccessible files

### PyTorch Model Management
- `save_pytorch_model(model_name, high_agent, low_agent, metadata=None)` - Save HRL agent models
  - Saves both high-level and low-level agent state dicts
  - Optionally saves metadata as JSON
  - Automatically creates models directory
  - Returns tuple of `(high_agent_path, low_agent_path)`

- `load_pytorch_model(model_name)` - Load HRL agent models
  - Loads to CPU by default (use `map_location` for GPU)
  - Returns tuple of `(high_agent_state_dict, low_agent_state_dict)`
  - Raises `FileNotFoundError` if either agent file is missing

- `delete_pytorch_model(model_name)` - Delete model files
  - Deletes high agent, low agent, metadata, and history files
  - Returns `True` if any files were deleted
  - Safely handles missing files

- `list_pytorch_models()` - List all trained models
  - Returns list of dicts with: `name`, `high_agent_path`, `low_agent_path`, `size_mb`, `modified`, `has_metadata`
  - Only includes models with both high and low agent files
  - Sorted by modification time (newest first)
  - Calculates total size in MB

### JSON Results Management
- `save_json_results(filename, data, subdir=None)` - Save simulation results to JSON
  - Automatically adds `.json` extension if missing
  - Supports optional subdirectory for organization
  - Automatically creates directories
  - Uses `indent=2` for readable output
  - Handles datetime serialization with `default=str`

- `read_json_results(filename, subdir=None)` - Read simulation results from JSON
  - Automatically adds `.json` extension if missing
  - Supports optional subdirectory
  - Raises `FileNotFoundError` if file doesn't exist
  - Raises `ValueError` for invalid JSON

- `list_json_results(subdir=None)` - List all JSON result files
  - Returns list of dicts with: `name`, `filename`, `path`, `size`, `modified`
  - Supports optional subdirectory
  - Sorted by modification time (newest first)

### Security & Validation
- `sanitize_filename(filename)` - Sanitize filenames to prevent path traversal attacks
  - Removes path separators (`/`, `\`)
  - Replaces dangerous characters with underscores
  - Allows only alphanumeric, underscore, hyphen, and dot
  - Prevents hidden files (starting with `.`)
  - Raises `ValueError` for empty or invalid filenames

- `validate_path(path, base_dir)` - Validate paths are within allowed directories
  - Resolves absolute paths
  - Checks path is within base directory using `relative_to()`
  - Raises `ValueError` if path is outside allowed directory
  - Returns resolved absolute path

- `ensure_directories()` - Ensure all required directories exist
  - Creates `configs/`, `configs/scenarios/`, `models/`, `results/`
  - Uses `mkdir(parents=True, exist_ok=True)` for safety

- `get_file_size_mb(file_path)` - Get file size in megabytes
  - Returns `0.0` if file doesn't exist
  - Calculates size in MB (1024 * 1024 bytes)

## Directory Structure

```
project_root/
├── configs/              # YAML configuration files
│   └── scenarios/        # Scenario-specific configs
├── models/               # PyTorch model files
└── results/              # JSON simulation results
```

## Usage Examples

### Reading a Scenario Configuration

```python
from backend.utils.file_manager import read_yaml_config

# Read from configs/ directory
config = read_yaml_config('personal_eur.yaml')
# Returns: {'environment': {...}, 'training': {...}, 'reward': {...}}

# Read from configs/scenarios/ directory
scenario = read_yaml_config('bologna_coppia.yaml', scenarios=True)

# Extension is automatically added
config = read_yaml_config('balanced')  # Reads 'balanced.yaml'
```

### Writing a New Scenario

```python
from backend.utils.file_manager import write_yaml_config

scenario_config = {
    'name': 'My Scenario',
    'environment': {
        'income': 3200,
        'fixed_expenses': 1400,
        # ... other parameters
    },
    'training': {...},
    'reward': {...}
}

# Write to configs/scenarios/ directory
path = write_yaml_config('my_scenario', scenario_config, scenarios=True)
print(f"Saved to: {path}")
```

### Listing All Scenarios

```python
from backend.utils.file_manager import list_yaml_configs

# List scenarios
scenarios = list_yaml_configs(scenarios=True)
for scenario in scenarios:
    print(f"{scenario['name']}: {scenario['size']} bytes, modified {scenario['modified']}")

# Output:
# bologna_coppia: 1234 bytes, modified 2025-11-06 10:30:00
# milano_senior: 1156 bytes, modified 2025-11-05 15:20:00
```

### Saving a Trained Model

```python
from backend.utils.file_manager import save_pytorch_model

# Assuming you have trained agents
high_agent = FinancialStrategist(training_config)
low_agent = BudgetExecutor(training_config)

# Save both agents with metadata
high_path, low_path = save_pytorch_model(
    model_name='bologna_coppia_trained',
    high_agent=high_agent,
    low_agent=low_agent,
    metadata={
        'scenario': 'bologna_coppia',
        'episodes': 1000,
        'final_reward': 168.5,
        'final_stability': 0.985,
        'trained_at': '2025-11-06T10:30:00'
    }
)

print(f"High agent saved to: {high_path}")
print(f"Low agent saved to: {low_path}")
```

### Loading a Trained Model

```python
from backend.utils.file_manager import load_pytorch_model

# Load model state dictionaries
high_state, low_state = load_pytorch_model('bologna_coppia_trained')

# Load into agent instances
high_agent.load_state_dict(high_state)
low_agent.load_state_dict(low_state)

# Set to evaluation mode
high_agent.eval()
low_agent.eval()
```

### Listing Available Models

```python
from backend.utils.file_manager import list_pytorch_models

models = list_pytorch_models()
for model in models:
    print(f"{model['name']}: {model['size_mb']} MB")
    print(f"  Modified: {model['modified']}")
    print(f"  Has metadata: {model['has_metadata']}")
    print(f"  Paths: {model['high_agent_path']}, {model['low_agent_path']}")

# Output:
# bologna_coppia_trained: 2.45 MB
#   Modified: 2025-11-06 10:30:00
#   Has metadata: True
#   Paths: models/bologna_coppia_trained_high_agent.pt, models/bologna_coppia_trained_low_agent.pt
```

### Saving Simulation Results

```python
from backend.utils.file_manager import save_json_results
from datetime import datetime

results = {
    'simulation_id': 'sim_20251106_103000',
    'scenario_name': 'bologna_coppia',
    'model_name': 'bologna_coppia_trained',
    'num_episodes': 10,
    'timestamp': datetime.now().isoformat(),
    'duration_mean': 27.3,
    'total_wealth_mean': 18842.5,
    'episodes': [...]
}

# Save to results/bologna_coppia/ directory
path = save_json_results(
    'simulation_20251106_103000',
    results,
    subdir='bologna_coppia'
)
print(f"Results saved to: {path}")
```

### Reading Simulation Results

```python
from backend.utils.file_manager import read_json_results

# Read from results/bologna_coppia/ directory
results = read_json_results(
    'simulation_20251106_103000',
    subdir='bologna_coppia'
)

print(f"Simulation ID: {results['simulation_id']}")
print(f"Mean wealth: {results['total_wealth_mean']}")
```

### Listing All Results

```python
from backend.utils.file_manager import list_json_results

# List all results in a subdirectory
results = list_json_results(subdir='bologna_coppia')
for result in results:
    print(f"{result['name']}: {result['size']} bytes")

# List all results in main results directory
all_results = list_json_results()
```

## Security Features

All file operations include multiple layers of security:

1. **Filename Sanitization**
   - Removes path separators (`/`, `\`) using `os.path.basename()`
   - Replaces dangerous characters with underscores
   - Only allows: alphanumeric, underscore, hyphen, dot
   - Prevents hidden files (starting with `.`)
   - Rejects empty or invalid filenames

2. **Path Validation**
   - Resolves all paths to absolute paths
   - Validates paths are within allowed base directories
   - Uses `Path.relative_to()` to detect directory traversal
   - Prevents `../` attacks and symlink exploits

3. **Safe File Extensions**
   - Automatically adds correct extensions (`.yaml`, `.json`)
   - Prevents extension confusion attacks

4. **Directory Isolation**
   - YAML configs: restricted to `configs/` and `configs/scenarios/`
   - PyTorch models: restricted to `models/`
   - JSON results: restricted to `results/` and subdirectories

5. **Safe YAML/JSON Parsing**
   - Uses `yaml.safe_load()` instead of `yaml.load()`
   - Uses `json.load()` with proper error handling
   - Handles datetime serialization safely with `default=str`

## Error Handling

All functions raise appropriate exceptions with descriptive messages:

### FileNotFoundError
Raised when requested files don't exist:
```python
try:
    config = read_yaml_config('nonexistent.yaml')
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Error: Configuration file not found: nonexistent.yaml
```

### ValueError
Raised for invalid inputs or file contents:
```python
# Invalid filename
try:
    sanitize_filename('')
except ValueError as e:
    print(f"Error: {e}")
    # Error: Filename cannot be empty

# Invalid YAML syntax
try:
    config = read_yaml_config('broken.yaml')
except ValueError as e:
    print(f"Error: {e}")
    # Error: Invalid YAML in broken.yaml: ...

# Path traversal attempt
try:
    validate_path(Path('/etc/passwd'), CONFIGS_DIR)
except ValueError as e:
    print(f"Error: {e}")
    # Error: Path /etc/passwd is outside allowed directory ...
```

### OSError
Raised for file system operation failures (automatically handled by try-except blocks in list functions)

## Best Practices

1. **Always use the provided functions** - Don't bypass security by using raw file operations
2. **Handle exceptions appropriately** - Catch and handle `FileNotFoundError` and `ValueError`
3. **Use scenarios parameter** - Set `scenarios=True` when working with scenario configs
4. **Provide metadata** - Include metadata when saving models for better tracking
5. **Use subdirectories** - Organize results by scenario using the `subdir` parameter
6. **Check return values** - Delete functions return `bool` indicating success

## Integration with API

These utilities are designed to be used by the service layer:

```python
# In scenario_service.py
from backend.utils.file_manager import read_yaml_config, write_yaml_config

def get_scenario(name: str) -> dict:
    """Get scenario configuration"""
    try:
        return read_yaml_config(name, scenarios=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario not found: {name}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def create_scenario(name: str, config: dict) -> str:
    """Create new scenario"""
    try:
        path = write_yaml_config(name, config, scenarios=True)
        return str(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## Testing

The module includes comprehensive error handling and edge case management. When testing:

1. Test with valid inputs
2. Test with invalid filenames (empty, path traversal attempts)
3. Test with missing files
4. Test with invalid YAML/JSON syntax
5. Test with missing model files (only high or only low agent)
6. Test directory creation
7. Test file listing with empty directories

## Performance Considerations

- **Listing functions** scan directories and stat files - may be slow with many files
- **Model loading** loads entire state dicts into memory - consider memory usage
- **JSON serialization** uses `indent=2` for readability - larger file sizes
- **Path validation** resolves paths - minimal overhead but happens on every operation
