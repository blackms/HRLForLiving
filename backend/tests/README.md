# Backend Tests

This directory contains comprehensive tests for the HRL Finance UI backend API.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_api_scenarios.py    # Scenarios API endpoint tests
├── test_api_models.py       # Models API endpoint tests
├── test_api_training.py     # Training API endpoint tests
├── test_api_simulation.py   # Simulation API endpoint tests
├── test_api_reports.py      # Reports API endpoint tests
├── test_services.py         # Service layer unit tests
├── test_file_manager.py     # File manager utility tests
└── test_integration.py      # Integration tests for complete workflows
```

## Running Tests

### Run all tests
```bash
cd backend
python3 -m pytest tests/ -v
```

### Run specific test file
```bash
python3 -m pytest tests/test_api_scenarios.py -v
```

### Run specific test class
```bash
python3 -m pytest tests/test_api_scenarios.py::TestScenariosAPI -v
```

### Run specific test
```bash
python3 -m pytest tests/test_api_scenarios.py::TestScenariosAPI::test_list_scenarios -v
```

### Run with coverage
```bash
python3 -m pytest tests/ --cov=backend --cov-report=html
```

## Test Categories

### Unit Tests
- **API Endpoints**: Test individual API endpoints with various inputs
- **Service Layer**: Test business logic in service classes
- **Utilities**: Test helper functions and utilities

### Integration Tests
- **Complete Workflows**: Test end-to-end user workflows
- **API Interactions**: Test multiple API calls in sequence
- **Data Persistence**: Test data storage and retrieval

## Fixtures

Common fixtures available in `conftest.py`:

- `client`: FastAPI test client
- `temp_configs_dir`: Temporary configs directory
- `temp_models_dir`: Temporary models directory
- `temp_results_dir`: Temporary results directory
- `sample_scenario_config`: Sample scenario configuration
- `sample_training_request`: Sample training request
- `sample_simulation_request`: Sample simulation request

## Test Coverage

The test suite covers:

- ✅ All API endpoints (scenarios, training, simulation, models, reports)
- ✅ Request validation and error handling
- ✅ Service layer business logic
- ✅ File management utilities
- ✅ Complete user workflows
- ✅ Edge cases and error conditions

## Writing New Tests

When adding new tests:

1. Follow the existing test structure
2. Use descriptive test names (test_<action>_<expected_result>)
3. Use fixtures for common setup
4. Test both success and error cases
5. Clean up any created resources
6. Add docstrings to test classes and methods

## Dependencies

The following testing dependencies are included in `backend/requirements.txt`:

- **pytest** (7.4.3) - Python testing framework
  - Provides test discovery, fixtures, and assertions
  - Used for all unit and integration tests
  
- **pytest-asyncio** (0.21.1) - Async test support
  - Enables testing of async/await functions
  - Required for FastAPI endpoint tests
  - Use `@pytest.mark.asyncio` decorator for async tests
  
- **httpx** (0.25.2) - HTTP client for testing
  - Async HTTP client for API endpoint testing
  - Compatible with FastAPI's TestClient
  - Supports both sync and async requests

### Installation

All testing dependencies are automatically installed with:

```bash
pip install -r requirements.txt
```

### Example Async Test

```python
import pytest
from httpx import AsyncClient
from backend.main import app

@pytest.mark.asyncio
async def test_list_scenarios():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/scenarios")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
```
