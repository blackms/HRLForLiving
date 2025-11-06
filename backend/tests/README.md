# Backend Tests

This directory contains comprehensive tests for the HRL Finance UI backend API.

## Test Structure

```
tests/
├── conftest.py                        # Pytest configuration and fixtures
├── test_api_scenarios.py              # Scenarios API endpoint tests
├── test_api_models.py                 # Models API endpoint tests
├── test_api_training.py               # Training API endpoint tests
├── test_api_simulation.py             # Simulation API endpoint tests
├── test_api_reports.py                # Reports API endpoint tests
├── test_services.py                   # Service layer unit tests (comprehensive)
├── test_file_manager.py               # File manager utility tests
├── test_websocket.py                  # WebSocket communication tests
├── test_integration.py                # Integration tests for complete workflows
├── README.md                          # This file - test suite overview
├── INTEGRATION_TESTS_SUMMARY.md       # Detailed integration test documentation
├── INTEGRATION_TEST_NOTES.md          # Known issues and recommendations
└── WEBSOCKET_TESTS_SUMMARY.md         # WebSocket tests documentation
```

### Service Layer Tests (`test_services.py`)

Comprehensive tests for all service layer components:

**TestScenarioService** (10 tests):
- Template retrieval and validation (5 templates)
- Template environment config validation
- Scenario listing with structure validation
- Scenario retrieval (existing and non-existent)
- Scenario name extraction from model names

**TestModelService** (7 tests):
- Model listing with metadata
- Model retrieval (existing and non-existent)
- Model deletion (existing and non-existent)
- Final metrics extraction from training history
- NaN/Infinity filtering in metrics
- Empty history handling

**TestSimulationService** (4 tests):
- Service initialization
- Simulation listing with required fields
- Simulation results retrieval
- Statistics calculation from episodes

**TestReportService** (5 tests):
- Service initialization
- Report listing with metadata
- Report retrieval (existing and non-existent)
- Report file path retrieval
- Report data aggregation
- HTML content generation

**Total Service Tests: 26**

### WebSocket Communication Tests (`test_websocket.py`)

Comprehensive tests for real-time training updates via WebSocket:

**TestTrainingSocketManager** (6 tests):
- Progress update emission
- Training started event emission
- Training completed event emission
- Training stopped event emission
- Training error event emission
- Multiple sequential progress updates

**TestWebSocketIntegration** (4 tests):
- Training service progress callback mechanism
- Complete training lifecycle event sequence
- Error handling and error event emission
- Early stop scenario with stopped event

**TestWebSocketConnectionHandlers** (2 tests):
- Connection handler setup verification
- Progress data structure and type validation

**TestWebSocketEventPayloads** (3 tests):
- Training started payload structure
- Training completed payload structure
- Training error payload structure

**Total WebSocket Tests: 15**

All tests use mocking to avoid actual WebSocket connections, ensuring fast and reliable test execution.

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

### Integration Tests (`test_integration.py`)

**TestIntegrationWorkflows** (4 tests):
- Scenario creation and retrieval workflow (CRUD operations)
- Model listing and details workflow
- Simulation history retrieval workflow
- Training status check workflow
- Report listing workflow

**TestCompleteUserWorkflows** (7 tests):
- Template to scenario creation workflow (5 steps)
- Scenario to model workflow (4 steps)
- Model to simulation workflow (5 steps)
- Simulation to report workflow (7 steps)
- Comparison workflow (4 steps)
- Error recovery workflow (6 steps)
- Data consistency workflow (6 steps)
- Concurrent operations workflow (6 steps)

**Total Integration Tests: 11**

These tests validate:
- Complete end-to-end user workflows as documented in API_QUICK_START.md
- Multi-step operations with proper data flow
- Error handling and recovery mechanisms
- Data consistency across operations
- Concurrent operations without conflicts
- System resilience after errors

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
- ✅ Service layer business logic (comprehensive)
- ✅ File management utilities
- ✅ WebSocket communication and real-time updates (15 tests)
- ✅ Complete user workflows (11 integration tests)
- ✅ End-to-end workflows as documented in API_QUICK_START.md
- ✅ Multi-step operations with data flow validation
- ✅ Error recovery and system resilience
- ✅ Data consistency across operations
- ✅ Concurrent operations without conflicts
- ✅ Edge cases and error conditions
- ✅ Data validation and sanitization
- ✅ Statistics calculation and aggregation
- ✅ Error handling with invalid data (NaN, Infinity)
- ✅ Template validation and structure
- ✅ Report generation and formatting
- ✅ Training lifecycle event broadcasting
- ✅ WebSocket event payload structures

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

## Documentation

### Test Documentation Files

- **README.md** (this file) - Test suite overview and quick reference
- **INTEGRATION_TESTS_SUMMARY.md** - Comprehensive documentation of all integration tests
  - Detailed workflow descriptions
  - Step-by-step test breakdowns
  - Expected behaviors and validations
  - API endpoints tested
  - Test statistics and coverage
- **INTEGRATION_TEST_NOTES.md** - Known issues and recommendations
  - Status code mismatches
  - Template structure verification needed
  - Future enhancement suggestions
  - Test improvement recommendations
- **WEBSOCKET_TESTS_SUMMARY.md** - WebSocket communication tests documentation
  - Test coverage breakdown (15 tests)
  - Event types tested (progress, started, completed, stopped, error)
  - Integration points with training service
  - Mocking strategy and approach
  - Event payload structures

### Related Documentation

- `backend/API_QUICK_START.md` - User workflows that integration tests validate
- `backend/API_DOCUMENTATION.md` - Complete API reference
- `backend/README.md` - Backend setup and overview

## Continuous Integration

When adding new features:
1. Write unit tests for new functions/methods
2. Write API tests for new endpoints
3. Add integration tests for new workflows
4. Update documentation to match
5. Ensure all tests pass before merging

## Test Maintenance

### Regular Tasks

- Review and update tests when API changes
- Keep integration tests in sync with API_QUICK_START.md
- Update documentation when adding new tests
- Fix any known issues documented in INTEGRATION_TEST_NOTES.md
- Monitor test coverage and add tests for uncovered code
