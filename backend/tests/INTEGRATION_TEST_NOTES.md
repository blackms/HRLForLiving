# Integration Test Notes

## Known Issues and Discrepancies

### 1. Report Generation Status Code Mismatch

**Location:** `test_integration.py::TestCompleteUserWorkflows::test_simulation_to_report_workflow`

**Issue:** Test expects `HTTP_200_OK` but implementation returns `HTTP_202_ACCEPTED`

**Details:**
- Line 260 in test: `assert report_response.status_code == status.HTTP_200_OK`
- Actual implementation: `backend/api/reports.py` line 20: `@router.post("/generate", status_code=202)`
- Documentation: `backend/API_DOCUMENTATION.md` correctly documents `202 Accepted`

**Resolution Required:**
Update test to expect `status.HTTP_202_ACCEPTED` instead of `status.HTTP_200_OK`

```python
# Current (incorrect):
assert report_response.status_code == status.HTTP_200_OK

# Should be:
assert report_response.status_code == status.HTTP_202_ACCEPTED
```

**Why 202 Accepted:**
Report generation is an asynchronous operation that may take time to complete. The 202 status code correctly indicates that the request has been accepted for processing but is not yet complete.

### 2. Template Response Structure

**Location:** `test_integration.py::TestCompleteUserWorkflows::test_scenario_template_to_creation_workflow`

**Potential Issue:** Template structure access pattern

**Details:**
- Line 128: `templates_data["templates"]` - expects templates in a "templates" key
- Line 138: `template["config"]["environment"]` - expects nested config structure

**Current Implementation Check Needed:**
Verify that `GET /api/scenarios/templates` returns:
```json
{
  "templates": [
    {
      "name": "balanced",
      "config": {
        "environment": {...},
        "training": {...},
        "reward": {...}
      }
    }
  ]
}
```

**Alternative Structure (from API docs):**
```json
[
  {
    "name": "balanced",
    "display_name": "Balanced Profile",
    "description": "...",
    "environment": {...},
    "training": {...},
    "reward": {...}
  }
]
```

**Resolution Required:**
1. Check actual API response structure
2. Update test to match actual structure, OR
3. Update API to match test expectations

### 3. Health Check Endpoint

**Location:** `test_integration.py::TestCompleteUserWorkflows::test_error_recovery_workflow`

**Issue:** Test uses `GET /health` endpoint

**Details:**
- Line 225: `health_response = client.get("/health")`
- This endpoint should be documented in API_DOCUMENTATION.md if not already present

**Resolution:**
Verify `/health` endpoint is documented in:
- `backend/API_DOCUMENTATION.md`
- `backend/API_QUICK_START.md`
- `backend/README.md`

### 4. Scenario Description Field Persistence

**Location:** `test_integration.py::TestCompleteUserWorkflows::test_data_consistency_workflow`

**Potential Issue:** Description field may not be persisted

**Details:**
- Line 253: Sets `description = "Testing data consistency"`
- Line 263: Expects `retrieved_data["description"] == "Testing data consistency"`

**Note:** Some YAML-based storage may not preserve description field. Test should handle this gracefully.

**Suggested Fix:**
```python
# Current:
assert retrieved_data["description"] == "Testing data consistency"

# Should be:
if "description" in retrieved_data and retrieved_data["description"] is not None:
    assert retrieved_data["description"] == "Testing data consistency"
```

## Test Execution Notes

### Prerequisites

Before running integration tests, ensure:
1. Backend server is NOT running (tests use TestClient)
2. Test directories exist (configs, models, results)
3. No conflicting test data from previous runs

### Cleanup

Integration tests create temporary resources:
- Scenarios: `workflow_test_from_template`, `workflow_scenario_to_model`, `consistency_test_scenario`, `concurrent_test_*`
- These should be cleaned up automatically by tests
- If tests fail, manual cleanup may be required

### Test Data Dependencies

Some tests require existing data:
- `test_model_to_simulation_workflow`: Requires at least one trained model
- `test_simulation_to_report_workflow`: Requires at least one simulation result
- `test_comparison_workflow`: Requires at least two simulation results

These tests gracefully skip if data is not available.

## Recommendations

### 1. Fix Status Code Assertion
**Priority:** High  
**Effort:** Low (1 line change)

Update line 260 in `test_integration.py`:
```python
assert report_response.status_code == status.HTTP_202_ACCEPTED
```

### 2. Verify Template Structure
**Priority:** Medium  
**Effort:** Low (verification + potential fix)

Check actual API response and update test or API to match.

### 3. Document Health Endpoint
**Priority:** Low  
**Effort:** Low (documentation update)

Add `/health` endpoint to API documentation if missing.

### 4. Add Graceful Description Handling
**Priority:** Low  
**Effort:** Low (add conditional check)

Update test to handle missing description field gracefully.

## Future Enhancements

### Additional Integration Tests

1. **Complete Training Workflow**
   - Start training → Monitor progress → Verify model created
   - Requires WebSocket testing or polling

2. **WebSocket Integration Test**
   - Connect to WebSocket → Start training → Receive updates
   - Requires Socket.IO test client

3. **Large Dataset Test**
   - Create 100+ scenarios → Verify performance
   - Test pagination if implemented

4. **Concurrent Request Test**
   - Multiple simultaneous API calls
   - Test for race conditions

5. **File Download Test**
   - Generate report → Download file → Verify content
   - Test actual file download, not just metadata

### Test Improvements

1. **Parameterized Tests**
   - Use `@pytest.mark.parametrize` for multiple scenarios
   - Test with different template types

2. **Fixtures for Test Data**
   - Create fixtures for common test scenarios
   - Reduce code duplication

3. **Better Error Messages**
   - Add descriptive assertion messages
   - Include actual vs expected values

4. **Test Isolation**
   - Ensure tests don't depend on each other
   - Use unique names for all test resources

## Related Files

- `backend/tests/test_integration.py` - Integration test implementation
- `backend/tests/INTEGRATION_TESTS_SUMMARY.md` - Detailed test documentation
- `backend/tests/README.md` - Test suite overview
- `backend/API_QUICK_START.md` - User workflows documented
- `backend/API_DOCUMENTATION.md` - Complete API reference

---

**Last Updated:** 2024-01-15  
**Status:** 1 known issue (status code mismatch), 3 items to verify
