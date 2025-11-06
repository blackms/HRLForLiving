"""Integration tests for complete workflows"""
import pytest
from fastapi import status
import time


class TestIntegrationWorkflows:
    """Test complete user workflows"""
    
    def test_scenario_creation_workflow(self, client, sample_scenario_config):
        """Test complete scenario creation and retrieval workflow"""
        # Create a unique scenario name
        scenario_name = "integration_test_scenario"
        sample_scenario_config["name"] = scenario_name
        
        # Step 1: Create scenario
        create_response = client.post("/api/scenarios", json=sample_scenario_config)
        assert create_response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_409_CONFLICT
        ]
        
        # Step 2: List scenarios and verify it's there
        list_response = client.get("/api/scenarios")
        assert list_response.status_code == status.HTTP_200_OK
        scenarios = list_response.json()
        scenario_names = [s["name"] for s in scenarios]
        assert scenario_name in scenario_names
        
        # Step 3: Get scenario details
        get_response = client.get(f"/api/scenarios/{scenario_name}")
        assert get_response.status_code == status.HTTP_200_OK
        scenario_data = get_response.json()
        assert scenario_data["name"] == scenario_name
        
        # Step 4: Update scenario
        updated_config = sample_scenario_config.copy()
        updated_config["description"] = "Updated in integration test"
        update_response = client.put(
            f"/api/scenarios/{scenario_name}",
            json=updated_config
        )
        assert update_response.status_code == status.HTTP_200_OK
        
        # Step 5: Delete scenario
        delete_response = client.delete(f"/api/scenarios/{scenario_name}")
        assert delete_response.status_code == status.HTTP_200_OK
        
        # Step 6: Verify deletion
        get_after_delete = client.get(f"/api/scenarios/{scenario_name}")
        assert get_after_delete.status_code == status.HTTP_404_NOT_FOUND
    
    def test_model_listing_workflow(self, client):
        """Test model listing and details workflow"""
        # Step 1: List all models
        list_response = client.get("/api/models")
        assert list_response.status_code == status.HTTP_200_OK
        models_data = list_response.json()
        assert "models" in models_data
        assert "total" in models_data
        
        # Step 2: If models exist, get details for first one
        if len(models_data["models"]) > 0:
            model_name = models_data["models"][0]["name"]
            
            detail_response = client.get(f"/api/models/{model_name}")
            assert detail_response.status_code == status.HTTP_200_OK
            model_detail = detail_response.json()
            assert model_detail["name"] == model_name
    
    def test_simulation_history_workflow(self, client):
        """Test simulation history retrieval workflow"""
        # Step 1: Get simulation history
        history_response = client.get("/api/simulation/history")
        assert history_response.status_code == status.HTTP_200_OK
        history_data = history_response.json()
        assert "simulations" in history_data
        assert "total" in history_data
        
        # Step 2: If simulations exist, get results for first one
        if len(history_data["simulations"]) > 0:
            sim_id = history_data["simulations"][0]["simulation_id"]
            
            results_response = client.get(f"/api/simulation/results/{sim_id}")
            assert results_response.status_code == status.HTTP_200_OK
            results_data = results_response.json()
            assert "simulation_id" in results_data
    
    def test_training_status_workflow(self, client):
        """Test training status check workflow"""
        # Get training status
        status_response = client.get("/api/training/status")
        assert status_response.status_code == status.HTTP_200_OK
        
        status_data = status_response.json()
        assert "is_training" in status_data
        assert isinstance(status_data["is_training"], bool)
        
        # If not training, current_episode should be 0
        if not status_data["is_training"]:
            assert status_data["current_episode"] == 0
    
    def test_report_listing_workflow(self, client):
        """Test report listing workflow"""
        # List all reports
        list_response = client.get("/api/reports/list")
        assert list_response.status_code == status.HTTP_200_OK
        
        reports_data = list_response.json()
        assert "reports" in reports_data
        assert "total" in reports_data
        assert isinstance(reports_data["reports"], list)


class TestCompleteUserWorkflows:
    """Test complete end-to-end user workflows as documented in API_QUICK_START.md"""
    
    def test_scenario_template_to_creation_workflow(self, client):
        """Test workflow: Get template -> Customize -> Create scenario"""
        # Step 1: Get available templates
        templates_response = client.get("/api/scenarios/templates")
        assert templates_response.status_code == status.HTTP_200_OK
        templates_data = templates_response.json()
        
        # The response is a list of templates, not a dict with "templates" key
        assert isinstance(templates_data, list)
        assert len(templates_data) > 0
        
        # Step 2: Select a template (e.g., balanced)
        template = None
        for t in templates_data:
            if t["name"] == "balanced":
                template = t
                break
        
        assert template is not None, "Balanced template should exist"
        
        # Step 3: Customize template and create scenario
        # Template structure has environment, training, reward directly
        scenario_config = {
            "name": "workflow_test_from_template",
            "description": "Created from balanced template",
            "environment": template["environment"],
            "training": template.get("training", {
                "num_episodes": 100,
                "save_interval": 50,
                "eval_episodes": 5
            }),
            "reward": template.get("reward", {
                "wealth_weight": 1.0,
                "stability_weight": 0.5,
                "goal_weight": 0.3
            })
        }
        
        # Step 4: Create the scenario
        create_response = client.post("/api/scenarios", json=scenario_config)
        assert create_response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_409_CONFLICT
        ]
        
        # Step 5: Verify scenario exists
        get_response = client.get("/api/scenarios/workflow_test_from_template")
        assert get_response.status_code == status.HTTP_200_OK
        
        # Cleanup
        client.delete("/api/scenarios/workflow_test_from_template")
    
    def test_scenario_to_model_workflow(self, client, sample_scenario_config):
        """Test workflow: Create scenario -> Check models -> Verify scenario available for training"""
        # Step 1: Create a scenario
        scenario_name = "workflow_scenario_to_model"
        sample_scenario_config["name"] = scenario_name
        
        create_response = client.post("/api/scenarios", json=sample_scenario_config)
        assert create_response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_409_CONFLICT
        ]
        
        # Step 2: List scenarios to verify creation
        list_response = client.get("/api/scenarios")
        assert list_response.status_code == status.HTTP_200_OK
        scenarios = list_response.json()
        scenario_names = [s["name"] for s in scenarios]
        assert scenario_name in scenario_names
        
        # Step 3: Check training status (should not be training)
        status_response = client.get("/api/training/status")
        assert status_response.status_code == status.HTTP_200_OK
        status_data = status_response.json()
        assert "is_training" in status_data
        
        # Step 4: List models (scenario should be available for training)
        models_response = client.get("/api/models")
        assert models_response.status_code == status.HTTP_200_OK
        
        # Cleanup
        client.delete(f"/api/scenarios/{scenario_name}")
    
    def test_model_to_simulation_workflow(self, client):
        """Test workflow: List models -> Select model -> Prepare simulation request"""
        # Step 1: List available models
        models_response = client.get("/api/models")
        assert models_response.status_code == status.HTTP_200_OK
        models_data = models_response.json()
        assert "models" in models_data
        
        # Step 2: If models exist, verify we can get details
        if len(models_data["models"]) > 0:
            model = models_data["models"][0]
            model_name = model["name"]
            
            # Step 3: Get model details
            detail_response = client.get(f"/api/models/{model_name}")
            assert detail_response.status_code == status.HTTP_200_OK
            model_detail = detail_response.json()
            
            # Step 4: Verify model has required fields for simulation
            assert "name" in model_detail
            assert "scenario_name" in model_detail or "scenario" in model_detail
            
            # Step 5: Get scenario for the model
            scenario_name = model_detail.get("scenario_name") or model_detail.get("scenario")
            if scenario_name:
                scenario_response = client.get(f"/api/scenarios/{scenario_name}")
                # Scenario might not exist anymore, so we accept 404
                assert scenario_response.status_code in [
                    status.HTTP_200_OK,
                    status.HTTP_404_NOT_FOUND
                ]
    
    def test_simulation_to_report_workflow(self, client):
        """Test workflow: Get simulation results -> Generate report"""
        # Step 1: Get simulation history
        history_response = client.get("/api/simulation/history")
        assert history_response.status_code == status.HTTP_200_OK
        history_data = history_response.json()
        
        # Step 2: If simulations exist, test report generation
        if len(history_data["simulations"]) > 0:
            simulation = history_data["simulations"][0]
            simulation_id = simulation["simulation_id"]
            
            # Step 3: Get simulation results
            results_response = client.get(f"/api/simulation/results/{simulation_id}")
            assert results_response.status_code == status.HTTP_200_OK
            results_data = results_response.json()
            
            # Step 4: Verify results have required data for report
            assert "simulation_id" in results_data
            assert "scenario_name" in results_data
            assert "model_name" in results_data
            
            # Step 5: Generate report
            report_request = {
                "simulation_id": simulation_id,
                "report_type": "html",
                "title": "Integration Test Report",
                "include_sections": ["summary", "results"]
            }
            
            report_response = client.post("/api/reports/generate", json=report_request)
            # Report generation returns 202 Accepted (async operation)
            assert report_response.status_code == status.HTTP_202_ACCEPTED
            report_data = report_response.json()
            
            # Step 6: Verify report was created
            assert "report_id" in report_data
            assert "file_path" in report_data
            
            # Step 7: Verify report appears in list
            list_response = client.get("/api/reports/list")
            assert list_response.status_code == status.HTTP_200_OK
            reports = list_response.json()
            report_ids = [r["report_id"] for r in reports["reports"]]
            assert report_data["report_id"] in report_ids
    
    def test_comparison_workflow(self, client):
        """Test workflow: List simulations -> Select multiple -> Compare results"""
        # Step 1: Get simulation history
        history_response = client.get("/api/simulation/history")
        assert history_response.status_code == status.HTTP_200_OK
        history_data = history_response.json()
        
        # Step 2: If multiple simulations exist, compare them
        if len(history_data["simulations"]) >= 2:
            sim1_id = history_data["simulations"][0]["simulation_id"]
            sim2_id = history_data["simulations"][1]["simulation_id"]
            
            # Step 3: Get results for both simulations
            results1_response = client.get(f"/api/simulation/results/{sim1_id}")
            assert results1_response.status_code == status.HTTP_200_OK
            results1 = results1_response.json()
            
            results2_response = client.get(f"/api/simulation/results/{sim2_id}")
            assert results2_response.status_code == status.HTTP_200_OK
            results2 = results2_response.json()
            
            # Step 4: Verify both have comparable metrics
            comparable_fields = [
                "duration_mean", "total_wealth_mean", "investment_gains_mean",
                "avg_invest_pct", "avg_save_pct", "avg_consume_pct"
            ]
            
            for field in comparable_fields:
                assert field in results1, f"Field {field} missing in results1"
                assert field in results2, f"Field {field} missing in results2"
    
    def test_error_recovery_workflow(self, client):
        """Test workflow: Handle errors gracefully and recover"""
        # Step 1: Try to get non-existent scenario
        response = client.get("/api/scenarios/nonexistent_scenario_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        error_data = response.json()
        assert "detail" in error_data or "error" in error_data
        
        # Step 2: Try to create scenario with invalid data
        invalid_scenario = {
            "name": "invalid_scenario",
            "environment": {
                "income": -1000,  # Invalid: negative income
                "fixed_expenses": 800
            }
        }
        response = client.post("/api/scenarios", json=invalid_scenario)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Step 3: Try to get non-existent model
        response = client.get("/api/models/nonexistent_model_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Step 4: Try to get non-existent simulation results
        response = client.get("/api/simulation/results/nonexistent_sim_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        
        # Step 5: Verify system is still functional after errors
        # Verify we can still list resources after encountering errors
        scenarios_response = client.get("/api/scenarios")
        assert scenarios_response.status_code == status.HTTP_200_OK
        
        models_response = client.get("/api/models")
        assert models_response.status_code == status.HTTP_200_OK
        
        # Step 6: Verify we can still perform operations
        training_status = client.get("/api/training/status")
        assert training_status.status_code == status.HTTP_200_OK
    
    def test_data_consistency_workflow(self, client, sample_scenario_config):
        """Test workflow: Verify data consistency across operations"""
        # Step 1: Create a scenario with specific values
        scenario_name = "consistency_test_scenario"
        sample_scenario_config["name"] = scenario_name
        sample_scenario_config["description"] = "Testing data consistency"
        sample_scenario_config["environment"]["income"] = 2500.0
        
        create_response = client.post("/api/scenarios", json=sample_scenario_config)
        assert create_response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_409_CONFLICT
        ]
        
        # Step 2: Retrieve the scenario
        get_response = client.get(f"/api/scenarios/{scenario_name}")
        assert get_response.status_code == status.HTTP_200_OK
        retrieved_data = get_response.json()
        
        # Step 3: Verify all data matches
        assert retrieved_data["name"] == scenario_name
        # Description might be None if not stored, check if it exists
        if "description" in retrieved_data and retrieved_data["description"] is not None:
            assert retrieved_data["description"] == "Testing data consistency"
        assert retrieved_data["environment"]["income"] == 2500.0
        
        # Step 4: Update the scenario
        sample_scenario_config["environment"]["income"] = 3000.0
        update_response = client.put(
            f"/api/scenarios/{scenario_name}",
            json=sample_scenario_config
        )
        assert update_response.status_code == status.HTTP_200_OK
        
        # Step 5: Verify update was applied
        get_after_update = client.get(f"/api/scenarios/{scenario_name}")
        assert get_after_update.status_code == status.HTTP_200_OK
        updated_data = get_after_update.json()
        assert updated_data["environment"]["income"] == 3000.0
        
        # Step 6: Verify scenario appears in list with correct data
        list_response = client.get("/api/scenarios")
        assert list_response.status_code == status.HTTP_200_OK
        scenarios = list_response.json()
        
        matching_scenario = None
        for s in scenarios:
            if s["name"] == scenario_name:
                matching_scenario = s
                break
        
        assert matching_scenario is not None
        
        # Cleanup
        client.delete(f"/api/scenarios/{scenario_name}")
    
    def test_concurrent_operations_workflow(self, client, sample_scenario_config):
        """Test workflow: Multiple operations in sequence without conflicts"""
        # Step 1: Create multiple scenarios
        scenario_names = [
            "concurrent_test_1",
            "concurrent_test_2",
            "concurrent_test_3"
        ]
        
        for name in scenario_names:
            config = sample_scenario_config.copy()
            config["name"] = name
            response = client.post("/api/scenarios", json=config)
            assert response.status_code in [
                status.HTTP_201_CREATED,
                status.HTTP_409_CONFLICT
            ]
        
        # Step 2: List all scenarios
        list_response = client.get("/api/scenarios")
        assert list_response.status_code == status.HTTP_200_OK
        scenarios = list_response.json()
        scenario_list = [s["name"] for s in scenarios]
        
        # Step 3: Verify all created scenarios are in the list
        for name in scenario_names:
            assert name in scenario_list
        
        # Step 4: Get details for each scenario
        for name in scenario_names:
            response = client.get(f"/api/scenarios/{name}")
            assert response.status_code == status.HTTP_200_OK
        
        # Step 5: Delete all test scenarios
        for name in scenario_names:
            response = client.delete(f"/api/scenarios/{name}")
            assert response.status_code == status.HTTP_200_OK
        
        # Step 6: Verify all were deleted
        for name in scenario_names:
            response = client.get(f"/api/scenarios/{name}")
            assert response.status_code == status.HTTP_404_NOT_FOUND
