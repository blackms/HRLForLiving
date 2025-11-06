"""Integration tests for complete workflows"""
import pytest
from fastapi import status


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
