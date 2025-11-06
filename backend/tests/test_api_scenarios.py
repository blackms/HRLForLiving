"""Tests for Scenarios API endpoints"""
import pytest
from fastapi import status


class TestScenariosAPI:
    """Test scenarios API endpoints"""
    
    def test_list_scenarios(self, client):
        """Test listing all scenarios"""
        response = client.get("/api/scenarios")
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.json(), list)
    
    def test_get_templates(self, client):
        """Test getting scenario templates"""
        response = client.get("/api/scenarios/templates")
        assert response.status_code == status.HTTP_200_OK
        templates = response.json()
        assert isinstance(templates, list)
        assert len(templates) > 0
        
        # Check template structure
        template = templates[0]
        assert "name" in template
        assert "display_name" in template
        assert "description" in template
        assert "environment" in template
    
    def test_create_scenario(self, client, sample_scenario_config):
        """Test creating a new scenario"""
        response = client.post("/api/scenarios", json=sample_scenario_config)
        
        # Should create successfully or conflict if already exists
        assert response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_409_CONFLICT
        ]
        
        if response.status_code == status.HTTP_201_CREATED:
            data = response.json()
            assert data["name"] == sample_scenario_config["name"]
            assert "path" in data
            assert "created_at" in data
    
    def test_create_scenario_invalid_data(self, client):
        """Test creating scenario with invalid data"""
        invalid_config = {
            "name": "invalid",
            "environment": {
                "income": -1000.0  # Invalid: negative income
            }
        }
        response = client.post("/api/scenarios", json=invalid_config)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_scenario(self, client):
        """Test getting a specific scenario"""
        # First, list scenarios to get a valid name
        list_response = client.get("/api/scenarios")
        scenarios = list_response.json()
        
        if len(scenarios) > 0:
            scenario_name = scenarios[0]["name"]
            response = client.get(f"/api/scenarios/{scenario_name}")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["name"] == scenario_name
            assert "environment" in data
            assert "training" in data
    
    def test_get_nonexistent_scenario(self, client):
        """Test getting a scenario that doesn't exist"""
        response = client.get("/api/scenarios/nonexistent_scenario_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_update_scenario(self, client, sample_scenario_config):
        """Test updating an existing scenario"""
        # First create a scenario
        create_response = client.post("/api/scenarios", json=sample_scenario_config)
        
        if create_response.status_code == status.HTTP_201_CREATED:
            # Update the scenario
            updated_config = sample_scenario_config.copy()
            updated_config["description"] = "Updated description"
            updated_config["environment"]["income"] = 2500.0
            
            response = client.put(
                f"/api/scenarios/{sample_scenario_config['name']}",
                json=updated_config
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "updated_at" in data
    
    def test_delete_scenario(self, client):
        """Test deleting a scenario"""
        # Create a test scenario first
        test_config = {
            "name": "test_delete_scenario",
            "description": "Scenario to be deleted",
            "environment": {
                "income": 2000.0,
                "fixed_expenses": 800.0,
                "variable_expense_mean": 400.0,
                "variable_expense_std": 100.0,
                "inflation": 0.02,
                "safety_threshold": 3000.0,
                "max_months": 120,
                "initial_cash": 5000.0,
                "risk_tolerance": 0.5
            },
            "training": {
                "num_episodes": 100
            },
            "reward": {
                "wealth_weight": 1.0
            }
        }
        
        create_response = client.post("/api/scenarios", json=test_config)
        
        if create_response.status_code == status.HTTP_201_CREATED:
            # Delete the scenario
            response = client.delete(f"/api/scenarios/{test_config['name']}")
            assert response.status_code == status.HTTP_200_OK
            
            # Verify it's deleted
            get_response = client.get(f"/api/scenarios/{test_config['name']}")
            assert get_response.status_code == status.HTTP_404_NOT_FOUND
