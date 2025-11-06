"""Tests for Simulation API endpoints"""
import pytest
from fastapi import status


class TestSimulationAPI:
    """Test simulation API endpoints"""
    
    def test_get_simulation_history(self, client):
        """Test getting simulation history"""
        response = client.get("/api/simulation/history")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "simulations" in data
        assert "total" in data
        assert isinstance(data["simulations"], list)
        assert data["total"] == len(data["simulations"])
        
        # Verify structure if simulations exist
        if len(data["simulations"]) > 0:
            sim = data["simulations"][0]
            assert "simulation_id" in sim
            assert "scenario_name" in sim
            assert "model_name" in sim
            assert "timestamp" in sim  # API uses timestamp instead of created_at
    
    def test_run_simulation_missing_model(self, client):
        """Test running simulation with non-existent model"""
        request = {
            "model_name": "nonexistent_model_xyz",
            "scenario_name": "test_scenario",
            "num_episodes": 5
        }
        
        response = client.post("/api/simulation/run", json=request)
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    def test_run_simulation_invalid_params(self, client):
        """Test running simulation with invalid parameters"""
        request = {
            "model_name": "test_model",
            "scenario_name": "test_scenario",
            "num_episodes": -5  # Invalid: negative episodes
        }
        
        response = client.post("/api/simulation/run", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_get_simulation_results_nonexistent(self, client):
        """Test getting results for non-existent simulation"""
        response = client.get("/api/simulation/results/nonexistent_sim_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_simulation_results(self, client):
        """Test getting simulation results"""
        # First get history to find a valid simulation
        history_response = client.get("/api/simulation/history")
        simulations = history_response.json()["simulations"]
        
        if len(simulations) > 0:
            sim_id = simulations[0]["simulation_id"]
            response = client.get(f"/api/simulation/results/{sim_id}")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert "simulation_id" in data
            assert "scenario_name" in data
            assert "model_name" in data
            assert "num_episodes" in data
            # Statistics are returned as individual fields, not nested
            assert "duration_mean" in data
            assert "total_wealth_mean" in data
    
    def test_run_simulation_missing_scenario(self, client):
        """Test running simulation with non-existent scenario"""
        request = {
            "model_name": "test_model",
            "scenario_name": "nonexistent_scenario_xyz",
            "num_episodes": 5
        }
        
        response = client.post("/api/simulation/run", json=request)
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    def test_run_simulation_missing_fields(self, client):
        """Test running simulation with missing required fields"""
        request = {
            "model_name": "test_model"
            # Missing scenario_name
        }
        
        response = client.post("/api/simulation/run", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_run_simulation_zero_episodes(self, client):
        """Test running simulation with zero episodes"""
        request = {
            "model_name": "test_model",
            "scenario_name": "test_scenario",
            "num_episodes": 0  # Invalid: must be > 0
        }
        
        response = client.post("/api/simulation/run", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
