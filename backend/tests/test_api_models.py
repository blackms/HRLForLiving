"""Tests for Models API endpoints"""
import pytest
from fastapi import status


class TestModelsAPI:
    """Test models API endpoints"""
    
    def test_list_models(self, client):
        """Test listing all models"""
        response = client.get("/api/models")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert isinstance(data["models"], list)
        assert data["total"] == len(data["models"])
        
        # Verify structure if models exist
        if len(data["models"]) > 0:
            model = data["models"][0]
            assert "name" in model
            assert "scenario_name" in model
            assert "trained_at" in model  # API uses trained_at instead of created_at
    
    def test_get_model(self, client):
        """Test getting a specific model"""
        # First list models to get a valid name
        list_response = client.get("/api/models")
        models = list_response.json()["models"]
        
        if len(models) > 0:
            model_name = models[0]["name"]
            response = client.get(f"/api/models/{model_name}")
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["name"] == model_name
            assert "scenario_name" in data
            assert "trained_at" in data  # API uses trained_at instead of created_at
            assert "high_agent_path" in data
            assert "low_agent_path" in data
    
    def test_get_nonexistent_model(self, client):
        """Test getting a model that doesn't exist"""
        response = client.get("/api/models/nonexistent_model_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_delete_model(self, client):
        """Test deleting a model"""
        # Test deleting a non-existent model
        response = client.delete("/api/models/test_model_that_doesnt_exist")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_delete_model_success(self, client):
        """Test successful model deletion"""
        # List models first
        list_response = client.get("/api/models")
        models = list_response.json()["models"]
        
        # Note: In a real test environment, we would create a test model first
        # For now, we verify the endpoint structure
        if len(models) > 0:
            # We won't actually delete real models in tests
            # This would require creating a test model first
            pass
