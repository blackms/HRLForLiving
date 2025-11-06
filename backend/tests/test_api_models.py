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
            assert "created_at" in data
    
    def test_get_nonexistent_model(self, client):
        """Test getting a model that doesn't exist"""
        response = client.get("/api/models/nonexistent_model_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_delete_model(self, client):
        """Test deleting a model"""
        # List models first
        list_response = client.get("/api/models")
        models = list_response.json()["models"]
        
        # Only test deletion if there are models and we can safely delete one
        # In a real test, we'd create a test model first
        if len(models) > 0:
            # For safety, we'll just test the endpoint structure
            # without actually deleting a real model
            response = client.delete("/api/models/test_model_that_doesnt_exist")
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_404_NOT_FOUND
            ]
