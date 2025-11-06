"""Tests for Training API endpoints"""
import pytest
from fastapi import status


class TestTrainingAPI:
    """Test training API endpoints"""
    
    def test_get_training_status(self, client):
        """Test getting training status"""
        response = client.get("/api/training/status")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "is_training" in data
        assert "scenario_name" in data
        assert "current_episode" in data
        assert "total_episodes" in data
        assert isinstance(data["is_training"], bool)
    
    def test_start_training_missing_scenario(self, client):
        """Test starting training with non-existent scenario"""
        request = {
            "scenario_name": "nonexistent_scenario_xyz",
            "num_episodes": 10,
            "save_interval": 5,
            "eval_episodes": 2
        }
        
        response = client.post("/api/training/start", json=request)
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_400_BAD_REQUEST
        ]
    
    def test_start_training_invalid_params(self, client):
        """Test starting training with invalid parameters"""
        request = {
            "scenario_name": "test_scenario",
            "num_episodes": -10,  # Invalid: negative episodes
            "save_interval": 5
        }
        
        response = client.post("/api/training/start", json=request)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_stop_training_when_not_training(self, client):
        """Test stopping training when no training is in progress"""
        response = client.post("/api/training/stop")
        # Should return 400 if no training in progress
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST
        ]
