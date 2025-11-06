"""Tests for Reports API endpoints"""
import pytest
from fastapi import status


class TestReportsAPI:
    """Test reports API endpoints"""
    
    def test_list_reports(self, client):
        """Test listing all reports"""
        response = client.get("/api/reports/list")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "reports" in data
        assert "total" in data
        assert isinstance(data["reports"], list)
    
    def test_generate_report_missing_simulation(self, client):
        """Test generating report for non-existent simulation"""
        request = {
            "simulation_id": "nonexistent_sim_xyz",
            "report_type": "html",
            "include_sections": ["summary", "results"],
            "title": "Test Report"
        }
        
        response = client.post("/api/reports/generate", json=request)
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]
    
    def test_generate_report_invalid_type(self, client):
        """Test generating report with invalid type"""
        request = {
            "simulation_id": "test_sim",
            "report_type": "invalid_type",  # Invalid type
            "include_sections": ["summary"],
            "title": "Test Report"
        }
        
        response = client.post("/api/reports/generate", json=request)
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_download_report_nonexistent(self, client):
        """Test downloading non-existent report"""
        response = client.get("/api/reports/nonexistent_report_xyz")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_report_metadata_nonexistent(self, client):
        """Test getting metadata for non-existent report"""
        response = client.get("/api/reports/nonexistent_report_xyz/metadata")
        assert response.status_code == status.HTTP_404_NOT_FOUND
