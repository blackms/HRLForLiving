"""Tests for service layer"""
import pytest
from pathlib import Path
from backend.services.scenario_service import ScenarioService
from backend.services.model_service import ModelService


class TestScenarioService:
    """Test scenario service layer"""
    
    def test_get_templates(self):
        """Test getting scenario templates"""
        templates = ScenarioService.get_templates()
        
        assert isinstance(templates, list)
        assert len(templates) > 0
        
        # Check template structure
        template = templates[0]
        assert "name" in template
        assert "display_name" in template
        assert "description" in template
        assert "environment" in template
        assert "training" in template
        assert "reward" in template
    
    def test_list_scenarios(self):
        """Test listing scenarios"""
        scenarios = ScenarioService.list_scenarios()
        
        assert isinstance(scenarios, list)
        # Each scenario should have required fields
        for scenario in scenarios:
            assert "name" in scenario
            assert "income" in scenario
            assert "fixed_expenses" in scenario
    
    def test_get_scenario_nonexistent(self):
        """Test getting non-existent scenario"""
        with pytest.raises(FileNotFoundError):
            ScenarioService.get_scenario("nonexistent_scenario_xyz")


class TestModelService:
    """Test model service layer"""
    
    def test_list_models(self):
        """Test listing models"""
        models = ModelService.list_models()
        
        assert isinstance(models, list)
        # Each model should have required fields
        for model in models:
            assert "name" in model
            assert "scenario_name" in model
            assert "created_at" in model
    
    def test_get_model_nonexistent(self):
        """Test getting non-existent model"""
        with pytest.raises(FileNotFoundError):
            ModelService.get_model("nonexistent_model_xyz")
    
    def test_delete_model_nonexistent(self):
        """Test deleting non-existent model"""
        result = ModelService.delete_model("nonexistent_model_xyz")
        assert result is False
