"""Tests for service layer"""
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from backend.services.scenario_service import ScenarioService
from backend.services.model_service import ModelService
from backend.services.simulation_service import SimulationService
from backend.services.report_service import ReportService
from backend.models.requests import ScenarioConfig, EnvironmentConfig, TrainingConfig, RewardConfig


class TestScenarioService:
    """Test scenario service layer"""
    
    def test_get_templates(self):
        """Test getting scenario templates"""
        templates = ScenarioService.get_templates()
        
        assert isinstance(templates, list)
        assert len(templates) == 5  # Should have 5 preset templates
        
        # Check template structure
        template = templates[0]
        assert "name" in template
        assert "display_name" in template
        assert "description" in template
        assert "environment" in template
        assert "training" in template
        assert "reward" in template
        
        # Verify template names
        template_names = [t['name'] for t in templates]
        assert 'conservative' in template_names
        assert 'balanced' in template_names
        assert 'aggressive' in template_names
        assert 'young_professional' in template_names
        assert 'young_couple' in template_names
    
    def test_template_environment_config(self):
        """Test that template environment configs are valid"""
        templates = ScenarioService.get_templates()
        
        for template in templates:
            env = template['environment']
            # Check required fields
            assert env['income'] > 0
            assert env['fixed_expenses'] >= 0
            assert env['variable_expense_mean'] >= 0
            assert env['variable_expense_std'] >= 0
            assert 0 <= env['risk_tolerance'] <= 1
            assert env['max_months'] > 0
            assert env['initial_cash'] >= 0
    
    def test_list_scenarios(self):
        """Test listing scenarios"""
        scenarios = ScenarioService.list_scenarios()
        
        assert isinstance(scenarios, list)
        # Each scenario should have required fields
        for scenario in scenarios:
            assert "name" in scenario
            assert "income" in scenario
            assert "fixed_expenses" in scenario
            assert "available_monthly" in scenario
            assert "available_pct" in scenario
    
    def test_get_scenario_existing(self):
        """Test getting an existing scenario"""
        scenarios = ScenarioService.list_scenarios()
        
        if scenarios:
            # Get first available scenario
            scenario_name = scenarios[0]['name']
            scenario = ScenarioService.get_scenario(scenario_name)
            
            # Verify structure
            assert scenario['name'] == scenario_name
            assert 'environment' in scenario
            assert 'training' in scenario
            assert 'reward' in scenario
            assert 'created_at' in scenario
            assert 'updated_at' in scenario
    
    def test_get_scenario_nonexistent(self):
        """Test getting non-existent scenario"""
        with pytest.raises(FileNotFoundError):
            ScenarioService.get_scenario("nonexistent_scenario_xyz_12345")
    
    def test_extract_scenario_name(self):
        """Test scenario name extraction from model name"""
        # Test various model name formats
        assert ModelService._extract_scenario_name("balanced") == "balanced"
        assert ModelService._extract_scenario_name("balanced_high") == "balanced"
        assert ModelService._extract_scenario_name("balanced_low") == "balanced"
        assert ModelService._extract_scenario_name("balanced_agent") == "balanced"


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
            assert "trained_at" in model
            assert "size_mb" in model
            assert "has_metadata" in model
    
    def test_get_model_nonexistent(self):
        """Test getting non-existent model"""
        with pytest.raises(FileNotFoundError):
            ModelService.get_model("nonexistent_model_xyz_12345")
    
    def test_delete_model_nonexistent(self):
        """Test deleting non-existent model"""
        result = ModelService.delete_model("nonexistent_model_xyz_12345")
        assert result is False
    
    def test_extract_final_metrics_valid_data(self):
        """Test extracting final metrics from valid history"""
        history = {
            'episode_rewards': [100.0, 150.0, 200.0],
            'episode_lengths': [50, 60, 70],
            'cash_balances': [5000.0, 6000.0, 7000.0],
            'total_invested': [10000.0, 12000.0, 15000.0]
        }
        
        metrics = ModelService._extract_final_metrics(history)
        
        assert metrics['final_reward'] == 200.0
        assert metrics['avg_reward'] == 150.0
        assert metrics['max_reward'] == 200.0
        assert metrics['min_reward'] == 100.0
        assert metrics['final_duration'] == 70
        assert metrics['final_cash'] == 7000.0
        assert metrics['final_invested'] == 15000.0
    
    def test_extract_final_metrics_with_nan(self):
        """Test extracting metrics with NaN values"""
        import math
        
        history = {
            'episode_rewards': [100.0, math.nan, 200.0, math.inf],
            'episode_lengths': [50, 60, math.nan],
        }
        
        metrics = ModelService._extract_final_metrics(history)
        
        # Should filter out NaN and Infinity
        assert metrics['final_reward'] == 200.0
        assert metrics['avg_reward'] == 150.0  # (100 + 200) / 2
        assert metrics['final_duration'] == 60
    
    def test_extract_final_metrics_empty(self):
        """Test extracting metrics from empty history"""
        history = {
            'episode_rewards': [],
            'episode_lengths': []
        }
        
        metrics = ModelService._extract_final_metrics(history)
        
        # Should return empty dict when no valid data
        assert 'final_reward' not in metrics
        assert 'final_duration' not in metrics


class TestSimulationService:
    """Test simulation service layer"""
    
    def test_simulation_service_initialization(self):
        """Test that simulation service initializes correctly"""
        service = SimulationService()
        assert service is not None
    
    def test_list_simulations(self):
        """Test listing simulations"""
        service = SimulationService()
        simulations = service.list_simulations()
        
        assert isinstance(simulations, list)
        # Each simulation should have required fields
        for sim in simulations:
            assert "simulation_id" in sim
            assert "scenario_name" in sim
            assert "model_name" in sim
            assert "num_episodes" in sim
            assert "timestamp" in sim
    
    def test_get_simulation_results_nonexistent(self):
        """Test getting non-existent simulation results"""
        service = SimulationService()
        
        with pytest.raises(FileNotFoundError):
            service.get_simulation_results("nonexistent_simulation_xyz_12345")
    
    def test_calculate_statistics(self):
        """Test statistics calculation from episodes"""
        service = SimulationService()
        
        episodes = [
            {
                'duration': 50,
                'final_cash': 5000.0,
                'final_invested': 10000.0,
                'final_portfolio_value': 12000.0,
                'total_wealth': 17000.0,
                'investment_gains': 2000.0,
                'actions': [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]
            },
            {
                'duration': 60,
                'final_cash': 6000.0,
                'final_invested': 12000.0,
                'final_portfolio_value': 14000.0,
                'total_wealth': 20000.0,
                'investment_gains': 2000.0,
                'actions': [[0.6, 0.2, 0.2], [0.5, 0.3, 0.2]]
            }
        ]
        
        stats = service._calculate_statistics(episodes)
        
        # Check that statistics are calculated
        assert 'duration_mean' in stats
        assert 'duration_std' in stats
        assert 'total_wealth_mean' in stats
        assert 'avg_invest_pct' in stats
        assert 'avg_save_pct' in stats
        assert 'avg_consume_pct' in stats
        
        # Verify calculations
        assert stats['duration_mean'] == 55.0  # (50 + 60) / 2
        assert stats['total_wealth_mean'] == 18500.0  # (17000 + 20000) / 2


class TestReportService:
    """Test report service layer"""
    
    def test_report_service_initialization(self):
        """Test that report service initializes correctly"""
        service = ReportService()
        assert service is not None
    
    def test_list_reports(self):
        """Test listing reports"""
        service = ReportService()
        reports = service.list_reports()
        
        assert isinstance(reports, list)
        # Each report should have required fields
        for report in reports:
            assert "report_id" in report
            assert "simulation_id" in report
            assert "report_type" in report
            assert "generated_at" in report
    
    def test_get_report_nonexistent(self):
        """Test getting non-existent report"""
        service = ReportService()
        
        with pytest.raises(FileNotFoundError):
            service.get_report("nonexistent_report_xyz_12345")
    
    def test_get_report_file_path_nonexistent(self):
        """Test getting file path for non-existent report"""
        service = ReportService()
        
        with pytest.raises(FileNotFoundError):
            service.get_report_file_path("nonexistent_report_xyz_12345")
    
    def test_aggregate_report_data(self):
        """Test aggregating report data"""
        service = ReportService()
        
        simulation_data = {
            'simulation_id': 'test_sim_123',
            'scenario_name': 'test_scenario',
            'model_name': 'test_model',
            'num_episodes': 10,
            'timestamp': '2024-01-01T00:00:00',
            'duration_mean': 50.0,
            'duration_std': 5.0,
            'total_wealth_mean': 20000.0,
            'total_wealth_std': 1000.0,
            'final_cash_mean': 5000.0,
            'final_cash_std': 500.0,
            'final_invested_mean': 10000.0,
            'final_invested_std': 1000.0,
            'final_portfolio_mean': 12000.0,
            'final_portfolio_std': 1200.0,
            'investment_gains_mean': 2000.0,
            'investment_gains_std': 200.0,
            'avg_invest_pct': 0.5,
            'avg_save_pct': 0.3,
            'avg_consume_pct': 0.2,
            'episodes': []
        }
        
        scenario_config = {
            'description': 'Test scenario',
            'environment': {
                'income': 3000.0,
                'fixed_expenses': 1500.0,
                'variable_expense_mean': 500.0,
                'variable_expense_std': 100.0,
                'inflation': 0.02,
                'safety_threshold': 5000.0,
                'initial_cash': 10000.0,
                'risk_tolerance': 0.5,
                'investment_return_mean': 0.005,
                'investment_return_std': 0.02,
                'investment_return_type': 'stochastic'
            },
            'training': {
                'num_episodes': 1000,
                'high_period': 6,
                'gamma_low': 0.95,
                'gamma_high': 0.99
            }
        }
        
        report_data = service._aggregate_report_data(
            simulation_data=simulation_data,
            scenario_config=scenario_config,
            title="Test Report"
        )
        
        # Verify structure
        assert report_data['title'] == "Test Report"
        assert report_data['simulation_id'] == 'test_sim_123'
        assert report_data['scenario_name'] == 'test_scenario'
        assert report_data['model_name'] == 'test_model'
        assert 'summary' in report_data
        assert 'strategy' in report_data
        assert 'scenario' in report_data
        assert 'training' in report_data
        
        # Verify summary data
        assert report_data['summary']['duration_mean'] == 50.0
        assert report_data['summary']['total_wealth_mean'] == 20000.0
        
        # Verify strategy data (converted to percentage)
        assert report_data['strategy']['avg_invest_pct'] == 50.0
        assert report_data['strategy']['avg_save_pct'] == 30.0
        assert report_data['strategy']['avg_consume_pct'] == 20.0
    
    def test_build_html_content(self):
        """Test HTML content generation"""
        service = ReportService()
        
        report_data = {
            'title': 'Test Report',
            'generated_at': '2024-01-01 00:00:00',
            'simulation_id': 'test_sim_123',
            'scenario_name': 'test_scenario',
            'model_name': 'test_model',
            'num_episodes': 10,
            'sections': ['summary', 'strategy'],
            'summary': {
                'duration_mean': 50.0,
                'duration_std': 5.0,
                'total_wealth_mean': 20000.0,
                'total_wealth_std': 1000.0,
                'final_cash_mean': 5000.0,
                'final_cash_std': 500.0,
                'investment_gains_mean': 2000.0,
                'investment_gains_std': 200.0
            },
            'strategy': {
                'avg_invest_pct': 50.0,
                'avg_save_pct': 30.0,
                'avg_consume_pct': 20.0
            },
            'episodes': []
        }
        
        html = service._build_html_content(report_data)
        
        # Verify HTML structure
        assert '<!DOCTYPE html>' in html
        assert '<html lang="en">' in html
        assert 'Test Report' in html
        assert 'test_sim_123' in html
        assert 'test_scenario' in html
        assert 'test_model' in html
        
        # Verify sections are included
        assert 'Summary Statistics' in html
        assert 'Strategy Learned' in html
