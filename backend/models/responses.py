"""Pydantic models for API responses"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class TrainingProgress(BaseModel):
    """Real-time training progress update"""
    episode: int = Field(description="Current episode number")
    total_episodes: int = Field(description="Total number of episodes")
    avg_reward: float = Field(description="Average reward over recent episodes")
    avg_duration: float = Field(description="Average episode duration in months")
    avg_cash: float = Field(description="Average final cash balance")
    avg_invested: float = Field(description="Average final invested amount")
    stability: float = Field(description="Stability metric (0-1)")
    goal_adherence: float = Field(description="Goal adherence metric (0-1)")
    elapsed_time: float = Field(description="Elapsed training time in seconds")


class TrainingStatus(BaseModel):
    """Current training status"""
    is_training: bool = Field(description="Whether training is currently active")
    scenario_name: Optional[str] = Field(None, description="Name of scenario being trained")
    current_episode: Optional[int] = Field(None, description="Current episode number")
    total_episodes: Optional[int] = Field(None, description="Total episodes planned")
    start_time: Optional[datetime] = Field(None, description="Training start timestamp")
    latest_progress: Optional[TrainingProgress] = Field(None, description="Latest progress update")


class EpisodeResult(BaseModel):
    """Results from a single simulation episode"""
    episode_id: int = Field(description="Episode identifier")
    duration: int = Field(description="Episode duration in months")
    final_cash: float = Field(description="Final cash balance")
    final_invested: float = Field(description="Final invested amount")
    final_portfolio_value: float = Field(description="Final portfolio value")
    total_wealth: float = Field(description="Total wealth (cash + portfolio)")
    investment_gains: float = Field(description="Total investment gains/losses")
    months: List[int] = Field(description="Month numbers")
    cash_history: List[float] = Field(description="Cash balance over time")
    invested_history: List[float] = Field(description="Invested amount over time")
    portfolio_history: List[float] = Field(description="Portfolio value over time")
    actions: List[List[float]] = Field(description="Actions taken [invest%, save%, consume%]")


class SimulationResults(BaseModel):
    """Aggregated results from multiple simulation episodes"""
    model_config = {"protected_namespaces": ()}
    
    simulation_id: str = Field(description="Unique simulation identifier")
    scenario_name: str = Field(description="Name of scenario used")
    model_name: str = Field(description="Name of model used")
    num_episodes: int = Field(description="Number of episodes run")
    timestamp: datetime = Field(description="Simulation completion timestamp")
    
    # Summary statistics
    duration_mean: float = Field(description="Mean episode duration")
    duration_std: float = Field(description="Standard deviation of duration")
    final_cash_mean: float = Field(description="Mean final cash balance")
    final_invested_mean: float = Field(description="Mean final invested amount")
    final_portfolio_mean: float = Field(description="Mean final portfolio value")
    total_wealth_mean: float = Field(description="Mean total wealth")
    total_wealth_std: float = Field(description="Standard deviation of total wealth")
    investment_gains_mean: float = Field(description="Mean investment gains")
    
    # Strategy metrics
    avg_invest_pct: float = Field(description="Average investment percentage")
    avg_save_pct: float = Field(description="Average save percentage")
    avg_consume_pct: float = Field(description="Average consume percentage")
    
    # Detailed episode data
    episodes: List[EpisodeResult] = Field(description="Individual episode results")


class ScenarioSummary(BaseModel):
    """Summary information about a scenario"""
    name: str = Field(description="Scenario name")
    description: Optional[str] = Field(None, description="Scenario description")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    # Key metrics from environment config
    income: float = Field(description="Monthly income")
    fixed_expenses: float = Field(description="Fixed expenses")
    available_income_pct: float = Field(description="Percentage of income available after fixed expenses")
    risk_tolerance: float = Field(description="Risk tolerance level")


class ModelSummary(BaseModel):
    """Summary information about a trained model"""
    model_config = {"protected_namespaces": ()}
    
    name: str = Field(description="Model name")
    scenario_name: str = Field(description="Scenario used for training")
    size_mb: float = Field(description="Model file size in MB")
    trained_at: str = Field(description="Training completion timestamp")
    has_metadata: bool = Field(description="Whether metadata file exists")
    episodes: Optional[int] = Field(None, description="Number of training episodes")
    income: Optional[float] = Field(None, description="Income from environment config")
    risk_tolerance: Optional[float] = Field(None, description="Risk tolerance from environment config")
    final_reward: Optional[float] = Field(None, description="Final average reward")
    avg_reward: Optional[float] = Field(None, description="Average reward across training")
    max_reward: Optional[float] = Field(None, description="Maximum reward achieved")
    final_duration: Optional[float] = Field(None, description="Final episode duration")
    final_cash: Optional[float] = Field(None, description="Final cash balance")
    final_invested: Optional[float] = Field(None, description="Final invested amount")


class ModelDetail(BaseModel):
    """Detailed information about a trained model"""
    model_config = {"protected_namespaces": ()}
    
    name: str = Field(description="Model name")
    scenario_name: str = Field(description="Scenario used for training")
    high_agent_path: str = Field(description="Path to high-level agent file")
    low_agent_path: str = Field(description="Path to low-level agent file")
    size_mb: float = Field(description="Total model file size in MB")
    trained_at: str = Field(description="Training completion timestamp")
    has_metadata: bool = Field(description="Whether metadata file exists")
    has_history: bool = Field(description="Whether training history file exists")
    episodes: Optional[int] = Field(None, description="Number of training episodes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Full metadata")
    environment_config: Optional[Dict[str, Any]] = Field(None, description="Environment configuration")
    training_config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")
    reward_config: Optional[Dict[str, Any]] = Field(None, description="Reward configuration")
    training_history: Optional[Dict[str, Any]] = Field(None, description="Processed training history")
    final_metrics: Optional[Dict[str, Any]] = Field(None, description="Final training metrics")


class ScenarioListResponse(BaseModel):
    """List of scenarios"""
    scenarios: List[ScenarioSummary] = Field(description="List of scenario summaries")
    total: int = Field(description="Total number of scenarios")


class ModelListResponse(BaseModel):
    """List of trained models"""
    models: List[ModelSummary] = Field(description="List of model summaries")
    total: int = Field(description="Total number of models")


class SimulationHistoryResponse(BaseModel):
    """List of past simulations"""
    simulations: List[Dict[str, Any]] = Field(description="List of simulation summaries")
    total: int = Field(description="Total number of simulations")


class ReportResponse(BaseModel):
    """Response for report generation"""
    report_id: str = Field(description="Unique report identifier")
    simulation_id: str = Field(description="Simulation ID used for report")
    report_type: str = Field(description="Report format (pdf or html)")
    title: str = Field(description="Report title")
    generated_at: str = Field(description="Report generation timestamp")
    file_path: str = Field(description="Path to generated report file")
    file_size_kb: float = Field(description="Report file size in KB")
    sections: List[str] = Field(description="Sections included in report")
    status: str = Field(description="Report generation status")
    message: str = Field(description="Status message")


class ReportListResponse(BaseModel):
    """List of generated reports"""
    reports: List[Dict[str, Any]] = Field(description="List of report metadata")
    total: int = Field(description="Total number of reports")


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(description="Current server timestamp")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(description="Error type or code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(description="Error timestamp")
