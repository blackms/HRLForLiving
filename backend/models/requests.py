"""Pydantic models for API requests"""
from typing import Optional
from pydantic import BaseModel, Field


class EnvironmentConfig(BaseModel):
    """Environment configuration for financial simulation"""
    income: float = Field(gt=0, description="Monthly income in currency units")
    fixed_expenses: float = Field(ge=0, description="Fixed monthly expenses")
    variable_expense_mean: float = Field(ge=0, description="Mean of variable expenses")
    variable_expense_std: float = Field(ge=0, description="Standard deviation of variable expenses")
    inflation: float = Field(ge=-1, le=1, description="Monthly inflation rate")
    safety_threshold: float = Field(ge=0, description="Minimum cash buffer threshold")
    max_months: int = Field(gt=0, description="Maximum simulation duration in months")
    initial_cash: float = Field(ge=0, default=0, description="Starting cash balance")
    risk_tolerance: float = Field(ge=0, le=1, description="Risk tolerance level (0-1)")
    investment_return_mean: float = Field(default=0.005, description="Mean monthly investment return")
    investment_return_std: float = Field(default=0.02, description="Standard deviation of investment returns")
    investment_return_type: str = Field(default="stochastic", pattern="^(fixed|stochastic|none)$", description="Type of investment returns")


class TrainingConfig(BaseModel):
    """Training configuration for HRL system"""
    num_episodes: int = Field(gt=0, default=5000, description="Number of training episodes")
    gamma_low: float = Field(gt=0, le=1, default=0.95, description="Discount factor for low-level agent")
    gamma_high: float = Field(gt=0, le=1, default=0.99, description="Discount factor for high-level agent")
    high_period: int = Field(gt=0, default=6, description="Planning horizon for high-level agent in months")
    batch_size: int = Field(gt=0, default=32, description="Batch size for training")
    learning_rate_low: float = Field(gt=0, default=3e-4, description="Learning rate for low-level agent")
    learning_rate_high: float = Field(gt=0, default=1e-4, description="Learning rate for high-level agent")


class RewardConfig(BaseModel):
    """Reward function configuration"""
    alpha: float = Field(default=10.0, description="Investment reward coefficient")
    beta: float = Field(default=0.1, description="Stability penalty coefficient")
    gamma: float = Field(default=5.0, description="Overspend penalty coefficient")
    delta: float = Field(default=20.0, description="Debt penalty coefficient")
    lambda_: float = Field(default=1.0, alias="lambda", description="Wealth growth coefficient")
    mu: float = Field(default=0.5, description="Stability bonus coefficient")

    class Config:
        populate_by_name = True


class ScenarioConfig(BaseModel):
    """Complete scenario configuration"""
    name: str = Field(min_length=1, max_length=100, description="Scenario name")
    description: Optional[str] = Field(None, max_length=500, description="Scenario description")
    environment: EnvironmentConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)


class TrainingRequest(BaseModel):
    """Request to start model training"""
    scenario_name: str = Field(min_length=1, description="Name of the scenario to train on")
    num_episodes: int = Field(gt=0, default=1000, description="Number of training episodes")
    save_interval: int = Field(gt=0, default=100, description="Save checkpoint every N episodes")
    eval_episodes: int = Field(gt=0, default=10, description="Number of evaluation episodes")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SimulationRequest(BaseModel):
    """Request to run a simulation"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(min_length=1, description="Name of the trained model to use")
    scenario_name: str = Field(min_length=1, description="Name of the scenario to simulate")
    num_episodes: int = Field(gt=0, default=10, description="Number of simulation episodes")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ReportRequest(BaseModel):
    """Request to generate a report"""
    simulation_id: str = Field(min_length=1, description="ID of the simulation results")
    report_type: str = Field(pattern="^(pdf|html)$", description="Report format (pdf or html)")
    include_sections: Optional[list[str]] = Field(None, description="Sections to include in report")
    title: Optional[str] = Field(None, max_length=200, description="Custom report title")
