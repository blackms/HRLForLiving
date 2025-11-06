"""Scenarios API endpoints

This module provides REST API endpoints for:
- Listing all scenarios
- Getting scenario details
- Creating new scenarios
- Updating existing scenarios
- Deleting scenarios
- Getting scenario templates
"""
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.models.requests import ScenarioConfig
from backend.services.scenario_service import ScenarioService


# Create router
router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])


# Response models
class ScenarioSummary(BaseModel):
    """Summary information for a scenario"""
    name: str
    description: str = ""
    income: float
    fixed_expenses: float
    variable_expenses: float
    available_monthly: float
    available_pct: float
    risk_tolerance: float
    updated_at: str
    size: int


class ScenarioDetail(BaseModel):
    """Detailed scenario information"""
    name: str
    description: Optional[str] = None
    environment: Dict[str, Any]
    training: Dict[str, Any]
    reward: Dict[str, Any]
    created_at: str
    updated_at: str
    size: int


class ScenarioCreateResponse(BaseModel):
    """Response after creating a scenario"""
    name: str
    description: Optional[str] = None
    path: str
    created_at: str
    updated_at: str
    message: str = "Scenario created successfully"


class ScenarioUpdateResponse(BaseModel):
    """Response after updating a scenario"""
    name: str
    description: Optional[str] = None
    path: str
    updated_at: str
    message: str = "Scenario updated successfully"


class ScenarioDeleteResponse(BaseModel):
    """Response after deleting a scenario"""
    name: str
    message: str = "Scenario deleted successfully"


class TemplateResponse(BaseModel):
    """Template information"""
    name: str
    display_name: str
    description: str
    environment: Dict[str, Any]
    training: Dict[str, Any]
    reward: Dict[str, Any]


# Endpoints

@router.get("", response_model=List[ScenarioSummary])
async def list_scenarios():
    """List all available scenarios
    
    Returns:
        List of scenario summaries with key metrics
    """
    try:
        scenarios = ScenarioService.list_scenarios()
        return scenarios
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list scenarios: {str(e)}"
        )


@router.get("/templates", response_model=List[TemplateResponse])
async def get_templates():
    """Get preset scenario templates
    
    Returns:
        List of available templates
    """
    try:
        templates = ScenarioService.get_templates()
        return templates
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get templates: {str(e)}"
        )


@router.get("/{name}", response_model=ScenarioDetail)
async def get_scenario(name: str):
    """Get detailed information for a specific scenario
    
    Args:
        name: Scenario name
        
    Returns:
        Complete scenario configuration
        
    Raises:
        404: If scenario not found
        500: If error reading scenario
    """
    try:
        scenario = ScenarioService.get_scenario(name)
        return scenario
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario '{name}' not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scenario: {str(e)}"
        )


@router.post("", response_model=ScenarioCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_scenario(scenario: ScenarioConfig):
    """Create a new scenario
    
    Args:
        scenario: Complete scenario configuration
        
    Returns:
        Created scenario information
        
    Raises:
        400: If validation fails
        409: If scenario name already exists
        500: If error creating scenario
    """
    try:
        result = ScenarioService.create_scenario(scenario)
        return {
            **result,
            "message": "Scenario created successfully"
        }
    except ValueError as e:
        # Check if it's a duplicate name error
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create scenario: {str(e)}"
        )


@router.put("/{name}", response_model=ScenarioUpdateResponse)
async def update_scenario(name: str, scenario: ScenarioConfig):
    """Update an existing scenario
    
    Args:
        name: Current scenario name
        scenario: Updated scenario configuration
        
    Returns:
        Updated scenario information
        
    Raises:
        400: If validation fails
        404: If scenario not found
        409: If trying to rename to existing name
        500: If error updating scenario
    """
    try:
        result = ScenarioService.update_scenario(name, scenario)
        return {
            **result,
            "message": "Scenario updated successfully"
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scenario '{name}' not found"
        )
    except ValueError as e:
        # Check if it's a duplicate name error
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update scenario: {str(e)}"
        )


@router.delete("/{name}", response_model=ScenarioDeleteResponse)
async def delete_scenario(name: str):
    """Delete a scenario
    
    Args:
        name: Scenario name to delete
        
    Returns:
        Deletion confirmation
        
    Raises:
        404: If scenario not found
        500: If error deleting scenario
    """
    try:
        deleted = ScenarioService.delete_scenario(name)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Scenario '{name}' not found"
            )
        return {
            "name": name,
            "message": "Scenario deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete scenario: {str(e)}"
        )
