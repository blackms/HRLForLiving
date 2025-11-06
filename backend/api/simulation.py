"""Simulation API endpoints"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from datetime import datetime

from backend.models.requests import SimulationRequest
from backend.models.responses import (
    SimulationResults,
    SimulationHistoryResponse,
    ErrorResponse
)
from backend.services.simulation_service import simulation_service


router = APIRouter(prefix="/api/simulation", tags=["simulation"])


@router.post("/run", status_code=202)
async def run_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Run a simulation with a trained model on a scenario
    
    Args:
        request: Simulation request with model_name, scenario_name, num_episodes, seed
        background_tasks: FastAPI background tasks
        
    Returns:
        dict: Simulation start confirmation with simulation_id
        
    Raises:
        HTTPException 404: If model or scenario not found
        HTTPException 500: If simulation fails to start
    """
    try:
        # Start simulation in background
        # For now, we'll run it synchronously but could be made async
        results = await simulation_service.run_simulation(
            model_name=request.model_name,
            scenario_name=request.scenario_name,
            num_episodes=request.num_episodes,
            seed=request.seed
        )
        
        return {
            "status": "completed",
            "simulation_id": results['simulation_id'],
            "message": f"Simulation completed with {request.num_episodes} episodes",
            "results": results
        }
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NotFound",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SimulationError",
                "message": f"Failed to run simulation: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/results/{simulation_id}")
async def get_simulation_results(simulation_id: str) -> Dict[str, Any]:
    """
    Get results for a specific simulation
    
    Args:
        simulation_id: Unique simulation identifier
        
    Returns:
        dict: Simulation results with statistics and episode data
        
    Raises:
        HTTPException 404: If simulation not found
        HTTPException 500: If error retrieving results
    """
    try:
        results = simulation_service.get_simulation_results(simulation_id)
        return results
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NotFound",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RetrievalError",
                "message": f"Failed to retrieve simulation results: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get("/history")
async def get_simulation_history() -> SimulationHistoryResponse:
    """
    Get list of all past simulations
    
    Returns:
        SimulationHistoryResponse: List of simulation summaries
        
    Raises:
        HTTPException 500: If error listing simulations
    """
    try:
        simulations = simulation_service.list_simulations()
        
        return SimulationHistoryResponse(
            simulations=simulations,
            total=len(simulations)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ListError",
                "message": f"Failed to list simulations: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )
