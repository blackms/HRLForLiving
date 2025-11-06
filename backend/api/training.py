"""Training API endpoints"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from backend.models.requests import TrainingRequest
from backend.models.responses import TrainingStatus, ErrorResponse
from backend.services import training_service
from backend.websocket import socket_manager

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def start_training(request: TrainingRequest):
    """
    Start training a model on a scenario
    
    Args:
        request: Training configuration
        
    Returns:
        Training start confirmation
        
    Raises:
        HTTPException 400: If training is already in progress
        HTTPException 404: If scenario not found
        HTTPException 500: If training fails to start
    """
    try:
        # Set progress callback to emit WebSocket updates
        training_service.set_progress_callback(socket_manager.emit_progress)
        
        # Start training
        result = await training_service.start_training(
            scenario_name=request.scenario_name,
            num_episodes=request.num_episodes,
            save_interval=request.save_interval,
            eval_episodes=request.eval_episodes,
            seed=request.seed
        )
        
        # Emit training started event
        await socket_manager.emit_training_started({
            'scenario_name': request.scenario_name,
            'num_episodes': request.num_episodes,
            'start_time': result['start_time']
        })
        
        return {
            'message': 'Training started successfully',
            'data': result
        }
        
    except ValueError as e:
        # Training already in progress
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        # Scenario not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected error
        await socket_manager.emit_training_error({
            'message': 'Failed to start training',
            'details': str(e)
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )


@router.post("/stop")
async def stop_training():
    """
    Stop the current training process
    
    Returns:
        Training stop confirmation
        
    Raises:
        HTTPException 400: If no training is in progress
        HTTPException 500: If stop fails
    """
    try:
        result = await training_service.stop_training()
        
        # Emit training stopped event
        await socket_manager.emit_training_stopped({
            'scenario_name': result['scenario_name'],
            'episodes_completed': result['episodes_completed']
        })
        
        return {
            'message': 'Training stopped successfully',
            'data': result
        }
        
    except ValueError as e:
        # No training in progress
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop training: {str(e)}"
        )


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """
    Get current training status
    
    Returns:
        Current training status including progress
    """
    try:
        status_data = training_service.get_status()
        
        # Convert to response model
        return TrainingStatus(
            is_training=status_data['is_training'],
            scenario_name=status_data['scenario_name'],
            current_episode=status_data['current_episode'],
            total_episodes=status_data['total_episodes'],
            start_time=(
                datetime.fromisoformat(status_data['start_time'])
                if status_data['start_time']
                else None
            ),
            latest_progress=status_data['latest_progress']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}"
        )
