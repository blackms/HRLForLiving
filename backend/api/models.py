"""Models API endpoints

This module provides REST API endpoints for:
- Listing trained models
- Getting model details
- Deleting models
"""
from fastapi import APIRouter, HTTPException, status
from typing import List
from datetime import datetime

from backend.models.responses import (
    ModelSummary,
    ModelDetail,
    ModelListResponse,
    ErrorResponse
)
from backend.services.model_service import ModelService

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List all trained models",
    description="Get a list of all trained models with summary information"
)
async def list_models():
    """List all trained models
    
    Returns:
        ModelListResponse with list of model summaries
    """
    try:
        models = ModelService.list_models()
        
        # Convert to Pydantic models
        model_summaries = [ModelSummary(**model) for model in models]
        
        return ModelListResponse(
            models=model_summaries,
            total=len(model_summaries)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.get(
    "/{name}",
    response_model=ModelDetail,
    summary="Get model details",
    description="Get detailed information about a specific trained model",
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"}
    }
)
async def get_model(name: str):
    """Get detailed model information
    
    Args:
        name: Model name
        
    Returns:
        ModelDetail with complete model information
        
    Raises:
        HTTPException: 404 if model not found
    """
    try:
        model_detail = ModelService.get_model(name)
        return ModelDetail(**model_detail)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model details: {str(e)}"
        )


@router.delete(
    "/{name}",
    summary="Delete a model",
    description="Delete a trained model and its associated files",
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"}
    }
)
async def delete_model(name: str):
    """Delete a trained model
    
    Args:
        name: Model name to delete
        
    Returns:
        Success message
        
    Raises:
        HTTPException: 404 if model not found
    """
    try:
        deleted = ModelService.delete_model(name)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{name}' not found"
            )
        
        return {
            "message": f"Model '{name}' deleted successfully",
            "deleted_at": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )
