# FastAPI application entry point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from backend.api.scenarios import router as scenarios_router
from backend.api.training import router as training_router
from backend.api.simulation import router as simulation_router
from backend.api.models import router as models_router
from backend.api.reports import router as reports_router
from backend.websocket import sio

# API metadata
tags_metadata = [
    {
        "name": "scenarios",
        "description": "Operations for managing financial scenarios. Scenarios define the environment parameters, training configuration, and reward structure for the HRL system.",
    },
    {
        "name": "training",
        "description": "Operations for training AI models on scenarios. Training uses hierarchical reinforcement learning to learn optimal financial strategies.",
    },
    {
        "name": "simulation",
        "description": "Operations for running simulations with trained models. Simulations evaluate model performance and generate detailed financial projections.",
    },
    {
        "name": "models",
        "description": "Operations for managing trained models. Models contain the learned policies for financial decision-making.",
    },
    {
        "name": "reports",
        "description": "Operations for generating and downloading reports. Reports provide comprehensive analysis of simulation results.",
    },
]

app = FastAPI(
    title="HRL Finance System API",
    description="""
# HRL Finance System API

A comprehensive API for Hierarchical Reinforcement Learning-based Personal Finance Optimization.

## Overview

This API provides a complete interface for:
- **Creating and managing financial scenarios** with customizable parameters
- **Training AI models** using hierarchical reinforcement learning
- **Running simulations** to evaluate financial strategies
- **Generating reports** with detailed analysis and visualizations
- **Real-time training updates** via WebSocket connections

## Key Features

### 🎯 Scenario Management
Create realistic financial scenarios with parameters like income, expenses, inflation, risk tolerance, and investment returns.

### 🤖 AI Training
Train hierarchical reinforcement learning models that learn optimal financial strategies through trial and error.

### 📊 Simulation & Analysis
Run simulations with trained models to see predicted financial outcomes over time.

### 📄 Report Generation
Generate comprehensive PDF or HTML reports with charts, statistics, and strategy analysis.

### 🔄 Real-time Updates
Monitor training progress in real-time using WebSocket connections.

## Getting Started

1. **Create a scenario** using `POST /api/scenarios`
2. **Train a model** using `POST /api/training/start`
3. **Run a simulation** using `POST /api/simulation/run`
4. **Generate a report** using `POST /api/reports/generate`

## WebSocket Connection

Connect to `/socket.io` for real-time training updates:
- Event: `training_started` - Training has begun
- Event: `training_progress` - Progress update with metrics
- Event: `training_completed` - Training finished successfully
- Event: `training_stopped` - Training stopped by user
- Event: `training_error` - Training encountered an error

## Authentication

Currently, this API does not require authentication. In production environments, implement appropriate authentication and authorization mechanisms.

## Rate Limiting

No rate limiting is currently enforced. Consider implementing rate limiting for production deployments.

## Support

For issues, questions, or contributions, please refer to the project documentation.
    """,
    version="1.0.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "HRL Finance System",
        "url": "https://github.com/yourusername/hrl-finance-system",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scenarios_router)
app.include_router(training_router)
app.include_router(simulation_router)
app.include_router(models_router)
app.include_router(reports_router)

# Mount Socket.IO
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path='/socket.io'
)

@app.get(
    "/",
    summary="API Root",
    description="Get basic API information and links to documentation",
    tags=["general"]
)
async def root():
    """
    API root endpoint providing basic information and navigation links.
    
    Returns:
        dict: API information with links to documentation
    
    Example:
        ```bash
        curl http://localhost:8000/
        ```
    """
    return {
        "message": "HRL Finance System API",
        "version": "1.0.0",
        "description": "Hierarchical Reinforcement Learning for Personal Finance Optimization",
        "documentation": {
            "interactive": {
                "swagger": "/docs",
                "redoc": "/redoc"
            },
            "files": {
                "index": "backend/API_DOCUMENTATION_INDEX.md",
                "complete": "backend/API_DOCUMENTATION.md",
                "quick_start": "backend/API_QUICK_START.md",
                "readme": "backend/README.md"
            },
            "openapi_schema": "/openapi.json"
        },
        "endpoints": {
            "scenarios": "/api/scenarios",
            "training": "/api/training",
            "simulation": "/api/simulation",
            "models": "/api/models",
            "reports": "/api/reports"
        },
        "websocket": "/socket.io",
        "status": "operational"
    }

@app.get(
    "/health",
    summary="Health Check",
    description="Check if the API service is running and healthy",
    tags=["general"],
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "version": "1.0.0",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        }
    }
)
async def health_check():
    """
    Health check endpoint for monitoring service availability.
    
    Returns:
        dict: Health status information
    """
    from datetime import datetime
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
