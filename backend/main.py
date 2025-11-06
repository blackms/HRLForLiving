# FastAPI application entry point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from backend.api.scenarios import router as scenarios_router
from backend.api.training import router as training_router
from backend.api.simulation import router as simulation_router
from backend.api.models import router as models_router
from backend.websocket import sio

app = FastAPI(
    title="HRL Finance System API",
    description="API for Hierarchical Reinforcement Learning Personal Finance Optimization",
    version="1.0.0"
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

# Mount Socket.IO
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path='/socket.io'
)

@app.get("/")
async def root():
    return {"message": "HRL Finance System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
