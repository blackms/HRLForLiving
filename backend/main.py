# FastAPI application entry point
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.scenarios import router as scenarios_router

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

@app.get("/")
async def root():
    return {"message": "HRL Finance System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
