# FastAPI application entry point
from fastapi import FastAPI

app = FastAPI(
    title="HRL Finance System API",
    description="API for Hierarchical Reinforcement Learning Personal Finance Optimization",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "HRL Finance System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
