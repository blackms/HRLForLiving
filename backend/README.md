# HRL Finance System Backend

FastAPI backend for the HRL Finance System.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --port 8000
```

## API Endpoints

### Core Endpoints

- `GET /` - Root endpoint returning API information
- `GET /health` - Health check endpoint for monitoring

### API Documentation

Once running, visit:
- API Root: http://localhost:8000
- Health Check: http://localhost:8000/health
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Current Implementation Status

✅ **Completed:**
- FastAPI application initialization
- Root endpoint with API information
- Health check endpoint
- CORS middleware configuration (ready for frontend integration)
- OpenAPI documentation (auto-generated)

🚧 **In Progress:**
- API endpoint implementations (scenarios, training, simulation, models, reports)
- WebSocket support for real-time training updates
- Service layer business logic
- Pydantic models for request/response validation
