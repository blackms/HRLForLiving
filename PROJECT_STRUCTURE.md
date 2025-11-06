# HRL Finance UI - Project Structure

## Overview

This project consists of two main components:
- **Backend**: FastAPI server exposing the HRL Finance System
- **Frontend**: React + TypeScript web application

## Directory Structure

```
.
├── backend/                    # FastAPI backend
│   ├── api/                   # API endpoint handlers
│   ├── models/                # Pydantic request/response models
│   ├── services/              # Business logic layer
│   ├── utils/                 # Utility functions
│   ├── websocket/             # WebSocket handlers
│   ├── main.py                # FastAPI application entry
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Backend documentation
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/        # Reusable React components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API and WebSocket clients
│   │   ├── types/             # TypeScript type definitions
│   │   ├── utils/             # Utility functions
│   │   ├── App.tsx            # Main app component
│   │   ├── main.tsx           # Application entry point
│   │   └── index.css          # Global styles (Tailwind)
│   ├── public/                # Static assets
│   ├── package.json           # Node dependencies
│   ├── tailwind.config.js     # Tailwind CSS configuration
│   ├── vite.config.ts         # Vite configuration
│   └── README.md              # Frontend documentation
│
└── PROJECT_STRUCTURE.md       # This file
```

## Backend Dependencies

- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **Pydantic**: Data validation using Python type annotations
- **python-socketio**: WebSocket support for real-time updates
- **PyYAML**: YAML file parsing for configurations
- **PyTorch**: Deep learning framework (for HRL models)

## Frontend Dependencies

### Core
- **React 19**: UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Build tool and dev server

### UI & Styling
- **Tailwind CSS**: Utility-first CSS framework
- **Recharts**: Charting library for data visualization

### Networking
- **Axios**: HTTP client for API requests
- **Socket.IO Client**: WebSocket client for real-time updates
- **React Router**: Client-side routing

## Getting Started

### Backend Setup

```bash
cd backend

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --port 8000
```

The API will be available at:
- API Root: http://localhost:8000
- Health Check: http://localhost:8000/health
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Current Status:**
- ✅ FastAPI application initialized
- ✅ Root endpoint (`/`) returning API information
- ✅ Health check endpoint (`/health`) for monitoring
- ✅ Pydantic request models (EnvironmentConfig, TrainingConfig, RewardConfig, ScenarioConfig, TrainingRequest, SimulationRequest, ReportRequest)
- ✅ Pydantic response models (TrainingProgress, SimulationResults, ScenarioSummary, ModelSummary, ErrorResponse, etc.)
- ✅ File management utilities (YAML, PyTorch models, JSON results) with security features
- ✅ **Scenarios API (complete CRUD operations)**
- ✅ **Scenario service layer with business logic**
- ✅ **Training API with WebSocket support** ⭐
- ✅ **Training service layer with HRL orchestration** ⭐
- ✅ **Real-time training progress updates via WebSocket** ⭐
- 🚧 Simulation API in development
- 🚧 Models API in development
- 🚧 Reports API in development

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The app will be available at http://localhost:5173

### Build for Production

```bash
# Frontend
cd frontend
npm run build

# Backend (with Gunicorn)
cd backend
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

## Next Steps

The project structure is now ready. The next tasks involve:

1. Implementing backend API endpoints
2. Creating frontend components and pages
3. Integrating WebSocket for real-time training updates
4. Building the user interface for scenario management
5. Implementing visualization components

Refer to `.kiro/specs/hrl-finance-ui/tasks.md` for the complete implementation plan.
