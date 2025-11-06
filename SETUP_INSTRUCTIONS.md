# Setup Instructions

Complete setup guide for the HRL Finance System (Core System + Web Interface).

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Setup](#quick-setup)
3. [Core System Setup](#core-system-setup)
4. [Web Interface Setup](#web-interface-setup)
5. [Verification](#verification)
6. [Next Steps](#next-steps)

## System Requirements

### Hardware Requirements

- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Disk Space**: Minimum 2GB free space
- **GPU**: Optional (CUDA-compatible GPU for faster training)

### Software Requirements

**Core System**:
- Python 3.10 or higher
- pip (Python package manager)
- Git (for cloning repository)

**Web Interface**:
- Node.js 18 or higher
- npm 9 or higher (comes with Node.js)

**Optional**:
- WeasyPrint dependencies (for PDF report generation)
- CUDA toolkit (for GPU acceleration)

### Operating System Support

- ✅ macOS (10.15+)
- ✅ Linux (Ubuntu 20.04+, Debian 11+, Fedora 35+)
- ✅ Windows 10/11 (with WSL2 recommended)

## Quick Setup

Get up and running in 5 minutes:

```bash
# 1. Clone repository
git clone <repository-url>
cd hrl-finance-system

# 2. Install core system dependencies
pip install -r requirements.txt

# 3. Install web interface dependencies
cd frontend
npm install
cd ..

# 4. Start backend server
cd backend
uvicorn main:socket_app --reload --port 8000 &
cd ..

# 5. Start frontend application
cd frontend
npm run dev
```

Open your browser to http://localhost:5173

## Core System Setup

### Step 1: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10 or higher

# Test imports
python -c "import torch; import gymnasium; import stable_baselines3; print('Core system ready!')"
```

### Step 3: Run Quick Test

```bash
# Train a model with balanced profile (1000 episodes, ~5-10 minutes)
python train.py --profile balanced --episodes 1000

# Evaluate the trained model
python evaluate.py \
  --high-agent models/balanced_high_agent.pt \
  --low-agent models/balanced_low_agent.pt \
  --episodes 20
```

### Core System Dependencies

The following packages will be installed:

- **gymnasium** (0.29.0+) - RL environment framework
- **numpy** (1.24.0+) - Numerical computing
- **stable-baselines3** (2.0.0+) - RL algorithms (PPO)
- **torch** (2.0.0+) - Neural network framework
- **pyyaml** (6.0+) - Configuration file parsing
- **tensorboard** (2.14.0+) - Experiment tracking

## Web Interface Setup

### Step 1: Install Node.js

**macOS** (using Homebrew):
```bash
brew install node
```

**Ubuntu/Debian**:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

**Windows**:
Download installer from https://nodejs.org/

### Step 2: Install Backend Dependencies

```bash
cd backend

# Install Python dependencies (if not already done)
pip install -r requirements.txt

# Verify installation
python -c "import fastapi; import uvicorn; print('Backend ready!')"
```

### Step 3: Install Frontend Dependencies

```bash
cd frontend

# Install Node dependencies
npm install

# Verify installation
npm list react typescript vite
```

### Step 4: Start Backend Server

```bash
cd backend

# Start with WebSocket support
uvicorn main:socket_app --reload --port 8000

# Or start in background
uvicorn main:socket_app --reload --port 8000 &
```

The backend API will be available at:
- API Root: http://localhost:8000
- Health Check: http://localhost:8000/health
- API Documentation: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/socket.io

### Step 5: Start Frontend Application

```bash
cd frontend

# Start development server
npm run dev
```

The web interface will be available at http://localhost:5173

### Web Interface Dependencies

**Backend**:
- **fastapi** - Modern web framework
- **uvicorn** - ASGI server
- **pydantic** - Data validation
- **python-socketio** - WebSocket support
- **pyyaml** - YAML parsing
- **torch** - Model loading

**Frontend**:
- **react** (19) - UI library
- **typescript** - Type safety
- **vite** - Build tool
- **tailwindcss** - Styling
- **recharts** - Data visualization
- **axios** - HTTP client
- **socket.io-client** - WebSocket client
- **react-router-dom** - Routing

## Verification

### Verify Core System

```bash
# Run tests
pytest tests/ -v

# Run sanity checks
pytest tests/test_sanity_checks.py -v

# Check for NaN issues
python debug_nan.py
```

### Verify Backend API

```bash
# Check health endpoint
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","timestamp":"..."}

# List scenarios
curl http://localhost:8000/api/scenarios

# View API documentation
open http://localhost:8000/docs  # macOS
xdg-open http://localhost:8000/docs  # Linux
start http://localhost:8000/docs  # Windows
```

### Verify Frontend

1. Open browser to http://localhost:5173
2. You should see the Dashboard page
3. Check browser console (F12) for errors
4. Try navigating to different pages

### Verify WebSocket Connection

1. Navigate to Training Monitor page
2. Check for green connection indicator
3. Browser console should show: "Connected to training updates"

## Optional Setup

### PDF Report Generation (WeasyPrint)

**macOS**:
```bash
brew install python3 cairo pango gdk-pixbuf libffi
pip install weasyprint
```

**Ubuntu/Debian**:
```bash
sudo apt-get install python3-dev python3-pip python3-cffi \
  libcairo2 libpango-1.0-0 libpangocairo-1.0-0 \
  libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
pip install weasyprint
```

**Windows**:
1. Download GTK3 runtime from https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
2. Install GTK3 runtime
3. `pip install weasyprint`

**Note**: If WeasyPrint installation fails, HTML reports will still work.

### GPU Acceleration (Optional)

If you have an NVIDIA GPU:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch (if needed)
# Visit: https://pytorch.org/get-started/locally/
# Select your CUDA version and follow instructions
```

### TensorBoard (Already Included)

TensorBoard is included in requirements.txt. To use:

```bash
# Start TensorBoard
tensorboard --logdir=runs

# Open browser to: http://localhost:6006
```

## Directory Structure

After setup, your directory structure should look like:

```
hrl-finance-system/
├── backend/                 # FastAPI backend
│   ├── api/                # API endpoints
│   ├── models/             # Pydantic models
│   ├── services/           # Business logic
│   ├── utils/              # Utilities
│   ├── websocket/          # WebSocket handlers
│   ├── main.py             # Application entry
│   └── requirements.txt    # Python dependencies
├── frontend/                # React frontend
│   ├── src/                # Source code
│   ├── public/             # Static assets
│   ├── node_modules/       # Node dependencies (after npm install)
│   ├── package.json        # Node dependencies
│   └── vite.config.ts      # Vite configuration
├── src/                     # Core system source
│   ├── agents/             # RL agents
│   ├── environment/        # Simulation environment
│   ├── training/           # Training orchestration
│   └── utils/              # Utilities
├── configs/                 # Configuration files
│   └── scenarios/          # Scenario configurations
├── models/                  # Trained models (created during training)
├── results/                 # Simulation results (created during use)
├── reports/                 # Generated reports (created during use)
├── tests/                   # Test suite
├── examples/                # Usage examples
├── requirements.txt         # Core system dependencies
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
└── README.md                # Main documentation
```

## Troubleshooting Setup

### Python Version Issues

**Problem**: `python --version` shows Python 2.x or < 3.10

**Solution**:
```bash
# Try python3 instead
python3 --version

# Or install Python 3.10+
# macOS: brew install python@3.10
# Ubuntu: sudo apt install python3.10
# Windows: Download from python.org
```

### pip Install Fails

**Problem**: `pip install -r requirements.txt` fails

**Solution**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install in virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install dependencies individually if needed
pip install torch gymnasium stable-baselines3 numpy pyyaml tensorboard
```

### Node/npm Not Found

**Problem**: `node: command not found`

**Solution**:
```bash
# Install Node.js
# macOS: brew install node
# Ubuntu: curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs
# Windows: Download from nodejs.org

# Verify installation
node --version
npm --version
```

### Port Already in Use

**Problem**: `Address already in use` error

**Solution**:
```bash
# Find process using port 8000
# macOS/Linux:
lsof -i :8000
kill -9 <PID>

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use different port
uvicorn main:socket_app --reload --port 8001
```

### Frontend Won't Start

**Problem**: `npm run dev` fails

**Solution**:
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite

# Try again
npm run dev
```

### CORS Errors

**Problem**: Browser shows CORS policy errors

**Solution**:
1. Ensure backend is running on port 8000
2. Ensure frontend is running on port 5173
3. Check `backend/main.py` CORS configuration
4. Restart both backend and frontend

### WebSocket Connection Fails

**Problem**: WebSocket connection indicator shows red

**Solution**:
1. Ensure backend is running with `socket_app` (not just `app`)
2. Check backend logs for WebSocket errors
3. Verify URL: `ws://localhost:8000/socket.io`
4. Try refreshing the page
5. Check browser console for errors

## Next Steps

After successful setup:

1. **Read the User Guide**: See `USER_GUIDE.md` for complete feature documentation
2. **Try Quick Start**: Follow `QUICK_START.md` for a 5-minute tutorial
3. **Explore Examples**: Check `examples/` directory for code examples
4. **Create Your First Scenario**: Use the web interface to create a financial scenario
5. **Train Your First Model**: Train a model on your scenario
6. **Run Simulations**: Evaluate your trained model
7. **Generate Reports**: Create professional reports of your results

## Getting Help

If you encounter issues:

1. **Check Troubleshooting Guide**: See `TROUBLESHOOTING.md`
2. **Check Documentation**: See `README.md` and `USER_GUIDE.md`
3. **Check API Docs**: Visit http://localhost:8000/docs
4. **Check Logs**: Review backend terminal output and browser console
5. **Run Tests**: `pytest tests/ -v` to verify system integrity
6. **Open Issue**: Report bugs on GitHub

## Development Setup

For development work:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

## Production Deployment

For production deployment:

```bash
# Build frontend
cd frontend
npm run build

# Serve with production server
cd backend
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:socket_app

# Or use Docker
docker-compose up -d
```

See `DEPLOYMENT.md` for complete deployment instructions.

---

**Last Updated**: November 2024
**Version**: 1.0
**For**: HRL Finance System
