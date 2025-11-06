# API Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Start the Server

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Access Documentation

Open your browser to:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 3. Try Your First Request

```bash
# Check API health
curl http://localhost:8000/health

# List available scenarios
curl http://localhost:8000/api/scenarios

# Get scenario templates
curl http://localhost:8000/api/scenarios/templates
```

## 📝 Complete Workflow Example

### Step 1: Create a Scenario

```bash
curl -X POST http://localhost:8000/api/scenarios \
  -H "Content-Type: application/json" \
  -d '{
    "name": "quickstart_scenario",
    "description": "Quick start example",
    "environment": {
      "income": 3000,
      "fixed_expenses": 1200,
      "variable_expense_mean": 500,
      "variable_expense_std": 100,
      "inflation": 0.002,
      "safety_threshold": 5000,
      "max_months": 120,
      "initial_cash": 10000,
      "risk_tolerance": 0.5,
      "investment_return_mean": 0.005,
      "investment_return_std": 0.02,
      "investment_return_type": "stochastic"
    }
  }'
```

### Step 2: Start Training

```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_name": "quickstart_scenario",
    "num_episodes": 100,
    "save_interval": 50,
    "eval_episodes": 10
  }'
```

### Step 3: Monitor Training

```bash
# Check status
curl http://localhost:8000/api/training/status

# Or connect via WebSocket for real-time updates
# See WebSocket examples in API_DOCUMENTATION.md
```

### Step 4: Run Simulation

```bash
curl -X POST http://localhost:8000/api/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "quickstart_scenario",
    "scenario_name": "quickstart_scenario",
    "num_episodes": 10
  }'
```

### Step 5: Generate Report

```bash
# Use the simulation_id from step 4 response
curl -X POST http://localhost:8000/api/reports/generate \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "YOUR_SIMULATION_ID",
    "report_type": "html",
    "title": "Quick Start Report"
  }'
```

### Step 6: Download Report

```bash
# Use the report_id from step 5 response
curl http://localhost:8000/api/reports/YOUR_REPORT_ID -o report.html
```

## 🐍 Python Client Example

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Create scenario
scenario = {
    "name": "python_example",
    "description": "Python client example",
    "environment": {
        "income": 3000,
        "fixed_expenses": 1200,
        "variable_expense_mean": 500,
        "variable_expense_std": 100,
        "inflation": 0.002,
        "safety_threshold": 5000,
        "max_months": 120,
        "initial_cash": 10000,
        "risk_tolerance": 0.5,
        "investment_return_mean": 0.005,
        "investment_return_std": 0.02,
        "investment_return_type": "stochastic"
    }
}

response = requests.post(f"{BASE_URL}/api/scenarios", json=scenario)
print(f"✓ Scenario created: {response.json()['name']}")

# 2. Start training
training = {
    "scenario_name": "python_example",
    "num_episodes": 100,
    "save_interval": 50,
    "eval_episodes": 10
}

response = requests.post(f"{BASE_URL}/api/training/start", json=training)
print(f"✓ Training started")

# 3. Wait for training to complete
while True:
    response = requests.get(f"{BASE_URL}/api/training/status")
    status = response.json()
    
    if not status['is_training']:
        print("✓ Training completed!")
        break
    
    progress = status.get('latest_progress', {})
    episode = progress.get('episode', 0)
    total = status['total_episodes']
    reward = progress.get('avg_reward', 0)
    
    print(f"  Episode {episode}/{total} - Reward: {reward:.2f}")
    time.sleep(5)

# 4. Run simulation
simulation = {
    "model_name": "python_example",
    "scenario_name": "python_example",
    "num_episodes": 10
}

response = requests.post(f"{BASE_URL}/api/simulation/run", json=simulation)
results = response.json()
simulation_id = results['simulation_id']
print(f"✓ Simulation completed: {simulation_id}")
print(f"  Total Wealth: ${results['results']['total_wealth_mean']:.2f}")

# 5. Generate report
report = {
    "simulation_id": simulation_id,
    "report_type": "html",
    "title": "Python Example Report"
}

response = requests.post(f"{BASE_URL}/api/reports/generate", json=report)
report_data = response.json()
report_id = report_data['report_id']
print(f"✓ Report generated: {report_id}")

# 6. Download report
response = requests.get(f"{BASE_URL}/api/reports/{report_id}")
with open("example_report.html", "wb") as f:
    f.write(response.content)
print(f"✓ Report saved: example_report.html")
```

## 🌐 JavaScript/TypeScript Client Example

```typescript
const BASE_URL = "http://localhost:8000";

async function runWorkflow() {
  // 1. Create scenario
  const scenario = {
    name: "js_example",
    description: "JavaScript client example",
    environment: {
      income: 3000,
      fixed_expenses: 1200,
      variable_expense_mean: 500,
      variable_expense_std: 100,
      inflation: 0.002,
      safety_threshold: 5000,
      max_months: 120,
      initial_cash: 10000,
      risk_tolerance: 0.5,
      investment_return_mean: 0.005,
      investment_return_std: 0.02,
      investment_return_type: "stochastic"
    }
  };

  let response = await fetch(`${BASE_URL}/api/scenarios`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(scenario)
  });
  console.log("✓ Scenario created");

  // 2. Start training
  const training = {
    scenario_name: "js_example",
    num_episodes: 100,
    save_interval: 50,
    eval_episodes: 10
  };

  response = await fetch(`${BASE_URL}/api/training/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(training)
  });
  console.log("✓ Training started");

  // 3. Monitor training (simplified)
  // In production, use WebSocket for real-time updates
  await new Promise(resolve => setTimeout(resolve, 60000)); // Wait 1 minute

  // 4. Run simulation
  const simulation = {
    model_name: "js_example",
    scenario_name: "js_example",
    num_episodes: 10
  };

  response = await fetch(`${BASE_URL}/api/simulation/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(simulation)
  });
  const results = await response.json();
  console.log(`✓ Simulation completed: ${results.simulation_id}`);

  // 5. Generate report
  const report = {
    simulation_id: results.simulation_id,
    report_type: "html",
    title: "JavaScript Example Report"
  };

  response = await fetch(`${BASE_URL}/api/reports/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(report)
  });
  const reportData = await response.json();
  console.log(`✓ Report generated: ${reportData.report_id}`);

  // 6. Download report
  window.open(`${BASE_URL}/api/reports/${reportData.report_id}`);
}

runWorkflow();
```

## 🔌 WebSocket Real-Time Updates

### Python

```python
import socketio

sio = socketio.Client()

@sio.on('connect')
def on_connect():
    print('Connected to training updates')

@sio.on('training_progress')
def on_progress(data):
    print(f"Episode {data['episode']}/{data['total_episodes']}")
    print(f"  Reward: {data['avg_reward']:.2f}")
    print(f"  Duration: {data['avg_duration']:.1f} months")
    print(f"  Stability: {data['stability']:.2%}")

@sio.on('training_completed')
def on_completed(data):
    print(f"Training completed: {data['scenario_name']}")
    sio.disconnect()

sio.connect('http://localhost:8000', socketio_path='/socket.io')
sio.wait()
```

### JavaScript

```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:8000', {
  path: '/socket.io',
  transports: ['websocket']
});

socket.on('connect', () => {
  console.log('Connected to training updates');
});

socket.on('training_progress', (data) => {
  console.log(`Episode ${data.episode}/${data.total_episodes}`);
  console.log(`  Reward: ${data.avg_reward.toFixed(2)}`);
  console.log(`  Duration: ${data.avg_duration.toFixed(1)} months`);
  console.log(`  Stability: ${(data.stability * 100).toFixed(1)}%`);
});

socket.on('training_completed', (data) => {
  console.log(`Training completed: ${data.scenario_name}`);
  socket.disconnect();
});
```

## 📚 Next Steps

1. **Read the full documentation:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
2. **Explore the interactive docs:** http://localhost:8000/docs
3. **Check out the frontend:** See `../frontend/README.md`
4. **Review data models:** See `models/requests.py` and `models/responses.py`

## 🆘 Common Issues

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -i :8000
kill -9 <PID>

# Or use a different port
uvicorn main:app --reload --port 8001
```

### Training Not Starting

```bash
# Check if training is already in progress
curl http://localhost:8000/api/training/status

# Stop existing training if needed
curl -X POST http://localhost:8000/api/training/stop
```

### Model Not Found

```bash
# List available models
curl http://localhost:8000/api/models

# Ensure training completed successfully
curl http://localhost:8000/api/training/status
```

## 💡 Tips

1. **Use templates:** Start with preset templates instead of creating scenarios from scratch
2. **Start small:** Use fewer episodes (100-200) for testing, more (1000+) for production
3. **Monitor training:** Use WebSocket for real-time updates instead of polling
4. **Save checkpoints:** Set appropriate `save_interval` to avoid losing progress
5. **Use seeds:** Set `seed` parameter for reproducible results

## 📖 Additional Resources

- **Complete API Documentation:** [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
- **Backend README:** [README.md](./README.md)
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
