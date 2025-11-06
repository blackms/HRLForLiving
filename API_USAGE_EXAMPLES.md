# API Usage Examples

Practical examples for using the HRL Finance System API.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Python Examples](#python-examples)
3. [JavaScript Examples](#javascript-examples)
4. [cURL Examples](#curl-examples)
5. [Complete Workflows](#complete-workflows)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

## Getting Started

### Prerequisites

- Backend running at `http://localhost:8000`
- API documentation available at `http://localhost:8000/docs`

### Base URL

```
http://localhost:8000
```

### Authentication

Currently, the API does not require authentication. For production use, implement proper authentication.

## Python Examples

### Setup

```python
import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def api_request(method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
    """Make API request with error handling"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        elif method == "PUT":
            response = requests.put(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return None
```


### Scenarios API

#### List All Scenarios

```python
def list_scenarios():
    """Get list of all scenarios"""
    scenarios = api_request("GET", "/api/scenarios")
    
    if scenarios:
        print(f"Found {len(scenarios)} scenarios:")
        for scenario in scenarios:
            print(f"  - {scenario['name']}: {scenario['income']}€/month, "
                  f"{scenario['available_pct']:.1f}% available")
    
    return scenarios

# Usage
scenarios = list_scenarios()
```

#### Get Scenario Details

```python
def get_scenario(name: str):
    """Get detailed scenario configuration"""
    scenario = api_request("GET", f"/api/scenarios/{name}")
    
    if scenario:
        print(f"Scenario: {scenario['name']}")
        print(f"Income: €{scenario['environment']['income']}")
        print(f"Fixed Expenses: €{scenario['environment']['fixed_expenses']}")
        print(f"Risk Tolerance: {scenario['environment']['risk_tolerance']}")
    
    return scenario

# Usage
scenario = get_scenario("bologna_coppia")
```

#### Create New Scenario

```python
def create_scenario(name: str, income: float, fixed_expenses: float):
    """Create a new financial scenario"""
    data = {
        "name": name,
        "description": f"Custom scenario: {name}",
        "environment": {
            "income": income,
            "fixed_expenses": fixed_expenses,
            "variable_expense_mean": 700,
            "variable_expense_std": 100,
            "inflation": 0.02,
            "safety_threshold": 1000,
            "max_months": 60,
            "initial_cash": 0,
            "risk_tolerance": 0.5,
            "investment_return_mean": 0.005,
            "investment_return_std": 0.02,
            "investment_return_type": "stochastic"
        }
    }
    
    result = api_request("POST", "/api/scenarios", data)
    
    if result:
        print(f"Created scenario: {result['name']}")
        print(f"Path: {result['path']}")
    
    return result

# Usage
new_scenario = create_scenario("my_scenario", 3000, 1500)
```

#### Update Scenario

```python
def update_scenario(name: str, updates: Dict):
    """Update an existing scenario"""
    # First get current scenario
    current = get_scenario(name)
    if not current:
        return None
    
    # Merge updates
    current['environment'].update(updates)
    
    result = api_request("PUT", f"/api/scenarios/{name}", current)
    
    if result:
        print(f"Updated scenario: {result['name']}")
    
    return result

# Usage
updated = update_scenario("my_scenario", {
    "income": 3500,
    "risk_tolerance": 0.7
})
```

#### Delete Scenario

```python
def delete_scenario(name: str):
    """Delete a scenario"""
    result = api_request("DELETE", f"/api/scenarios/{name}")
    
    if result:
        print(f"Deleted scenario: {result['name']}")
    
    return result

# Usage
delete_scenario("my_scenario")
```

#### Get Templates

```python
def get_templates():
    """Get preset scenario templates"""
    templates = api_request("GET", "/api/scenarios/templates")
    
    if templates:
        print(f"Available templates:")
        for template in templates:
            print(f"  - {template['display_name']}: {template['description']}")
    
    return templates

# Usage
templates = get_templates()
```

### Training API

#### Start Training

```python
def start_training(scenario_name: str, episodes: int = 1000):
    """Start training a model"""
    data = {
        "scenario_name": scenario_name,
        "num_episodes": episodes,
        "save_interval": 100,
        "eval_episodes": 10,
        "seed": 42
    }
    
    result = api_request("POST", "/api/training/start", data)
    
    if result:
        print(f"Training started: {result['message']}")
        print(f"Scenario: {result['data']['scenario_name']}")
        print(f"Episodes: {result['data']['num_episodes']}")
    
    return result

# Usage
training = start_training("bologna_coppia", 1000)
```

#### Get Training Status

```python
import time

def monitor_training():
    """Monitor training progress"""
    while True:
        status = api_request("GET", "/api/training/status")
        
        if not status or not status['is_training']:
            print("Training not active")
            break
        
        progress = status.get('latest_progress', {})
        episode = progress.get('episode', 0)
        total = status.get('total_episodes', 0)
        reward = progress.get('avg_reward', 0)
        
        print(f"Episode {episode}/{total} - Reward: {reward:.2f}")
        
        if episode >= total:
            print("Training complete!")
            break
        
        time.sleep(5)  # Poll every 5 seconds

# Usage
monitor_training()
```

#### Stop Training

```python
def stop_training():
    """Stop current training"""
    result = api_request("POST", "/api/training/stop")
    
    if result:
        print(f"Training stopped: {result['message']}")
        print(f"Episodes completed: {result['data']['episodes_completed']}")
    
    return result

# Usage
stop_training()
```

### WebSocket Training Updates (Python)

```python
import socketio

def monitor_training_realtime(scenario_name: str):
    """Monitor training with real-time WebSocket updates"""
    sio = socketio.Client()
    
    @sio.on('connect')
    def on_connect():
        print("Connected to training updates")
    
    @sio.on('training_started')
    def on_started(data):
        print(f"Training started: {data['scenario_name']}")
        print(f"Episodes: {data['num_episodes']}")
    
    @sio.on('training_progress')
    def on_progress(data):
        episode = data['episode']
        total = data['total_episodes']
        reward = data['avg_reward']
        stability = data['stability']
        
        print(f"Episode {episode}/{total}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Stability: {stability:.1%}")
    
    @sio.on('training_completed')
    def on_completed(data):
        print(f"Training completed!")
        print(f"Final reward: {data['final_reward']:.2f}")
        sio.disconnect()
    
    @sio.on('training_error')
    def on_error(data):
        print(f"Training error: {data['message']}")
        sio.disconnect()
    
    # Connect and start training
    sio.connect('http://localhost:8000', socketio_path='/socket.io')
    start_training(scenario_name)
    sio.wait()

# Usage
monitor_training_realtime("bologna_coppia")
```

### Simulation API

#### Run Simulation

```python
def run_simulation(model_name: str, scenario_name: str, episodes: int = 10):
    """Run simulation with trained model"""
    data = {
        "model_name": model_name,
        "scenario_name": scenario_name,
        "num_episodes": episodes,
        "seed": 42
    }
    
    result = api_request("POST", "/api/simulation/run", data)
    
    if result and result['status'] == 'completed':
        results = result['results']
        print(f"Simulation completed: {result['simulation_id']}")
        print(f"Mean wealth: €{results['total_wealth_mean']:.2f}")
        print(f"Mean duration: {results['duration_mean']:.1f} months")
        print(f"Investment strategy: {results['avg_invest_pct']:.1%}")
    
    return result

# Usage
simulation = run_simulation("bologna_coppia", "bologna_coppia", 10)
```

#### Get Simulation Results

```python
def get_simulation_results(simulation_id: str):
    """Get detailed simulation results"""
    results = api_request("GET", f"/api/simulation/results/{simulation_id}")
    
    if results:
        print(f"Simulation: {simulation_id}")
        print(f"Episodes: {results['num_episodes']}")
        print(f"Mean wealth: €{results['total_wealth_mean']:.2f}")
        print(f"Strategy: Invest {results['avg_invest_pct']:.1%}, "
              f"Save {results['avg_save_pct']:.1%}, "
              f"Consume {results['avg_consume_pct']:.1%}")
    
    return results

# Usage
results = get_simulation_results("bologna_coppia_bologna_coppia_1730901234")
```

#### List Simulation History

```python
def list_simulations():
    """Get list of all past simulations"""
    history = api_request("GET", "/api/simulation/history")
    
    if history:
        print(f"Found {history['total']} simulations:")
        for sim in history['simulations']:
            print(f"  - {sim['simulation_id']}")
            print(f"    Model: {sim['model_name']}, Scenario: {sim['scenario_name']}")
            print(f"    Wealth: €{sim['total_wealth_mean']:.2f}")
    
    return history

# Usage
simulations = list_simulations()
```

### Models API

#### List Models

```python
def list_models():
    """Get list of all trained models"""
    models = api_request("GET", "/api/models")
    
    if models:
        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model['name']}")
            print(f"    Scenario: {model['scenario_name']}")
            print(f"    Episodes: {model['episodes']}")
            print(f"    Final reward: {model['final_reward']:.2f}")
    
    return models

# Usage
models = list_models()
```

#### Get Model Details

```python
def get_model(name: str):
    """Get detailed model information"""
    model = api_request("GET", f"/api/models/{name}")
    
    if model:
        print(f"Model: {model['name']}")
        print(f"Scenario: {model['scenario_name']}")
        print(f"Episodes: {model['episodes']}")
        print(f"Final metrics:")
        print(f"  Reward: {model['final_reward']:.2f}")
        print(f"  Duration: {model['final_duration']:.1f} months")
        print(f"  Cash: €{model['final_cash']:.2f}")
    
    return model

# Usage
model = get_model("bologna_coppia")
```

#### Delete Model

```python
def delete_model(name: str):
    """Delete a trained model"""
    result = api_request("DELETE", f"/api/models/{name}")
    
    if result:
        print(f"Deleted model: {result['name']}")
        print(f"Files removed: {result['files_deleted']}")
    
    return result

# Usage
delete_model("old_model")
```

### Reports API

#### Generate Report

```python
def generate_report(simulation_id: str, report_type: str = "html"):
    """Generate a report from simulation results"""
    data = {
        "simulation_id": simulation_id,
        "report_type": report_type,
        "include_sections": [
            "summary",
            "scenario",
            "training",
            "results",
            "strategy",
            "charts"
        ],
        "title": f"Financial Report - {simulation_id}"
    }
    
    result = api_request("POST", "/api/reports/generate", data)
    
    if result:
        print(f"Report generated: {result['report_id']}")
        print(f"Type: {result['report_type']}")
        print(f"Size: {result['file_size_kb']:.1f} KB")
        print(f"Download: http://localhost:8000/api/reports/{result['report_id']}")
    
    return result

# Usage
report = generate_report("bologna_coppia_bologna_coppia_1730901234", "html")
```

#### Download Report

```python
def download_report(report_id: str, output_file: str):
    """Download a generated report"""
    url = f"{BASE_URL}/api/reports/{report_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"Report downloaded: {output_file}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Download error: {e}")
        return False

# Usage
download_report("report_bologna_coppia_1730901234", "my_report.html")
```

#### List Reports

```python
def list_reports():
    """Get list of all generated reports"""
    reports = api_request("GET", "/api/reports/list")
    
    if reports:
        print(f"Found {reports['total']} reports:")
        for report in reports['reports']:
            print(f"  - {report['report_id']}")
            print(f"    Simulation: {report['simulation_id']}")
            print(f"    Type: {report['report_type']}")
            print(f"    Size: {report['file_size_kb']:.1f} KB")
    
    return reports

# Usage
reports = list_reports()
```


## JavaScript Examples

### Setup

```javascript
// Using fetch API
const BASE_URL = 'http://localhost:8000';

async function apiRequest(method, endpoint, data = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  if (data) {
    options.body = JSON.stringify(data);
  }
  
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, options);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API Error:', error);
    return null;
  }
}
```

### Scenarios API

```javascript
// List scenarios
async function listScenarios() {
  const scenarios = await apiRequest('GET', '/api/scenarios');
  console.log('Scenarios:', scenarios);
  return scenarios;
}

// Get scenario
async function getScenario(name) {
  const scenario = await apiRequest('GET', `/api/scenarios/${name}`);
  console.log('Scenario:', scenario);
  return scenario;
}

// Create scenario
async function createScenario(name, income, fixedExpenses) {
  const data = {
    name,
    description: `Custom scenario: ${name}`,
    environment: {
      income,
      fixed_expenses: fixedExpenses,
      variable_expense_mean: 700,
      variable_expense_std: 100,
      inflation: 0.02,
      safety_threshold: 1000,
      max_months: 60,
      initial_cash: 0,
      risk_tolerance: 0.5,
      investment_return_mean: 0.005,
      investment_return_std: 0.02,
      investment_return_type: 'stochastic'
    }
  };
  
  const result = await apiRequest('POST', '/api/scenarios', data);
  console.log('Created:', result);
  return result;
}

// Usage
listScenarios();
getScenario('bologna_coppia');
createScenario('my_scenario', 3000, 1500);
```

### Training API

```javascript
// Start training
async function startTraining(scenarioName, episodes = 1000) {
  const data = {
    scenario_name: scenarioName,
    num_episodes: episodes,
    save_interval: 100,
    eval_episodes: 10,
    seed: 42
  };
  
  const result = await apiRequest('POST', '/api/training/start', data);
  console.log('Training started:', result);
  return result;
}

// Get training status
async function getTrainingStatus() {
  const status = await apiRequest('GET', '/api/training/status');
  console.log('Training status:', status);
  return status;
}

// Monitor training with polling
async function monitorTraining() {
  const interval = setInterval(async () => {
    const status = await getTrainingStatus();
    
    if (!status || !status.is_training) {
      console.log('Training not active');
      clearInterval(interval);
      return;
    }
    
    const progress = status.latest_progress || {};
    console.log(`Episode ${progress.episode}/${status.total_episodes}`);
    console.log(`Reward: ${progress.avg_reward?.toFixed(2)}`);
    
    if (progress.episode >= status.total_episodes) {
      console.log('Training complete!');
      clearInterval(interval);
    }
  }, 5000); // Poll every 5 seconds
}

// Usage
startTraining('bologna_coppia', 1000);
monitorTraining();
```

### WebSocket Training Updates (JavaScript)

```javascript
import io from 'socket.io-client';

function monitorTrainingRealtime(scenarioName) {
  const socket = io('http://localhost:8000', { path: '/socket.io' });
  
  socket.on('connect', () => {
    console.log('Connected to training updates');
  });
  
  socket.on('training_started', (data) => {
    console.log('Training started:', data.scenario_name);
    console.log('Episodes:', data.num_episodes);
  });
  
  socket.on('training_progress', (data) => {
    console.log(`Episode ${data.episode}/${data.total_episodes}`);
    console.log(`Reward: ${data.avg_reward.toFixed(2)}`);
    console.log(`Stability: ${(data.stability * 100).toFixed(1)}%`);
  });
  
  socket.on('training_completed', (data) => {
    console.log('Training completed!');
    console.log('Final reward:', data.final_reward.toFixed(2));
    socket.disconnect();
  });
  
  socket.on('training_error', (data) => {
    console.error('Training error:', data.message);
    socket.disconnect();
  });
  
  // Start training
  startTraining(scenarioName);
}

// Usage
monitorTrainingRealtime('bologna_coppia');
```

### Simulation API

```javascript
// Run simulation
async function runSimulation(modelName, scenarioName, episodes = 10) {
  const data = {
    model_name: modelName,
    scenario_name: scenarioName,
    num_episodes: episodes,
    seed: 42
  };
  
  const result = await apiRequest('POST', '/api/simulation/run', data);
  
  if (result && result.status === 'completed') {
    const results = result.results;
    console.log('Simulation completed:', result.simulation_id);
    console.log('Mean wealth: €' + results.total_wealth_mean.toFixed(2));
    console.log('Investment: ' + (results.avg_invest_pct * 100).toFixed(1) + '%');
  }
  
  return result;
}

// Get simulation results
async function getSimulationResults(simulationId) {
  const results = await apiRequest('GET', `/api/simulation/results/${simulationId}`);
  console.log('Results:', results);
  return results;
}

// Usage
runSimulation('bologna_coppia', 'bologna_coppia', 10);
```

### Reports API

```javascript
// Generate report
async function generateReport(simulationId, reportType = 'html') {
  const data = {
    simulation_id: simulationId,
    report_type: reportType,
    include_sections: ['summary', 'scenario', 'results', 'strategy'],
    title: `Financial Report - ${simulationId}`
  };
  
  const result = await apiRequest('POST', '/api/reports/generate', data);
  
  if (result) {
    console.log('Report generated:', result.report_id);
    console.log('Download:', `http://localhost:8000/api/reports/${result.report_id}`);
  }
  
  return result;
}

// Download report
async function downloadReport(reportId, filename) {
  const url = `${BASE_URL}/api/reports/${reportId}`;
  
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    
    // Create download link
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    
    console.log('Report downloaded:', filename);
  } catch (error) {
    console.error('Download error:', error);
  }
}

// Usage
generateReport('bologna_coppia_bologna_coppia_1730901234', 'html');
downloadReport('report_bologna_coppia_1730901234', 'report.html');
```

## cURL Examples

### Scenarios API

```bash
# List scenarios
curl http://localhost:8000/api/scenarios

# Get scenario
curl http://localhost:8000/api/scenarios/bologna_coppia

# Create scenario
curl -X POST http://localhost:8000/api/scenarios \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_scenario",
    "description": "Custom scenario",
    "environment": {
      "income": 3000,
      "fixed_expenses": 1500,
      "variable_expense_mean": 700,
      "variable_expense_std": 100,
      "inflation": 0.02,
      "safety_threshold": 1000,
      "max_months": 60,
      "initial_cash": 0,
      "risk_tolerance": 0.5,
      "investment_return_mean": 0.005,
      "investment_return_std": 0.02,
      "investment_return_type": "stochastic"
    }
  }'

# Update scenario
curl -X PUT http://localhost:8000/api/scenarios/my_scenario \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_scenario",
    "environment": {
      "income": 3500,
      "risk_tolerance": 0.7
    }
  }'

# Delete scenario
curl -X DELETE http://localhost:8000/api/scenarios/my_scenario

# Get templates
curl http://localhost:8000/api/scenarios/templates
```

### Training API

```bash
# Start training
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_name": "bologna_coppia",
    "num_episodes": 1000,
    "save_interval": 100,
    "eval_episodes": 10,
    "seed": 42
  }'

# Get training status
curl http://localhost:8000/api/training/status

# Stop training
curl -X POST http://localhost:8000/api/training/stop
```

### Simulation API

```bash
# Run simulation
curl -X POST http://localhost:8000/api/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "bologna_coppia",
    "scenario_name": "bologna_coppia",
    "num_episodes": 10,
    "seed": 42
  }'

# Get simulation results
curl http://localhost:8000/api/simulation/results/bologna_coppia_bologna_coppia_1730901234

# List simulation history
curl http://localhost:8000/api/simulation/history
```

### Models API

```bash
# List models
curl http://localhost:8000/api/models

# Get model details
curl http://localhost:8000/api/models/bologna_coppia

# Delete model
curl -X DELETE http://localhost:8000/api/models/old_model
```

### Reports API

```bash
# Generate report
curl -X POST http://localhost:8000/api/reports/generate \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_id": "bologna_coppia_bologna_coppia_1730901234",
    "report_type": "html",
    "include_sections": ["summary", "scenario", "results", "strategy"],
    "title": "Financial Report"
  }'

# Download report
curl -O http://localhost:8000/api/reports/report_bologna_coppia_1730901234

# List reports
curl http://localhost:8000/api/reports/list
```

## Complete Workflows

### Workflow 1: Create Scenario and Train Model

```python
# 1. Create scenario
scenario = create_scenario("my_scenario", 3000, 1500)

# 2. Start training
training = start_training("my_scenario", 2000)

# 3. Monitor progress (WebSocket or polling)
monitor_training_realtime("my_scenario")

# 4. Check final model
model = get_model("my_scenario")
print(f"Final reward: {model['final_reward']:.2f}")
```

### Workflow 2: Run Simulation and Generate Report

```python
# 1. List available models
models = list_models()
model_name = models[0]['name']

# 2. Run simulation
simulation = run_simulation(model_name, "bologna_coppia", 20)
simulation_id = simulation['simulation_id']

# 3. Get detailed results
results = get_simulation_results(simulation_id)

# 4. Generate report
report = generate_report(simulation_id, "html")

# 5. Download report
download_report(report['report_id'], "financial_report.html")
```

### Workflow 3: Compare Multiple Scenarios

```python
# 1. Create multiple scenarios
scenarios = [
    ("conservative", 3000, 1500, 0.3),
    ("balanced", 3000, 1500, 0.5),
    ("aggressive", 3000, 1500, 0.8)
]

for name, income, expenses, risk in scenarios:
    create_scenario(name, income, expenses)
    # Update risk tolerance
    update_scenario(name, {"risk_tolerance": risk})

# 2. Train models for each
for name, _, _, _ in scenarios:
    start_training(name, 2000)
    # Wait for completion...

# 3. Run simulations
simulation_ids = []
for name, _, _, _ in scenarios:
    sim = run_simulation(name, name, 20)
    simulation_ids.append(sim['simulation_id'])

# 4. Compare results
for sim_id in simulation_ids:
    results = get_simulation_results(sim_id)
    print(f"{sim_id}: €{results['total_wealth_mean']:.2f}")
```

### Workflow 4: Batch Processing

```python
import concurrent.futures

def process_scenario(scenario_name):
    """Train and simulate a scenario"""
    # Train
    start_training(scenario_name, 1000)
    # Wait for completion (implement proper waiting)
    time.sleep(600)  # 10 minutes
    
    # Simulate
    simulation = run_simulation(scenario_name, scenario_name, 10)
    
    # Generate report
    report = generate_report(simulation['simulation_id'], "html")
    
    return {
        'scenario': scenario_name,
        'simulation_id': simulation['simulation_id'],
        'report_id': report['report_id']
    }

# Process multiple scenarios in parallel
scenarios = ["scenario1", "scenario2", "scenario3"]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_scenario, scenarios))

for result in results:
    print(f"Completed: {result['scenario']}")
    print(f"  Simulation: {result['simulation_id']}")
    print(f"  Report: {result['report_id']}")
```

## Error Handling

### Python Error Handling

```python
import requests
from typing import Optional, Dict, Any

class APIError(Exception):
    """Custom API error"""
    pass

def safe_api_request(method: str, endpoint: str, data: Dict = None) -> Optional[Dict[str, Any]]:
    """API request with comprehensive error handling"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        elif method == "PUT":
            response = requests.put(url, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse JSON
        return response.json()
        
    except requests.exceptions.Timeout:
        print(f"Request timeout for {endpoint}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Connection error - is backend running?")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error {e.response.status_code}: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
    except ValueError as e:
        print(f"JSON decode error: {e}")
        return None

# Usage with retry logic
def api_request_with_retry(method: str, endpoint: str, data: Dict = None, max_retries: int = 3):
    """API request with automatic retry"""
    for attempt in range(max_retries):
        result = safe_api_request(method, endpoint, data)
        if result is not None:
            return result
        
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retry {attempt + 1}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
    
    raise APIError(f"Failed after {max_retries} attempts")
```

### JavaScript Error Handling

```javascript
class APIError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

async function safeApiRequest(method, endpoint, data = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: 30000,
  };
  
  if (data) {
    options.body = JSON.stringify(data);
  }
  
  try {
    const response = await fetch(`${BASE_URL}${endpoint}`, options);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new APIError(
        `HTTP ${response.status}: ${errorText}`,
        response.status
      );
    }
    
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    
    if (error.name === 'AbortError') {
      throw new APIError('Request timeout', 408);
    }
    
    throw new APIError(`Network error: ${error.message}`, 0);
  }
}

// Usage with retry
async function apiRequestWithRetry(method, endpoint, data = null, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await safeApiRequest(method, endpoint, data);
    } catch (error) {
      if (attempt === maxRetries - 1) {
        throw error;
      }
      
      const waitTime = Math.pow(2, attempt) * 1000;
      console.log(`Retry ${attempt + 1}/${maxRetries} in ${waitTime}ms...`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
  }
}
```

## Best Practices

### 1. Always Check Response Status

```python
result = api_request("GET", "/api/scenarios")
if result is None:
    print("Request failed")
    return

# Process result
for scenario in result:
    print(scenario['name'])
```

### 2. Use Timeouts

```python
response = requests.get(url, timeout=30)  # 30 second timeout
```

### 3. Handle Errors Gracefully

```python
try:
    result = api_request("POST", "/api/training/start", data)
except APIError as e:
    print(f"Training failed: {e}")
    # Fallback or retry logic
```

### 4. Validate Data Before Sending

```python
def validate_scenario(data):
    """Validate scenario data before sending"""
    required_fields = ['name', 'environment']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    env = data['environment']
    if env['income'] <= 0:
        raise ValueError("Income must be positive")
    if env['risk_tolerance'] < 0 or env['risk_tolerance'] > 1:
        raise ValueError("Risk tolerance must be between 0 and 1")
    
    return True

# Usage
if validate_scenario(scenario_data):
    create_scenario(**scenario_data)
```

### 5. Use Async for Multiple Requests

```python
import asyncio
import aiohttp

async def fetch_scenarios():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/scenarios") as response:
            return await response.json()

async def fetch_models():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/api/models") as response:
            return await response.json()

# Fetch both concurrently
scenarios, models = await asyncio.gather(
    fetch_scenarios(),
    fetch_models()
)
```

### 6. Log API Calls

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def api_request(method, endpoint, data=None):
    logger.info(f"{method} {endpoint}")
    if data:
        logger.debug(f"Data: {json.dumps(data, indent=2)}")
    
    result = # ... make request
    
    logger.info(f"Response: {result.status_code}")
    return result
```

### 7. Cache Responses

```python
from functools import lru_cache
import time

@lru_cache(maxsize=128)
def get_scenario_cached(name, cache_time):
    """Cache scenario for 5 minutes"""
    return get_scenario(name)

# Usage - cache_time changes every 5 minutes
cache_key = int(time.time() / 300)
scenario = get_scenario_cached("bologna_coppia", cache_key)
```

### 8. Use Environment Variables

```python
import os

BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))
```

### 9. Implement Rate Limiting

```python
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            now = time.time()
            
            # Remove old calls
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            # Check limit
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                time.sleep(sleep_time)
            
            self.calls.append(time.time())
            return func(*args, **kwargs)
        
        return wrapper

# Usage: max 10 calls per minute
@RateLimiter(max_calls=10, period=60)
def api_request(method, endpoint, data=None):
    # ... make request
    pass
```

### 10. Document Your API Usage

```python
def create_scenario(name: str, income: float, fixed_expenses: float) -> Dict[str, Any]:
    """
    Create a new financial scenario.
    
    Args:
        name: Unique scenario name
        income: Monthly income in EUR (must be positive)
        fixed_expenses: Fixed monthly expenses in EUR (must be non-negative)
    
    Returns:
        Dict containing scenario details and creation status
    
    Raises:
        APIError: If request fails or validation errors occur
        ValueError: If parameters are invalid
    
    Example:
        >>> scenario = create_scenario("my_scenario", 3000, 1500)
        >>> print(scenario['name'])
        'my_scenario'
    """
    # Implementation...
```

---

**Last Updated**: November 2024
**Version**: 1.0
**For**: HRL Finance System API
