# Quick Start: Training API

## Installation

```bash
# Install dependencies
pip install -r backend/requirements.txt
```

## Start Server

```bash
# From project root
uvicorn backend.main:socket_app --reload --port 8000
```

## Test the API

### 1. Check Health

```bash
curl http://localhost:8000/health
```

### 2. List Available Scenarios

```bash
curl http://localhost:8000/api/scenarios
```

### 3. Start Training

```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_name": "bologna_coppia",
    "num_episodes": 100,
    "save_interval": 50,
    "eval_episodes": 10
  }'
```

### 4. Check Training Status

```bash
curl http://localhost:8000/api/training/status
```

### 5. Stop Training

```bash
curl -X POST http://localhost:8000/api/training/stop
```

## WebSocket Client (Python)

```python
import socketio
import requests
import time

# Create Socket.IO client
sio = socketio.Client()

@sio.on('connect')
def on_connect():
    print('Connected to server')
    sio.emit('subscribe_training', {})

@sio.on('training_started')
def on_started(data):
    print(f"Training started: {data['scenario_name']}")

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

@sio.on('training_error')
def on_error(data):
    print(f"Error: {data['message']}")

# Connect to server
sio.connect('http://localhost:8000', socketio_path='/socket.io')

# Start training
response = requests.post('http://localhost:8000/api/training/start', json={
    'scenario_name': 'bologna_coppia',
    'num_episodes': 100,
    'save_interval': 50,
    'eval_episodes': 10
})

print(response.json())

# Wait for training to complete
sio.wait()
```

## WebSocket Client (JavaScript)

```javascript
import io from 'socket.io-client';
import axios from 'axios';

// Connect to WebSocket
const socket = io('http://localhost:8000', {
  path: '/socket.io'
});

socket.on('connect', () => {
  console.log('Connected to server');
  socket.emit('subscribe_training', {});
});

socket.on('training_started', (data) => {
  console.log(`Training started: ${data.scenario_name}`);
});

socket.on('training_progress', (data) => {
  console.log(`Episode ${data.episode}/${data.total_episodes}`);
  console.log(`  Reward: ${data.avg_reward.toFixed(2)}`);
  console.log(`  Duration: ${data.avg_duration.toFixed(1)} months`);
  console.log(`  Stability: ${(data.stability * 100).toFixed(1)}%`);
});

socket.on('training_completed', (data) => {
  console.log(`Training completed: ${data.scenario_name}`);
});

socket.on('training_error', (data) => {
  console.error(`Error: ${data.message}`);
});

// Start training
axios.post('http://localhost:8000/api/training/start', {
  scenario_name: 'bologna_coppia',
  num_episodes: 100,
  save_interval: 50,
  eval_episodes: 10
}).then(response => {
  console.log(response.data);
}).catch(error => {
  console.error('Failed to start training:', error.response?.data);
});
```

## Expected Output

### Training Progress
```
Episode 10/100
  Reward: 145.32
  Duration: 115.4 months
  Stability: 94.5%

Episode 20/100
  Reward: 152.18
  Duration: 117.2 months
  Stability: 96.2%

...

Episode 100/100
  Reward: 168.45
  Duration: 118.8 months
  Stability: 98.1%
```

### Saved Models
After training completes, models are saved to:
- `models/bologna_coppia_high_agent.pt`
- `models/bologna_coppia_low_agent.pt`
- `models/bologna_coppia_history.json`

### Checkpoints
Checkpoints saved to:
- `models/checkpoints/bologna_coppia/checkpoint_episode_50/`
- `models/checkpoints/bologna_coppia/checkpoint_episode_100/`

## Troubleshooting

### "Training already in progress"
Only one training session can run at a time. Stop the current training first:
```bash
curl -X POST http://localhost:8000/api/training/stop
```

### "Scenario not found"
Make sure the scenario exists:
```bash
curl http://localhost:8000/api/scenarios
```

### WebSocket connection fails
Check that the server is running and accessible:
```bash
curl http://localhost:8000/health
```

### Import errors
Install all dependencies:
```bash
pip install -r backend/requirements.txt
```

## API Documentation

For complete API documentation, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Training API: [TRAINING_API.md](api/TRAINING_API.md)

## Next Steps

1. **Create a scenario** using POST /api/scenarios
2. **Start training** on your scenario
3. **Monitor progress** via WebSocket
4. **Use trained models** for simulation (coming soon)
5. **Generate reports** from results (coming soon)
