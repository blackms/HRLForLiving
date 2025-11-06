# Training API Documentation

## Overview

The Training API provides endpoints for training HRL models on financial scenarios with real-time progress updates via WebSocket.

## Endpoints

### POST /api/training/start

Start training a model on a scenario.

**Request Body:**
```json
{
  "scenario_name": "bologna_coppia",
  "num_episodes": 1000,
  "save_interval": 100,
  "eval_episodes": 10,
  "seed": 42
}
```

**Response (202 Accepted):**
```json
{
  "message": "Training started successfully",
  "data": {
    "status": "started",
    "scenario_name": "bologna_coppia",
    "num_episodes": 1000,
    "start_time": "2025-11-06T10:30:00"
  }
}
```

**Error Responses:**
- `400 Bad Request`: Training already in progress
- `404 Not Found`: Scenario not found
- `500 Internal Server Error`: Training failed to start

### POST /api/training/stop

Stop the current training process gracefully.

**Response (200 OK):**
```json
{
  "message": "Training stopped successfully",
  "data": {
    "status": "stopped",
    "scenario_name": "bologna_coppia",
    "episodes_completed": 450,
    "total_episodes": 1000
  }
}
```

**Error Responses:**
- `400 Bad Request`: No training in progress
- `500 Internal Server Error`: Failed to stop training

### GET /api/training/status

Get current training status.

**Response (200 OK):**
```json
{
  "is_training": true,
  "scenario_name": "bologna_coppia",
  "current_episode": 450,
  "total_episodes": 1000,
  "start_time": "2025-11-06T10:30:00",
  "latest_progress": {
    "episode": 450,
    "total_episodes": 1000,
    "avg_reward": 168.5,
    "avg_duration": 118.2,
    "avg_cash": 5234.67,
    "avg_invested": 12500.00,
    "stability": 0.985,
    "goal_adherence": 0.0234,
    "elapsed_time": 323.45
  }
}
```

## WebSocket Events

Connect to WebSocket at: `ws://localhost:8000/socket.io`

### Client Events

#### connect
Emitted automatically when client connects.

#### subscribe_training
Subscribe to training updates.

```javascript
socket.emit('subscribe_training', {});
```

### Server Events

#### connection_established
Sent when client successfully connects.

```json
{
  "message": "Connected to training updates",
  "sid": "session_id"
}
```

#### training_started
Sent when training begins.

```json
{
  "scenario_name": "bologna_coppia",
  "num_episodes": 1000,
  "start_time": "2025-11-06T10:30:00"
}
```

#### training_progress
Sent periodically during training (every episode).

```json
{
  "episode": 450,
  "total_episodes": 1000,
  "avg_reward": 168.5,
  "avg_duration": 118.2,
  "avg_cash": 5234.67,
  "avg_invested": 12500.00,
  "stability": 0.985,
  "goal_adherence": 0.0234,
  "elapsed_time": 323.45
}
```

#### training_completed
Sent when training finishes successfully.

```json
{
  "scenario_name": "bologna_coppia",
  "episodes_completed": 1000,
  "final_metrics": {
    "avg_reward": 175.3,
    "avg_duration": 120.5,
    "stability": 0.992
  }
}
```

#### training_stopped
Sent when training is stopped manually.

```json
{
  "scenario_name": "bologna_coppia",
  "episodes_completed": 450
}
```

#### training_error
Sent when an error occurs during training.

```json
{
  "message": "Training failed",
  "details": "Error details here"
}
```

## Usage Example

### Python Client

```python
import requests
import socketio

# Connect to WebSocket
sio = socketio.Client()

@sio.on('training_progress')
def on_progress(data):
    print(f"Episode {data['episode']}/{data['total_episodes']}")
    print(f"Reward: {data['avg_reward']:.2f}")

@sio.on('training_completed')
def on_complete(data):
    print(f"Training completed: {data['scenario_name']}")

sio.connect('http://localhost:8000', socketio_path='/socket.io')

# Start training
response = requests.post('http://localhost:8000/api/training/start', json={
    'scenario_name': 'bologna_coppia',
    'num_episodes': 1000,
    'save_interval': 100,
    'eval_episodes': 10
})

print(response.json())

# Wait for training to complete
sio.wait()
```

### JavaScript Client

```javascript
import io from 'socket.io-client';
import axios from 'axios';

// Connect to WebSocket
const socket = io('http://localhost:8000', {
  path: '/socket.io'
});

socket.on('connect', () => {
  console.log('Connected to training updates');
  socket.emit('subscribe_training', {});
});

socket.on('training_progress', (data) => {
  console.log(`Episode ${data.episode}/${data.total_episodes}`);
  console.log(`Reward: ${data.avg_reward.toFixed(2)}`);
});

socket.on('training_completed', (data) => {
  console.log(`Training completed: ${data.scenario_name}`);
});

// Start training
axios.post('http://localhost:8000/api/training/start', {
  scenario_name: 'bologna_coppia',
  num_episodes: 1000,
  save_interval: 100,
  eval_episodes: 10
}).then(response => {
  console.log(response.data);
});
```

## Training Process

1. **Initialization**: Load scenario configuration and create environment
2. **Agent Creation**: Initialize high-level (Strategist) and low-level (Executor) agents
3. **Training Loop**: Execute episodes with hierarchical learning
4. **Progress Updates**: Send real-time updates via WebSocket every episode
5. **Checkpointing**: Save model checkpoints at specified intervals
6. **Completion**: Save final model and training history

## Model Storage

Trained models are saved in the `models/` directory:

- `{scenario_name}_high_agent.pt`: High-level agent model
- `{scenario_name}_low_agent.pt`: Low-level agent model
- `{scenario_name}_history.json`: Training history and metrics

Checkpoints are saved in `models/checkpoints/{scenario_name}/`:

- `checkpoint_episode_{N}/`: Periodic checkpoints
- `checkpoint_best/`: Best performing model
- `checkpoint_final/`: Final model after training

## Performance Considerations

- Training runs asynchronously in the background
- Only one training session can run at a time
- WebSocket updates are sent every episode (can be throttled if needed)
- Large models may take significant time to train (1000 episodes ≈ 5-10 minutes)

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful operation
- `202 Accepted`: Training started (async operation)
- `400 Bad Request`: Invalid request or state
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

WebSocket errors are emitted via the `training_error` event.
