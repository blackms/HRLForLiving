"""WebSocket handlers for real-time training updates"""
import socketio
from typing import Dict, Any

# Create Socket.IO server instance
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',  # Configure appropriately for production
    logger=True,
    engineio_logger=False
)


class TrainingSocketManager:
    """Manager for training WebSocket connections and events"""
    
    def __init__(self, sio_server: socketio.AsyncServer):
        """
        Initialize the training socket manager
        
        Args:
            sio_server: Socket.IO server instance
        """
        self.sio = sio_server
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up Socket.IO event handlers"""
        
        @self.sio.event
        async def connect(sid, environ):
            """
            Handle client connection
            
            Args:
                sid: Session ID
                environ: WSGI environment dict
            """
            print(f"Client connected: {sid}")
            await self.sio.emit('connection_established', {
                'message': 'Connected to training updates',
                'sid': sid
            }, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """
            Handle client disconnection
            
            Args:
                sid: Session ID
            """
            print(f"Client disconnected: {sid}")
        
        @self.sio.event
        async def subscribe_training(sid, data):
            """
            Handle subscription to training updates
            
            Args:
                sid: Session ID
                data: Subscription data (optional scenario filter)
            """
            print(f"Client {sid} subscribed to training updates")
            await self.sio.emit('subscription_confirmed', {
                'message': 'Subscribed to training updates'
            }, room=sid)
    
    async def emit_progress(self, progress: Dict[str, Any]):
        """
        Emit training progress update to all connected clients
        
        Args:
            progress: Progress data dictionary containing:
                - episode: Current episode number
                - total_episodes: Total episodes
                - avg_reward: Average reward
                - avg_duration: Average duration
                - avg_cash: Average cash balance
                - avg_invested: Average invested amount
                - stability: Stability metric
                - goal_adherence: Goal adherence metric
                - elapsed_time: Elapsed time in seconds
        """
        await self.sio.emit('training_progress', progress)
        print(f"Emitted progress: Episode {progress['episode']}/{progress['total_episodes']}")
    
    async def emit_training_started(self, data: Dict[str, Any]):
        """
        Emit training started event
        
        Args:
            data: Training start data containing:
                - scenario_name: Name of scenario
                - num_episodes: Total episodes
                - start_time: Start timestamp
        """
        await self.sio.emit('training_started', data)
        print(f"Emitted training started: {data['scenario_name']}")
    
    async def emit_training_completed(self, data: Dict[str, Any]):
        """
        Emit training completed event
        
        Args:
            data: Training completion data containing:
                - scenario_name: Name of scenario
                - episodes_completed: Number of episodes completed
                - final_metrics: Final performance metrics
        """
        await self.sio.emit('training_completed', data)
        print(f"Emitted training completed: {data['scenario_name']}")
    
    async def emit_training_stopped(self, data: Dict[str, Any]):
        """
        Emit training stopped event
        
        Args:
            data: Training stop data containing:
                - scenario_name: Name of scenario
                - episodes_completed: Number of episodes completed
        """
        await self.sio.emit('training_stopped', data)
        print(f"Emitted training stopped: {data['scenario_name']}")
    
    async def emit_training_error(self, error: Dict[str, Any]):
        """
        Emit training error event
        
        Args:
            error: Error data containing:
                - message: Error message
                - details: Additional error details
        """
        await self.sio.emit('training_error', error)
        print(f"Emitted training error: {error['message']}")


# Global socket manager instance
socket_manager = TrainingSocketManager(sio)
