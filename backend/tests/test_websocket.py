"""Tests for WebSocket communication"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import socketio


class TestTrainingSocketManager:
    """Test TrainingSocketManager class"""
    
    @pytest.fixture
    def mock_sio(self):
        """Create a mock Socket.IO server"""
        mock = AsyncMock(spec=socketio.AsyncServer)
        mock.emit = AsyncMock()
        mock.event = MagicMock()
        return mock
    
    @pytest.fixture
    def socket_manager(self, mock_sio):
        """Create a TrainingSocketManager with mock server"""
        from backend.websocket.training_socket import TrainingSocketManager
        return TrainingSocketManager(mock_sio)
    
    @pytest.mark.asyncio
    async def test_emit_progress(self, socket_manager, mock_sio):
        """Test emitting training progress updates"""
        progress = {
            'episode': 10,
            'total_episodes': 100,
            'avg_reward': 150.5,
            'avg_duration': 120.3,
            'avg_cash': 5000.0,
            'avg_invested': 10000.0,
            'stability': 0.95,
            'goal_adherence': 0.88,
            'elapsed_time': 300.5
        }
        
        await socket_manager.emit_progress(progress)
        
        # Verify emit was called with correct event and data
        mock_sio.emit.assert_called_once_with('training_progress', progress)
    
    @pytest.mark.asyncio
    async def test_emit_training_started(self, socket_manager, mock_sio):
        """Test emitting training started event"""
        data = {
            'scenario_name': 'test_scenario',
            'num_episodes': 100,
            'start_time': '2024-01-01T00:00:00'
        }
        
        await socket_manager.emit_training_started(data)
        
        # Verify emit was called with correct event and data
        mock_sio.emit.assert_called_once_with('training_started', data)
    
    @pytest.mark.asyncio
    async def test_emit_training_completed(self, socket_manager, mock_sio):
        """Test emitting training completed event"""
        data = {
            'scenario_name': 'test_scenario',
            'episodes_completed': 100,
            'final_metrics': {
                'avg_reward': 200.0,
                'avg_duration': 115.5,
                'stability': 0.98
            }
        }
        
        await socket_manager.emit_training_completed(data)
        
        # Verify emit was called with correct event and data
        mock_sio.emit.assert_called_once_with('training_completed', data)
    
    @pytest.mark.asyncio
    async def test_emit_training_stopped(self, socket_manager, mock_sio):
        """Test emitting training stopped event"""
        data = {
            'scenario_name': 'test_scenario',
            'episodes_completed': 50
        }
        
        await socket_manager.emit_training_stopped(data)
        
        # Verify emit was called with correct event and data
        mock_sio.emit.assert_called_once_with('training_stopped', data)
    
    @pytest.mark.asyncio
    async def test_emit_training_error(self, socket_manager, mock_sio):
        """Test emitting training error event"""
        error = {
            'message': 'Training failed',
            'details': 'Out of memory'
        }
        
        await socket_manager.emit_training_error(error)
        
        # Verify emit was called with correct event and data
        mock_sio.emit.assert_called_once_with('training_error', error)
    
    @pytest.mark.asyncio
    async def test_multiple_progress_updates(self, socket_manager, mock_sio):
        """Test emitting multiple progress updates"""
        progress_updates = [
            {
                'episode': i,
                'total_episodes': 100,
                'avg_reward': 100.0 + i,
                'avg_duration': 120.0,
                'avg_cash': 5000.0,
                'avg_invested': 10000.0,
                'stability': 0.9,
                'goal_adherence': 0.85,
                'elapsed_time': i * 10.0
            }
            for i in range(1, 6)
        ]
        
        for progress in progress_updates:
            await socket_manager.emit_progress(progress)
        
        # Verify emit was called 5 times
        assert mock_sio.emit.call_count == 5
        
        # Verify each call had correct event name
        for call in mock_sio.emit.call_args_list:
            assert call[0][0] == 'training_progress'


class TestWebSocketIntegration:
    """Test WebSocket integration with training service"""
    
    @pytest.mark.asyncio
    async def test_training_service_progress_callback(self):
        """Test that training service calls progress callback"""
        from backend.services.training_service import TrainingService
        
        # Create training service
        service = TrainingService()
        
        # Create mock callback
        mock_callback = AsyncMock()
        service.set_progress_callback(mock_callback)
        
        # Verify callback was set
        assert service._progress_callback is not None
        
        # Test callback invocation
        test_progress = {
            'episode': 1,
            'total_episodes': 10,
            'avg_reward': 100.0,
            'avg_duration': 120.0,
            'avg_cash': 5000.0,
            'avg_invested': 10000.0,
            'stability': 0.9,
            'goal_adherence': 0.85,
            'elapsed_time': 10.0
        }
        
        await service._progress_callback(test_progress)
        mock_callback.assert_called_once_with(test_progress)
    
    @pytest.mark.asyncio
    async def test_websocket_events_during_training_lifecycle(self):
        """Test WebSocket events are emitted during training lifecycle"""
        from backend.websocket.training_socket import TrainingSocketManager
        
        # Create mock Socket.IO server
        mock_sio = AsyncMock(spec=socketio.AsyncServer)
        mock_sio.emit = AsyncMock()
        
        # Create socket manager
        manager = TrainingSocketManager(mock_sio)
        
        # Simulate training lifecycle
        # 1. Training started
        await manager.emit_training_started({
            'scenario_name': 'test_scenario',
            'num_episodes': 10,
            'start_time': '2024-01-01T00:00:00'
        })
        
        # 2. Progress updates
        for episode in range(1, 11):
            await manager.emit_progress({
                'episode': episode,
                'total_episodes': 10,
                'avg_reward': 100.0 + episode,
                'avg_duration': 120.0,
                'avg_cash': 5000.0,
                'avg_invested': 10000.0,
                'stability': 0.9,
                'goal_adherence': 0.85,
                'elapsed_time': episode * 10.0
            })
        
        # 3. Training completed
        await manager.emit_training_completed({
            'scenario_name': 'test_scenario',
            'episodes_completed': 10,
            'final_metrics': {
                'avg_reward': 110.0,
                'avg_duration': 120.0,
                'stability': 0.9
            }
        })
        
        # Verify all events were emitted
        # 1 started + 10 progress + 1 completed = 12 total
        assert mock_sio.emit.call_count == 12
        
        # Verify event types
        event_types = [call[0][0] for call in mock_sio.emit.call_args_list]
        assert event_types[0] == 'training_started'
        assert all(event_types[i] == 'training_progress' for i in range(1, 11))
        assert event_types[11] == 'training_completed'
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self):
        """Test WebSocket error event emission"""
        from backend.websocket.training_socket import TrainingSocketManager
        
        # Create mock Socket.IO server
        mock_sio = AsyncMock(spec=socketio.AsyncServer)
        mock_sio.emit = AsyncMock()
        
        # Create socket manager
        manager = TrainingSocketManager(mock_sio)
        
        # Simulate training error
        await manager.emit_training_error({
            'message': 'Training failed due to invalid configuration',
            'details': 'Scenario not found'
        })
        
        # Verify error event was emitted
        mock_sio.emit.assert_called_once()
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'training_error'
        assert 'message' in call_args[0][1]
        assert 'details' in call_args[0][1]
    
    @pytest.mark.asyncio
    async def test_websocket_stopped_event(self):
        """Test WebSocket stopped event emission"""
        from backend.websocket.training_socket import TrainingSocketManager
        
        # Create mock Socket.IO server
        mock_sio = AsyncMock(spec=socketio.AsyncServer)
        mock_sio.emit = AsyncMock()
        
        # Create socket manager
        manager = TrainingSocketManager(mock_sio)
        
        # Simulate training lifecycle with early stop
        # 1. Training started
        await manager.emit_training_started({
            'scenario_name': 'test_scenario',
            'num_episodes': 100,
            'start_time': '2024-01-01T00:00:00'
        })
        
        # 2. Some progress updates
        for episode in range(1, 6):
            await manager.emit_progress({
                'episode': episode,
                'total_episodes': 100,
                'avg_reward': 100.0,
                'avg_duration': 120.0,
                'avg_cash': 5000.0,
                'avg_invested': 10000.0,
                'stability': 0.9,
                'goal_adherence': 0.85,
                'elapsed_time': episode * 10.0
            })
        
        # 3. Training stopped early
        await manager.emit_training_stopped({
            'scenario_name': 'test_scenario',
            'episodes_completed': 5
        })
        
        # Verify events were emitted
        # 1 started + 5 progress + 1 stopped = 7 total
        assert mock_sio.emit.call_count == 7
        
        # Verify last event was stopped
        last_call = mock_sio.emit.call_args_list[-1]
        assert last_call[0][0] == 'training_stopped'
        assert last_call[0][1]['episodes_completed'] == 5


class TestWebSocketConnectionHandlers:
    """Test WebSocket connection event handlers"""
    
    @pytest.mark.asyncio
    async def test_connection_handler_setup(self):
        """Test that connection handlers are properly set up"""
        from backend.websocket.training_socket import TrainingSocketManager
        
        # Create mock Socket.IO server
        mock_sio = AsyncMock(spec=socketio.AsyncServer)
        mock_sio.event = MagicMock(side_effect=lambda func: func)
        mock_sio.emit = AsyncMock()
        
        # Create socket manager (this should set up handlers)
        manager = TrainingSocketManager(mock_sio)
        
        # Verify event decorator was called for handlers
        # The _setup_handlers method should register connect, disconnect, subscribe_training
        assert mock_sio.event.call_count >= 3
    
    @pytest.mark.asyncio
    async def test_progress_data_structure(self):
        """Test that progress data has correct structure"""
        from backend.websocket.training_socket import TrainingSocketManager
        
        # Create mock Socket.IO server
        mock_sio = AsyncMock(spec=socketio.AsyncServer)
        mock_sio.emit = AsyncMock()
        
        # Create socket manager
        manager = TrainingSocketManager(mock_sio)
        
        # Define expected progress structure
        progress = {
            'episode': 10,
            'total_episodes': 100,
            'avg_reward': 150.5,
            'avg_duration': 120.3,
            'avg_cash': 5000.0,
            'avg_invested': 10000.0,
            'stability': 0.95,
            'goal_adherence': 0.88,
            'elapsed_time': 300.5
        }
        
        await manager.emit_progress(progress)
        
        # Verify data structure
        call_args = mock_sio.emit.call_args
        emitted_data = call_args[0][1]
        
        # Check all required fields are present
        assert 'episode' in emitted_data
        assert 'total_episodes' in emitted_data
        assert 'avg_reward' in emitted_data
        assert 'avg_duration' in emitted_data
        assert 'avg_cash' in emitted_data
        assert 'avg_invested' in emitted_data
        assert 'stability' in emitted_data
        assert 'goal_adherence' in emitted_data
        assert 'elapsed_time' in emitted_data
        
        # Check data types
        assert isinstance(emitted_data['episode'], int)
        assert isinstance(emitted_data['total_episodes'], int)
        assert isinstance(emitted_data['avg_reward'], (int, float))
        assert isinstance(emitted_data['avg_duration'], (int, float))
        assert isinstance(emitted_data['stability'], (int, float))
        assert isinstance(emitted_data['goal_adherence'], (int, float))
        assert isinstance(emitted_data['elapsed_time'], (int, float))


class TestWebSocketEventPayloads:
    """Test WebSocket event payload structures"""
    
    @pytest.mark.asyncio
    async def test_training_started_payload(self):
        """Test training_started event payload structure"""
        from backend.websocket.training_socket import TrainingSocketManager
        
        mock_sio = AsyncMock(spec=socketio.AsyncServer)
        mock_sio.emit = AsyncMock()
        manager = TrainingSocketManager(mock_sio)
        
        payload = {
            'scenario_name': 'test_scenario',
            'num_episodes': 100,
            'start_time': '2024-01-01T00:00:00'
        }
        
        await manager.emit_training_started(payload)
        
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'training_started'
        emitted_data = call_args[0][1]
        
        assert 'scenario_name' in emitted_data
        assert 'num_episodes' in emitted_data
        assert 'start_time' in emitted_data
        assert isinstance(emitted_data['scenario_name'], str)
        assert isinstance(emitted_data['num_episodes'], int)
    
    @pytest.mark.asyncio
    async def test_training_completed_payload(self):
        """Test training_completed event payload structure"""
        from backend.websocket.training_socket import TrainingSocketManager
        
        mock_sio = AsyncMock(spec=socketio.AsyncServer)
        mock_sio.emit = AsyncMock()
        manager = TrainingSocketManager(mock_sio)
        
        payload = {
            'scenario_name': 'test_scenario',
            'episodes_completed': 100,
            'final_metrics': {
                'avg_reward': 200.0,
                'avg_duration': 115.5,
                'stability': 0.98
            }
        }
        
        await manager.emit_training_completed(payload)
        
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'training_completed'
        emitted_data = call_args[0][1]
        
        assert 'scenario_name' in emitted_data
        assert 'episodes_completed' in emitted_data
        assert 'final_metrics' in emitted_data
        assert isinstance(emitted_data['final_metrics'], dict)
    
    @pytest.mark.asyncio
    async def test_training_error_payload(self):
        """Test training_error event payload structure"""
        from backend.websocket.training_socket import TrainingSocketManager
        
        mock_sio = AsyncMock(spec=socketio.AsyncServer)
        mock_sio.emit = AsyncMock()
        manager = TrainingSocketManager(mock_sio)
        
        payload = {
            'message': 'Training failed',
            'details': 'Configuration error'
        }
        
        await manager.emit_training_error(payload)
        
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'training_error'
        emitted_data = call_args[0][1]
        
        assert 'message' in emitted_data
        assert 'details' in emitted_data
        assert isinstance(emitted_data['message'], str)
