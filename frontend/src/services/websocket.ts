import { io, Socket } from 'socket.io-client';
import type { TrainingProgress } from '../types';

// WebSocket event types
export type TrainingEvent = 'started' | 'progress' | 'completed' | 'stopped' | 'error';

export interface TrainingEventData {
  event: TrainingEvent;
  scenario_name?: string;
  progress?: TrainingProgress;
  error?: string;
  message?: string;
}

// Callback types
type TrainingEventCallback = (data: TrainingEventData) => void;
type ConnectionCallback = () => void;
type ErrorCallback = (error: Error) => void;

class WebSocketClient {
  private socket: Socket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isManualDisconnect = false;

  // Event listeners
  private eventListeners: Map<string, Set<TrainingEventCallback>> = new Map();
  private connectionListeners: Set<ConnectionCallback> = new Set();
  private disconnectionListeners: Set<ConnectionCallback> = new Set();
  private errorListeners: Set<ErrorCallback> = new Set();

  constructor(url: string = 'http://localhost:8000') {
    this.url = url;
  }

  /**
   * Connect to the WebSocket server
   */
  connect(): void {
    if (this.socket?.connected) {
      console.log('WebSocket already connected');
      return;
    }

    this.isManualDisconnect = false;

    this.socket = io(this.url, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
      reconnectionDelayMax: 5000,
      timeout: 10000,
    });

    this.setupEventHandlers();
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    if (!this.socket) {
      return;
    }

    this.isManualDisconnect = true;
    this.socket.disconnect();
    this.socket = null;
    this.reconnectAttempts = 0;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected || false;
  }

  /**
   * Setup event handlers for Socket.IO events
   */
  private setupEventHandlers(): void {
    if (!this.socket) return;

    // Connection events
    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.connectionListeners.forEach(callback => callback());
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.disconnectionListeners.forEach(callback => callback());

      // Auto-reconnect if not manual disconnect
      if (!this.isManualDisconnect && reason === 'io server disconnect') {
        this.attemptReconnect();
      }
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.errorListeners.forEach(callback => callback(error));
      this.attemptReconnect();
    });

    // Training events
    this.socket.on('training_started', (data: TrainingEventData) => {
      this.notifyListeners('training_started', data);
    });

    this.socket.on('training_progress', (data: TrainingEventData) => {
      this.notifyListeners('training_progress', data);
    });

    this.socket.on('training_completed', (data: TrainingEventData) => {
      this.notifyListeners('training_completed', data);
    });

    this.socket.on('training_stopped', (data: TrainingEventData) => {
      this.notifyListeners('training_stopped', data);
    });

    this.socket.on('training_error', (data: TrainingEventData) => {
      this.notifyListeners('training_error', data);
    });
  }

  /**
   * Attempt to reconnect with exponential backoff
   */
  private attemptReconnect(): void {
    if (this.isManualDisconnect || this.reconnectAttempts >= this.maxReconnectAttempts) {
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Attempting reconnect ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);

    setTimeout(() => {
      if (!this.isManualDisconnect && !this.socket?.connected) {
        this.socket?.connect();
      }
    }, delay);
  }

  /**
   * Notify all listeners for a specific event
   */
  private notifyListeners(event: string, data: TrainingEventData): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Subscribe to training events
   */
  on(event: 'training_started' | 'training_progress' | 'training_completed' | 'training_stopped' | 'training_error', callback: TrainingEventCallback): () => void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }

    this.eventListeners.get(event)!.add(callback);

    // Return unsubscribe function
    return () => {
      this.eventListeners.get(event)?.delete(callback);
    };
  }

  /**
   * Subscribe to connection events
   */
  onConnect(callback: ConnectionCallback): () => void {
    this.connectionListeners.add(callback);
    return () => {
      this.connectionListeners.delete(callback);
    };
  }

  /**
   * Subscribe to disconnection events
   */
  onDisconnect(callback: ConnectionCallback): () => void {
    this.disconnectionListeners.add(callback);
    return () => {
      this.disconnectionListeners.delete(callback);
    };
  }

  /**
   * Subscribe to error events
   */
  onError(callback: ErrorCallback): () => void {
    this.errorListeners.add(callback);
    return () => {
      this.errorListeners.delete(callback);
    };
  }

  /**
   * Remove all event listeners
   */
  removeAllListeners(): void {
    this.eventListeners.clear();
    this.connectionListeners.clear();
    this.disconnectionListeners.clear();
    this.errorListeners.clear();
  }

  /**
   * Emit a custom event (for testing or future features)
   */
  emit(event: string, data: any): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('Cannot emit event: WebSocket not connected');
    }
  }
}

// Export singleton instance
export const websocket = new WebSocketClient();

// Export class for testing
export { WebSocketClient };
