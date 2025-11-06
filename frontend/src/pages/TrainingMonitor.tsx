import { useState, useEffect } from 'react';
import { api } from '../services/api';
import { websocket } from '../services/websocket';
import type { TrainingProgress, TrainingStatus } from '../types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ScenarioOption {
  name: string;
  description?: string;
}

interface ChartDataPoint {
  episode: number;
  reward: number;
  duration: number;
  stability: number;
  cash: number;
  invested: number;
}

export default function TrainingMonitor() {
  // Form state
  const [scenarios, setScenarios] = useState<ScenarioOption[]>([]);
  const [selectedScenario, setSelectedScenario] = useState('');
  const [numEpisodes, setNumEpisodes] = useState(1000);
  const [saveInterval, setSaveInterval] = useState(100);
  const [evalEpisodes, setEvalEpisodes] = useState(10);
  const [seed, setSeed] = useState<number | undefined>(undefined);

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [currentProgress, setCurrentProgress] = useState<TrainingProgress | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Load scenarios on mount
  useEffect(() => {
    loadScenarios();
  }, []);

  // Setup WebSocket connection
  useEffect(() => {
    websocket.connect();

    const unsubConnect = websocket.onConnect(() => {
      setWsConnected(true);
      console.log('WebSocket connected');
    });

    const unsubDisconnect = websocket.onDisconnect(() => {
      setWsConnected(false);
      console.log('WebSocket disconnected');
    });

    const unsubError = websocket.onError((error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error');
    });

    return () => {
      unsubConnect();
      unsubDisconnect();
      unsubError();
    };
  }, []);

  // Subscribe to training events
  useEffect(() => {
    const unsubStarted = websocket.on('training_started', (data) => {
      console.log('Training started:', data);
      setIsTraining(true);
      setError(null);
      setChartData([]);
    });

    const unsubProgress = websocket.on('training_progress', (data) => {
      if (data.progress) {
        setCurrentProgress(data.progress);
        
        // Add to chart data
        setChartData(prev => [...prev, {
          episode: data.progress!.episode,
          reward: data.progress!.avg_reward,
          duration: data.progress!.avg_duration,
          stability: data.progress!.stability * 100, // Convert to percentage
          cash: data.progress!.avg_cash,
          invested: data.progress!.avg_invested,
        }]);
      }
    });

    const unsubCompleted = websocket.on('training_completed', (data) => {
      console.log('Training completed:', data);
      setIsTraining(false);
      setError(null);
    });

    const unsubStopped = websocket.on('training_stopped', (data) => {
      console.log('Training stopped:', data);
      setIsTraining(false);
    });

    const unsubError = websocket.on('training_error', (data) => {
      console.error('Training error:', data);
      setIsTraining(false);
      setError(data.error || 'Training error occurred');
    });

    return () => {
      unsubStarted();
      unsubProgress();
      unsubCompleted();
      unsubStopped();
      unsubError();
    };
  }, []);

  // Poll training status periodically
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const status = await api.getTrainingStatus();
        setTrainingStatus(status);
        setIsTraining(status.is_training);
        
        if (status.is_training && status.progress) {
          setCurrentProgress(status.progress);
        }
      } catch (err) {
        console.error('Failed to fetch training status:', err);
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 5000);

    return () => clearInterval(interval);
  }, []);

  const loadScenarios = async () => {
    try {
      const data = await api.listScenarios();
      setScenarios(data);
      if (data.length > 0 && !selectedScenario) {
        setSelectedScenario(data[0].name);
      }
    } catch (err) {
      setError('Failed to load scenarios');
      console.error(err);
    }
  };

  const handleStartTraining = async () => {
    if (!selectedScenario) {
      setError('Please select a scenario');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await api.startTraining({
        scenario_name: selectedScenario,
        num_episodes: numEpisodes,
        save_interval: saveInterval,
        eval_episodes: evalEpisodes,
        seed: seed,
      });

      setIsTraining(true);
      setChartData([]);
      setCurrentProgress(null);
    } catch (err: any) {
      setError(err.message || 'Failed to start training');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    setLoading(true);
    setError(null);

    try {
      await api.stopTraining();
      setIsTraining(false);
    } catch (err: any) {
      setError(err.message || 'Failed to stop training');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  const progressPercentage = currentProgress 
    ? (currentProgress.episode / currentProgress.total_episodes) * 100 
    : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Training Monitor</h1>
          <p className="text-gray-600 dark:text-gray-400">Train AI models on financial scenarios</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`h-3 w-3 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {wsConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-800 dark:text-red-200">{error}</p>
        </div>
      )}

      {/* Training Configuration Form */}
      {!isTraining && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Training Configuration</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Scenario
              </label>
              <select
                value={selectedScenario}
                onChange={(e) => setSelectedScenario(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select a scenario</option>
                {scenarios.map((scenario) => (
                  <option key={scenario.name} value={scenario.name}>
                    {scenario.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Number of Episodes
              </label>
              <input
                type="number"
                value={numEpisodes}
                onChange={(e) => setNumEpisodes(parseInt(e.target.value))}
                min="1"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Save Interval
              </label>
              <input
                type="number"
                value={saveInterval}
                onChange={(e) => setSaveInterval(parseInt(e.target.value))}
                min="1"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Evaluation Episodes
              </label>
              <input
                type="number"
                value={evalEpisodes}
                onChange={(e) => setEvalEpisodes(parseInt(e.target.value))}
                min="1"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Random Seed (optional)
              </label>
              <input
                type="number"
                value={seed || ''}
                onChange={(e) => setSeed(e.target.value ? parseInt(e.target.value) : undefined)}
                placeholder="Leave empty for random"
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="mt-6">
            <button
              onClick={handleStartTraining}
              disabled={loading || !selectedScenario}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors"
            >
              {loading ? 'Starting...' : 'Start Training'}
            </button>
          </div>
        </div>
      )}

      {/* Training Progress */}
      {isTraining && currentProgress && (
        <>
          {/* Status Bar */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                  Training: {trainingStatus?.scenario_name || selectedScenario}
                </h2>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Episode {currentProgress.episode} / {currentProgress.total_episodes} • 
                  Elapsed: {formatTime(currentProgress.elapsed_time)}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <div className="animate-pulse h-3 w-3 rounded-full bg-blue-500" />
                <span className="text-sm font-medium text-blue-600 dark:text-blue-400">Training</span>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="relative w-full h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="absolute top-0 left-0 h-full bg-blue-600 transition-all duration-300"
                style={{ width: `${progressPercentage}%` }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
                  {progressPercentage.toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="mt-4">
              <button
                onClick={handleStopTraining}
                disabled={loading}
                className="px-6 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors"
              >
                {loading ? 'Stopping...' : 'Stop Training'}
              </button>
            </div>
          </div>

          {/* Current Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Average Reward</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                {currentProgress.avg_reward.toFixed(2)}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Duration</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                {currentProgress.avg_duration.toFixed(1)} <span className="text-lg">months</span>
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Stability</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                {(currentProgress.stability * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Average Cash</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                €{currentProgress.avg_cash.toFixed(0)}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Average Invested</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                €{currentProgress.avg_invested.toFixed(0)}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Goal Adherence</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                {(currentProgress.goal_adherence * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Training Charts */}
          {chartData.length > 0 && (
            <div className="space-y-6">
              {/* Reward Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Average Reward Over Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="episode" 
                      stroke="#9CA3AF"
                      label={{ value: 'Episode', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      label={{ value: 'Reward', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '0.5rem'
                      }}
                      labelStyle={{ color: '#F3F4F6' }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="reward" 
                      stroke="#3B82F6" 
                      strokeWidth={2}
                      dot={false}
                      name="Avg Reward"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Duration Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Average Duration Over Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="episode" 
                      stroke="#9CA3AF"
                      label={{ value: 'Episode', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      label={{ value: 'Months', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '0.5rem'
                      }}
                      labelStyle={{ color: '#F3F4F6' }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="duration" 
                      stroke="#10B981" 
                      strokeWidth={2}
                      dot={false}
                      name="Avg Duration (months)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Stability Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Stability Over Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="episode" 
                      stroke="#9CA3AF"
                      label={{ value: 'Episode', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      label={{ value: 'Stability (%)', angle: -90, position: 'insideLeft' }}
                      domain={[0, 100]}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '0.5rem'
                      }}
                      labelStyle={{ color: '#F3F4F6' }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="stability" 
                      stroke="#8B5CF6" 
                      strokeWidth={2}
                      dot={false}
                      name="Stability (%)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Cash and Investment Chart */}
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Cash vs Investment Over Time</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="episode" 
                      stroke="#9CA3AF"
                      label={{ value: 'Episode', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      stroke="#9CA3AF"
                      label={{ value: 'Amount (€)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1F2937', 
                        border: '1px solid #374151',
                        borderRadius: '0.5rem'
                      }}
                      labelStyle={{ color: '#F3F4F6' }}
                      formatter={(value: number) => `€${value.toFixed(0)}`}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="cash" 
                      stroke="#F59E0B" 
                      strokeWidth={2}
                      dot={false}
                      name="Avg Cash"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="invested" 
                      stroke="#EF4444" 
                      strokeWidth={2}
                      dot={false}
                      name="Avg Invested"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
