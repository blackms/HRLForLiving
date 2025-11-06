import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';

// API response types (matching what the API actually returns)
interface ScenarioSummary {
  name: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
}

interface ModelSummary {
  name: string;
  scenario_name: string;
  episodes?: number;
  final_reward?: number;
  final_stability?: number;
  trained_at?: string;
  file_size_mb?: number;
}

interface SimulationHistoryItem {
  id: string;
  scenario_name: string;
  model_name: string;
  num_episodes: number;
  created_at: string;
}

interface ActivityItem {
  type: 'training' | 'simulation' | 'scenario' | 'model';
  message: string;
  timestamp: string;
}

export default function Dashboard() {
  const navigate = useNavigate();
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [simulations, setSimulations] = useState<SimulationHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [scenariosData, modelsData, simulationsData] = await Promise.all([
        api.listScenarios().catch(() => []),
        api.listModels().catch(() => []),
        api.getSimulationHistory().catch(() => []),
      ]);

      setScenarios(scenariosData);
      setModels(modelsData);
      setSimulations(simulationsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const getRecentActivity = (): ActivityItem[] => {
    const activities: ActivityItem[] = [];

    // Add recent simulations
    simulations.slice(0, 3).forEach((sim) => {
      activities.push({
        type: 'simulation',
        message: `Simulation run: ${sim.model_name} on ${sim.scenario_name} (${sim.num_episodes} episodes)`,
        timestamp: sim.created_at,
      });
    });

    // Add recent models
    models.slice(0, 2).forEach((model) => {
      if (model.trained_at) {
        activities.push({
          type: 'model',
          message: `Model trained: ${model.name} (${model.episodes || 0} episodes)`,
          timestamp: model.trained_at,
        });
      }
    });

    // Sort by timestamp descending
    return activities
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, 5);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
    return date.toLocaleDateString();
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'training':
        return '🎯';
      case 'simulation':
        return '🔬';
      case 'scenario':
        return '📝';
      case 'model':
        return '🤖';
      default:
        return '📊';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <div className="flex items-center space-x-2">
          <span className="text-red-600 dark:text-red-400">⚠️</span>
          <p className="text-red-800 dark:text-red-300">{error}</p>
        </div>
        <button
          onClick={loadDashboardData}
          className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  const recentActivity = getRecentActivity();

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
          <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400 mt-1">
            Welcome to HRL Finance System
          </p>
        </div>
        <button
          onClick={loadDashboardData}
          className="px-4 py-2 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 self-start sm:self-auto"
          aria-label="Refresh dashboard data"
        >
          <span role="img" aria-hidden="true">🔄</span> Refresh
        </button>
      </div>

      {/* Statistics Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
        <div
          className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 sm:p-6"
          role="region"
          aria-label="Scenarios statistics"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Scenarios</p>
              <p className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mt-1">
                {scenarios.length}
              </p>
            </div>
            <div className="text-3xl sm:text-4xl" role="img" aria-label="Scenarios icon">📝</div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
            Financial configurations
          </p>
        </div>

        <div
          className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 sm:p-6"
          role="region"
          aria-label="Models statistics"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Models</p>
              <p className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mt-1">
                {models.length}
              </p>
            </div>
            <div className="text-3xl sm:text-4xl" role="img" aria-label="Models icon">🤖</div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
            Trained AI agents
          </p>
        </div>

        <div
          className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 sm:p-6 sm:col-span-2 lg:col-span-1"
          role="region"
          aria-label="Simulations statistics"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Simulations</p>
              <p className="text-2xl sm:text-3xl font-bold text-gray-900 dark:text-white mt-1">
                {simulations.length}
              </p>
            </div>
            <div className="text-3xl sm:text-4xl" role="img" aria-label="Simulations icon">🔬</div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
            Evaluation runs
          </p>
        </div>
      </div>

      {/* Quick Actions */}
      <section
        className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg border border-blue-200 dark:border-blue-800 p-4 sm:p-6"
        aria-labelledby="quick-actions-heading"
      >
        <h2 id="quick-actions-heading" className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          <button
            onClick={() => navigate('/scenarios')}
            className="flex items-center space-x-3 px-4 py-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 hover:shadow-md transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
            aria-label="Create new scenario"
          >
            <span className="text-xl sm:text-2xl" role="img" aria-hidden="true">➕</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              New Scenario
            </span>
          </button>

          <button
            onClick={() => navigate('/training')}
            className="flex items-center space-x-3 px-4 py-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 hover:shadow-md transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
            aria-label="Start training a model"
          >
            <span className="text-xl sm:text-2xl" role="img" aria-hidden="true">🎯</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Start Training
            </span>
          </button>

          <button
            onClick={() => navigate('/simulation')}
            className="flex items-center space-x-3 px-4 py-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 hover:shadow-md transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
            aria-label="Run a simulation"
          >
            <span className="text-xl sm:text-2xl" role="img" aria-hidden="true">▶️</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Run Simulation
            </span>
          </button>

          <button
            onClick={() => navigate('/comparison')}
            className="flex items-center space-x-3 px-4 py-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-500 dark:hover:border-blue-500 hover:shadow-md transition-all focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
            aria-label="Compare simulation results"
          >
            <span className="text-xl sm:text-2xl" role="img" aria-hidden="true">⚖️</span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              Compare Results
            </span>
          </button>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* Recent Scenarios */}
        <section
          className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700"
          aria-labelledby="recent-scenarios-heading"
        >
          <div className="p-4 sm:p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex justify-between items-center">
              <h2 id="recent-scenarios-heading" className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
                Recent Scenarios
              </h2>
              <button
                onClick={() => navigate('/scenarios')}
                className="text-sm text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 rounded"
                aria-label="View all scenarios"
              >
                View all →
              </button>
            </div>
          </div>
          <div className="p-4 sm:p-6">
            {scenarios.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  No scenarios yet
                </p>
                <button
                  onClick={() => navigate('/scenarios')}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
                  aria-label="Create your first scenario"
                >
                  Create your first scenario
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                {scenarios.slice(0, 3).map((scenario) => (
                  <button
                    key={scenario.name}
                    className="w-full text-left p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-blue-500 dark:hover:border-blue-500 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset"
                    onClick={() => navigate('/scenarios')}
                    aria-label={`View scenario: ${scenario.name}`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900 dark:text-white">
                          {scenario.name}
                        </h3>
                        {scenario.description && (
                          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                            {scenario.description}
                          </p>
                        )}
                        {scenario.created_at && (
                          <div className="mt-2 text-xs text-gray-500 dark:text-gray-500">
                            Created {formatDate(scenario.created_at)}
                          </div>
                        )}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>

        {/* Recent Models */}
        <section
          className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700"
          aria-labelledby="recent-models-heading"
        >
          <div className="p-4 sm:p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex justify-between items-center">
              <h2 id="recent-models-heading" className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
                Recent Models
              </h2>
              <button
                onClick={() => navigate('/training')}
                className="text-sm text-blue-600 dark:text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900 rounded"
                aria-label="View all models"
              >
                View all →
              </button>
            </div>
          </div>
          <div className="p-4 sm:p-6">
            {models.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-500 dark:text-gray-400 mb-4">
                  No trained models yet
                </p>
                <button
                  onClick={() => navigate('/training')}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900"
                  aria-label="Train your first model"
                >
                  Train your first model
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                {models.slice(0, 3).map((model) => (
                  <button
                    key={model.name}
                    className="w-full text-left p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg border border-gray-200 dark:border-gray-600 hover:border-blue-500 dark:hover:border-blue-500 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset"
                    onClick={() => navigate('/simulation')}
                    aria-label={`View model: ${model.name}`}
                  >
                    <div className="flex justify-between items-start">
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900 dark:text-white">
                          {model.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          Scenario: {model.scenario_name}
                        </p>
                        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-xs text-gray-500 dark:text-gray-500">
                          {model.episodes && (
                            <span><span role="img" aria-hidden="true">📊</span> {model.episodes} episodes</span>
                          )}
                          {model.trained_at && (
                            <span><span role="img" aria-hidden="true">🕒</span> {formatDate(model.trained_at)}</span>
                          )}
                        </div>
                      </div>
                      {model.final_reward !== undefined && (
                        <div className="ml-4 text-right flex-shrink-0">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {model.final_reward.toFixed(1)}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-500">
                            reward
                          </div>
                        </div>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>

      {/* Recent Activity Feed */}
      {recentActivity.length > 0 && (
        <section
          className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700"
          aria-labelledby="recent-activity-heading"
        >
          <div className="p-4 sm:p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 id="recent-activity-heading" className="text-base sm:text-lg font-semibold text-gray-900 dark:text-white">
              Recent Activity
            </h2>
          </div>
          <div className="p-4 sm:p-6">
            <div className="space-y-4" role="list">
              {recentActivity.map((activity, index) => (
                <div key={index} className="flex items-start space-x-3" role="listitem">
                  <div className="flex-shrink-0 text-xl sm:text-2xl" role="img" aria-label={`${activity.type} activity`}>
                    {getActivityIcon(activity.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-900 dark:text-white break-words">
                      {activity.message}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-500 mt-1">
                      <time dateTime={activity.timestamp}>{formatDate(activity.timestamp)}</time>
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
