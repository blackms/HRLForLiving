import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import ReportModal from '../components/ReportModal';
import type { SimulationResult } from '../types';

interface ModelOption {
  name: string;
  scenario_name: string;
  episodes?: number;
}

interface ScenarioOption {
  name: string;
  description?: string;
}

export default function SimulationRunner() {
  const navigate = useNavigate();

  // Form state
  const [models, setModels] = useState<ModelOption[]>([]);
  const [scenarios, setScenarios] = useState<ScenarioOption[]>([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedScenario, setSelectedScenario] = useState('');
  const [numEpisodes, setNumEpisodes] = useState(10);
  const [seed, setSeed] = useState<number | undefined>(undefined);

  // Simulation state
  const [isRunning, setIsRunning] = useState(false);
  const [simulationId, setSimulationId] = useState<string | null>(null);
  const [results, setResults] = useState<SimulationResult | null>(null);

  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);

  // Load models and scenarios on mount
  useEffect(() => {
    loadModels();
    loadScenarios();
  }, []);

  const loadModels = async () => {
    try {
      const data = await api.listModels();
      setModels(data);
      if (data.length > 0 && !selectedModel) {
        setSelectedModel(data[0].name);
      }
    } catch (err) {
      setError('Failed to load models');
      console.error(err);
    }
  };

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

  const handleRunSimulation = async () => {
    if (!selectedModel || !selectedScenario) {
      setError('Please select both a model and a scenario');
      return;
    }

    setLoading(true);
    setIsRunning(true);
    setError(null);
    setResults(null);

    try {
      const response: any = await api.runSimulation({
        model_name: selectedModel,
        scenario_name: selectedScenario,
        num_episodes: numEpisodes,
        seed: seed,
      });

      // The API returns results in the response.results field
      if (response.simulation_id) {
        setSimulationId(response.simulation_id);
        
        // Results are included in the response
        if (response.results) {
          setResults(response.results);
        } else {
          // Fallback: fetch the results separately
          const simulationResults = await api.getSimulationResults(response.simulation_id);
          setResults(simulationResults);
        }
      }
    } catch (err: any) {
      setError(err.message || 'Failed to run simulation');
      console.error(err);
    } finally {
      setLoading(false);
      setIsRunning(false);
    }
  };

  const handleViewResults = () => {
    if (simulationId) {
      navigate(`/results?id=${simulationId}`);
    }
  };

  const handleReset = () => {
    setResults(null);
    setSimulationId(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Simulation Runner</h1>
        <p className="text-gray-600 dark:text-gray-400">Run simulations with trained models</p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-800 dark:text-red-200">{error}</p>
        </div>
      )}

      {/* Configuration Form */}
      {!results && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Simulation Configuration</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Trained Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isRunning}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              >
                <option value="">Select a model</option>
                {models.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name} ({model.scenario_name})
                  </option>
                ))}
              </select>
              {models.length === 0 && (
                <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                  No trained models available. Train a model first.
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Scenario
              </label>
              <select
                value={selectedScenario}
                onChange={(e) => setSelectedScenario(e.target.value)}
                disabled={isRunning}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
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
                max="100"
                disabled={isRunning}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              />
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Recommended: 10-50 episodes for reliable statistics
              </p>
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
                disabled={isRunning}
                className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              />
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Set a seed for reproducible results
              </p>
            </div>
          </div>

          <div className="mt-6">
            <button
              onClick={handleRunSimulation}
              disabled={loading || isRunning || !selectedModel || !selectedScenario}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors"
            >
              {isRunning ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Running Simulation...
                </span>
              ) : (
                'Run Simulation'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Progress Indicator */}
      {isRunning && !results && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div className="flex items-center justify-center gap-4">
            <svg className="animate-spin h-8 w-8 text-blue-600" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <div>
              <p className="text-lg font-medium text-gray-900 dark:text-white">Running simulation...</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Evaluating {numEpisodes} episodes with {selectedModel}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Summary Statistics */}
      {results && (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Simulation Complete</h2>
              <button
                onClick={handleReset}
                className="px-4 py-2 text-sm bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-white rounded-lg transition-colors"
              >
                Run Another
              </button>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Model: {results.model_name} • Scenario: {results.scenario_name} • Episodes: {results.num_episodes}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Duration</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                {results.duration_mean.toFixed(1)}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                ± {results.duration_std.toFixed(1)} months
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Total Wealth</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                €{results.total_wealth_mean.toFixed(0)}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                ± €{results.total_wealth_std.toFixed(0)}
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Investment Gains</h3>
              <p className={`text-3xl font-bold ${results.investment_gains_mean >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                {results.investment_gains_mean >= 0 ? '+' : ''}€{results.investment_gains_mean.toFixed(0)}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {((results.investment_gains_mean / results.final_invested_mean) * 100).toFixed(1)}% return
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">Final Portfolio</h3>
              <p className="text-3xl font-bold text-gray-900 dark:text-white">
                €{results.final_portfolio_mean.toFixed(0)}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Cash: €{results.final_cash_mean.toFixed(0)}
              </p>
            </div>
          </div>

          {/* Strategy Breakdown */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Strategy Learned</h2>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Invest</span>
                  <span className="text-sm font-bold text-gray-900 dark:text-white">
                    {(results.avg_invest_pct * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-600"
                    style={{ width: `${results.avg_invest_pct * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Save</span>
                  <span className="text-sm font-bold text-gray-900 dark:text-white">
                    {(results.avg_save_pct * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-600"
                    style={{ width: `${results.avg_save_pct * 100}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Consume</span>
                  <span className="text-sm font-bold text-gray-900 dark:text-white">
                    {(results.avg_consume_pct * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-purple-600"
                    style={{ width: `${results.avg_consume_pct * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={handleViewResults}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              View Detailed Results
            </button>
            <button
              onClick={() => setIsReportModalOpen(true)}
              className="px-6 py-3 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
            >
              Generate Report
            </button>
            <button
              onClick={() => navigate('/comparison')}
              className="px-6 py-3 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-white rounded-lg font-medium transition-colors"
            >
              Compare Scenarios
            </button>
          </div>

          {/* Report Modal */}
          {results && simulationId && (
            <ReportModal
              isOpen={isReportModalOpen}
              onClose={() => setIsReportModalOpen(false)}
              simulationId={simulationId}
              scenarioName={results.scenario_name}
              modelName={results.model_name}
            />
          )}
        </>
      )}
    </div>
  );
}
