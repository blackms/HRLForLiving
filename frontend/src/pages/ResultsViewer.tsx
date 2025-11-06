import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { api } from '../services/api';
import type { SimulationResult } from '../types';

type ChartView = 'cash' | 'portfolio' | 'wealth' | 'actions';

export default function ResultsViewer() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const simulationId = searchParams.get('id');

  // Data state
  const [results, setResults] = useState<SimulationResult | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState<number>(0);

  // UI state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ChartView>('cash');

  // Load simulation results
  useEffect(() => {
    if (simulationId) {
      loadResults();
    } else {
      setError('No simulation ID provided');
      setLoading(false);
    }
  }, [simulationId]);

  const loadResults = async () => {
    if (!simulationId) return;

    setLoading(true);
    setError(null);

    try {
      const data = await api.getSimulationResults(simulationId);
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load simulation results');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    if (!results) return;

    const dataStr = JSON.stringify(results, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `simulation_${simulationId}_results.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <svg className="animate-spin h-12 w-12 text-blue-600 mx-auto mb-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <p className="text-gray-600 dark:text-gray-400">Loading results...</p>
        </div>
      </div>
    );
  }

  if (error || !results) {
    return (
      <div className="space-y-6">
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-red-800 dark:text-red-200 mb-2">Error Loading Results</h2>
          <p className="text-red-700 dark:text-red-300">{error || 'No results found'}</p>
          <button
            onClick={() => navigate('/simulation')}
            className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
          >
            Back to Simulation Runner
          </button>
        </div>
      </div>
    );
  }

  const episode = results.episodes[selectedEpisode];

  // Prepare chart data
  const cashBalanceData = episode.months?.map((month, idx) => ({
    month,
    cash: episode.cash?.[idx] || 0,
  })) || [];

  const portfolioData = episode.months?.map((month, idx) => ({
    month,
    invested: episode.invested?.[idx] || 0,
    portfolioValue: episode.portfolio_value?.[idx] || 0,
  })) || [];

  const wealthData = episode.months?.map((month, idx) => ({
    month,
    cash: episode.cash?.[idx] || 0,
    portfolio: episode.portfolio_value?.[idx] || 0,
    total: (episode.cash?.[idx] || 0) + (episode.portfolio_value?.[idx] || 0),
  })) || [];

  const actionData = [
    { name: 'Invest', value: results.avg_invest_pct * 100, color: '#3B82F6' },
    { name: 'Save', value: results.avg_save_pct * 100, color: '#10B981' },
    { name: 'Consume', value: results.avg_consume_pct * 100, color: '#8B5CF6' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <button
            onClick={() => navigate('/simulation')}
            className="text-blue-600 dark:text-blue-400 hover:underline mb-2 flex items-center gap-1"
          >
            ← Back to Simulation Runner
          </button>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Results: {results.scenario_name}
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Model: {results.model_name} • {results.num_episodes} Episodes
          </p>
        </div>
      </div>

      {/* Summary Statistics */}
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

      {/* Episode Selector */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Episode Details</h2>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Viewing episode {selectedEpisode + 1} of {results.num_episodes}
            </p>
          </div>
          <select
            value={selectedEpisode}
            onChange={(e) => setSelectedEpisode(parseInt(e.target.value))}
            className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500"
          >
            {results.episodes.map((ep, idx) => (
              <option key={idx} value={idx}>
                Episode {idx + 1} ({ep.duration} months)
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md">
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="flex -mb-px">
            <button
              onClick={() => setActiveTab('cash')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'cash'
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:border-gray-300'
              }`}
            >
              Cash Balance
            </button>
            <button
              onClick={() => setActiveTab('portfolio')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'portfolio'
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:border-gray-300'
              }`}
            >
              Portfolio Evolution
            </button>
            <button
              onClick={() => setActiveTab('wealth')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'wealth'
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:border-gray-300'
              }`}
            >
              Wealth Accumulation
            </button>
            <button
              onClick={() => setActiveTab('actions')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'actions'
                  ? 'border-blue-600 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:border-gray-300'
              }`}
            >
              Action Distribution
            </button>
          </nav>
        </div>

        {/* Chart Content */}
        <div className="p-6">
          {activeTab === 'cash' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Cash Balance Over Time
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={cashBalanceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis
                    dataKey="month"
                    label={{ value: 'Month', position: 'insideBottom', offset: -5 }}
                    stroke="#9CA3AF"
                  />
                  <YAxis
                    label={{ value: 'Cash (EUR)', angle: -90, position: 'insideLeft' }}
                    stroke="#9CA3AF"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                      color: '#F9FAFB',
                    }}
                    formatter={(value: number) => [`€${value.toFixed(2)}`, 'Cash']}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="cash"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    dot={false}
                    name="Cash Balance"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeTab === 'portfolio' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Portfolio Evolution
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={portfolioData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis
                    dataKey="month"
                    label={{ value: 'Month', position: 'insideBottom', offset: -5 }}
                    stroke="#9CA3AF"
                  />
                  <YAxis
                    label={{ value: 'Value (EUR)', angle: -90, position: 'insideLeft' }}
                    stroke="#9CA3AF"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                      color: '#F9FAFB',
                    }}
                    formatter={(value: number) => `€${value.toFixed(2)}`}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="invested"
                    stroke="#10B981"
                    strokeWidth={2}
                    dot={false}
                    name="Amount Invested"
                  />
                  <Line
                    type="monotone"
                    dataKey="portfolioValue"
                    stroke="#8B5CF6"
                    strokeWidth={2}
                    dot={false}
                    name="Portfolio Value"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeTab === 'wealth' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Wealth Accumulation
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={wealthData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis
                    dataKey="month"
                    label={{ value: 'Month', position: 'insideBottom', offset: -5 }}
                    stroke="#9CA3AF"
                  />
                  <YAxis
                    label={{ value: 'Wealth (EUR)', angle: -90, position: 'insideLeft' }}
                    stroke="#9CA3AF"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                      color: '#F9FAFB',
                    }}
                    formatter={(value: number) => `€${value.toFixed(2)}`}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="cash"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    dot={false}
                    name="Cash"
                  />
                  <Line
                    type="monotone"
                    dataKey="portfolio"
                    stroke="#10B981"
                    strokeWidth={2}
                    dot={false}
                    name="Portfolio"
                  />
                  <Line
                    type="monotone"
                    dataKey="total"
                    stroke="#F59E0B"
                    strokeWidth={3}
                    dot={false}
                    name="Total Wealth"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {activeTab === 'actions' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Action Distribution
              </h3>
              <div className="flex items-center justify-center">
                <ResponsiveContainer width="100%" height={400}>
                  <PieChart>
                    <Pie
                      data={actionData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                      outerRadius={120}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {actionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '0.5rem',
                        color: '#F9FAFB',
                      }}
                      formatter={(value: number) => `${value.toFixed(1)}%`}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Strategy Learned */}
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
          onClick={() => navigate('/comparison')}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
        >
          Compare Scenarios
        </button>
        <button
          onClick={handleExport}
          className="px-6 py-3 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-white rounded-lg font-medium transition-colors"
        >
          Export Data
        </button>
      </div>
    </div>
  );
}
