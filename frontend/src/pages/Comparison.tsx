import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { api } from '../services/api';
import type { SimulationResult } from '../types';

interface SimulationHistoryItem {
  id: string;
  scenario_name: string;
  model_name: string;
  num_episodes: number;
  created_at: string;
}

interface ComparisonMetric {
  name: string;
  [key: string]: number | string;
}

export default function Comparison() {
  const navigate = useNavigate();

  // Data state
  const [availableSimulations, setAvailableSimulations] = useState<SimulationHistoryItem[]>([]);
  const [selectedSimulations, setSelectedSimulations] = useState<string[]>([]);
  const [simulationResults, setSimulationResults] = useState<Map<string, SimulationResult>>(new Map());

  // UI state
  const [loading, setLoading] = useState(true);
  const [loadingResults, setLoadingResults] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load available simulations
  useEffect(() => {
    loadSimulations();
  }, []);

  const loadSimulations = async () => {
    setLoading(true);
    setError(null);

    try {
      const simulations = await api.getSimulationHistory();
      setAvailableSimulations(simulations);
    } catch (err: any) {
      setError(err.message || 'Failed to load simulations');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Load results for selected simulations
  const loadSelectedResults = async () => {
    if (selectedSimulations.length === 0) return;

    setLoadingResults(true);
    setError(null);

    try {
      const newResults = new Map(simulationResults);

      for (const simId of selectedSimulations) {
        if (!newResults.has(simId)) {
          const result = await api.getSimulationResults(simId);
          newResults.set(simId, result);
        }
      }

      setSimulationResults(newResults);
    } catch (err: any) {
      setError(err.message || 'Failed to load simulation results');
      console.error(err);
    } finally {
      setLoadingResults(false);
    }
  };

  // Load results when selections change
  useEffect(() => {
    if (selectedSimulations.length > 0) {
      loadSelectedResults();
    }
  }, [selectedSimulations]);

  const handleSimulationToggle = (simId: string) => {
    setSelectedSimulations(prev => {
      if (prev.includes(simId)) {
        return prev.filter(id => id !== simId);
      } else {
        if (prev.length >= 4) {
          setError('Maximum 4 simulations can be compared at once');
          return prev;
        }
        return [...prev, simId];
      }
    });
  };

  const handleClearAll = () => {
    setSelectedSimulations([]);
    setSimulationResults(new Map());
  };

  const handleExportCSV = () => {
    if (selectedSimulations.length === 0) return;

    const headers = ['Metric', ...selectedSimulations.map(id => {
      const sim = availableSimulations.find(s => s.id === id);
      return sim ? `${sim.scenario_name} (${sim.model_name})` : id;
    })];

    const metrics = [
      'Duration (months)',
      'Total Wealth (EUR)',
      'Investment Gains (EUR)',
      'Final Cash (EUR)',
      'Final Portfolio (EUR)',
      'Invest %',
      'Save %',
      'Consume %',
    ];

    const rows = metrics.map(metric => {
      const row = [metric];
      selectedSimulations.forEach(simId => {
        const result = simulationResults.get(simId);
        if (result) {
          switch (metric) {
            case 'Duration (months)':
              row.push(result.duration_mean.toFixed(1));
              break;
            case 'Total Wealth (EUR)':
              row.push(result.total_wealth_mean.toFixed(2));
              break;
            case 'Investment Gains (EUR)':
              row.push(result.investment_gains_mean.toFixed(2));
              break;
            case 'Final Cash (EUR)':
              row.push(result.final_cash_mean.toFixed(2));
              break;
            case 'Final Portfolio (EUR)':
              row.push(result.final_portfolio_mean.toFixed(2));
              break;
            case 'Invest %':
              row.push((result.avg_invest_pct * 100).toFixed(1));
              break;
            case 'Save %':
              row.push((result.avg_save_pct * 100).toFixed(1));
              break;
            case 'Consume %':
              row.push((result.avg_consume_pct * 100).toFixed(1));
              break;
          }
        } else {
          row.push('N/A');
        }
      });
      return row;
    });

    const csvContent = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `comparison_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    if (selectedSimulations.length === 0) return;

    const exportData = selectedSimulations.map(simId => {
      const result = simulationResults.get(simId);
      const sim = availableSimulations.find(s => s.id === simId);
      return {
        simulation_id: simId,
        scenario_name: sim?.scenario_name,
        model_name: sim?.model_name,
        results: result,
      };
    });

    const dataStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `comparison_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Prepare comparison data
  const getComparisonMetrics = (): ComparisonMetric[] => {
    const metrics: ComparisonMetric[] = [
      { name: 'Duration (months)' },
      { name: 'Total Wealth (EUR)' },
      { name: 'Investment Gains (EUR)' },
      { name: 'Final Cash (EUR)' },
      { name: 'Final Portfolio (EUR)' },
      { name: 'Invest %' },
      { name: 'Save %' },
      { name: 'Consume %' },
    ];

    selectedSimulations.forEach(simId => {
      const result = simulationResults.get(simId);
      const sim = availableSimulations.find(s => s.id === simId);
      const label = sim ? `${sim.scenario_name}` : simId.substring(0, 8);

      if (result) {
        metrics[0][label] = parseFloat(result.duration_mean.toFixed(1));
        metrics[1][label] = parseFloat(result.total_wealth_mean.toFixed(0));
        metrics[2][label] = parseFloat(result.investment_gains_mean.toFixed(0));
        metrics[3][label] = parseFloat(result.final_cash_mean.toFixed(0));
        metrics[4][label] = parseFloat(result.final_portfolio_mean.toFixed(0));
        metrics[5][label] = parseFloat((result.avg_invest_pct * 100).toFixed(1));
        metrics[6][label] = parseFloat((result.avg_save_pct * 100).toFixed(1));
        metrics[7][label] = parseFloat((result.avg_consume_pct * 100).toFixed(1));
      }
    });

    return metrics;
  };

  // Prepare wealth over time comparison data
  const getWealthComparisonData = () => {
    if (selectedSimulations.length === 0) return [];

    // Find the maximum duration across all simulations
    let maxDuration = 0;
    selectedSimulations.forEach(simId => {
      const result = simulationResults.get(simId);
      if (result && result.episodes.length > 0) {
        const episode = result.episodes[0];
        if (episode.months && episode.months.length > maxDuration) {
          maxDuration = episode.months.length;
        }
      }
    });

    // Build data points for each month
    const data: any[] = [];
    for (let month = 0; month < maxDuration; month++) {
      const point: any = { month };

      selectedSimulations.forEach(simId => {
        const result = simulationResults.get(simId);
        const sim = availableSimulations.find(s => s.id === simId);
        const label = sim ? sim.scenario_name : simId.substring(0, 8);

        if (result && result.episodes.length > 0) {
          const episode = result.episodes[0];
          if (episode.months && episode.cash && episode.portfolio_value && month < episode.months.length) {
            point[label] = (episode.cash[month] || 0) + (episode.portfolio_value[month] || 0);
          }
        }
      });

      data.push(point);
    }

    return data;
  };

  const colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6'];

  // Calculate differences (percentage change from first simulation)
  const calculateDifference = (value: number, baseValue: number): string => {
    if (baseValue === 0) return 'N/A';
    const diff = ((value - baseValue) / baseValue) * 100;
    return `${diff >= 0 ? '+' : ''}${diff.toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <svg className="animate-spin h-12 w-12 text-blue-600 mx-auto mb-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <p className="text-gray-600 dark:text-gray-400">Loading simulations...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <button
            onClick={() => navigate('/dashboard')}
            className="text-blue-600 dark:text-blue-400 hover:underline mb-2 flex items-center gap-1"
          >
            ← Back to Dashboard
          </button>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Scenario Comparison
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Compare multiple simulation results side-by-side
          </p>
        </div>
      </div>

      {error && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
          <p className="text-yellow-800 dark:text-yellow-200">{error}</p>
        </div>
      )}

      {/* Simulation Selector */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Select Simulations to Compare
          </h2>
          <div className="flex gap-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {selectedSimulations.length} / 4 selected
            </span>
            {selectedSimulations.length > 0 && (
              <button
                onClick={handleClearAll}
                className="text-sm text-red-600 dark:text-red-400 hover:underline"
              >
                Clear All
              </button>
            )}
          </div>
        </div>

        {availableSimulations.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              No simulations available. Run some simulations first.
            </p>
            <button
              onClick={() => navigate('/simulation')}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
            >
              Go to Simulation Runner
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {availableSimulations.map(sim => (
              <div
                key={sim.id}
                onClick={() => handleSimulationToggle(sim.id)}
                className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  selectedSimulations.includes(sim.id)
                    ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/20'
                    : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-gray-900 dark:text-white">
                    {sim.scenario_name}
                  </h3>
                  <input
                    type="checkbox"
                    checked={selectedSimulations.includes(sim.id)}
                    onChange={() => {}}
                    className="mt-1"
                  />
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                  Model: {sim.model_name}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-500">
                  {sim.num_episodes} episodes • {new Date(sim.created_at).toLocaleDateString()}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Loading Results */}
      {loadingResults && (
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div className="flex items-center justify-center">
            <svg className="animate-spin h-8 w-8 text-blue-600 mr-3" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span className="text-gray-600 dark:text-gray-400">Loading simulation results...</span>
          </div>
        </div>
      )}

      {/* Comparison Results */}
      {selectedSimulations.length > 0 && !loadingResults && (
        <>
          {/* Metrics Comparison Table */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Metrics Comparison
              </h2>
              <div className="flex gap-2">
                <button
                  onClick={handleExportCSV}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Export CSV
                </button>
                <button
                  onClick={handleExportJSON}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Export JSON
                </button>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700">
                    <th className="text-left py-3 px-4 font-semibold text-gray-900 dark:text-white">
                      Metric
                    </th>
                    {selectedSimulations.map((simId, idx) => {
                      const sim = availableSimulations.find(s => s.id === simId);
                      return (
                        <th key={simId} className="text-right py-3 px-4 font-semibold text-gray-900 dark:text-white">
                          <div className="flex items-center justify-end gap-2">
                            <div
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: colors[idx] }}
                            />
                            <span>{sim?.scenario_name || simId.substring(0, 8)}</span>
                          </div>
                          <div className="text-xs font-normal text-gray-500 dark:text-gray-400 mt-1">
                            {sim?.model_name}
                          </div>
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {getComparisonMetrics().map((metric, metricIdx) => {
                    const baseValue = selectedSimulations.length > 0
                      ? (simulationResults.get(selectedSimulations[0]) as any)?.[
                          metricIdx === 0 ? 'duration_mean' :
                          metricIdx === 1 ? 'total_wealth_mean' :
                          metricIdx === 2 ? 'investment_gains_mean' :
                          metricIdx === 3 ? 'final_cash_mean' :
                          metricIdx === 4 ? 'final_portfolio_mean' :
                          metricIdx === 5 ? 'avg_invest_pct' :
                          metricIdx === 6 ? 'avg_save_pct' : 'avg_consume_pct'
                        ]
                      : 0;

                    return (
                      <tr
                        key={metric.name}
                        className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700/50"
                      >
                        <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">
                          {metric.name}
                        </td>
                        {selectedSimulations.map((simId, idx) => {
                          const sim = availableSimulations.find(s => s.id === simId);
                          const label = sim ? sim.scenario_name : simId.substring(0, 8);
                          const value = metric[label] as number;

                          let displayValue = 'N/A';
                          let difference = '';

                          if (value !== undefined) {
                            if (metric.name.includes('%')) {
                              displayValue = `${value.toFixed(1)}%`;
                            } else if (metric.name.includes('EUR')) {
                              displayValue = `€${value.toFixed(0)}`;
                            } else {
                              displayValue = value.toFixed(1);
                            }

                            // Calculate difference from first simulation
                            if (idx > 0 && baseValue) {
                              const adjustedBase = metricIdx >= 5 ? baseValue * 100 : baseValue;
                              difference = calculateDifference(value, adjustedBase);
                            }
                          }

                          return (
                            <td key={simId} className="py-3 px-4 text-right">
                              <div className="font-semibold text-gray-900 dark:text-white">
                                {displayValue}
                              </div>
                              {difference && (
                                <div className={`text-xs ${
                                  difference.startsWith('+')
                                    ? 'text-green-600 dark:text-green-400'
                                    : difference.startsWith('-')
                                    ? 'text-red-600 dark:text-red-400'
                                    : 'text-gray-500 dark:text-gray-400'
                                }`}>
                                  {difference}
                                </div>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Comparative Bar Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Wealth Comparison */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Total Wealth Comparison
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[getComparisonMetrics()[1]]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                      color: '#F9FAFB',
                    }}
                    formatter={(value: number) => `€${value.toFixed(0)}`}
                  />
                  <Legend />
                  {selectedSimulations.map((simId, idx) => {
                    const sim = availableSimulations.find(s => s.id === simId);
                    const label = sim ? sim.scenario_name : simId.substring(0, 8);
                    return (
                      <Bar
                        key={simId}
                        dataKey={label}
                        fill={colors[idx]}
                        name={label}
                      />
                    );
                  })}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Duration Comparison */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Duration Comparison
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[getComparisonMetrics()[0]]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                      color: '#F9FAFB',
                    }}
                    formatter={(value: number) => `${value.toFixed(1)} months`}
                  />
                  <Legend />
                  {selectedSimulations.map((simId, idx) => {
                    const sim = availableSimulations.find(s => s.id === simId);
                    const label = sim ? sim.scenario_name : simId.substring(0, 8);
                    return (
                      <Bar
                        key={simId}
                        dataKey={label}
                        fill={colors[idx]}
                        name={label}
                      />
                    );
                  })}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Investment Gains Comparison */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Investment Gains Comparison
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[getComparisonMetrics()[2]]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                      color: '#F9FAFB',
                    }}
                    formatter={(value: number) => `€${value.toFixed(0)}`}
                  />
                  <Legend />
                  {selectedSimulations.map((simId, idx) => {
                    const sim = availableSimulations.find(s => s.id === simId);
                    const label = sim ? sim.scenario_name : simId.substring(0, 8);
                    return (
                      <Bar
                        key={simId}
                        dataKey={label}
                        fill={colors[idx]}
                        name={label}
                      />
                    );
                  })}
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Strategy Comparison */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Strategy Distribution Comparison
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getComparisonMetrics().slice(5, 8)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                  <XAxis dataKey="name" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" label={{ value: 'Percentage', angle: -90, position: 'insideLeft' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                      color: '#F9FAFB',
                    }}
                    formatter={(value: number) => `${value.toFixed(1)}%`}
                  />
                  <Legend />
                  {selectedSimulations.map((simId, idx) => {
                    const sim = availableSimulations.find(s => s.id === simId);
                    const label = sim ? sim.scenario_name : simId.substring(0, 8);
                    return (
                      <Bar
                        key={simId}
                        dataKey={label}
                        fill={colors[idx]}
                        name={label}
                      />
                    );
                  })}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Wealth Over Time Comparison */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Wealth Accumulation Over Time
            </h3>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={getWealthComparisonData()}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
                <XAxis
                  dataKey="month"
                  label={{ value: 'Month', position: 'insideBottom', offset: -5 }}
                  stroke="#9CA3AF"
                />
                <YAxis
                  label={{ value: 'Total Wealth (EUR)', angle: -90, position: 'insideLeft' }}
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
                {selectedSimulations.map((simId, idx) => {
                  const sim = availableSimulations.find(s => s.id === simId);
                  const label = sim ? sim.scenario_name : simId.substring(0, 8);
                  return (
                    <Line
                      key={simId}
                      type="monotone"
                      dataKey={label}
                      stroke={colors[idx]}
                      strokeWidth={2}
                      dot={false}
                      name={label}
                    />
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Summary Insights */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Key Insights
            </h3>
            <div className="space-y-3">
              {selectedSimulations.length > 1 && (() => {
                const results = selectedSimulations.map(id => simulationResults.get(id)).filter(Boolean) as SimulationResult[];
                if (results.length < 2) return null;

                const bestWealth = results.reduce((best, curr) =>
                  curr.total_wealth_mean > best.total_wealth_mean ? curr : best
                );
                const bestGains = results.reduce((best, curr) =>
                  curr.investment_gains_mean > best.investment_gains_mean ? curr : best
                );
                const shortest = results.reduce((best, curr) =>
                  curr.duration_mean < best.duration_mean ? curr : best
                );

                return (
                  <>
                    <div className="flex items-start gap-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
                      <span className="text-green-600 dark:text-green-400 text-xl">🏆</span>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">
                          Highest Total Wealth
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {bestWealth.scenario_name} achieved €{bestWealth.total_wealth_mean.toFixed(0)}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-start gap-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <span className="text-blue-600 dark:text-blue-400 text-xl">📈</span>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">
                          Best Investment Returns
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {bestGains.scenario_name} gained €{bestGains.investment_gains_mean.toFixed(0)}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-start gap-3 p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg">
                      <span className="text-purple-600 dark:text-purple-400 text-xl">⚡</span>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">
                          Shortest Duration
                        </p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {shortest.scenario_name} completed in {shortest.duration_mean.toFixed(1)} months
                        </p>
                      </div>
                    </div>
                  </>
                );
              })()}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
