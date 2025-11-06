import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { api } from '../services/api';
import type { Scenario, EnvironmentConfig, TrainingConfig, RewardConfig } from '../types';

interface FormErrors {
  [key: string]: string;
}

export default function ScenarioBuilder() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const editMode = searchParams.get('edit');
  
  // Form state
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [environment, setEnvironment] = useState<EnvironmentConfig>({
    income: 2000,
    fixed_expenses: 770,
    variable_expense_mean: 500,
    variable_expense_std: 120,
    inflation: 0.02,
    safety_threshold: 5000,
    max_months: 360,
    initial_cash: 5000,
    risk_tolerance: 0.5,
    investment_return_mean: 0.005,
    investment_return_std: 0.02,
    investment_return_type: 'stochastic',
  });
  
  const [training, setTraining] = useState<TrainingConfig>({
    num_episodes: 1000,
    gamma_low: 0.99,
    gamma_high: 0.99,
    high_period: 12,
    batch_size: 64,
    learning_rate_low: 0.0003,
    learning_rate_high: 0.0003,
  });
  
  const [reward, setReward] = useState<RewardConfig>({
    alpha: 1.0,
    beta: 0.5,
    gamma: 0.3,
    delta: 0.2,
    lambda_: 0.1,
    mu: 0.05,
  });

  // UI state
  const [templates, setTemplates] = useState<Record<string, Partial<Scenario>>>({});
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [errors, setErrors] = useState<FormErrors>({});
  const [saving, setSaving] = useState(false);
  const [loadingScenario, setLoadingScenario] = useState(false);

  // Load templates on mount
  useEffect(() => {
    loadTemplates();
  }, []);

  // Load scenario if in edit mode
  useEffect(() => {
    if (editMode) {
      loadScenario(editMode);
    }
  }, [editMode]);

  const loadTemplates = async () => {
    try {
      const templatesData = await api.getScenarioTemplates();
      setTemplates(templatesData);
    } catch (err) {
      console.error('Failed to load templates:', err);
    }
  };

  const loadScenario = async (scenarioName: string) => {
    try {
      setLoadingScenario(true);
      const scenario = await api.getScenario(scenarioName);
      setName(scenario.name);
      setDescription(scenario.description || '');
      setEnvironment(scenario.environment);
      if (scenario.training) setTraining(scenario.training);
      if (scenario.reward) setReward(scenario.reward);
    } catch (err) {
      console.error('Failed to load scenario:', err);
    } finally {
      setLoadingScenario(false);
    }
  };

  const handleTemplateChange = (templateKey: string) => {
    setSelectedTemplate(templateKey);
    if (templateKey && templates[templateKey]) {
      const template = templates[templateKey];
      if (template.environment) setEnvironment(template.environment);
      if (template.training) setTraining(template.training);
      if (template.reward) setReward(template.reward);
      if (template.description) setDescription(template.description);
    }
  };

  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    // Name validation
    if (!name.trim()) {
      newErrors.name = 'Scenario name is required';
    } else if (name.length < 3) {
      newErrors.name = 'Name must be at least 3 characters';
    }

    // Environment validation
    if (environment.income <= 0) {
      newErrors.income = 'Income must be positive';
    }
    if (environment.fixed_expenses < 0) {
      newErrors.fixed_expenses = 'Fixed expenses cannot be negative';
    }
    if (environment.variable_expense_mean < 0) {
      newErrors.variable_expense_mean = 'Variable expenses cannot be negative';
    }
    if (environment.variable_expense_std < 0) {
      newErrors.variable_expense_std = 'Standard deviation cannot be negative';
    }
    if (environment.inflation < -1 || environment.inflation > 1) {
      newErrors.inflation = 'Inflation must be between -100% and 100%';
    }
    if (environment.safety_threshold < 0) {
      newErrors.safety_threshold = 'Safety threshold cannot be negative';
    }
    if (environment.max_months <= 0) {
      newErrors.max_months = 'Max months must be positive';
    }
    if (environment.initial_cash < 0) {
      newErrors.initial_cash = 'Initial cash cannot be negative';
    }
    if (environment.risk_tolerance < 0 || environment.risk_tolerance > 1) {
      newErrors.risk_tolerance = 'Risk tolerance must be between 0 and 1';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSave = async () => {
    if (!validateForm()) {
      return;
    }

    try {
      setSaving(true);
      const scenario: Omit<Scenario, 'created_at' | 'updated_at'> = {
        name,
        description,
        environment,
        training,
        reward,
      };

      if (editMode) {
        await api.updateScenario(editMode, scenario);
      } else {
        await api.createScenario(scenario);
      }

      navigate('/scenarios');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to save scenario';
      setErrors({ submit: errorMessage });
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    navigate('/scenarios');
  };

  // Calculate preview metrics
  const availableIncome = environment.income - environment.fixed_expenses - environment.variable_expense_mean;
  const availableIncomePct = (availableIncome / environment.income) * 100;
  const riskProfile = environment.risk_tolerance < 0.3 ? 'Conservative' : 
                      environment.risk_tolerance < 0.7 ? 'Balanced' : 'Aggressive';

  if (loadingScenario) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading scenario...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <button
          onClick={handleCancel}
          className="text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
        >
          ← Back
        </button>
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            {editMode ? 'Edit Scenario' : 'Create New Scenario'}
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Configure financial parameters and training settings
          </p>
        </div>
      </div>

      {/* Error message */}
      {errors.submit && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-800 dark:text-red-300">{errors.submit}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form Section */}
        <div className="lg:col-span-2 space-y-6">
          {/* Basic Information */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Basic Information
            </h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Scenario Name *
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={!!editMode}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.name ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  } ${editMode ? 'opacity-50 cursor-not-allowed' : ''}`}
                  placeholder="e.g., young_professional"
                />
                {errors.name && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.name}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  placeholder="Brief description of this scenario"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Load from Template
                </label>
                <select
                  value={selectedTemplate}
                  onChange={(e) => handleTemplateChange(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="">-- Select a template --</option>
                  {Object.keys(templates).map((key) => (
                    <option key={key} value={key}>
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Environment Configuration */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Environment Configuration
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Monthly Income (EUR) *
                </label>
                <input
                  type="number"
                  value={environment.income}
                  onChange={(e) => setEnvironment({ ...environment, income: parseFloat(e.target.value) || 0 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.income ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="0"
                  step="100"
                />
                {errors.income && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.income}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Fixed Expenses (EUR) *
                </label>
                <input
                  type="number"
                  value={environment.fixed_expenses}
                  onChange={(e) => setEnvironment({ ...environment, fixed_expenses: parseFloat(e.target.value) || 0 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.fixed_expenses ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="0"
                  step="50"
                />
                {errors.fixed_expenses && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.fixed_expenses}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Variable Expenses Mean (EUR) *
                </label>
                <input
                  type="number"
                  value={environment.variable_expense_mean}
                  onChange={(e) => setEnvironment({ ...environment, variable_expense_mean: parseFloat(e.target.value) || 0 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.variable_expense_mean ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="0"
                  step="50"
                />
                {errors.variable_expense_mean && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.variable_expense_mean}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Variable Expenses Std Dev (EUR) *
                </label>
                <input
                  type="number"
                  value={environment.variable_expense_std}
                  onChange={(e) => setEnvironment({ ...environment, variable_expense_std: parseFloat(e.target.value) || 0 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.variable_expense_std ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="0"
                  step="10"
                />
                {errors.variable_expense_std && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.variable_expense_std}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Inflation Rate (%) *
                </label>
                <input
                  type="number"
                  value={environment.inflation * 100}
                  onChange={(e) => setEnvironment({ ...environment, inflation: (parseFloat(e.target.value) || 0) / 100 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.inflation ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="-100"
                  max="100"
                  step="0.1"
                />
                {errors.inflation && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.inflation}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Safety Threshold (EUR) *
                </label>
                <input
                  type="number"
                  value={environment.safety_threshold}
                  onChange={(e) => setEnvironment({ ...environment, safety_threshold: parseFloat(e.target.value) || 0 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.safety_threshold ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="0"
                  step="500"
                />
                {errors.safety_threshold && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.safety_threshold}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Max Months *
                </label>
                <input
                  type="number"
                  value={environment.max_months}
                  onChange={(e) => setEnvironment({ ...environment, max_months: parseInt(e.target.value) || 0 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.max_months ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="1"
                  step="12"
                />
                {errors.max_months && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.max_months}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Initial Cash (EUR) *
                </label>
                <input
                  type="number"
                  value={environment.initial_cash}
                  onChange={(e) => setEnvironment({ ...environment, initial_cash: parseFloat(e.target.value) || 0 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.initial_cash ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="0"
                  step="500"
                />
                {errors.initial_cash && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.initial_cash}</p>
                )}
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Risk Tolerance (0-1) *
                </label>
                <input
                  type="number"
                  value={environment.risk_tolerance}
                  onChange={(e) => setEnvironment({ ...environment, risk_tolerance: parseFloat(e.target.value) || 0 })}
                  className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white ${
                    errors.risk_tolerance ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
                  }`}
                  min="0"
                  max="1"
                  step="0.1"
                />
                {errors.risk_tolerance && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.risk_tolerance}</p>
                )}
              </div>
            </div>
          </div>

          {/* Investment Returns Configuration */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Investment Returns
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Return Type
                </label>
                <select
                  value={environment.investment_return_type}
                  onChange={(e) => setEnvironment({ ...environment, investment_return_type: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="stochastic">Stochastic</option>
                  <option value="fixed">Fixed</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Mean Return (% per month)
                </label>
                <input
                  type="number"
                  value={environment.investment_return_mean * 100}
                  onChange={(e) => setEnvironment({ ...environment, investment_return_mean: (parseFloat(e.target.value) || 0) / 100 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  step="0.1"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Std Dev (% per month)
                </label>
                <input
                  type="number"
                  value={environment.investment_return_std * 100}
                  onChange={(e) => setEnvironment({ ...environment, investment_return_std: (parseFloat(e.target.value) || 0) / 100 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  min="0"
                  step="0.1"
                />
              </div>
            </div>
          </div>

          {/* Training Configuration */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Training Configuration
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Number of Episodes
                </label>
                <input
                  type="number"
                  value={training.num_episodes}
                  onChange={(e) => setTraining({ ...training, num_episodes: parseInt(e.target.value) || 1000 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  min="1"
                  step="100"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  High-Level Period (months)
                </label>
                <input
                  type="number"
                  value={training.high_period}
                  onChange={(e) => setTraining({ ...training, high_period: parseInt(e.target.value) || 12 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  min="1"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={training.batch_size}
                  onChange={(e) => setTraining({ ...training, batch_size: parseInt(e.target.value) || 64 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  min="1"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Learning Rate (Low)
                </label>
                <input
                  type="number"
                  value={training.learning_rate_low}
                  onChange={(e) => setTraining({ ...training, learning_rate_low: parseFloat(e.target.value) || 0.0003 })}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                  step="0.0001"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Preview Panel */}
        <div className="lg:col-span-1">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 sticky top-6">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Preview
            </h2>
            
            {/* Monthly Cash Flow */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Monthly Cash Flow
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Income:</span>
                  <span className="text-sm font-medium text-green-600 dark:text-green-400">
                    +{environment.income.toLocaleString()} EUR
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Fixed:</span>
                  <span className="text-sm font-medium text-red-600 dark:text-red-400">
                    -{environment.fixed_expenses.toLocaleString()} EUR
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Variable (avg):</span>
                  <span className="text-sm font-medium text-red-600 dark:text-red-400">
                    -{environment.variable_expense_mean.toLocaleString()} EUR
                  </span>
                </div>
                <div className="border-t border-gray-200 dark:border-gray-700 pt-2 mt-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-900 dark:text-white">Available:</span>
                    <span className={`text-sm font-bold ${availableIncome >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      {availableIncome >= 0 ? '+' : ''}{availableIncome.toLocaleString()} EUR
                    </span>
                  </div>
                  <div className="flex justify-between items-center mt-1">
                    <span className="text-xs text-gray-500 dark:text-gray-500">
                      ({availableIncomePct.toFixed(1)}% of income)
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Risk Profile */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Risk Profile
              </h3>
              <div className="flex items-center space-x-2">
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                  riskProfile === 'Conservative' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300' :
                  riskProfile === 'Balanced' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300' :
                  'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-300'
                }`}>
                  {riskProfile}
                </div>
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  ({(environment.risk_tolerance * 100).toFixed(0)}%)
                </span>
              </div>
            </div>

            {/* Key Metrics */}
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Key Metrics
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Safety Buffer:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {environment.safety_threshold.toLocaleString()} EUR
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Initial Cash:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {environment.initial_cash.toLocaleString()} EUR
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Time Horizon:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {environment.max_months} months ({(environment.max_months / 12).toFixed(1)} years)
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Inflation:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {(environment.inflation * 100).toFixed(1)}% / year
                  </span>
                </div>
              </div>
            </div>

            {/* Investment Returns */}
            <div>
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Investment Returns
              </h3>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Type:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white capitalize">
                    {environment.investment_return_type}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Expected:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {(environment.investment_return_mean * 100).toFixed(2)}% / month
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600 dark:text-gray-400">Volatility:</span>
                  <span className="text-sm font-medium text-gray-900 dark:text-white">
                    {(environment.investment_return_std * 100).toFixed(2)}% / month
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-end space-x-4 pb-6">
        <button
          onClick={handleCancel}
          disabled={saving}
          className="px-6 py-2 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center space-x-2"
        >
          {saving ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              <span>Saving...</span>
            </>
          ) : (
            <span>{editMode ? 'Update Scenario' : 'Save Scenario'}</span>
          )}
        </button>
      </div>
    </div>
  );
}
