import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Dashboard from '../../pages/Dashboard';
import * as api from '../../services/api';

// Mock the API module
vi.mock('../../services/api');

const mockApi = api as any;

describe('Dashboard Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  
  it('renders dashboard with loading state', () => {
    mockApi.listScenarios.mockReturnValue(new Promise(() => {})); // Never resolves
    mockApi.listModels.mockReturnValue(new Promise(() => {}));
    mockApi.getSimulationHistory.mockReturnValue(new Promise(() => {}));
    
    render(
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    );
    
    expect(screen.getByText(/dashboard/i)).toBeInTheDocument();
  });
  
  it('displays scenarios when loaded', async () => {
    const mockScenarios = [
      {
        name: 'test_scenario',
        description: 'Test scenario',
        income: 2000,
        fixed_expenses: 800,
        variable_expenses: 400,
        available_monthly: 800,
        available_pct: 40,
        risk_tolerance: 0.5,
        updated_at: '2024-01-01T00:00:00',
        size: 1024,
      },
    ];
    
    mockApi.listScenarios.mockResolvedValue(mockScenarios);
    mockApi.listModels.mockResolvedValue({ models: [], total: 0 });
    mockApi.getSimulationHistory.mockResolvedValue({ simulations: [], total: 0 });
    
    render(
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    );
    
    await waitFor(() => {
      expect(screen.getByText('test_scenario')).toBeInTheDocument();
    });
  });
  
  it('displays models when loaded', async () => {
    const mockModels = {
      models: [
        {
          name: 'test_model',
          scenario_name: 'test_scenario',
          episodes: 100,
          final_reward: 150,
          final_stability: 0.95,
          created_at: '2024-01-01T00:00:00',
        },
      ],
      total: 1,
    };
    
    mockApi.listScenarios.mockResolvedValue([]);
    mockApi.listModels.mockResolvedValue(mockModels);
    mockApi.getSimulationHistory.mockResolvedValue({ simulations: [], total: 0 });
    
    render(
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    );
    
    await waitFor(() => {
      expect(screen.getByText('test_model')).toBeInTheDocument();
    });
  });
  
  it('handles API errors gracefully', async () => {
    mockApi.listScenarios.mockRejectedValue(new Error('API Error'));
    mockApi.listModels.mockRejectedValue(new Error('API Error'));
    mockApi.getSimulationHistory.mockRejectedValue(new Error('API Error'));
    
    render(
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    );
    
    await waitFor(() => {
      // Should show error state or empty state
      expect(screen.queryByText(/error/i) || screen.queryByText(/no scenarios/i)).toBeTruthy();
    });
  });
});
