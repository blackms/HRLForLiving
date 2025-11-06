import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';
import * as api from '../../services/api';

// Mock axios
vi.mock('axios');
const mockedAxios = axios as any;

describe('API Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  
  describe('listScenarios', () => {
    it('fetches scenarios successfully', async () => {
      const mockScenarios = [
        { name: 'scenario1', income: 2000 },
        { name: 'scenario2', income: 3000 },
      ];
      
      mockedAxios.get.mockResolvedValue({ data: mockScenarios });
      
      const result = await api.listScenarios();
      
      expect(mockedAxios.get).toHaveBeenCalledWith('/api/scenarios');
      expect(result).toEqual(mockScenarios);
    });
    
    it('handles errors gracefully', async () => {
      mockedAxios.get.mockRejectedValue(new Error('Network error'));
      
      await expect(api.listScenarios()).rejects.toThrow('Network error');
    });
  });
  
  describe('getScenario', () => {
    it('fetches a specific scenario', async () => {
      const mockScenario = {
        name: 'test_scenario',
        environment: { income: 2000 },
      };
      
      mockedAxios.get.mockResolvedValue({ data: mockScenario });
      
      const result = await api.getScenario('test_scenario');
      
      expect(mockedAxios.get).toHaveBeenCalledWith('/api/scenarios/test_scenario');
      expect(result).toEqual(mockScenario);
    });
  });
  
  describe('createScenario', () => {
    it('creates a new scenario', async () => {
      const newScenario = {
        name: 'new_scenario',
        environment: { income: 2000 },
      };
      
      const mockResponse = {
        name: 'new_scenario',
        path: '/configs/new_scenario.yaml',
        created_at: '2024-01-01T00:00:00',
      };
      
      mockedAxios.post.mockResolvedValue({ data: mockResponse });
      
      const result = await api.createScenario(newScenario as any);
      
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/scenarios', newScenario);
      expect(result).toEqual(mockResponse);
    });
  });
  
  describe('listModels', () => {
    it('fetches models successfully', async () => {
      const mockModels = {
        models: [
          { name: 'model1', scenario_name: 'scenario1' },
          { name: 'model2', scenario_name: 'scenario2' },
        ],
        total: 2,
      };
      
      mockedAxios.get.mockResolvedValue({ data: mockModels });
      
      const result = await api.listModels();
      
      expect(mockedAxios.get).toHaveBeenCalledWith('/api/models');
      expect(result).toEqual(mockModels);
    });
  });
  
  describe('startTraining', () => {
    it('starts training successfully', async () => {
      const trainingRequest = {
        scenario_name: 'test_scenario',
        num_episodes: 100,
        save_interval: 50,
        eval_episodes: 5,
      };
      
      const mockResponse = {
        message: 'Training started',
        data: { start_time: '2024-01-01T00:00:00' },
      };
      
      mockedAxios.post.mockResolvedValue({ data: mockResponse });
      
      const result = await api.startTraining(trainingRequest as any);
      
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/training/start', trainingRequest);
      expect(result).toEqual(mockResponse);
    });
  });
  
  describe('getTrainingStatus', () => {
    it('fetches training status', async () => {
      const mockStatus = {
        is_training: true,
        scenario_name: 'test_scenario',
        current_episode: 50,
        total_episodes: 100,
      };
      
      mockedAxios.get.mockResolvedValue({ data: mockStatus });
      
      const result = await api.getTrainingStatus();
      
      expect(mockedAxios.get).toHaveBeenCalledWith('/api/training/status');
      expect(result).toEqual(mockStatus);
    });
  });
  
  describe('runSimulation', () => {
    it('runs simulation successfully', async () => {
      const simulationRequest = {
        model_name: 'test_model',
        scenario_name: 'test_scenario',
        num_episodes: 10,
      };
      
      const mockResponse = {
        status: 'completed',
        simulation_id: 'sim_123',
        results: {},
      };
      
      mockedAxios.post.mockResolvedValue({ data: mockResponse });
      
      const result = await api.runSimulation(simulationRequest as any);
      
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/simulation/run', simulationRequest);
      expect(result).toEqual(mockResponse);
    });
  });
  
  describe('getSimulationResults', () => {
    it('fetches simulation results', async () => {
      const mockResults = {
        simulation_id: 'sim_123',
        scenario_name: 'test_scenario',
        model_name: 'test_model',
        episodes: [],
      };
      
      mockedAxios.get.mockResolvedValue({ data: mockResults });
      
      const result = await api.getSimulationResults('sim_123');
      
      expect(mockedAxios.get).toHaveBeenCalledWith('/api/simulation/results/sim_123');
      expect(result).toEqual(mockResults);
    });
  });
});
