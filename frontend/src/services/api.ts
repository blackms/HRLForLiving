import axios from 'axios';
import type { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  Scenario,
  TrainingStatus,
  SimulationResult,
  Report,
  TrainingConfig,
  EnvironmentConfig,
  RewardConfig,
} from '../types';

// API Response types
interface ScenarioSummary {
  name: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
}

interface ScenarioDetail extends ScenarioSummary {
  environment: EnvironmentConfig;
  training: TrainingConfig;
  reward: RewardConfig;
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

interface ModelDetail extends ModelSummary {
  metadata?: Record<string, any>;
  history?: Record<string, any>;
}

interface SimulationHistoryItem {
  id: string;
  scenario_name: string;
  model_name: string;
  num_episodes: number;
  created_at: string;
}

interface TemplateItem {
  name: string;
  display_name: string;
  description: string;
  environment: EnvironmentConfig;
  training: TrainingConfig;
  reward: RewardConfig;
}



// API Error type
export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, any>;
  timestamp?: string;
}

class ApiClient {
  private client: AxiosInstance;
  private maxRetries = 3;
  private retryDelay = 1000;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Add timestamp to requests for debugging
        config.headers['X-Request-Time'] = new Date().toISOString();
        return config;
      },
      (error: AxiosError) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError<ApiError>) => {
        const config = error.config;

        // Retry logic for network errors and 5xx errors
        if (config && this.shouldRetry(error)) {
          const retryCount = (config as any).__retryCount || 0;

          if (retryCount < this.maxRetries) {
            (config as any).__retryCount = retryCount + 1;

            // Exponential backoff
            const delay = this.retryDelay * Math.pow(2, retryCount);
            await new Promise(resolve => setTimeout(resolve, delay));

            return this.client(config);
          }
        }

        return Promise.reject(this.handleError(error));
      }
    );
  }

  private shouldRetry(error: AxiosError): boolean {
    if (!error.response) {
      // Network error
      return true;
    }

    const status = error.response.status;
    // Retry on 5xx errors and 429 (rate limit)
    return status >= 500 || status === 429;
  }

  private handleError(error: AxiosError<ApiError>): Error {
    if (error.response?.data) {
      const apiError = error.response.data;
      return new Error(apiError.message || apiError.error || 'An error occurred');
    }

    if (error.request) {
      return new Error('No response from server. Please check your connection.');
    }

    return new Error(error.message || 'An unexpected error occurred');
  }

  // Scenarios API
  async listScenarios(): Promise<ScenarioSummary[]> {
    const response = await this.client.get<{ scenarios: ScenarioSummary[] }>('/api/scenarios');
    return response.data.scenarios;
  }

  async getScenario(name: string): Promise<ScenarioDetail> {
    const response = await this.client.get<ScenarioDetail>(`/api/scenarios/${name}`);
    return response.data;
  }

  async createScenario(scenario: Omit<Scenario, 'created_at' | 'updated_at'>): Promise<{ name: string; message: string }> {
    const response = await this.client.post<{ name: string; message: string }>('/api/scenarios', scenario);
    return response.data;
  }

  async updateScenario(name: string, scenario: Partial<Scenario>): Promise<{ name: string; message: string }> {
    const response = await this.client.put<{ name: string; message: string }>(`/api/scenarios/${name}`, scenario);
    return response.data;
  }

  async deleteScenario(name: string): Promise<{ message: string }> {
    const response = await this.client.delete<{ message: string }>(`/api/scenarios/${name}`);
    return response.data;
  }

  async getScenarioTemplates(): Promise<Record<string, Partial<Scenario>>> {
    const response = await this.client.get<TemplateItem[]>('/api/scenarios/templates');
    // Convert array to dictionary keyed by name
    const templates: Record<string, Partial<Scenario>> = {};
    response.data.forEach((template) => {
      templates[template.name] = {
        name: template.name,
        description: template.description,
        environment: template.environment,
        training: template.training,
        reward: template.reward,
      };
    });
    return templates;
  }

  // Training API
  async startTraining(request: {
    scenario_name: string;
    num_episodes?: number;
    save_interval?: number;
    eval_episodes?: number;
    seed?: number;
  }): Promise<{ message: string; scenario_name: string }> {
    const response = await this.client.post<{ message: string; scenario_name: string }>('/api/training/start', request);
    return response.data;
  }

  async stopTraining(): Promise<{ message: string }> {
    const response = await this.client.post<{ message: string }>('/api/training/stop');
    return response.data;
  }

  async getTrainingStatus(): Promise<TrainingStatus> {
    const response = await this.client.get<TrainingStatus>('/api/training/status');
    return response.data;
  }

  // Simulation API
  async runSimulation(request: {
    model_name: string;
    scenario_name: string;
    num_episodes?: number;
    seed?: number;
  }): Promise<{ message: string; simulation_id: string }> {
    const response = await this.client.post<{ message: string; simulation_id: string }>('/api/simulation/run', request);
    return response.data;
  }

  async getSimulationResults(id: string): Promise<SimulationResult> {
    const response = await this.client.get<SimulationResult>(`/api/simulation/results/${id}`);
    return response.data;
  }

  async getSimulationHistory(): Promise<SimulationHistoryItem[]> {
    const response = await this.client.get<{ simulations: SimulationHistoryItem[] }>('/api/simulation/history');
    return response.data.simulations;
  }

  // Models API
  async listModels(): Promise<ModelSummary[]> {
    const response = await this.client.get<{ models: ModelSummary[] }>('/api/models');
    return response.data.models;
  }

  async getModel(name: string): Promise<ModelDetail> {
    const response = await this.client.get<ModelDetail>(`/api/models/${name}`);
    return response.data;
  }

  async deleteModel(name: string): Promise<{ message: string }> {
    const response = await this.client.delete<{ message: string }>(`/api/models/${name}`);
    return response.data;
  }

  // Reports API
  async generateReport(request: {
    simulation_id: string;
    report_type: 'pdf' | 'html';
    include_sections?: string[];
    title?: string;
  }): Promise<{ report_id: string; message: string }> {
    const response = await this.client.post<{ report_id: string; message: string }>('/api/reports/generate', request);
    return response.data;
  }

  async getReport(id: string): Promise<Blob> {
    const response = await this.client.get(`/api/reports/${id}`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async listReports(): Promise<Report[]> {
    const response = await this.client.get<{ reports: Report[] }>('/api/reports/list');
    return response.data.reports;
  }

  // Health check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.client.get<{ status: string; timestamp: string }>('/health');
    return response.data;
  }
}

// Export singleton instance
export const api = new ApiClient();

// Export class for testing
export { ApiClient };
