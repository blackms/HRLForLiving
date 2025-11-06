// Core type definitions for the HRL Finance UI

export interface EnvironmentConfig {
  income: number;
  fixed_expenses: number;
  variable_expense_mean: number;
  variable_expense_std: number;
  inflation: number;
  safety_threshold: number;
  max_months: number;
  initial_cash: number;
  risk_tolerance: number;
  investment_return_mean: number;
  investment_return_std: number;
  investment_return_type: string;
}

export interface TrainingConfig {
  num_episodes: number;
  gamma_low?: number;
  gamma_high?: number;
  high_period?: number;
  batch_size?: number;
  learning_rate_low?: number;
  learning_rate_high?: number;
}

export interface TrainingRequest {
  scenario_name: string;
  num_episodes: number;
  save_interval: number;
  eval_episodes: number;
  seed?: number;
}

export interface RewardConfig {
  alpha?: number;
  beta?: number;
  gamma?: number;
  delta?: number;
  lambda_?: number;
  mu?: number;
}

export interface Scenario {
  name: string;
  description?: string;
  environment: EnvironmentConfig;
  training?: TrainingConfig;
  reward?: RewardConfig;
  created_at?: string;
  updated_at?: string;
}

export interface ScenarioSummary {
  name: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
  income: number;
  fixed_expenses: number;
  available_income_pct: number;
  risk_tolerance: number;
}

export interface ModelSummary {
  name: string;
  scenario_name: string;
  size_mb: number;
  trained_at: string;
  has_metadata: boolean;
  episodes?: number;
  income?: number;
  risk_tolerance?: number;
  final_reward?: number;
  avg_reward?: number;
  max_reward?: number;
  final_duration?: number;
  final_cash?: number;
  final_invested?: number;
}

export interface ModelDetail {
  name: string;
  scenario_name: string;
  high_agent_path: string;
  low_agent_path: string;
  size_mb: number;
  trained_at: string;
  has_metadata: boolean;
  has_history: boolean;
  episodes?: number;
  metadata?: Record<string, any>;
  environment_config?: EnvironmentConfig;
  training_config?: TrainingConfig;
  reward_config?: RewardConfig;
  training_history?: Record<string, any>;
  final_metrics?: Record<string, any>;
}

export interface TrainingProgress {
  episode: number;
  total_episodes: number;
  avg_reward: number;
  avg_duration: number;
  avg_cash: number;
  avg_invested: number;
  stability: number;
  goal_adherence: number;
  elapsed_time: number;
}

export interface TrainingStatus {
  is_training: boolean;
  scenario_name?: string;
  current_episode?: number;
  total_episodes?: number;
  progress?: TrainingProgress;
}

export interface EpisodeData {
  episode_num: number;
  duration: number;
  final_cash: number;
  final_invested: number;
  final_portfolio_value: number;
  total_wealth: number;
  investment_gains: number;
  avg_invest_pct: number;
  avg_save_pct: number;
  avg_consume_pct: number;
  months?: number[];
  cash?: number[];
  invested?: number[];
  portfolio_value?: number[];
  actions?: number[][];
}

export interface SimulationResult {
  simulation_id: string;
  scenario_name: string;
  model_name: string;
  num_episodes: number;
  timestamp: string;
  duration_mean: number;
  duration_std: number;
  final_cash_mean: number;
  final_invested_mean: number;
  final_portfolio_mean: number;
  total_wealth_mean: number;
  total_wealth_std: number;
  investment_gains_mean: number;
  avg_invest_pct: number;
  avg_save_pct: number;
  avg_consume_pct: number;
  episodes: EpisodeData[];
}

export interface SimulationRequest {
  model_name: string;
  scenario_name: string;
  num_episodes: number;
  seed?: number;
}

export interface ReportRequest {
  simulation_id: string;
  report_type: 'pdf' | 'html';
  include_sections?: string[];
  title?: string;
}

export interface Report {
  report_id: string;
  simulation_id: string;
  report_type: 'pdf' | 'html';
  title: string;
  generated_at: string;
  file_path: string;
  file_size_kb: number;
  sections: string[];
  status?: string;
  message?: string;
}

export type Theme = 'light' | 'dark';
