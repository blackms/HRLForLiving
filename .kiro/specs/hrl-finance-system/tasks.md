# Implementation Plan

- [x] 1. Set up project structure and core data models
  - Create directory structure: `src/environment/`, `src/agents/`, `src/training/`, `src/utils/`, `tests/`
  - Create `__init__.py` files for all packages
  - Set up `requirements.txt` with dependencies: gymnasium, numpy, stable-baselines3, torch, pyyaml
  - Create configuration dataclasses in `src/utils/config.py` for EnvironmentConfig, TrainingConfig, RewardConfig
  - Create Transition dataclass in `src/utils/data_models.py`
  - Create BehavioralProfile enum in `src/utils/config.py`
  - _Requirements: 6.2, 6.3_

- [x] 2. Implement BudgetEnv (Gymnasium environment)
  - [x] 2.1 Create base environment class structure
    - Implement `BudgetEnv` class inheriting from `gym.Env` in `src/environment/budget_env.py`
    - Define observation space as Box(7) with appropriate bounds
    - Define action space as Box(3) with bounds [0, 1]
    - Implement `__init__` method accepting EnvironmentConfig
    - _Requirements: 1.1, 7.1, 7.2, 7.3_
  
  - [x] 2.2 Implement state management and reset logic
    - Implement `reset()` method to initialize cash balance, month counter, and return initial state
    - Implement `_get_state()` helper to construct observation vector
    - Initialize income, fixed expenses, inflation from config
    - _Requirements: 1.5, 7.5_
  
  - [x] 2.3 Implement action processing and normalization
    - Implement `_normalize_action()` using softmax to ensure action sums to 1
    - Add validation to handle negative or invalid action values
    - _Requirements: 2.5_
  
  - [x] 2.4 Implement expense simulation
    - Implement `_sample_variable_expense()` using normal distribution with configured mean and std
    - Apply inflation adjustments to expenses in step function
    - _Requirements: 1.2, 1.3_
  
  - [x] 2.5 Implement step function and episode termination
    - Implement `step(action)` method to process allocation, update cash balance, compute expenses
    - Calculate new cash balance: `cash = cash + income - fixed - variable - invest_amount`
    - Implement termination logic for negative cash or max months reached
    - Return tuple (next_state, reward, done, info) per Gymnasium interface
    - _Requirements: 1.4, 7.4_
  
  - [x] 2.6 Write unit tests for BudgetEnv
    - Test state initialization and reset functionality
    - Test action normalization with various inputs
    - Test episode termination conditions
    - Test variable expense sampling distribution
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 3. Implement Reward Engine
  - [ ] 3.1 Create RewardEngine class
    - Implement `RewardEngine` class in `src/environment/reward_engine.py`
    - Initialize with RewardConfig containing coefficients (α, β, γ, δ, λ, μ)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [ ] 3.2 Implement low-level reward computation
    - Implement `compute_low_level_reward(action, state, next_state)` method
    - Calculate investment reward: `α * invest_amount`
    - Calculate stability penalty: `β * max(0, threshold - cash)`
    - Calculate overspend penalty: `γ * overspend`
    - Calculate debt penalty: `δ * abs(min(0, cash))`
    - Return combined reward
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [ ] 3.3 Implement high-level reward computation
    - Implement `compute_high_level_reward(episode_history)` method
    - Aggregate low-level rewards over strategic period
    - Add wealth change term: `λ * Δwealth`
    - Add stability bonus for consistent positive balance
    - _Requirements: 3.5_
  
  - [ ] 3.4 Write unit tests for RewardEngine
    - Test low-level reward with various scenarios (high investment, low cash, overspend, debt)
    - Test high-level reward aggregation
    - Test reward coefficient effects
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 4. Integrate RewardEngine with BudgetEnv
  - Modify `BudgetEnv.__init__` to accept RewardConfig and create RewardEngine instance
  - Update `BudgetEnv.step()` to use `reward_engine.compute_low_level_reward()` instead of internal calculation
  - Pass action, current state, and next state to reward engine
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 5. Implement Low-Level Agent (Budget Executor)
  - [ ] 5.1 Create BudgetExecutor class structure
    - Implement `BudgetExecutor` class in `src/agents/budget_executor.py`
    - Initialize with neural network policy (use Stable-Baselines3 PPO)
    - Accept TrainingConfig for hyperparameters
    - _Requirements: 2.4, 2.5, 4.1_
  
  - [ ] 5.2 Implement action generation
    - Implement `act(state, goal)` method that concatenates state and goal vectors
    - Pass concatenated input (10-dimensional) to PPO policy
    - Return action vector [invest, save, consume]
    - _Requirements: 2.4, 2.5_
  
  - [ ] 5.3 Implement learning method
    - Implement `learn(transitions)` method to update PPO policy
    - Apply discount factor γ_low = 0.95
    - Return training metrics (loss, policy entropy)
    - _Requirements: 4.1, 4.3_
  
  - [ ] 5.4 Write unit tests for BudgetExecutor
    - Test action generation within valid ranges
    - Test input concatenation (state + goal)
    - Test policy update mechanics
    - _Requirements: 2.4, 2.5, 4.1, 4.3_

- [ ] 6. Implement High-Level Agent (Financial Strategist)
  - [ ] 6.1 Create FinancialStrategist class structure
    - Implement `FinancialStrategist` class in `src/agents/financial_strategist.py`
    - Initialize with neural network policy (custom implementation or Option-Critic)
    - Accept TrainingConfig for hyperparameters
    - _Requirements: 2.1, 2.2, 2.3, 4.2_
  
  - [ ] 6.2 Implement state aggregation
    - Implement `aggregate_state(history)` method to compute macro features
    - Calculate average cash over last N months
    - Calculate average investment return
    - Calculate spending trend
    - Return 5-dimensional aggregated state
    - _Requirements: 2.2_
  
  - [ ] 6.3 Implement goal generation
    - Implement `select_goal(state)` method to generate goal vector
    - Output 3-dimensional goal: [target_invest_ratio, safety_buffer, aggressiveness]
    - Ensure values are within valid ranges
    - _Requirements: 2.1, 2.3_
  
  - [ ] 6.4 Implement learning method
    - Implement `learn(transitions)` method to update high-level policy
    - Apply discount factor γ_high = 0.99
    - Use HIRO or Option-Critic algorithm
    - Return training metrics
    - _Requirements: 4.2, 4.4_
  
  - [ ] 6.5 Write unit tests for FinancialStrategist
    - Test goal generation within valid ranges
    - Test state aggregation from history
    - Test policy update mechanics
    - _Requirements: 2.1, 2.2, 2.3, 4.2, 4.4_

- [ ] 7. Implement Training Orchestrator
  - [ ] 7.1 Create HRLTrainer class structure
    - Implement `HRLTrainer` class in `src/training/hrl_trainer.py`
    - Initialize with BudgetEnv, FinancialStrategist, BudgetExecutor, RewardEngine, TrainingConfig
    - Set up episode buffer for storing transitions
    - _Requirements: 4.5, 4.6_
  
  - [ ] 7.2 Implement main training loop
    - Implement `train(num_episodes)` method
    - For each episode: reset environment, get initial state
    - Generate initial goal from high-level agent
    - Execute monthly steps with low-level agent
    - Store transitions in episode buffer
    - _Requirements: 4.5_
  
  - [ ] 7.3 Implement policy update coordination
    - Update low-level policy when buffer reaches batch size
    - Every high_period steps: compute high-level reward, update high-level policy, generate new goal
    - Handle episode termination and cleanup
    - Return training history with metrics
    - _Requirements: 4.6_
  
  - [ ] 7.4 Implement evaluation method
    - Implement `evaluate(num_episodes)` method to run episodes without learning
    - Collect performance metrics during evaluation
    - Return evaluation results
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 7.5 Write integration tests for training loop
    - Test complete episode execution
    - Test high-level/low-level coordination
    - Test policy updates occur correctly
    - _Requirements: 4.5, 4.6_

- [ ] 8. Implement Analytics Module
  - [ ] 8.1 Create AnalyticsModule class
    - Implement `AnalyticsModule` class in `src/utils/analytics.py`
    - Initialize metric trackers (lists for states, actions, rewards, cash balances)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 8.2 Implement data recording
    - Implement `record_step(state, action, reward)` method to store step data
    - Track cash balance history, action history, reward history
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 8.3 Implement metric computation
    - Implement `compute_episode_metrics()` method
    - Calculate cumulative wealth growth (total invested)
    - Calculate cash stability index (% months with positive balance)
    - Calculate Sharpe-like ratio (mean return / std balance)
    - Calculate goal adherence (mean absolute difference)
    - Calculate policy stability (variance of actions)
    - Return metrics dictionary
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 8.4 Implement reset functionality
    - Implement `reset()` method to clear episode data
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 8.5 Write unit tests for AnalyticsModule
    - Test metric computation with known data
    - Test reset functionality
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 9. Integrate Analytics Module with Training Orchestrator
  - Modify `HRLTrainer.__init__` to create AnalyticsModule instance
  - Update training loop to call `analytics.record_step()` after each environment step
  - Call `analytics.compute_episode_metrics()` at episode end
  - Store metrics in training history
  - Call `analytics.reset()` at start of each episode
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 10. Implement Configuration Manager
  - [ ] 10.1 Create configuration loading utilities
    - Implement `load_config(yaml_path)` function in `src/utils/config_manager.py`
    - Parse YAML configuration file
    - Create EnvironmentConfig, TrainingConfig, RewardConfig instances
    - _Requirements: 6.2_
  
  - [ ] 10.2 Implement behavioral profile loading
    - Implement `load_behavioral_profile(profile_name)` function
    - Support "conservative", "balanced", "aggressive" profiles
    - Return appropriate configuration with adjusted risk tolerance and reward coefficients
    - _Requirements: 6.2, 6.3_
  
  - [ ] 10.3 Implement configuration validation
    - Validate all required parameters are present
    - Check parameter ranges (e.g., income > 0, gamma in [0, 1])
    - Raise descriptive errors for invalid configurations
    - _Requirements: 6.1, 6.4, 6.5_
  
  - [ ] 10.4 Write configuration tests
    - Test loading different behavioral profiles
    - Test parameter validation
    - Test configuration overrides
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 11. Create main training script
  - Create `train.py` script in project root
  - Parse command-line arguments for config path and behavioral profile
  - Load configuration using Configuration Manager
  - Instantiate BudgetEnv, RewardEngine, agents, and HRLTrainer
  - Execute training with specified number of episodes
  - Save trained models and training history
  - Print final evaluation metrics
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 12. Create evaluation script
  - Create `evaluate.py` script in project root
  - Load trained models from checkpoint
  - Run evaluation episodes without learning
  - Compute and display all performance metrics
  - Generate visualization of episode trajectories (cash balance, allocations over time)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 13. Create example configuration files
  - Create `configs/conservative.yaml` with conservative behavioral profile parameters
  - Create `configs/balanced.yaml` with balanced behavioral profile parameters
  - Create `configs/aggressive.yaml` with aggressive behavioral profile parameters
  - Include all required parameters: environment, training, and reward configs
  - _Requirements: 6.2, 6.3_

- [ ] 14. Create validation tests
  - [ ] 14.1 Implement sanity check tests
    - Test that random policy does not accumulate wealth
    - Test that conservative profile maintains higher cash balance
    - Test that aggressive profile invests more
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 6.2, 6.3_
  
  - [ ] 14.2 Implement edge case tests
    - Test very low income scenarios
    - Test very high expense scenarios
    - Test extreme inflation rates
    - Test maximum episode length
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 15. Add logging and monitoring
  - Integrate TensorBoard or Weights & Biases for experiment tracking
  - Log training curves (rewards, losses) during training
  - Log episode metrics (wealth, stability) after each episode
  - Log action distributions and goal adherence
  - Save hyperparameters with each experiment
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 16. Implement checkpointing
  - Add model saving every N episodes in HRLTrainer
  - Save both high-level and low-level agent models
  - Save configuration with each checkpoint
  - Keep best model based on evaluation performance
  - Implement checkpoint loading for resume functionality
  - _Requirements: 4.1, 4.2_

- [ ] 17. Create README and documentation
  - Write README.md with project overview, installation instructions, usage examples
  - Document configuration parameters and their effects
  - Provide example commands for training and evaluation
  - Document behavioral profiles and when to use each
  - _Requirements: 6.2, 6.3_
