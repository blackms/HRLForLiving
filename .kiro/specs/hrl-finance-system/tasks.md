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

- [x] 3. Implement Reward Engine
  - [x] 3.1 Create RewardEngine class
    - Implement `RewardEngine` class in `src/environment/reward_engine.py`
    - Initialize with RewardConfig containing coefficients (α, β, γ, δ, λ, μ)
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [x] 3.2 Implement low-level reward computation
    - Implement `compute_low_level_reward(action, state, next_state)` method
    - Calculate investment reward: `α * invest_amount`
    - Calculate stability penalty: `β * max(0, threshold - cash)`
    - Calculate overspend penalty: `γ * overspend`
    - Calculate debt penalty: `δ * abs(min(0, cash))`
    - Return combined reward
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  
  - [x] 3.3 Implement high-level reward computation
    - Implement `compute_high_level_reward(episode_history)` method
    - Aggregate low-level rewards over strategic period
    - Add wealth change term: `λ * Δwealth`
    - Add stability bonus for consistent positive balance
    - _Requirements: 3.5_
  
  - [x] 3.4 Write unit tests for RewardEngine
    - Test low-level reward with various scenarios (high investment, low cash, overspend, debt)
    - Test high-level reward aggregation
    - Test reward coefficient effects
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 4. Integrate RewardEngine with BudgetEnv
  - Modify `BudgetEnv.__init__` to accept RewardConfig and create RewardEngine instance
  - Update `BudgetEnv.step()` to use `reward_engine.compute_low_level_reward()` instead of internal calculation
  - Pass action, current state, and next state to reward engine
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 5. Implement Low-Level Agent (Budget Executor) ✅ COMPLETE
  - [x] 5.1 Create BudgetExecutor class structure
    - ✅ Implemented `BudgetExecutor` class in `src/agents/budget_executor.py`
    - ✅ Custom PolicyNetwork with [128, 128] hidden layers and softmax output
    - ✅ Accepts TrainingConfig for hyperparameters
    - ✅ 10-dimensional observation space (state + goal)
    - ✅ 3-dimensional action space with automatic normalization
    - _Requirements: 2.4, 2.5, 4.1_
  
  - [x] 5.2 Implement action generation
    - ✅ Implemented `act(state, goal, deterministic)` method
    - ✅ Concatenates state (7D) and goal (3D) vectors to create 10D input
    - ✅ Passes through policy network to generate action probabilities
    - ✅ Returns normalized action vector [invest, save, consume] that sums to 1
    - ✅ Input validation for state and goal dimensions
    - ✅ Deterministic mode support for evaluation
    - _Requirements: 2.4, 2.5_
  
  - [x] 5.3 Implement learning method
    - ✅ Implemented `learn(transitions)` method with simplified policy gradient
    - ✅ Applies discount factor γ_low = 0.95 for temporal credit assignment
    - ✅ Return normalization for stable training
    - ✅ Entropy bonus (0.01 coefficient) for exploration
    - ✅ Returns training metrics (loss, policy_entropy, n_updates)
    - ✅ Handles empty transitions and terminal states correctly
    - _Requirements: 4.1, 4.3_
  
  - [x] 5.4 Write unit tests for BudgetExecutor
    - ✅ Test initialization and configuration
    - ✅ Test action generation (basic, deterministic, input concatenation)
    - ✅ Test action normalization (including negative values)
    - ✅ Test action within valid ranges (multiple iterations)
    - ✅ Test input validation (invalid state/goal dimensions)
    - ✅ Test learning (basic, empty, single transition, terminal states)
    - ✅ Test policy update mechanics
    - ✅ Test goal influence on actions
    - _Requirements: 2.4, 2.5, 4.1, 4.3_
  
  - [x] 5.5 Additional features implemented
    - ✅ Model save/load functionality for checkpointing
    - ✅ Training metrics tracking (loss, policy entropy)
    - ✅ Robust error handling and validation

- [x] 6. Implement High-Level Agent (Financial Strategist) ✅ COMPLETE
  - [x] 6.1 Create FinancialStrategist class structure
    - ✅ Implemented `FinancialStrategist` class in `src/agents/financial_strategist.py`
    - ✅ Custom StrategistNetwork with [64, 64] hidden layers
    - ✅ Accepts TrainingConfig for hyperparameters
    - ✅ 5-dimensional aggregated state input
    - ✅ 3-dimensional goal output
    - _Requirements: 2.1, 2.2, 2.3, 4.2_
  
  - [x] 6.2 Implement state aggregation
    - ✅ Implemented `aggregate_state(history)` method to compute macro features
    - ✅ Calculates average cash over last N months
    - ✅ Calculates average investment return from cash changes
    - ✅ Calculates spending trend using linear fit
    - ✅ Returns 5-dimensional aggregated state: [avg_cash, avg_investment_return, spending_trend, current_wealth, months_elapsed]
    - _Requirements: 2.2_
  
  - [x] 6.3 Implement goal generation
    - ✅ Implemented `select_goal(state)` method to generate goal vector
    - ✅ Outputs 3-dimensional goal: [target_invest_ratio, safety_buffer, aggressiveness]
    - ✅ Ensures values are within valid ranges using sigmoid/softplus constraints
    - ✅ target_invest_ratio: [0, 1] using sigmoid
    - ✅ safety_buffer: [0, ∞) using softplus
    - ✅ aggressiveness: [0, 1] using sigmoid
    - _Requirements: 2.1, 2.3_
  
  - [x] 6.4 Implement learning method
    - ✅ Implemented `learn(transitions)` method to update high-level policy
    - ✅ Applies discount factor γ_high = 0.99
    - ✅ Uses simplified HIRO-style algorithm
    - ✅ Return normalization for stable training
    - ✅ Gradient clipping (max_norm=1.0) for stability
    - ✅ Returns training metrics (loss, policy_entropy, n_updates)
    - _Requirements: 4.2, 4.4_
  
  - [x] 6.5 Write unit tests for FinancialStrategist
    - ✅ Test goal generation within valid ranges
    - ✅ Test state aggregation from history (basic, empty, single state, averages)
    - ✅ Test policy update mechanics
    - ✅ Test deterministic goal generation
    - ✅ Test invalid state dimension handling
    - ✅ Test learning with terminal states
    - ✅ Test different states produce different goals
    - _Requirements: 2.1, 2.2, 2.3, 4.2, 4.4_

- [x] 7. Implement Training Orchestrator
  - [x] 7.1 Create HRLTrainer class structure ✅ COMPLETE
    - ✅ Implemented `HRLTrainer` class in `src/training/hrl_trainer.py`
    - ✅ Initialize with BudgetEnv, FinancialStrategist, BudgetExecutor, RewardEngine, TrainingConfig
    - ✅ Set up episode buffer for storing transitions
    - ✅ Set up state history for high-level agent aggregation
    - ✅ Initialize training metrics tracking (episode_rewards, episode_lengths, cash_balances, total_invested, losses)
    - _Requirements: 4.5, 4.6_
  
  - [x] 7.2 Implement main training loop ✅ COMPLETE
    - ✅ Implemented `train(num_episodes)` method
    - ✅ For each episode: reset environment, get initial state
    - ✅ Generate initial goal from high-level agent
    - ✅ Execute monthly steps with low-level agent
    - ✅ Store transitions in episode buffer
    - ✅ Track episode metrics (reward, length, cash balance, investments)
    - ✅ Print progress every 100 episodes
    - _Requirements: 4.5_
  
  - [x] 7.3 Implement policy update coordination ✅ COMPLETE
    - ✅ Update low-level policy when buffer reaches batch size
    - ✅ Every high_period steps: compute high-level reward, update high-level policy, generate new goal
    - ✅ Handle episode termination with final high-level and low-level updates
    - ✅ Return training history with all metrics
    - ✅ Coordinate high-level transitions and strategic goal updates
    - _Requirements: 4.6_
  
  - [x] 7.4 Implement evaluation method ✅ COMPLETE
    - ✅ Implemented `evaluate(num_episodes)` method to run episodes without learning
    - ✅ Deterministic policy execution for consistent evaluation
    - ✅ Collect all 5 performance metrics (wealth growth, stability, Sharpe ratio, goal adherence, policy stability)
    - ✅ Return comprehensive evaluation summary with mean/std statistics
    - ✅ Detailed per-episode results included in output
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ] 7.5 Write integration tests for training loop
    - Test complete episode execution
    - Test high-level/low-level coordination
    - Test policy updates occur correctly
    - Test analytics integration in training loop
    - _Requirements: 4.5, 4.6_

- [x] 8. Implement Analytics Module ✅ COMPLETE
  - [x] 8.1 Create AnalyticsModule class
    - ✅ Implemented `AnalyticsModule` class in `src/utils/analytics.py`
    - ✅ Initialize metric trackers (states, actions, rewards, cash_balances, goals, invested_amounts)
    - ✅ Comprehensive docstrings explaining all metrics
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 8.2 Implement data recording
    - ✅ Implemented `record_step(state, action, reward, goal, invested_amount)` method
    - ✅ Track cash balance history (extracted from state[3])
    - ✅ Track action history with copy() for safety
    - ✅ Track reward history
    - ✅ Optional goal and invested_amount tracking
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 8.3 Implement metric computation
    - ✅ Implemented `compute_episode_metrics()` method returning Dict[str, float]
    - ✅ Calculate cumulative wealth growth (sum of invested_amounts)
    - ✅ Calculate cash stability index (positive_months / total_months)
    - ✅ Calculate Sharpe-like ratio (mean_balance / std_balance)
    - ✅ Calculate goal adherence (mean absolute difference between goal[0] and action[0])
    - ✅ Calculate policy stability (mean variance of actions over time)
    - ✅ Return comprehensive metrics dictionary
    - ✅ Handle edge cases (empty data, single data point)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 8.4 Implement reset functionality
    - ✅ Implemented `reset()` method using .clear() on all lists
    - ✅ Clears states, actions, rewards, cash_balances, goals, invested_amounts
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [x] 8.5 Write unit tests for AnalyticsModule ✅ COMPLETE
    - ✅ Test initialization
    - ✅ Test basic step recording
    - ✅ Test step recording with goals and invested amounts
    - ✅ Test multiple step recording
    - ✅ Test metric computation with empty data
    - ✅ Test cumulative wealth growth calculation
    - ✅ Test cash stability index calculation
    - ✅ Test Sharpe-like ratio calculation (including zero std edge case)
    - ✅ Test goal adherence calculation
    - ✅ Test policy stability calculation
    - ✅ Test reset functionality
    - ✅ Test metrics after reset
    - ✅ Test single step edge cases (positive and negative cash)
    - ✅ Test goal adherence without goals
    - ✅ Test goal adherence with mismatched lengths
    - ✅ Test array copying to prevent reference issues
    - ✅ Test cumulative wealth without invested amounts
    - ✅ Test policy stability with identical actions (zero variance)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 9. Integrate Analytics Module with Training Orchestrator ✅ COMPLETE
  - ✅ Modified `HRLTrainer.__init__` to create AnalyticsModule instance
  - ✅ Updated training loop to call `analytics.record_step()` after each environment step
  - ✅ Call `analytics.compute_episode_metrics()` at episode end
  - ✅ Store all 5 analytics metrics in training history (cumulative_wealth_growth, cash_stability_index, sharpe_ratio, goal_adherence, policy_stability)
  - ✅ Call `analytics.reset()` at start of each episode
  - ✅ Enhanced progress printing to include stability and goal adherence metrics
  - ✅ Implemented evaluate() method with full analytics integration
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 10. Implement Configuration Manager ✅ COMPLETE
  - [x] 10.1 Create configuration loading utilities
    - ✅ Implemented `load_config(yaml_path)` function in `src/utils/config_manager.py`
    - ✅ Parse YAML configuration file with error handling
    - ✅ Create EnvironmentConfig, TrainingConfig, RewardConfig instances
    - ✅ Comprehensive docstrings with example YAML structure
    - ✅ Handles missing files, empty files, and YAML parsing errors
    - _Requirements: 6.2_
  
  - [x] 10.2 Implement behavioral profile loading
    - ✅ Implemented `load_behavioral_profile(profile_name)` function
    - ✅ Support "conservative", "balanced", "aggressive" profiles
    - ✅ Return appropriate configuration with adjusted risk tolerance and reward coefficients
    - ✅ Profile mapping with descriptive error messages
    - ✅ Uses BehavioralProfile enum from config.py
    - _Requirements: 6.2, 6.3_
  
  - [x] 10.3 Implement configuration validation
    - ✅ Validate all required parameters are present
    - ✅ Check parameter ranges (e.g., income > 0, gamma in [0, 1])
    - ✅ Raise descriptive ConfigurationError for invalid configurations
    - ✅ Separate validation functions: _validate_environment_config, _validate_training_config, _validate_reward_config
    - ✅ Validates: income > 0, expenses >= 0, inflation in [-1, 1], gamma in [0, 1], risk_tolerance in [0, 1], learning_rates > 0, reward coefficients >= 0
    - _Requirements: 6.1, 6.4, 6.5_
  
  - [x] 10.4 Write configuration tests ✅ COMPLETE
    - ✅ Test loading different behavioral profiles (conservative, balanced, aggressive)
    - ✅ Test parameter validation (positive, negative, boundary cases)
    - ✅ Test configuration overrides
    - ✅ Test YAML parsing errors
    - ✅ Test missing file handling
    - ✅ Test empty configuration file
    - ✅ 50+ comprehensive test cases covering all validation rules
    - ✅ Environment validation: income, expenses, inflation, safety_threshold, max_months, initial_cash, risk_tolerance
    - ✅ Training validation: num_episodes, gamma_low, gamma_high, high_period, batch_size, learning_rate_low, learning_rate_high
    - ✅ Reward validation: alpha, beta, gamma, delta, lambda_, mu
    - ✅ Boundary value testing for all range-constrained parameters
    - ✅ Case-insensitive profile name handling
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 11. Create main training script ✅ COMPLETE
  - ✅ Created `train.py` script in project root with comprehensive CLI interface
  - ✅ Parse command-line arguments for config path and behavioral profile (mutually exclusive)
  - ✅ Load configuration using Configuration Manager with error handling
  - ✅ Instantiate BudgetEnv, RewardEngine, agents, and HRLTrainer with progress feedback
  - ✅ Execute training with specified number of episodes and progress monitoring
  - ✅ Save trained models (high-level agent, low-level agent) and training history (JSON)
  - ✅ Print final evaluation metrics with comprehensive summary
  - ✅ Additional features:
    - Command-line options: --config, --profile, --episodes, --output, --eval-episodes, --save-interval, --seed
    - Configuration summary display before training
    - System initialization with component-by-component feedback
    - Training progress updates every 100 episodes
    - Training summary with last 100 episodes statistics (all 9 metrics)
    - Automatic model saving with proper directory creation
    - JSON serialization with numpy array conversion
    - Optional evaluation after training with comprehensive metrics display
    - Helpful usage examples in --help output
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [ ] 12. Create evaluation script
  - Create `evaluate.py` script in project root
  - Load trained models from checkpoint
  - Run evaluation episodes without learning
  - Compute and display all performance metrics
  - Generate visualization of episode trajectories (cash balance, allocations over time)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 13. Create example configuration files ✅ COMPLETE
  - ✅ Created `configs/conservative.yaml` with conservative behavioral profile parameters
  - ✅ Created `configs/balanced.yaml` with balanced behavioral profile parameters
  - ✅ Created `configs/aggressive.yaml` with aggressive behavioral profile parameters
  - ✅ Include all required parameters: environment, training, and reward configs
  - ✅ Added descriptive comments explaining each profile's focus
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
