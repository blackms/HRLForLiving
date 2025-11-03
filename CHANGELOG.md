# Changelog

All notable changes to the Personal Finance Optimization HRL System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- BudgetExecutor (Low-Level Agent) implementation
  - PPO-based agent for monthly allocation decisions
  - Custom PolicyNetwork with [128, 128] hidden layers and softmax output
  - 10-dimensional input (7-dimensional state + 3-dimensional goal)
  - 3-dimensional continuous action output with automatic normalization
  - act() method for action generation with deterministic mode support
  - learn() method implementing simplified policy gradient with PPO
  - Discount factor γ_low = 0.95 for temporal credit assignment
  - Entropy bonus for exploration (0.01 coefficient)
  - Model save/load functionality for checkpointing
  - Input validation for state and goal dimensions
  - Training metrics tracking (loss, policy entropy)
- Unit tests for BudgetExecutor (`tests/test_budget_executor.py`)
  - Initialization tests
  - Action generation tests (basic, deterministic, input concatenation)
  - Action normalization tests (including negative values)
  - Learning tests (basic, empty, single transition, terminal states)
  - Policy update mechanics tests
  - Goal influence tests
- BudgetEnv Gymnasium environment implementation
  - 7-dimensional continuous state space
  - 3-dimensional continuous action space with automatic normalization
  - Variable expense sampling from normal distribution
  - Inflation adjustments applied each step
  - Episode termination on negative cash or max months
  - Comprehensive info dictionary with financial metrics
  - Integrated RewardEngine for multi-objective reward computation
- RewardEngine implementation
  - Multi-objective reward computation for low-level agent
  - Strategic reward aggregation for high-level agent
  - Configurable reward coefficients (α, β, γ, δ, λ, μ)
  - Investment rewards, stability penalties, overspend penalties, debt penalties
  - Wealth growth tracking and stability bonus computation
- Environment module public API (`src/environment/__init__.py`)
- Basic usage example (`examples/basic_budget_env_usage.py`)
- Examples documentation (`examples/README.md`)
- RewardEngine usage example (`examples/reward_engine_usage.py`)
- Unit tests for BudgetEnv (`tests/test_budget_env.py`)
- Unit tests for RewardEngine (`tests/test_reward_engine.py`)

### Changed
- Marked Task 5 (Implement Low-Level Agent) as complete in tasks.md
- Updated README.md with BudgetExecutor usage examples and API documentation
- Updated HLD/LLD document with BudgetExecutor implementation status
- Integrated RewardEngine with BudgetEnv for production-ready reward computation
- BudgetEnv now accepts optional RewardConfig parameter in constructor
- BudgetEnv.step() now uses RewardEngine.compute_low_level_reward() for all reward calculations
- Updated README.md with BudgetEnv and RewardEngine usage examples and API documentation
- Updated HLD/LLD document with implementation details and status tracking
- Updated project structure documentation to reflect new examples directory
- Marked Task 2 (Implement BudgetEnv) as complete in tasks.md
- Marked Task 3 (Implement Reward Engine) as complete in tasks.md
- Marked Task 4 (Integrate RewardEngine with BudgetEnv) as complete in tasks.md

### Documentation
- Added Quick Start section to README.md
- Added detailed BudgetEnv API documentation
- Added implementation status section to HLD/LLD document
- Created examples directory with usage demonstrations

## [0.1.0] - 2025-11-03

### Added
- Initial project structure
- Configuration system with dataclasses
  - EnvironmentConfig
  - TrainingConfig
  - RewardConfig
  - BehavioralProfile enum
- Core data models
  - Transition dataclass
- Project documentation
  - Requirements document
  - Design document
  - Implementation tasks
  - HLD/LLD document
- Package initialization files
- Dependencies in requirements.txt
  - gymnasium>=0.29.0
  - numpy>=1.24.0
  - stable-baselines3>=2.0.0
  - torch>=2.0.0
  - pyyaml>=6.0

### Documentation
- Created comprehensive requirements document
- Created detailed design document with architecture diagrams
- Created implementation task breakdown
- Created README.md with project overview

---

## Version History

- **0.1.0** (2025-11-03): Initial project setup with configuration system and documentation
- **Unreleased**: BudgetEnv and RewardEngine implementations complete with full integration, ready for agent implementation
