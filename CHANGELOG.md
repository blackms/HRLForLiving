# Changelog

All notable changes to the Personal Finance Optimization HRL System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- BudgetEnv Gymnasium environment implementation
  - 7-dimensional continuous state space
  - 3-dimensional continuous action space with automatic normalization
  - Variable expense sampling from normal distribution
  - Inflation adjustments applied each step
  - Episode termination on negative cash or max months
  - Comprehensive info dictionary with financial metrics
- Environment module public API (`src/environment/__init__.py`)
- Basic usage example (`examples/basic_budget_env_usage.py`)
- Examples documentation (`examples/README.md`)
- Unit tests for BudgetEnv (`tests/test_budget_env.py`)

### Changed
- Updated README.md with BudgetEnv usage examples and API documentation
- Updated HLD/LLD document with implementation details and status tracking
- Updated project structure documentation to reflect new examples directory
- Marked Task 2 (Implement BudgetEnv) as complete in tasks.md

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
- **Unreleased**: BudgetEnv implementation complete, ready for Reward Engine integration
