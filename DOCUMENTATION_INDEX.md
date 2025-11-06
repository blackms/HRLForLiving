# Documentation Index

Complete guide to all documentation in the Personal Finance Optimization HRL System.

## Getting Started

| Document | Description | Audience |
|----------|-------------|----------|
| [Quick Start Guide](QUICK_START.md) | 5-minute guide to get running | New users |
| [README.md](README.md) | Comprehensive system documentation | All users |
| [Installation](#installation) | Setup and dependencies | All users |

## Core Documentation

### System Overview
- [README.md - Overview](README.md#overview) - System architecture and components
- [README.md - Key Features](README.md#key-features) - Main capabilities
- [README.md - Architecture](README.md#architecture) - System design diagram

### Configuration
- [README.md - Configuration](README.md#configuration) - Configuration system overview
- [README.md - Behavioral Profiles](README.md#behavioral-profiles) - Conservative, Balanced, Aggressive profiles
- [README.md - Configuration Parameters Reference](README.md#configuration-parameters-reference) - Complete parameter documentation
- [README.md - YAML Configuration Format](README.md#yaml-configuration-format) - Configuration file structure
- [configs/conservative.yaml](configs/conservative.yaml) - Conservative profile example
- [configs/balanced.yaml](configs/balanced.yaml) - Balanced profile example
- [configs/aggressive.yaml](configs/aggressive.yaml) - Aggressive profile example
- [configs/scenarios/README.md](configs/scenarios/README.md) - Italian financial scenarios (ISTAT/Numbeo 2024 data)
- [configs/scenarios/](configs/scenarios/) - 5 realistic Italian scenario configurations

### Components
- [README.md - Core Components](README.md#core-components) - All system components
- [README.md - BudgetEnv](README.md#budgetenv---financial-simulation-environment) - Financial environment
- [README.md - RewardEngine](README.md#rewardengine---multi-objective-reward-computation) - Reward system
- [README.md - BudgetExecutor](README.md#budgetexecutor---low-level-agent) - Low-level agent
- [README.md - FinancialStrategist](README.md#financialstrategist---high-level-agent) - High-level agent
- [README.md - HRLTrainer](README.md#hrltrainer---training-orchestrator) - Training system
- [README.md - AnalyticsModule](README.md#analyticsmodule---performance-metrics-tracking) - Performance metrics
- [README.md - Configuration Manager](README.md#configuration-manager---system-configuration) - Config loading

### Usage
- [README.md - Quick Start](README.md#quick-start) - Training and evaluation
- [README.md - Training Examples](README.md#2-training-the-hrl-system) - Command-line examples
- [README.md - Evaluation Examples](README.md#evaluating-trained-models) - Model evaluation
- [README.md - Strategy Analysis](README.md#analyzing-learned-strategy) - Analyzing learned policies
- [README.md - Logging and Monitoring](README.md#logging-and-monitoring) - TensorBoard usage
- [README.md - Utility Scripts](README.md#utility-scripts) - All utility scripts overview

### Performance
- [README.md - Performance Metrics](README.md#performance-metrics) - Metric definitions
- [README.md - Development Status](README.md#development-status) - Implementation status

## Specification Documents

| Document | Description | Audience |
|----------|-------------|----------|
| [Requirements](.kiro/specs/hrl-finance-system/requirements.md) | EARS-compliant system requirements | Developers, Researchers |
| [Design](.kiro/specs/hrl-finance-system/design.md) | Detailed architecture and design | Developers |
| [Tasks](.kiro/specs/hrl-finance-system/tasks.md) | Implementation task list | Developers |
| [HLD/LLD](Requirements/HRL_Finance_System_Design.md) | High and low-level design | Developers |

## Examples and Tutorials

| Example | Description | Run Command |
|---------|-------------|-------------|
| [Basic BudgetEnv](examples/basic_budget_env_usage.py) | Environment basics | `PYTHONPATH=. python3 examples/basic_budget_env_usage.py` |
| [RewardEngine](examples/reward_engine_usage.py) | Reward computation | `PYTHONPATH=. python3 examples/reward_engine_usage.py` |
| [Analytics](examples/analytics_usage.py) | Performance metrics | `PYTHONPATH=. python3 examples/analytics_usage.py` |
| [Training with Analytics](examples/training_with_analytics.py) | Complete training | `PYTHONPATH=. python3 examples/training_with_analytics.py` |
| [Logging](examples/logging_usage.py) | TensorBoard logging | `PYTHONPATH=. python3 examples/logging_usage.py` |
| [Checkpointing](examples/checkpointing_usage.py) | Save/load/resume | `PYTHONPATH=. python3 examples/checkpointing_usage.py` |
| [Examples README](examples/README.md) | All examples overview | - |

## Testing Documentation

### Backend API Tests

| Document | Description | Run Command |
|----------|-------------|-------------|
| [Backend Tests README](backend/tests/README.md) | Complete testing guide | - |
| [API Tests Summary](backend/tests/API_TESTS_SUMMARY.md) | 67+ tests overview | `pytest backend/tests/ -v` |
| [Service Tests Enhancement](backend/tests/SERVICE_TESTS_ENHANCEMENT_SUMMARY.md) | Service layer tests (26 tests) | `pytest backend/tests/test_services.py -v` |
| [Testing Setup Summary](backend/TESTING_SETUP_SUMMARY.md) | Testing infrastructure | - |
| [API Endpoint Tests](backend/tests/) | REST API tests (41 tests) | `pytest backend/tests/test_api_*.py -v` |
| [Service Layer Tests](backend/tests/test_services.py) | Business logic tests | `pytest backend/tests/test_services.py -v` |
| [Integration Tests](backend/tests/test_integration.py) | End-to-end workflows | `pytest backend/tests/test_integration.py -v` |

### Core System Tests

| Document | Description | Run Command |
|----------|-------------|-------------|
| [Test Coverage](tests/TEST_COVERAGE.md) | Coverage summary | - |
| [Unit Tests](tests/) | Component tests | `pytest tests/ -v` |
| [Integration Tests](tests/test_hrl_trainer.py) | Training pipeline tests | `pytest tests/test_hrl_trainer.py -v` |
| [Sanity Checks](tests/test_sanity_checks.py) | System validation | `pytest tests/test_sanity_checks.py -v` |

## Troubleshooting and Support

| Document | Description | Audience |
|----------|-------------|----------|
| [Troubleshooting](README.md#troubleshooting) | Common issues and solutions | All users |
| [Debug Scripts](README.md#debug-scripts) | Diagnostic tools for debugging | Developers |
| [FAQ](README.md#frequently-asked-questions) | Frequently asked questions | All users |
| [Extending the System](README.md#extending-the-system) | Customization guide | Developers |
| [Contributing](README.md#contributing) | Contribution guidelines | Contributors |

### Utility Scripts

| Script | Description | Run Command |
|--------|-------------|-------------|
| [train.py](train.py) | Main training script with CLI | `python3 train.py --profile balanced` |
| [evaluate.py](evaluate.py) | Model evaluation with visualizations | `python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt` |
| [visualize_results.py](visualize_results.py) | Publication-quality charts for papers and presentations | `python3 visualize_results.py` |
| [analyze_strategy.py](analyze_strategy.py) | Strategy analysis and recommendations | `python3 analyze_strategy.py` |
| [explain_failure.py](explain_failure.py) | Explainable AI failure analysis with month-by-month breakdown | `python3 explain_failure.py` |
| [study_italian_scenarios.py](study_italian_scenarios.py) | Comparative study of 5 Italian financial scenarios | `python3 study_italian_scenarios.py` |
| [debug_nan.py](debug_nan.py) | NaN detection in environment and rewards | `python3 debug_nan.py` |

## Reference

| Document | Description | Audience |
|----------|-------------|----------|
| [Changelog](CHANGELOG.md) | Version history | All users |
| [Requirements.txt](requirements.txt) | Python dependencies | Developers |
| [Future Enhancements](README.md#future-enhancements) | Roadmap | All users |
| [Citation](README.md#citation) | Research citation | Researchers |

## Quick Reference

### Training Commands
```bash
# Basic training
python3 train.py --profile balanced --episodes 5000

# With checkpointing
python3 train.py --profile balanced --episodes 10000 --save-interval 1000

# Custom configuration
python3 train.py --config my_config.yaml
```

### Evaluation Commands
```bash
# Basic evaluation
python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt

# With custom episodes
python3 evaluate.py --high-agent models/balanced_high_agent.pt --low-agent models/balanced_low_agent.pt --episodes 50

# Analyze learned strategy
python3 analyze_strategy.py

# Explain why agent fails (detailed month-by-month analysis)
python3 explain_failure.py

# Study Italian financial scenarios (comparative analysis)
python3 study_italian_scenarios.py
```

### Testing Commands
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test category
pytest tests/test_sanity_checks.py -v
```

### TensorBoard Commands
```bash
# Start TensorBoard
tensorboard --logdir=runs

# Open browser to: http://localhost:6006
```

### Debug Commands
```bash
# Check for NaN issues
python3 debug_nan.py

# Training diagnostics
python3 debug_training.py

# Explainable AI failure analysis
python3 explain_failure.py
```

## Documentation by Role

### For New Users
1. Start with [Quick Start Guide](QUICK_START.md)
2. Read [README.md - Overview](README.md#overview)
3. Try [Basic Examples](examples/README.md)
4. Review [Troubleshooting](README.md#troubleshooting)

### For Researchers
1. Read [Requirements](.kiro/specs/hrl-finance-system/requirements.md)
2. Study [Design Document](.kiro/specs/hrl-finance-system/design.md)
3. Review [Performance Metrics](README.md#performance-metrics)
4. Check [Citation](README.md#citation)

### For Developers
1. Read [Design Document](.kiro/specs/hrl-finance-system/design.md)
2. Review [Implementation Tasks](.kiro/specs/hrl-finance-system/tasks.md)
3. Study [Core Components](README.md#core-components)
4. Check [Test Coverage](tests/TEST_COVERAGE.md)
5. Read [Contributing Guidelines](README.md#contributing)

### For Contributors
1. Read [Contributing](README.md#contributing)
2. Review [Code Style Guidelines](README.md#code-style)
3. Study [Testing Guidelines](README.md#testing-guidelines)
4. Check [Development Status](README.md#development-status)

## Documentation Maintenance

### When to Update Documentation

| Change Type | Documents to Update |
|-------------|---------------------|
| New feature | README.md, CHANGELOG.md, examples/ |
| Configuration change | README.md (Configuration section), config files |
| API change | README.md (Core Components), docstrings |
| Bug fix | CHANGELOG.md |
| Performance improvement | README.md (Performance), CHANGELOG.md |
| New example | examples/README.md, DOCUMENTATION_INDEX.md |
| New test | tests/TEST_COVERAGE.md |

### Documentation Standards

- **README.md**: User-facing documentation, comprehensive but accessible
- **Specification docs**: Technical requirements and design, EARS-compliant
- **Examples**: Runnable code with clear comments and output
- **Docstrings**: Follow Google style, include types and examples
- **CHANGELOG.md**: Follow Keep a Changelog format
- **Comments**: Explain why, not what (code should be self-documenting)

## Getting Help

If you can't find what you're looking for:

1. Check the [FAQ](README.md#frequently-asked-questions)
2. Review [Troubleshooting](README.md#troubleshooting)
3. Search this documentation index
4. Check the [examples/](examples/) directory
5. Review the [test files](tests/) for usage patterns
6. Open an issue on GitHub

---

**Last Updated:** Task 17 completion
**Documentation Version:** 1.0
**System Version:** See [CHANGELOG.md](CHANGELOG.md)
