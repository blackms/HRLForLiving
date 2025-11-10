<div align="center">

# HRLForLiving

### AI-Powered Personal Finance Optimization

*Master your money with Hierarchical Reinforcement Learning*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

[Quick Start](#-quick-start) • [Features](#-features) • [Architecture](#-architecture) • [Documentation](#-documentation) • [Examples](#-examples)

---

</div>

## What is HRLForLiving?

HRLForLiving is a **cutting-edge hierarchical reinforcement learning (HRL) system** that learns optimal financial allocation strategies. Think of it as your AI financial advisor that simulates months of decision-making to maximize long-term wealth while maintaining stability.

### The Problem

Managing personal finances involves complex trade-offs:
- 💰 Maximize investments for long-term growth
- 🏦 Maintain sufficient savings for emergencies
- 🛍️ Balance discretionary spending
- 📊 Adapt to changing economic conditions

### Our Solution

A two-level AI system that **thinks strategically** and **acts tactically**:

```
┌─────────────────────────────────────────┐
│   High-Level Agent (Strategist)         │
│   "What should my 6-month plan be?"     │
└──────────────┬──────────────────────────┘
               │ Strategic Goals
               ▼
┌─────────────────────────────────────────┐
│   Low-Level Agent (Executor)            │
│   "How do I allocate this month?"       │
└─────────────────────────────────────────┘
```

## Why HRLForLiving?

<table>
<tr>
<td width="33%" align="center">
<h3>🧠 Intelligent</h3>
<p>Uses state-of-the-art PPO and HIRO algorithms to learn optimal strategies</p>
</td>
<td width="33%" align="center">
<h3>🎯 Adaptive</h3>
<p>Adjusts to your risk profile, income, and expenses automatically</p>
</td>
<td width="33%" align="center">
<h3>📈 Data-Driven</h3>
<p>Makes decisions based on simulated outcomes, not rules of thumb</p>
</td>
</tr>
<tr>
<td width="33%" align="center">
<h3>⚡ Production-Ready</h3>
<p>Full FastAPI backend + React frontend with WebSocket support</p>
</td>
<td width="33%" align="center">
<h3>🔬 Research-Grade</h3>
<p>Comprehensive test suite with 100+ test cases</p>
</td>
<td width="33%" align="center">
<h3>🌍 Real-World</h3>
<p>Includes Italian market scenarios and inflation modeling</p>
</td>
</tr>
</table>

---

## Features

### Core Capabilities

- **Hierarchical Decision Making**: Strategic planning (6-12 months) + tactical execution (monthly)
- **Multi-Objective Optimization**: Balance investment growth, cash stability, and spending
- **Risk Profiling**: Conservative, Balanced, and Aggressive behavioral profiles
- **Economic Simulation**: Realistic modeling of inflation, market returns, and expenses
- **Explainable AI**: Understand why the system makes specific recommendations

### Technical Features

- **Modern ML Stack**: PyTorch, Stable-Baselines3, Gymnasium
- **Web Interface**: FastAPI backend + React frontend
- **Real-Time Updates**: WebSocket communication for live training monitoring
- **Configuration Management**: YAML-based configs with validation
- **Comprehensive Analytics**: Sharpe ratios, wealth growth, stability metrics
- **Docker Support**: Fully containerized deployment

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip or conda
- (Optional) Docker for containerized deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/AIgen-Solutions-s-r-l/HRLForLiving.git
cd HRLForLiving

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Simulation

```bash
# Train with the balanced profile (5,000 episodes, ~5 minutes)
python train.py --profile balanced --episodes 5000

# Evaluate the trained model
python evaluate.py --model models/hrl_agent.pth

# Visualize results
python visualize_results.py --results results/training_history.json
```

### Launch the Web UI

```bash
# Start the backend
cd backend
uvicorn main:app --reload

# In another terminal, start the frontend
cd frontend
npm install
npm start
```

Visit `http://localhost:3000` to interact with the system!

---

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (React Frontend + REST API)                 │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│                  FastAPI Backend                         │
│  • Scenario Management  • WebSocket Events               │
│  • Training Orchestration  • Model Serving               │
└─────────────────────┬────────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────────┐
│               HRL Training System                        │
│                                                          │
│  ┌────────────────┐         ┌─────────────────┐        │
│  │ Strategist     │────────▶│  Executor       │        │
│  │ (High-Level)   │  Goals  │  (Low-Level)    │        │
│  │                │         │                 │        │
│  │ 5D State       │         │ 10D State       │        │
│  │ 3D Goal Output │         │ 3D Action Output│        │
│  └────────┬───────┘         └────────┬────────┘        │
│           │                          │                  │
│           └──────────┬───────────────┘                  │
│                      │                                  │
│           ┌──────────▼──────────┐                       │
│           │   BudgetEnv         │                       │
│           │   • State tracking  │                       │
│           │   • Reward engine   │                       │
│           │   • Inflation sim   │                       │
│           └─────────────────────┘                       │
└──────────────────────────────────────────────────────────┘
```

### Agent Architecture

#### High-Level Agent (Strategist)
- **Input**: Aggregated financial state (5 dimensions)
- **Output**: Strategic goals for 6-12 months
- **Algorithm**: HIRO-style hierarchical learning
- **Network**: [64, 64] fully connected layers

#### Low-Level Agent (Executor)
- **Input**: Current state + high-level goals (10 dimensions)
- **Output**: Monthly allocation [invest, save, consume]
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Network**: [128, 128] fully connected layers

### Reward System

The system uses a sophisticated multi-objective reward function:

```
R_low = α·invest - β·max(0, threshold - cash)
        - γ·overspend - δ·|min(0, cash)|

R_high = Σ(R_low) + λ·Δwealth + μ·stability_bonus
```

**Coefficients**:
- α: Investment reward (encourages wealth growth)
- β: Stability penalty (maintains cash buffer)
- γ: Overspend penalty (prevents excessive consumption)
- δ: Debt penalty (avoids negative balance)
- λ: Wealth growth bonus (long-term objective)
- μ: Stability bonus (consistent positive balance)

---

## Project Structure

```
HRLForLiving/
│
├── src/                           # Core HRL system
│   ├── agents/                    # Agent implementations
│   │   ├── budget_executor.py     # Low-Level Agent (PPO)
│   │   └── financial_strategist.py # High-Level Agent (HIRO)
│   ├── environment/               # Simulation environment
│   │   ├── budget_env.py          # Gymnasium environment
│   │   └── reward_engine.py       # Multi-objective rewards
│   ├── training/                  # Training orchestration
│   │   └── hrl_trainer.py         # HRL training loop
│   └── utils/                     # Configuration & analytics
│       ├── config_manager.py      # YAML config loading
│       ├── analytics.py           # Performance metrics
│       └── data_models.py         # Core data structures
│
├── backend/                       # FastAPI application
│   ├── api/                       # REST endpoints
│   ├── websocket/                 # Real-time communication
│   ├── services/                  # Business logic
│   └── main.py                    # Application entry
│
├── frontend/                      # React application
│   └── src/                       # UI components
│
├── configs/                       # Behavioral profiles
│   ├── conservative.yaml          # Low-risk profile
│   ├── balanced.yaml              # Medium-risk profile
│   └── aggressive.yaml            # High-risk profile
│
├── tests/                         # Comprehensive test suite
│   └── test_*.py                  # 100+ test cases
│
├── examples/                      # Usage examples
│   ├── basic_budget_env_usage.py
│   ├── training_with_analytics.py
│   └── README.md
│
└── docs/                          # Documentation
    ├── QUICK_START.md
    ├── API.md
    └── DEPLOYMENT.md
```

---

## Behavioral Profiles

Choose a profile that matches your financial goals and risk tolerance:

### Conservative Profile
**For**: Risk-averse individuals prioritizing stability

```yaml
risk_tolerance: 0.3        # Low risk
safety_threshold: $1,500   # High cash buffer
investment_reward: 5.0     # Moderate investment incentive
```

**Expected Outcome**: Lower returns, higher stability

### Balanced Profile (Default)
**For**: Most users seeking reasonable growth with safety

```yaml
risk_tolerance: 0.5        # Medium risk
safety_threshold: $1,000   # Standard cash buffer
investment_reward: 10.0    # Standard investment incentive
```

**Expected Outcome**: Moderate returns, good stability

### Aggressive Profile
**For**: Risk-tolerant individuals maximizing growth

```yaml
risk_tolerance: 0.8        # High risk
safety_threshold: $500     # Minimal cash buffer
investment_reward: 15.0    # Strong investment incentive
```

**Expected Outcome**: Higher returns, lower stability

---

## Configuration

### Quick Configuration

```python
from src.utils.config_manager import load_behavioral_profile

# Load a predefined profile
env_config, training_config, reward_config = load_behavioral_profile('balanced')
```

### Custom Configuration

```yaml
# my_config.yaml
environment:
  income: 3200                    # Monthly salary
  fixed_expenses: 1400            # Rent, utilities, etc.
  variable_expense_mean: 700      # Groceries, entertainment
  inflation: 0.02                 # 2% monthly inflation
  max_months: 60                  # 5-year simulation

training:
  num_episodes: 10000             # Training iterations
  batch_size: 32                  # Update batch size
  learning_rate_low: 0.0003       # Executor learning rate
  learning_rate_high: 0.0001      # Strategist learning rate

reward:
  alpha: 10.0                     # Investment reward
  beta: 0.1                       # Stability penalty
  gamma: 5.0                      # Overspend penalty
  delta: 20.0                     # Debt penalty
```

Load with:
```python
from src.utils.config_manager import load_config

env_config, training_config, reward_config = load_config('my_config.yaml')
```

---

## Usage Examples

### Basic Training

```python
from src.training import HRLTrainer
from src.utils.config_manager import load_behavioral_profile

# Load configuration
env_config, training_config, reward_config = load_behavioral_profile('balanced')

# Initialize trainer
trainer = HRLTrainer(env_config, training_config, reward_config)

# Train the system
history = trainer.train()

# Save the model
trainer.save('models/my_agent.pth')
```

### Evaluation

```python
from src.training import HRLTrainer

# Load trained model
trainer = HRLTrainer.load('models/my_agent.pth')

# Evaluate over 100 episodes
results = trainer.evaluate(num_episodes=100)

print(f"Average cumulative wealth: ${results['avg_wealth']:.2f}")
print(f"Cash stability: {results['stability']:.2%}")
print(f"Sharpe ratio: {results['sharpe']:.2f}")
```

### Scenario Analysis

```python
from src.environment import BudgetEnv
from src.utils.config import EnvironmentConfig

# Create a high-inflation scenario
high_inflation_config = EnvironmentConfig(
    income=3200,
    fixed_expenses=1400,
    inflation=0.05,  # 5% monthly inflation!
    max_months=24
)

env = BudgetEnv(high_inflation_config)
# ... run simulation ...
```

---

## API Reference

### REST Endpoints

```
POST   /api/scenarios              # Create scenario
GET    /api/scenarios              # List scenarios
GET    /api/scenarios/{id}         # Get scenario
PUT    /api/scenarios/{id}         # Update scenario
DELETE /api/scenarios/{id}         # Delete scenario
POST   /api/scenarios/{id}/train   # Start training
GET    /api/scenarios/{id}/results # Get results
```

### WebSocket Events

```javascript
// Connect to training updates
const ws = new WebSocket('ws://localhost:8000/ws/training/{scenario_id}');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log(`Episode ${update.episode}: Reward = ${update.reward}`);
};
```

---

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

- **Backend**: `http://localhost:8000`
- **Frontend**: `http://localhost:3000`
- **PostgreSQL**: `localhost:5432`

---

## Performance Metrics

The system tracks comprehensive performance indicators:

| Metric | Description | Target |
|--------|-------------|--------|
| **Cumulative Wealth Growth** | Total invested capital | Maximize |
| **Cash Stability Index** | % months with positive balance | > 95% |
| **Sharpe-like Ratio** | Risk-adjusted returns | > 1.0 |
| **Goal Adherence** | Alignment with strategy | > 0.8 |
| **Policy Stability** | Consistency of decisions | < 0.1 |

---

## Advanced Features

### Explainable AI

```bash
# Analyze why the system failed to meet goals
python explain_failure.py --episode 1234 --threshold 0.8
```

### Italian Scenarios

```bash
# Compare different Italian market scenarios
python study_italian_scenarios.py --scenarios configs/italian_*.yaml
```

### Debug Tools

```bash
# Debug NaN/Inf issues in training
python debug_nan.py --model models/problematic_agent.pth
```

---

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_budget_env.py -v
pytest tests/test_hrl_trainer.py -v
```

**Test Coverage**: 100+ test cases across all components

---

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Ensure tests pass and code is formatted
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Format code
black src/ tests/

# Run linters
flake8 src/ tests/
mypy src/
```

---

## Roadmap

- [ ] Multi-currency support
- [ ] Tax optimization strategies
- [ ] Integration with real brokerage APIs
- [ ] Mobile app (React Native)
- [ ] Portfolio rebalancing strategies
- [ ] Social security optimization
- [ ] Retirement planning module

---

## Research & Citations

This project implements concepts from:

- **HIRO**: Data Efficient Hierarchical Reinforcement Learning ([Nachum et al., 2018](https://arxiv.org/abs/1805.08296))
- **PPO**: Proximal Policy Optimization Algorithms ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347))

If you use this project in your research, please cite:

```bibtex
@software{hrlforliving2025,
  title={HRLForLiving: Hierarchical Reinforcement Learning for Personal Finance},
  author={AIgen Solutions},
  year={2025},
  url={https://github.com/AIgen-Solutions-s-r-l/HRLForLiving}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

- **Documentation**: [Full documentation](https://github.com/AIgen-Solutions-s-r-l/HRLForLiving/wiki)
- **Issues**: [Report bugs](https://github.com/AIgen-Solutions-s-r-l/HRLForLiving/issues)
- **Discussions**: [Ask questions](https://github.com/AIgen-Solutions-s-r-l/HRLForLiving/discussions)
- **Email**: support@aigensolutions.com

---

<div align="center">

**Made with ❤️ by [AIgen Solutions](https://github.com/AIgen-Solutions-s-r-l)**

*Empowering financial decisions through AI*

[⬆ Back to Top](#hrlforliving)

</div>
