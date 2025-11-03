# Quick Start Guide

This guide will get you up and running with the Personal Finance Optimization HRL System in 5 minutes.

## 1. Installation (1 minute)

```bash
# Clone the repository
git clone <repository-url>
cd hrl-finance-system

# Install dependencies
pip install -r requirements.txt
```

## 2. Run Your First Training (2 minutes)

```bash
# Train with the balanced profile (recommended for first-time users)
python3 train.py --profile balanced --episodes 1000
```

This will:
- Initialize the HRL system with balanced risk settings
- Train for 1000 episodes (~5-10 minutes)
- Save trained models to `models/` directory
- Display training progress every 100 episodes

## 3. Evaluate Your Trained Model (1 minute)

```bash
# Evaluate the trained model
python3 evaluate.py \
  --high-agent models/balanced_high_agent.pt \
  --low-agent models/balanced_low_agent.pt \
  --episodes 20
```

This will:
- Load your trained models
- Run 20 evaluation episodes
- Display comprehensive performance metrics
- Generate visualization plots in `results/` directory

## 4. View Training Progress with TensorBoard (1 minute)

```bash
# Start TensorBoard (in a new terminal)
tensorboard --logdir=runs

# Open your browser to: http://localhost:6006
```

You'll see:
- Training curves (rewards, losses)
- Episode metrics (cash balance, investments)
- Action and goal distributions
- All 5 performance metrics over time

## What's Next?

### Try Different Behavioral Profiles

```bash
# Conservative: Low risk, high stability
python3 train.py --profile conservative --episodes 1000

# Aggressive: High risk, maximum growth
python3 train.py --profile aggressive --episodes 1000
```

### Customize Your Configuration

Create a custom YAML file based on `configs/balanced.yaml`:

```yaml
environment:
  income: 4000              # Your monthly income
  fixed_expenses: 1800      # Your fixed costs
  variable_expense_mean: 800 # Your average variable expenses
  risk_tolerance: 0.6       # Your risk appetite (0-1)

training:
  num_episodes: 5000        # More episodes = better learning

reward:
  alpha: 12.0               # Higher = more investment
  beta: 0.15                # Higher = more conservative
```

Then train with your custom config:

```bash
python3 train.py --config my_config.yaml
```

### Run Examples

```bash
# See individual components in action
PYTHONPATH=. python3 examples/basic_budget_env_usage.py
PYTHONPATH=. python3 examples/analytics_usage.py
PYTHONPATH=. python3 examples/training_with_analytics.py
```

### Run Tests

```bash
# Verify everything works correctly
pytest tests/ -v

# Run sanity checks
pytest tests/test_sanity_checks.py -v
```

## Understanding Your Results

### Key Metrics to Watch

| Metric | Good Value | What It Means |
|--------|-----------|---------------|
| **Episode Reward** | Increasing over time | Agent is learning |
| **Cash Stability Index** | > 0.9 | Rarely goes bankrupt |
| **Cumulative Wealth Growth** | High positive value | Successfully investing |
| **Sharpe Ratio** | > 1.0 | Good risk-adjusted returns |
| **Goal Adherence** | < 0.1 | Following strategic goals well |

### Typical Training Progress

- **Episodes 0-1000**: Agent learns to avoid bankruptcy
- **Episodes 1000-3000**: Agent learns to balance cash and investment
- **Episodes 3000-5000**: Agent optimizes for long-term wealth growth

### When to Stop Training

Stop training when:
- Episode rewards plateau (no improvement for 1000+ episodes)
- Cash stability index consistently > 0.9
- Goal adherence consistently < 0.1
- You're satisfied with the investment strategy

## Common First-Time Issues

### "Agent keeps going bankrupt"
**Solution:** Use conservative profile or increase `safety_threshold` in config

### "Agent doesn't invest enough"
**Solution:** Use aggressive profile or increase `alpha` (investment reward) in config

### "Training is too slow"
**Solution:** Reduce `num_episodes` to 1000-2000 for initial experiments

### "I don't understand the metrics"
**Solution:** Read the [Performance Metrics](README.md#performance-metrics) section in README.md

## Need Help?

- **Full Documentation**: See [README.md](README.md)
- **Examples**: Check the [examples/](examples/) directory
- **Configuration Help**: See [Configuration Parameters Reference](README.md#configuration-parameters-reference)
- **Troubleshooting**: See [Troubleshooting](README.md#troubleshooting) section
- **FAQ**: See [Frequently Asked Questions](README.md#frequently-asked-questions)

## Next Steps

1. **Experiment with profiles**: Try conservative, balanced, and aggressive
2. **Customize configuration**: Adjust parameters to match your scenario
3. **Analyze results**: Use TensorBoard to understand agent behavior
4. **Read the docs**: Dive deeper into [README.md](README.md) for advanced features
5. **Extend the system**: Add custom rewards, metrics, or profiles

Happy training! 🚀
