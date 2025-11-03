---
title: "Personal Finance Optimization HRL System"
author: "Alessio Rocchi"
version: "1.0"
date: "2025-11-03"
description: >
  High-Level Design (HLD) and Low-Level Design (LLD) document for the HRL system
  that optimizes personal finance allocation between investment, saving, and spending.
---

# 🧭 Personal Finance Optimization HRL System

## 1. System Overview

### Objective
The HRL agent simulates and learns to allocate a monthly salary optimally among:
- **Investments**
- **Savings**
- **Discretionary Spending**

Goal: maximize long-term investments while avoiding negative cash balance.

### Key Features
- Realistic monthly economic simulation (fixed/variable expenses, inflation, random events)
- Hierarchical decision-making (strategic + tactical)
- Configurable behavioral profiles (conservative, balanced, aggressive)
- Performance metrics for financial growth and stability
- Scalable for multi-agent simulations (families, organizations)

---

## 2. System Architecture (HLD)

### 2.1 HRL Architecture Overview

```mermaid
graph TD
A[Financial Environment] -->|state| B[High-Level Agent (Stratega Finanziario)]
B -->|goal vector (budget target, risk profile)| C[Low-Level Agent (Esecutore di Bilancio)]
C -->|action (allocation per month)| A
A -->|reward_low, next_state| C
A -->|reward_high (aggregated)| B
```

### 2.2 Components

| Component | Description |
|------------|-------------|
| **Environment (BudgetEnv)** | Simulates the financial environment: income, expenses, inflation, random events. |
| **High-Level Agent (Strategist)** | Defines medium-term strategy (6–12 months). |
| **Low-Level Agent (Executor)** | Executes concrete monthly actions respecting the strategy. |
| **Reward Engine** | Computes rewards for both short- and long-term objectives. |
| **Training Orchestrator** | Coordinates PPO/HIRO training for HRL. |
| **Analytics Module** | Evaluates performance and stability metrics. |

---

## 3. High-Level Design (HLD)

### 3.1 Environment: `BudgetEnv`
**Type:** Custom Gymnasium environment

**Observation space:**
```python
[income, fixed_expenses, variable_expenses, cash_balance, inflation, risk_tolerance, t_remaining]
```

**Action space:**
\[a_{invest}, a_{save}, a_{consume}\] with sum = 1.

**Reward function:**
\[
r_t = \alpha \cdot a_{invest} - \beta \cdot risk(cash) - \gamma \cdot overspend
\]

### 3.2 High-Level Agent
- Decision every 6–12 steps
- Input: average cash, investment return, spending trend
- Output: goal vector = [target_invest_ratio, target_safety_buffer, aggressiveness]
- Reward: long-term wealth + stability

### 3.3 Low-Level Agent
- Decision every month
- Input: monthly financial state + goal vector
- Output: continuous action [invest, save, consume]
- Reward: monthly investment gain - penalties for low balance

### 3.4 Training
Framework: **RLlib** or **Stable-Baselines3 + custom HRL wrapper**

Algorithms:
- PPO (low-level)
- HIRO / Option-Critic (high-level)

Discount factors:
- gamma_low = 0.95
- gamma_high = 0.99

### 3.5 Data Flow
1. Reset environment → generate initial scenario
2. High-level policy sets goal
3. Low-level acts monthly
4. Reward engine computes metrics
5. Orchestrator updates both policies

---

## 4. Low-Level Design (LLD)

### 4.1 Class: `BudgetEnv`

```python
class BudgetEnv(gym.Env):
    def __init__(self, config):
        self.income = config["income"]
        self.fixed = config["fixed"]
        self.var_mean = config["var_mean"]
        self.inflation = config["inflation"]
        self.safety_threshold = config["safety_threshold"]
        self.reset()

    def step(self, action):
        invest, save, consume = self._normalize(action)
        expenses = self._sample_variable_expense()
        self.cash = self.cash + self.income - self.fixed - expenses - invest*self.income
        reward = self._calculate_reward(invest, self.cash)
        done = self.cash < 0 or self.month >= self.max_months
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, invest, cash):
        risk_penalty = max(0, (self.safety_threshold - cash))
        return invest*10 - risk_penalty*0.1

    def reset(self):
        self.month = 0
        self.cash = 0
        return self._get_state()
```

### 4.2 High-Level Policy (Strategist)

```python
class FinancialStrategist:
    def __init__(self, model):
        self.model = model

    def select_goal(self, state):
        goal_vector = self.model.predict(state)
        return goal_vector  # [target_invest_ratio, safety_buffer, aggressiveness]

    def learn(self, transition_batch):
        self.model.update(transition_batch)
```

### 4.3 Low-Level Policy (Executor)

```python
class BudgetExecutor:
    def __init__(self, model):
        self.model = model

    def act(self, state, goal_vector):
        input_vec = np.concatenate([state, goal_vector])
        return self.model.predict(input_vec)  # action [invest, save, consume]

    def learn(self, transition_batch):
        self.model.update(transition_batch)
```

### 4.4 Trainer

```python
class HRLTrainer:
    def __init__(self, env, high_agent, low_agent, cfg):
        self.env = env
        self.high = high_agent
        self.low = low_agent
        self.cfg = cfg

    def train(self, episodes):
        for ep in range(episodes):
            s = self.env.reset()
            goal = self.high.select_goal(s)
            for t in range(self.cfg["T"]):
                a = self.low.act(s, goal)
                s_next, r_low, done, _ = self.env.step(a)
                self.low.learn((s, goal, a, r_low, s_next))

                if t % self.cfg["high_period"] == 0:
                    r_high = self.env.evaluate_macro_performance()
                    self.high.learn((s, goal, r_high, s_next))
                    goal = self.high.select_goal(s_next)

                s = s_next
                if done:
                    break
```

---

## 5. Reward Design

| Component | Description | Formula |
|------------|-------------|----------|
| Invest Reward | Encourages monthly investment | `+α * a_invest` |
| Stability Penalty | Penalty for low cash | `-β * max(0, threshold - cash)` |
| Overspend Penalty | Penalty for excessive consumption | `-γ * overspend` |
| Debt Penalty | Negative cash balance | `-δ * abs(cash)` |
| Long-term Reward | Aggregated over 6–12 months | `Σr_t + λ * Δwealth` |

---

## 6. Metrics

| Metric | Description |
|---------|-------------|
| Cumulative Wealth Growth | Total invested capital |
| Cash Stability Index | % months with positive balance |
| Sharpe-like Ratio | Return / std of balance |
| Goal Adherence | Difference between target and realized allocation |
| Policy Stability | Action variance over time |

---

## 7. Configurations

```yaml
env:
  income: 3200
  fixed: 1400
  var_mean: 700
  inflation: 0.02
  safety_threshold: 1000
  max_months: 60

training:
  episodes: 5000
  gamma_high: 0.99
  gamma_low: 0.95
  high_period: 6
  algorithm_high: HIRO
  algorithm_low: PPO
```

---

## 8. Future Extensions
- Multi-agent simulation (family / household)
- Integration with real macroeconomic data
- Multi-objective reward (comfort + wealth)
- Export results to dashboard
- Integration with forecasting models (RL + LSTM)
