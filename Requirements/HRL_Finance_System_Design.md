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

**Status:** ✅ **IMPLEMENTED** - Fully functional and ready for integration

**Observation space (7-dimensional):**
```python
[income, fixed_expenses, variable_expenses, cash_balance, inflation, risk_tolerance, t_remaining]
```

**Action space (3-dimensional, continuous [0, 1]):**
\[a_{invest}, a_{save}, a_{consume}\] with sum = 1 (automatically normalized via softmax).

**Key Features:**
- Automatic action normalization using softmax
- Variable expense sampling from normal distribution
- Inflation adjustments applied each step
- Episode termination on negative cash or max months
- Comprehensive info dictionary with cash balance, investments, and expenses

**Reward function (placeholder - will be replaced by RewardEngine):**
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

### 4.1 Class: `BudgetEnv` ✅ IMPLEMENTED

**Location:** `src/environment/budget_env.py`

**Implementation Details:**

```python
class BudgetEnv(gym.Env):
    """
    Custom Gymnasium environment for personal finance simulation.
    
    State Space: 7-dimensional continuous
    Action Space: 3-dimensional continuous [0, 1]
    """
    
    def __init__(self, config: EnvironmentConfig):
        """Initialize with EnvironmentConfig dataclass"""
        super().__init__()
        self.config = config
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -np.inf, 0, 0, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf, 1, 1, config.max_months], dtype=np.float32),
            shape=(7,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # Initialize state
        self.cash_balance = config.initial_cash
        self.current_month = 0
        self.total_invested = 0

    def step(self, action):
        """Execute one time step"""
        # Normalize action to ensure sum = 1
        action = self._normalize_action(action)
        invest_ratio, save_ratio, consume_ratio = action
        
        # Calculate allocations
        invest_amount = invest_ratio * self.income
        
        # Sample variable expense
        self.current_variable_expense = self._sample_variable_expense()
        
        # Apply inflation
        self._apply_inflation()
        
        # Update cash balance
        self.cash_balance = (
            self.cash_balance + self.income 
            - self.fixed_expenses 
            - self.current_variable_expense 
            - invest_amount
        )
        
        # Track investments
        self.total_invested += invest_amount
        self.current_month += 1
        
        # Calculate reward (placeholder)
        reward = self._calculate_reward(invest_amount, self.cash_balance)
        
        # Check termination
        terminated = self.cash_balance < 0
        truncated = self.current_month >= self.max_months
        
        # Return observation, reward, flags, info
        observation = self._get_state()
        info = {
            'cash_balance': self.cash_balance,
            'total_invested': self.total_invested,
            'month': self.current_month,
            'action': action,
            'invest_amount': invest_amount,
            'total_expenses': self.fixed_expenses + self.current_variable_expense
        }
        
        return observation, reward, terminated, truncated, info

    def _normalize_action(self, action):
        """Normalize action using softmax to ensure sum = 1"""
        action = np.clip(action, 1e-8, None)
        exp_action = np.exp(action - np.max(action))
        return exp_action / np.sum(exp_action)
    
    def _sample_variable_expense(self):
        """Sample from normal distribution"""
        expense = np.random.normal(
            self.config.variable_expense_mean,
            self.config.variable_expense_std
        )
        return max(0, expense)
    
    def _apply_inflation(self):
        """Apply inflation to expenses"""
        self.fixed_expenses *= (1 + self.inflation)
        self.current_variable_expense *= (1 + self.inflation)

    def reset(self, seed=None, options=None):
        """Reset to initial state"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.cash_balance = self.config.initial_cash
        self.current_month = 0
        self.total_invested = 0
        self.income = self.config.income
        self.fixed_expenses = self.config.fixed_expenses
        self.inflation = self.config.inflation
        self.current_variable_expense = self._sample_variable_expense()
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Construct state observation vector"""
        t_remaining = self.max_months - self.current_month
        return np.array([
            self.income,
            self.fixed_expenses,
            self.current_variable_expense,
            self.cash_balance,
            self.inflation,
            self.risk_tolerance,
            t_remaining
        ], dtype=np.float32)
```

**Usage:**
```python
from src.environment import BudgetEnv
from src.utils.config import EnvironmentConfig

config = EnvironmentConfig(
    income=3200,
    fixed_expenses=1400,
    variable_expense_mean=700,
    variable_expense_std=100,
    inflation=0.02,
    safety_threshold=1000,
    max_months=60,
    initial_cash=0,
    risk_tolerance=0.5
)

env = BudgetEnv(config)
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step([0.3, 0.5, 0.2])
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

## 8. Implementation Status

### ✅ Completed Components

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **BudgetEnv** | ✅ Complete | `src/environment/budget_env.py` | Fully functional Gymnasium environment with state management, action normalization, expense simulation, and episode termination |
| **Configuration System** | ✅ Complete | `src/utils/config.py` | EnvironmentConfig, TrainingConfig, RewardConfig, BehavioralProfile |
| **Data Models** | ✅ Complete | `src/utils/data_models.py` | Transition dataclass |
| **Unit Tests** | ✅ Complete | `tests/test_budget_env.py` | Comprehensive tests for BudgetEnv |
| **Examples** | ✅ Complete | `examples/basic_budget_env_usage.py` | Basic usage demonstration |

### 🚧 In Progress

| Component | Status | Next Steps |
|-----------|--------|------------|
| **Reward Engine** | Not started | Implement RewardEngine class with configurable coefficients |
| **Low-Level Agent** | Not started | Implement BudgetExecutor with PPO |
| **High-Level Agent** | Not started | Implement FinancialStrategist with HIRO/Option-Critic |
| **Training Orchestrator** | Not started | Implement HRLTrainer for coordinated training |
| **Analytics Module** | Not started | Implement performance metrics tracking |

### 📋 Next Immediate Tasks

1. **Implement Reward Engine** (Task 3)
   - Create RewardEngine class
   - Implement low-level reward computation
   - Implement high-level reward aggregation
   - Integrate with BudgetEnv

2. **Implement Low-Level Agent** (Task 5)
   - Create BudgetExecutor class
   - Integrate with Stable-Baselines3 PPO
   - Implement action generation and learning

3. **Implement High-Level Agent** (Task 6)
   - Create FinancialStrategist class
   - Implement state aggregation
   - Implement goal generation

## 9. Future Extensions
- Multi-agent simulation (family / household)
- Integration with real macroeconomic data
- Multi-objective reward (comfort + wealth)
- Export results to dashboard
- Integration with forecasting models (RL + LSTM)
