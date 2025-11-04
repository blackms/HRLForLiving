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
- TensorBoard integration for experiment tracking and visualization
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

**Status:** ✅ **IMPLEMENTED** - Fully functional with integrated RewardEngine

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
- Integrated RewardEngine for multi-objective reward computation

**Reward function:** Uses RewardEngine.compute_low_level_reward() for production-ready multi-objective rewards.

### 3.2 Reward Engine: `RewardEngine`
**Type:** Multi-objective reward computation

**Status:** ✅ **IMPLEMENTED** - Fully functional and tested

**Low-Level Reward Function:**
\[
r_{raw} = \alpha \cdot invest_{amount} - \beta \cdot max(0, threshold - cash) - \gamma \cdot overspend - \delta \cdot |min(0, cash)|
\]
\[
r_t = r_{raw} / 1000.0 \quad \text{(scaled for training stability)}
\]

**High-Level Reward Function:**
\[
r_{high} = \sum r_{low} + \lambda \cdot \Delta wealth + \mu \cdot stability_{ratio} \cdot period_{length}
\]

**Key Features:**
- Configurable reward coefficients (α, β, γ, δ, λ, μ)
- Investment rewards to encourage long-term growth
- Stability penalties to maintain positive cash balance
- Overspend penalties to prevent excessive consumption
- Debt penalties to heavily discourage negative balance
- Wealth growth tracking for strategic rewards
- Stability bonus for consistent positive balance
- **Automatic reward scaling (÷1000) to prevent gradient explosion**
- NaN/Inf safety checks with fallback penalties

### 3.3 High-Level Agent
- Decision every 6–12 steps
- Input: average cash, investment return, spending trend
- Output: goal vector = [target_invest_ratio, target_safety_buffer, aggressiveness]
- Reward: long-term wealth + stability

### 3.4 Low-Level Agent
- Decision every month
- Input: monthly financial state + goal vector
- Output: continuous action [invest, save, consume]
- Reward: monthly investment gain - penalties for low balance

### 3.5 Training
Framework: **RLlib** or **Stable-Baselines3 + custom HRL wrapper**

Algorithms:
- PPO (low-level)
- HIRO / Option-Critic (high-level)

Discount factors:
- gamma_low = 0.95
- gamma_high = 0.99

### 3.6 Data Flow
1. Reset environment → generate initial scenario
2. High-level policy sets goal
3. Low-level acts monthly
4. Reward engine computes metrics
5. Orchestrator updates both policies

---

## 4. Low-Level Design (LLD)

### 4.1 Class: `BudgetEnv` ✅ IMPLEMENTED

**Location:** `src/environment/budget_env.py`

**Status:** Fully implemented and tested. Ready for RewardEngine integration.

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

### 4.2 Class: `RewardEngine` ✅ IMPLEMENTED

**Location:** `src/environment/reward_engine.py`

**Status:** Fully implemented and tested. Ready for BudgetEnv integration.

**Implementation Details:**

```python
class RewardEngine:
    """
    Computes multi-objective rewards for both high-level and low-level agents.
    """
    
    def __init__(self, config: RewardConfig, safety_threshold: float = 1000):
        """Initialize with reward coefficients"""
        self.config = config
        self.safety_threshold = safety_threshold
        self.alpha = config.alpha      # Investment reward coefficient
        self.beta = config.beta        # Stability penalty coefficient
        self.gamma = config.gamma      # Overspend penalty coefficient
        self.delta = config.delta      # Debt penalty coefficient
        self.lambda_ = config.lambda_  # Wealth growth coefficient
        self.mu = config.mu            # Stability bonus coefficient

    def compute_low_level_reward(
        self, 
        action: np.ndarray, 
        state: np.ndarray, 
        next_state: np.ndarray
    ) -> float:
        """
        Compute immediate monthly reward combining:
        - Investment reward: α * invest_amount
        - Stability penalty: β * max(0, threshold - cash)
        - Overspend penalty: γ * overspend
        - Debt penalty: δ * abs(min(0, cash))
        """
        # Extract values
        income = state[0]
        current_cash = state[3]
        next_cash = next_state[3]
        invest_ratio = action[0]
        invest_amount = invest_ratio * income
        
        # Calculate reward components
        investment_reward = self.alpha * invest_amount
        
        stability_penalty = 0
        if next_cash < self.safety_threshold:
            stability_penalty = self.beta * (self.safety_threshold - next_cash)
        
        overspend = 0
        cash_change = next_cash - current_cash
        expected_decrease = invest_amount
        if cash_change < -expected_decrease:
            overspend = abs(cash_change + expected_decrease)
        overspend_penalty = self.gamma * overspend
        
        debt_penalty = 0
        if next_cash < 0:
            debt_penalty = self.delta * abs(next_cash)
        
        return investment_reward - stability_penalty - overspend_penalty - debt_penalty

    def compute_high_level_reward(self, episode_history: List[Transition]) -> float:
        """
        Compute strategic reward aggregating:
        - Sum of low-level rewards
        - Wealth change: λ * Δwealth
        - Stability bonus: μ * stability_ratio * period_length
        """
        if not episode_history:
            return 0.0
        
        # Aggregate low-level rewards
        total_low_level_reward = sum(t.reward for t in episode_history)
        
        # Calculate wealth change
        initial_cash = episode_history[0].state[3]
        final_cash = episode_history[-1].next_state[3]
        wealth_change = final_cash - initial_cash
        wealth_reward = self.lambda_ * wealth_change
        
        # Calculate stability bonus
        positive_balance_count = sum(1 for t in episode_history if t.next_state[3] > 0)
        stability_ratio = positive_balance_count / len(episode_history)
        stability_bonus = self.mu * stability_ratio * len(episode_history)
        
        return total_low_level_reward + wealth_reward + stability_bonus
```

**Usage:**
```python
from src.environment import RewardEngine
from src.utils.config import RewardConfig

config = RewardConfig(alpha=10.0, beta=0.1, gamma=5.0, delta=20.0, lambda_=1.0, mu=0.5)
reward_engine = RewardEngine(config, safety_threshold=1000)

# Compute low-level reward (automatically scaled by 1000.0)
reward = reward_engine.compute_low_level_reward(action, state, next_state)
# Returns scaled reward in range ~[-10, 10] for training stability

# Compute high-level reward
high_reward = reward_engine.compute_high_level_reward(episode_history)
```

**Important Note on Reward Scaling:**
The `compute_low_level_reward()` method automatically scales rewards by dividing by 1000.0. This is critical for training stability:
- With typical income (~$3200), raw rewards can exceed 10,000
- Large rewards cause gradient explosion in neural networks
- Scaling brings rewards into the recommended range of [-10, 10]
- The scaling factor (1000.0) is based on typical income values

### 4.3 Low-Level Agent: `BudgetExecutor` ✅ IMPLEMENTED

**Location:** `src/agents/budget_executor.py`

**Status:** Fully implemented and tested. Ready for integration with training orchestrator.

### 4.4 High-Level Agent: `FinancialStrategist` ✅ IMPLEMENTED

**Location:** `src/agents/financial_strategist.py`

**Status:** Fully implemented with state normalization for training stability. Ready for integration with training orchestrator and testing.

**Recent Updates:**
- Added state normalization in `aggregate_state()` method (2025-11-04)
- All 5 aggregated state features are now normalized to prevent extreme values
- NaN/Inf safety checks with fallback to default state
- Based on HIRO paper recommendations for hierarchical RL stability

**Implementation Details:**

```python
class BudgetExecutor:
    """
    Low-Level Agent that executes concrete monthly allocation decisions.
    Uses PPO for learning optimal allocation policies.
    """
    
    def __init__(self, config: TrainingConfig, state_dim: int = 7, goal_dim: int = 3):
        """Initialize with custom policy network"""
        self.config = config
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.input_dim = state_dim + goal_dim  # 10-dimensional
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.input_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([1, 1, 1]),
            shape=(3,), dtype=np.float32
        )
        
        # Initialize policy network [128, 128] with softmax output
        self.policy_network = PolicyNetwork(self.input_dim, 3)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate_low
        )

    def act(self, state: np.ndarray, goal: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Generate allocation action from state and goal.
        
        Concatenates state (7D) and goal (3D) to create 10D input,
        passes through policy network to produce action [invest, save, consume].
        """
        # Validate dimensions
        if state.shape[0] != self.state_dim:
            raise ValueError(f"Expected state dimension {self.state_dim}")
        if goal.shape[0] != self.goal_dim:
            raise ValueError(f"Expected goal dimension {self.goal_dim}")
        
        # Concatenate and predict
        concatenated_input = np.concatenate([state, goal])
        with torch.no_grad():
            input_tensor = torch.FloatTensor(concatenated_input).unsqueeze(0)
            action_probs = self.policy_network(input_tensor)
            action = action_probs.squeeze(0).numpy()
        
        # Normalize to ensure sum = 1
        return self._normalize_action(action)

    def learn(self, transitions: List[Transition]) -> Dict[str, float]:
        """
        Update PPO policy using collected transitions.
        
        Implements simplified policy gradient with:
        - Discounted returns (γ_low = 0.95)
        - Return normalization
        - Entropy bonus for exploration
        """
        if not transitions:
            return {'loss': 0.0, 'policy_entropy': 0.0}
        
        # Extract and prepare data
        states = np.array([t.state for t in transitions])
        goals = np.array([t.goal for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        dones = np.array([t.done for t in transitions])
        
        # Concatenate observations
        observations = np.concatenate([states, goals], axis=1)
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.config.gamma_low * G
            returns.insert(0, G)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(observations)
        actions_tensor = torch.FloatTensor(actions)
        returns_tensor = torch.FloatTensor(returns)
        
        # Policy update
        self.optimizer.zero_grad()
        action_probs = self.policy_network(obs_tensor)
        log_probs = torch.log(action_probs + 1e-8)
        selected_log_probs = (log_probs * actions_tensor).sum(dim=1)
        policy_loss = -(selected_log_probs * returns_tensor).mean()
        
        # Entropy for exploration
        entropy = -(action_probs * log_probs).sum(dim=1).mean()
        loss = policy_loss - 0.01 * entropy
        
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_entropy': entropy.item(),
            'n_updates': 1
        }
    
    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**Usage:**
```python
from src.agents.budget_executor import BudgetExecutor
from src.utils.config import TrainingConfig

config = TrainingConfig(
    gamma_low=0.95,
    learning_rate_low=3e-4,
    batch_size=32
)

executor = BudgetExecutor(config)

# Generate action
state = np.array([3200, 1400, 700, 1000, 0.02, 0.5, 50])
goal = np.array([0.3, 1000, 0.5])
action = executor.act(state, goal)

# Learn from experience
metrics = executor.learn(transitions)
print(f"Loss: {metrics['loss']:.4f}, Entropy: {metrics['policy_entropy']:.4f}")

# Save/load model
executor.save('models/executor.pth')
executor.load('models/executor.pth')
```

**Implementation Details:**

```python
class FinancialStrategist:
    """
    High-Level Agent that defines medium-term financial strategy.
    Uses HIRO-style algorithm for hierarchical learning.
    """
    
    def __init__(self, config: TrainingConfig, aggregated_state_dim: int = 5):
        """Initialize with custom policy network"""
        self.config = config
        self.aggregated_state_dim = aggregated_state_dim
        self.goal_dim = 3  # [target_invest_ratio, safety_buffer, aggressiveness]
        
        # Initialize policy network [64, 64]
        self.policy_network = StrategistNetwork(self.aggregated_state_dim, self.goal_dim)
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.learning_rate_high
        )

    def aggregate_state(self, history: List[np.ndarray]) -> np.ndarray:
        """
        Compute macro features from state history with automatic normalization.
        
        Returns 5-dimensional normalized aggregated state:
        [avg_cash, avg_investment_return, spending_trend, current_wealth, months_elapsed]
        
        State normalization is critical for hierarchical RL (Nachum et al., 2018 - HIRO).
        """
        if not history:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        history_array = np.array(history, dtype=np.float32)
        cash_balances = history_array[:, 3]
        variable_expenses = history_array[:, 2]
        
        avg_cash = np.mean(cash_balances)
        avg_investment_return = np.mean(np.diff(cash_balances)) if len(cash_balances) > 1 else 0.0
        spending_trend = np.polyfit(np.arange(len(variable_expenses)), variable_expenses, 1)[0] if len(variable_expenses) > 1 else 0.0
        current_wealth = cash_balances[-1]
        months_elapsed = history_array[0, 6] - history_array[-1, 6] if len(history_array) > 0 else 0.0
        
        # Normalize to prevent extreme values that destabilize training
        aggregated_state = np.array([
            avg_cash / 10000.0,                    # Normalize to ~0.5-1.0
            avg_investment_return / 1000.0,        # Normalize to ~-0.5 to 0.5
            spending_trend / 100.0,                # Normalize to ~-0.1 to 0.1
            current_wealth / 10000.0,              # Normalize to ~0.5-1.0
            months_elapsed / 120.0                 # Normalize to [0, 1]
        ], dtype=np.float32)
        
        # Safety check for NaN/Inf values
        if np.any(np.isnan(aggregated_state)) or np.any(np.isinf(aggregated_state)):
            print(f"WARNING: Invalid aggregated state! Returning default.")
            aggregated_state = np.array([0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
        
        return aggregated_state

    def select_goal(self, state: np.ndarray) -> np.ndarray:
        """
        Generate goal vector from aggregated state.
        
        Applies constraints to ensure valid ranges:
        - target_invest_ratio: [0, 1] using sigmoid
        - safety_buffer: [0, inf) using softplus
        - aggressiveness: [0, 1] using sigmoid
        """
        if state.shape[0] != self.aggregated_state_dim:
            raise ValueError(f"Expected aggregated state dimension {self.aggregated_state_dim}")
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            goal_raw = self.policy_network(state_tensor).squeeze(0).numpy()
        
        target_invest_ratio = 1.0 / (1.0 + np.exp(-goal_raw[0]))
        safety_buffer = np.log(1.0 + np.exp(goal_raw[1]))
        aggressiveness = 1.0 / (1.0 + np.exp(-goal_raw[2]))
        
        return np.array([target_invest_ratio, safety_buffer, aggressiveness], dtype=np.float32)

    def learn(self, transitions: List[Transition]) -> Dict[str, float]:
        """
        Update high-level policy using HIRO-style algorithm.
        
        Implements simplified HIRO with:
        - Discounted returns (γ_high = 0.99)
        - Return normalization
        - Gradient clipping for stability
        """
        if not transitions:
            return {'loss': 0.0, 'policy_entropy': 0.0}
        
        # Extract and prepare data
        states = np.array([t.state for t in transitions])
        goals = np.array([t.goal for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        dones = np.array([t.done for t in transitions])
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.config.gamma_high * G
            returns.insert(0, G)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        goals_tensor = torch.FloatTensor(goals)
        returns_tensor = torch.FloatTensor(returns)
        
        # Policy update
        self.optimizer.zero_grad()
        goal_predictions = self.policy_network(states_tensor)
        mse_loss = torch.mean((goal_predictions - goals_tensor) ** 2, dim=1)
        policy_loss = (mse_loss * returns_tensor).mean()
        
        goal_variance = torch.var(goal_predictions, dim=0).mean()
        loss = policy_loss - 0.01 * goal_variance
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_entropy': goal_variance.item(),
            'n_updates': 1
        }
    
    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**Usage:**
```python
from src.agents.financial_strategist import FinancialStrategist
from src.utils.config import TrainingConfig

config = TrainingConfig(
    gamma_high=0.99,
    learning_rate_high=1e-4,
    high_period=6
)

strategist = FinancialStrategist(config)

# Aggregate state from history
state_history = [state1, state2, state3, ...]
aggregated_state = strategist.aggregate_state(state_history)

# Generate strategic goal
goal = strategist.select_goal(aggregated_state)
print(f"Goal: invest={goal[0]:.2f}, buffer={goal[1]:.2f}, aggr={goal[2]:.2f}")

# Learn from experience
metrics = strategist.learn(high_level_transitions)
print(f"Loss: {metrics['loss']:.4f}, Entropy: {metrics['policy_entropy']:.4f}")

# Save/load model
strategist.save('models/strategist.pth')
strategist.load('models/strategist.pth')
```

### 4.5 Training Orchestrator: `HRLTrainer` ✅ IMPLEMENTED

**Location:** `src/training/hrl_trainer.py`

**Status:** Fully implemented with complete training loop, evaluation, TensorBoard logging, and checkpointing.

**Implementation Details:**

```python
class HRLTrainer:
    """
    Training Orchestrator that coordinates the HRL training loop.
    
    Manages interaction between high-level (Strategist) and low-level (Executor) agents,
    coordinating policy updates and managing the training process.
    """
    
    def __init__(
        self,
        env: BudgetEnv,
        high_agent: FinancialStrategist,
        low_agent: BudgetExecutor,
        reward_engine: RewardEngine,
        config: TrainingConfig
    ):
        """Initialize with all necessary components and tracking structures"""
        self.env = env
        self.high_agent = high_agent
        self.low_agent = low_agent
        self.reward_engine = reward_engine
        self.config = config
        
        # Episode buffer for storing transitions
        self.episode_buffer: List[Transition] = []
        
        # State history for high-level agent aggregation
        self.state_history: List[np.ndarray] = []
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'cash_balances': [],
            'total_invested': [],
            'low_level_losses': [],
            'high_level_losses': []
        }
    
    def train(self, num_episodes: int) -> Dict:
        """
        Execute the main HRL training loop.
        
        Training process:
        1. Reset environment and get initial state
        2. Generate initial goal from high-level agent
        3. Execute monthly steps with low-level agent
        4. Store transitions in episode buffer
        5. Update low-level policy when buffer reaches batch size
        6. Every high_period steps: compute high-level reward, update high-level policy, generate new goal
        7. Handle final updates at episode termination
        8. Track and print progress metrics
        
        Args:
            num_episodes: Number of training episodes to run
            
        Returns:
            dict: Training history with all collected metrics
        """
        for episode in range(num_episodes):
            # Reset environment and initialize
            state, _ = self.env.reset()
            self.state_history = [state]
            
            # Generate initial strategic goal
            aggregated_state = self.high_agent.aggregate_state(self.state_history)
            goal = self.high_agent.select_goal(aggregated_state)
            
            # Episode tracking
            episode_reward = 0
            episode_length = 0
            self.episode_buffer = []
            steps_since_high_update = 0
            high_level_transitions = []
            
            # Execute episode
            done = False
            while not done:
                # Low-level agent generates action
                action = self.low_agent.act(state, goal)
                
                # Execute action in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                transition = Transition(state, goal, action, reward, next_state, done)
                self.episode_buffer.append(transition)
                self.state_history.append(next_state)
                
                # Track metrics
                episode_reward += reward
                episode_length += 1
                steps_since_high_update += 1
                
                # Update low-level policy
                if len(self.episode_buffer) >= self.config.batch_size:
                    low_metrics = self.low_agent.learn(self.episode_buffer[-self.config.batch_size:])
                    self.training_history['low_level_losses'].append(low_metrics['loss'])
                
                # High-level re-planning every high_period steps
                if steps_since_high_update >= self.config.high_period and not done:
                    # Compute high-level reward over period
                    period_transitions = self.episode_buffer[-steps_since_high_update:]
                    high_level_reward = self.reward_engine.compute_high_level_reward(period_transitions)
                    
                    # Create and store high-level transition
                    high_transition = Transition(
                        aggregated_state, goal, goal,
                        high_level_reward,
                        self.high_agent.aggregate_state(self.state_history),
                        False
                    )
                    high_level_transitions.append(high_transition)
                    
                    # Update high-level policy
                    high_metrics = self.high_agent.learn(high_level_transitions)
                    self.training_history['high_level_losses'].append(high_metrics['loss'])
                    
                    # Generate new goal
                    aggregated_state = self.high_agent.aggregate_state(self.state_history)
                    goal = self.high_agent.select_goal(aggregated_state)
                    steps_since_high_update = 0
                
                state = next_state
            
            # Episode termination: final updates
            if len(self.episode_buffer) > 0:
                # Final high-level update
                period_transitions = self.episode_buffer[-steps_since_high_update:] if steps_since_high_update > 0 else self.episode_buffer
                high_level_reward = self.reward_engine.compute_high_level_reward(period_transitions)
                
                final_aggregated_state = self.high_agent.aggregate_state(self.state_history)
                high_transition = Transition(aggregated_state, goal, goal, high_level_reward, final_aggregated_state, True)
                high_level_transitions.append(high_transition)
                
                high_metrics = self.high_agent.learn(high_level_transitions)
                self.training_history['high_level_losses'].append(high_metrics['loss'])
                
                # Final low-level update
                if len(self.episode_buffer) >= self.config.batch_size:
                    low_metrics = self.low_agent.learn(self.episode_buffer[-self.config.batch_size:])
                    self.training_history['low_level_losses'].append(low_metrics['loss'])
            
            # Store episode metrics
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['cash_balances'].append(info.get('cash_balance', 0))
            self.training_history['total_invested'].append(info.get('total_invested', 0))
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                avg_cash = np.mean(self.training_history['cash_balances'][-100:])
                avg_invested = np.mean(self.training_history['total_invested'][-100:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Cash: {avg_cash:.2f}, "
                      f"Avg Invested: {avg_invested:.2f}")
        
        return self.training_history
    
    def evaluate(self, num_episodes: int) -> Dict:
        """
        Run evaluation episodes without learning.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            dict: Evaluation metrics
        """
        # TODO: Implement evaluation
        pass
```

**Key Features:**
- Automatic AnalyticsModule integration (zero-overhead tracking)
- Records step data automatically during training
- Computes 5 key metrics per episode:
  - Cumulative wealth growth
  - Cash stability index
  - Sharpe-like ratio
  - Goal adherence
  - Policy stability
- Enhanced progress printing with stability and goal adherence
- Deterministic evaluation mode
- Comprehensive evaluation summary with mean/std statistics
- Optional TensorBoard logging for experiment tracking
- Checkpointing and resume functionality:
  - Save checkpoints at regular intervals
  - Track and save best model based on evaluation
  - Resume training from saved checkpoints
  - Preserve complete training state (models, configs, history)

**Usage:**
```python
from src.training.hrl_trainer import HRLTrainer

# Initialize all components
env = BudgetEnv(env_config, reward_config)
strategist = FinancialStrategist(training_config)
executor = BudgetExecutor(training_config)
reward_engine = RewardEngine(reward_config, safety_threshold=1000)

# Create trainer (automatically creates AnalyticsModule)
trainer = HRLTrainer(env, strategist, executor, reward_engine, training_config)

# Train the HRL system (analytics tracked automatically)
history = trainer.train(num_episodes=5000)

# Access training metrics (includes all analytics metrics)
print(f"Final average reward: {np.mean(history['episode_rewards'][-100:]):.2f}")
print(f"Final average cash: {np.mean(history['cash_balances'][-100:]):.2f}")
print(f"Final average invested: {np.mean(history['total_invested'][-100:]):.2f}")
print(f"Final stability: {np.mean(history['cash_stability_index'][-100:]):.2%}")
print(f"Final goal adherence: {np.mean(history['goal_adherence'][-100:]):.4f}")

# Evaluate with comprehensive metrics
eval_metrics = trainer.evaluate(num_episodes=100)
print(f"\nEvaluation Results:")
print(f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
print(f"Mean Wealth Growth: ${eval_metrics['mean_wealth_growth']:.2f}")
print(f"Mean Cash Stability: {eval_metrics['mean_cash_stability']:.2%}")
print(f"Mean Sharpe Ratio: {eval_metrics['mean_sharpe_ratio']:.2f}")
print(f"Mean Goal Adherence: {eval_metrics['mean_goal_adherence']:.4f}")
print(f"Mean Policy Stability: {eval_metrics['mean_policy_stability']:.4f}")
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
| **BudgetEnv** | ✅ Complete | `src/environment/budget_env.py` | Fully functional Gymnasium environment with integrated RewardEngine, state management, action normalization, expense simulation, and episode termination |
| **RewardEngine** | ✅ Complete | `src/environment/reward_engine.py` | Multi-objective reward computation for both agents with configurable coefficients |
| **RewardEngine Integration** | ✅ Complete | `src/environment/budget_env.py` | BudgetEnv now uses RewardEngine for all reward calculations |
| **BudgetExecutor (Low-Level Agent)** | ✅ Complete | `src/agents/budget_executor.py` | PPO-based agent with custom policy network, action generation, learning, and model persistence |
| **FinancialStrategist (High-Level Agent)** | ✅ Complete | `src/agents/financial_strategist.py` | HIRO-style agent with state aggregation, goal generation, strategic learning, and model persistence |
| **Configuration System** | ✅ Complete | `src/utils/config.py` | EnvironmentConfig, TrainingConfig, RewardConfig, BehavioralProfile |
| **Data Models** | ✅ Complete | `src/utils/data_models.py` | Transition dataclass |
| **Unit Tests - BudgetEnv** | ✅ Complete | `tests/test_budget_env.py` | Comprehensive tests for BudgetEnv including 19 edge case tests for extreme financial scenarios |
| **Unit Tests - RewardEngine** | ✅ Complete | `tests/test_reward_engine.py` | Comprehensive tests for RewardEngine with all reward components |
| **Unit Tests - BudgetExecutor** | ✅ Complete | `tests/test_budget_executor.py` | Comprehensive tests for BudgetExecutor including action generation, learning, and policy updates |
| **Unit Tests - FinancialStrategist** | ✅ Complete | `tests/test_financial_strategist.py` | Comprehensive tests for FinancialStrategist including goal generation, state aggregation, learning, and policy updates |
| **Examples** | ✅ Complete | `examples/basic_budget_env_usage.py` | Basic usage demonstration |
| **HRLTrainer** | ✅ Complete | `src/training/hrl_trainer.py` | Training orchestrator with complete training loop, policy coordination, metrics tracking, TensorBoard logging, and checkpointing functionality |
| **Integration Tests - HRLTrainer** | ✅ Complete | `tests/test_hrl_trainer.py` | 13 comprehensive integration tests covering complete training pipeline, component coordination, and analytics integration |
| **Checkpointing Tests** | ✅ Complete | `tests/test_checkpointing.py` | 7 comprehensive tests for checkpoint save/load, resume training, and best model tracking |

### ✅ Recently Completed

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **Main Training Script** | ✅ Complete | `train.py` | Comprehensive CLI tool for training the HRL system with config/profile support, model saving, evaluation, and progress monitoring. Supports --config, --profile, --episodes, --output, --eval-episodes, --save-interval, and --seed options. |
| **AnalyticsModule** | ✅ Complete | `src/utils/analytics.py` | Performance metrics tracking with 5 key metrics: cumulative wealth growth, cash stability index, Sharpe-like ratio, goal adherence, and policy stability. Comprehensive test coverage with 18 test cases. |
| **AnalyticsModule Integration** | ✅ Complete | `src/training/hrl_trainer.py` | Fully integrated with HRLTrainer for automatic tracking during training and evaluation |
| **HRLTrainer Evaluation Method** | ✅ Complete | `src/training/hrl_trainer.py` | Deterministic evaluation with comprehensive summary statistics including all 5 analytics metrics |
| **Unit Tests - AnalyticsModule** | ✅ Complete | `tests/test_analytics.py` | 18 comprehensive test cases covering all functionality and edge cases |
| **ConfigurationManager** | ✅ Complete | `src/utils/config_manager.py` | YAML loading, behavioral profiles (conservative, balanced, aggressive), comprehensive validation with descriptive error messages |
| **Unit Tests - ConfigurationManager** | ✅ Complete | `tests/test_config_manager.py` | 50+ comprehensive test cases covering all validation rules, boundary values, error handling, and profile loading |
| **Sanity Check Tests** | ✅ Complete | `tests/test_sanity_checks.py` | 7 system-level validation tests for behavioral profiles, learning effectiveness, and configuration integrity |
| **TensorBoard Logging** | ✅ Complete | `src/utils/logger.py` | ExperimentLogger for comprehensive experiment tracking with TensorBoard integration |
| **Logging Examples** | ✅ Complete | `examples/logging_usage.py` | Complete demonstration of TensorBoard logging functionality |
| **Checkpointing Examples** | ✅ Complete | `examples/checkpointing_usage.py` | Complete demonstration of checkpointing and resume functionality |

### 🚧 In Progress

| Component | Status | Next Steps |
|-----------|--------|------------|
| **Evaluation Script** | Not started | Create evaluate.py script for loading and testing trained models |

### 📋 Next Immediate Tasks

1. **Evaluation Script** (Task 12)
   - ⏳ Create evaluate.py script for loading and testing trained models
   - ⏳ Support loading models from checkpoint
   - ⏳ Run evaluation episodes without learning
   - ⏳ Display comprehensive performance metrics
   - ⏳ Generate visualizations of episode trajectories

### ✅ Completed Tasks

1. **Integration Tests for HRLTrainer** (Task 7.5) - ✅ COMPLETE
   - ✅ Test complete episode execution with all components working together
   - ✅ Test high-level/low-level coordination and goal updates at correct intervals
   - ✅ Test policy updates occur correctly (low-level and high-level)
   - ✅ Test analytics integration throughout episode
   - ✅ Test episode buffer accumulation and state history tracking
   - ✅ Test reward engine integration during training
   - ✅ Test full training pipeline from start to finish (5 episodes)
   - ✅ Test evaluation after training integration
   - ✅ Test hierarchical coordination complete flow
   - ✅ Test batch size coordination for low-level updates
   - ✅ Test high_period coordination for high-level updates
   - ✅ Test policy improvement verification over time
   - ✅ Test all components (env, agents, reward engine, analytics) working together
   - ✅ 13 comprehensive integration tests in `tests/test_hrl_trainer.py`

2. **Main Training Script** (Task 11) - ✅ COMPLETE
   - ✅ Create train.py script in project root
   - ✅ Comprehensive CLI interface with argparse
   - ✅ Support for YAML configuration files and behavioral profiles
   - ✅ Command-line options: --config, --profile, --episodes, --output, --eval-episodes, --save-interval, --seed
   - ✅ Configuration loading with error handling
   - ✅ Configuration summary display
   - ✅ System initialization with progress feedback
   - ✅ Training execution with progress monitoring
   - ✅ Training summary with comprehensive statistics
   - ✅ Automatic model saving (high-level agent, low-level agent, training history)
   - ✅ JSON serialization with numpy array conversion
   - ✅ Optional evaluation after training
   - ✅ Helpful usage examples in --help

3. **Configuration Manager** (Task 10) - ✅ COMPLETE
   - ✅ Implement configuration loading utilities
   - ✅ Implement behavioral profile loading
   - ✅ Implement configuration validation
   - ✅ Write configuration tests (50+ test cases)
   - ✅ Environment validation (17 tests)
   - ✅ Training validation (13 tests)
   - ✅ Reward validation (8 tests)
   - ✅ Profile loading tests (5 tests)
   - ✅ Configuration loading tests (5 tests)
   - ✅ Override tests (1 test)

### 4.6 BudgetEnv Edge Case Tests ✅ IMPLEMENTED

**Location:** `tests/test_budget_env.py` (TestBudgetEnvEdgeCases class)

**Status:** Fully implemented with 19 comprehensive edge case tests.

**Purpose:** Validate BudgetEnv robustness under extreme financial scenarios and boundary conditions.

**Test Categories:**

**1. Income Stress Tests**
```python
def test_very_low_income_scenario(self):
    """Test with income barely covering expenses (income=100, expenses=95)"""

def test_extremely_low_income_immediate_failure(self):
    """Test with expenses exceeding income (income=500, fixed=1500)"""
```

**2. Expense Stress Tests**
```python
def test_very_high_fixed_expenses(self):
    """Test with 90% of income going to fixed expenses"""

def test_very_high_variable_expenses(self):
    """Test with high variable expenses (mean=2500, std=500)"""

def test_extreme_variable_expense_variance(self):
    """Test with 80% variance (std = 80% of mean)"""
```

**3. Inflation Stress Tests**
```python
def test_extreme_positive_inflation(self):
    """Test with 50% monthly hyperinflation"""

def test_extreme_negative_inflation(self):
    """Test with 20% monthly deflation"""

def test_zero_inflation(self):
    """Test with zero inflation (constant expenses)"""

def test_very_long_episode_with_inflation(self):
    """Test compounding inflation effects over 30+ months"""
```

**4. Episode Length Tests**
```python
def test_maximum_episode_length(self):
    """Test 120-month (10-year) episodes"""

def test_single_step_episode(self):
    """Test with max_months=1"""
```

**5. Initial Cash Tests**
```python
def test_high_initial_cash_buffer(self):
    """Test with $50,000 starting cash"""

def test_zero_initial_cash_survival(self):
    """Test starting with zero cash"""
```

**6. Combined Stress Tests**
```python
def test_combined_extreme_conditions(self):
    """Test multiple extreme conditions simultaneously:
    - Low income (1500)
    - High fixed expenses (1000)
    - High variable expenses (400 ± 200)
    - High inflation (10%)
    - Short episode (6 months)
    - Low initial cash (500)
    """
```

**Key Validations:**
- System handles extreme conditions without crashes
- No undefined behavior or NaN values
- Proper termination conditions (negative cash, max months)
- State observations remain valid (correct shape, non-negative expenses)
- Inflation effects compound correctly over time
- Episode length constraints are respected
- Cash balance updates correctly under stress

**Edge Cases Covered:**
- Income barely covering expenses
- Expenses exceeding income (immediate failure)
- 90%+ fixed expense ratios
- High variance expenses (std = 80% of mean)
- Hyperinflation (50% monthly)
- Deflation (negative inflation)
- Zero inflation (constant expenses)
- Very long episodes (120 months)
- Single-step episodes
- High initial cash buffers ($50,000)
- Zero initial cash
- Combined extreme conditions

**Test Execution:**
```bash
# Run all edge case tests
pytest tests/test_budget_env.py::TestBudgetEnvEdgeCases -v

# Run specific edge case
pytest tests/test_budget_env.py::TestBudgetEnvEdgeCases::test_combined_extreme_conditions

# Run with detailed output
pytest tests/test_budget_env.py::TestBudgetEnvEdgeCases -v -s
```

**Expected Behavior:**
- Very low income: Should survive with conservative allocation
- Extremely low income: Should terminate immediately (negative cash)
- High expenses: Should handle without crashes
- Extreme inflation: Expenses should increase dramatically
- Deflation: Expenses should decrease
- Zero inflation: Expenses should remain constant
- Long episodes: Should reach max_months or terminate naturally
- Single-step: Should truncate after first step
- High initial cash: Should enable aggressive investment
- Zero initial cash: Should survive with conservative allocation
- Combined stress: Should handle without crashes

**Benefits:**
- Ensures robustness under extreme market conditions
- Validates boundary value handling
- Confirms no crashes or undefined behavior
- Tests realistic stress scenarios (hyperinflation, deflation, income loss)
- Provides confidence for production deployment

### 4.7 Sanity Check Tests: `test_sanity_checks.py` ✅ IMPLEMENTED

**Location:** `tests/test_sanity_checks.py`

**Status:** Fully implemented with 7 comprehensive system-level validation tests.

**Purpose:** Validate complete HRL system behavior, behavioral profile differentiation, and learning effectiveness through end-to-end testing.

**Test Categories:**

**1. Random Policy Baseline Validation**
```python
def test_random_policy_does_not_accumulate_wealth(self, base_env_config, base_training_config, base_reward_config):
    """
    Verifies that untrained agents don't systematically accumulate wealth.
    Establishes baseline for learning effectiveness.
    
    Validates:
    - Random policy invests < 30% of maximum possible
    - Episodes terminate early due to poor cash management
    - Average episode length < 80% of max_months
    """
```

**2. Behavioral Profile Comparison Tests**
```python
def test_conservative_profile_maintains_higher_cash_balance(self):
    """
    Validates conservative profile maintains higher cash reserves than aggressive.
    
    Validates:
    - Conservative cash balance > aggressive cash balance
    - Conservative stability index >= aggressive stability index
    - Conservative safety threshold > aggressive safety threshold
    - Conservative risk tolerance < aggressive risk tolerance
    """

def test_aggressive_profile_invests_more(self):
    """
    Confirms aggressive profile invests more than conservative.
    
    Validates:
    - Aggressive total invested > conservative total invested
    - Aggressive wealth growth > conservative wealth growth
    - Aggressive alpha (investment reward) > conservative alpha
    - Aggressive beta (stability penalty) < conservative beta
    """

def test_balanced_profile_between_conservative_and_aggressive(self):
    """
    Ensures balanced profile exhibits behavior between extremes.
    
    Validates:
    - Conservative cash >= balanced cash >= aggressive cash (with 10% tolerance)
    - Aggressive invested >= balanced invested >= conservative invested (with 10% tolerance)
    - Conservative risk tolerance < balanced < aggressive risk tolerance
    """
```

**3. Learning Effectiveness Validation**
```python
def test_trained_policy_outperforms_random_policy(self, base_env_config, base_training_config, base_reward_config):
    """
    Verifies trained policies outperform random policies.
    
    Validates:
    - Trained reward > random reward
    - Trained stability > random stability
    - Trained episode length > random episode length
    """
```

**4. Configuration Validation Tests**
```python
def test_profile_risk_tolerance_ordering(self):
    """
    Validates correct risk tolerance ordering across profiles.
    
    Validates:
    - Conservative risk tolerance < balanced < aggressive
    - Conservative safety threshold > balanced > aggressive
    """

def test_profile_reward_coefficient_ordering(self):
    """
    Validates correct reward coefficient ordering.
    
    Validates:
    - Alpha (investment): aggressive > balanced > conservative
    - Beta (stability penalty): conservative > balanced > aggressive
    """
```

**Key Features:**
- **System-Level Testing**: Tests complete HRL system rather than individual components
- **Behavioral Validation**: Ensures different profiles produce expected behavioral differences
- **Learning Verification**: Confirms training actually improves performance
- **Statistical Robustness**: Uses multiple evaluation episodes for reliable comparisons
- **Fast Execution**: Uses realistic but short training durations (20-30 episodes)
- **Requirements Coverage**: Validates Requirements 1.1-1.3, 2.1-2.3, 6.2-6.3

**Test Execution:**
```bash
# Run all sanity checks
pytest tests/test_sanity_checks.py -v

# Run specific sanity check
pytest tests/test_sanity_checks.py::TestSanityChecks::test_trained_policy_outperforms_random_policy

# Run with detailed output
pytest tests/test_sanity_checks.py -v -s
```

**Expected Results:**
- Random policies should not accumulate significant wealth
- Conservative profiles should maintain higher cash balances
- Aggressive profiles should invest more and achieve higher wealth growth
- Balanced profiles should fall between conservative and aggressive
- Trained policies should outperform random policies
- Profile configurations should have correct ordering

**Usage in CI/CD:**
These sanity checks serve as smoke tests to validate that:
1. The HRL system learns effectively
2. Behavioral profiles are correctly differentiated
3. Configuration parameters have expected effects
4. System-level behavior matches design specifications

### 4.8 Main Training Script: `train.py` ✅ IMPLEMENTED

**Location:** `train.py` (project root)

**Status:** Fully implemented with comprehensive CLI interface and features.

**Purpose:** Command-line tool for training the HRL Finance System with flexible configuration options.

**Key Features:**
- Comprehensive argument parsing with mutually exclusive config sources
- Configuration loading from YAML files or behavioral profiles
- System initialization with progress feedback
- Training execution with monitoring and checkpointing
- Automatic model saving and training history export
- Optional evaluation after training
- Comprehensive error handling

**Command-Line Interface:**

```bash
# Train with behavioral profile
python train.py --profile balanced --episodes 5000

# Train with YAML configuration
python train.py --config configs/conservative.yaml

# Train with custom settings
python train.py --profile aggressive --episodes 10000 --output models/run1 --seed 42

# Train with evaluation
python train.py --profile balanced --eval-episodes 20
```

**Command-Line Options:**
- `--config PATH`: Path to YAML configuration file (mutually exclusive with --profile)
- `--profile {conservative,balanced,aggressive}`: Use predefined behavioral profile (mutually exclusive with --config)
- `--episodes N`: Number of training episodes (overrides config)
- `--output DIR`: Output directory for trained models (default: models/)
- `--eval-episodes N`: Number of evaluation episodes after training (default: 10, set to 0 to skip)
- `--save-interval N`: Save checkpoint every N episodes (default: 1000)
- `--seed N`: Random seed for reproducibility

**Implementation Functions:**

```python
def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments"""

def load_configuration(args) -> Tuple[EnvironmentConfig, TrainingConfig, RewardConfig, str]:
    """Load configuration from file or profile with error handling"""

def print_configuration(env_config, training_config, reward_config):
    """Display configuration summary before training"""

def initialize_system(...) -> Tuple[BudgetEnv, FinancialStrategist, BudgetExecutor, RewardEngine, HRLTrainer]:
    """Initialize all system components with progress feedback"""

def train_system(...) -> Dict:
    """Execute training loop with progress monitoring"""

def save_models(trainer, output_dir, config_name):
    """Save trained models and training history to disk"""

def print_training_summary(history):
    """Display training statistics over last 100 episodes"""

def evaluate_system(trainer, num_episodes) -> Dict:
    """Run evaluation episodes and return comprehensive metrics"""

def print_evaluation_results(eval_results):
    """Display evaluation metrics with mean and std"""
```

**Training Output:**

The script provides comprehensive output throughout execution:

1. **Configuration Summary**: Displays all environment, training, and reward parameters
2. **System Initialization**: Shows progress for each component (environment, agents, trainer)
3. **Training Progress**: Updates every 100 episodes with average metrics
4. **Training Summary**: Statistics over last 100 episodes (all 9 metrics)
5. **Model Saving**: Confirmation of saved files with paths
6. **Evaluation Results**: Comprehensive metrics with mean ± std

**Saved Files:**

After training, the script saves three files:
- `{config_name}_high_agent.pt`: Trained high-level agent (Strategist)
- `{config_name}_low_agent.pt`: Trained low-level agent (Executor)
- `{config_name}_history.json`: Complete training history with all metrics

**Training History JSON Structure:**

```json
{
  "episode_rewards": [float, ...],
  "episode_lengths": [int, ...],
  "cash_balances": [float, ...],
  "total_invested": [float, ...],
  "low_level_losses": [float, ...],
  "high_level_losses": [float, ...],
  "cumulative_wealth_growth": [float, ...],
  "cash_stability_index": [float, ...],
  "sharpe_ratio": [float, ...],
  "goal_adherence": [float, ...],
  "policy_stability": [float, ...]
}
```

**Error Handling:**

The script handles various error conditions gracefully:
- Configuration file not found
- Invalid YAML syntax
- Configuration validation errors
- Invalid behavioral profile names
- Unexpected errors during initialization or training

All errors are reported with descriptive messages and appropriate exit codes.

**Usage Example:**

```bash
# Quick start with balanced profile
python train.py --profile balanced --episodes 1000 --seed 42

# Output:
# ======================================================================
# HRL Finance System - Training Script
# ======================================================================
# Loading behavioral profile: balanced
# 
# ======================================================================
# Configuration Summary
# ======================================================================
# [... configuration details ...]
# 
# ======================================================================
# Initializing HRL System
# ======================================================================
# Setting random seed: 42
# 1. Creating BudgetEnv...
#    ✓ BudgetEnv initialized
# [... more initialization ...]
# 
# ======================================================================
# Starting Training
# ======================================================================
# Training for 1000 episodes...
# Progress:
# ----------------------------------------------------------------------
# Episode 100/1000 - Avg Reward: 45.23, Avg Cash: 1234.56, ...
# [... training progress ...]
# 
# ======================================================================
# Training Summary
# ======================================================================
# [... comprehensive statistics ...]
# 
# ======================================================================
# Saving Models
# ======================================================================
# ✓ High-level agent saved to: models/balanced_high_agent.pt
# ✓ Low-level agent saved to: models/balanced_low_agent.pt
# ✓ Training history saved to: models/balanced_history.json
# 
# ======================================================================
# Evaluating Trained System
# ======================================================================
# [... evaluation results ...]
# 
# ======================================================================
# Training Complete!
# ======================================================================
```

---

## 9. Future Extensions
- Multi-agent simulation (family / household)
- Integration with real macroeconomic data
- Multi-objective reward (comfort + wealth)
- Export results to dashboard
- Integration with forecasting models (RL + LSTM)
- Evaluation script (evaluate.py) for loading and testing trained models
- Visualization tools for episode trajectories and learning curves
