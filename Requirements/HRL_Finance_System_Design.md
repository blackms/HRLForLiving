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
r_t = \alpha \cdot invest_{amount} - \beta \cdot max(0, threshold - cash) - \gamma \cdot overspend - \delta \cdot |min(0, cash)|
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

# Compute low-level reward
reward = reward_engine.compute_low_level_reward(action, state, next_state)

# Compute high-level reward
high_reward = reward_engine.compute_high_level_reward(episode_history)
```

### 4.3 Low-Level Agent: `BudgetExecutor` ✅ IMPLEMENTED

**Location:** `src/agents/budget_executor.py`

**Status:** Fully implemented and tested. Ready for integration with training orchestrator.

### 4.4 High-Level Agent: `FinancialStrategist` ✅ IMPLEMENTED

**Location:** `src/agents/financial_strategist.py`

**Status:** Fully implemented. Ready for integration with training orchestrator and testing.

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
        Compute macro features from state history.
        
        Returns 5-dimensional aggregated state:
        [avg_cash, avg_investment_return, spending_trend, current_wealth, months_elapsed]
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
        
        return np.array([avg_cash, avg_investment_return, spending_trend, current_wealth, months_elapsed], dtype=np.float32)

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

**Status:** Fully implemented with complete training loop. Evaluation method pending.

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

**Usage:**
```python
from src.training.hrl_trainer import HRLTrainer

# Initialize all components
env = BudgetEnv(env_config, reward_config)
strategist = FinancialStrategist(training_config)
executor = BudgetExecutor(training_config)
reward_engine = RewardEngine(reward_config, safety_threshold=1000)

# Create trainer
trainer = HRLTrainer(env, strategist, executor, reward_engine, training_config)

# Train the HRL system
history = trainer.train(num_episodes=5000)

# Access training metrics
print(f"Final average reward: {np.mean(history['episode_rewards'][-100:]):.2f}")
print(f"Final average cash: {np.mean(history['cash_balances'][-100:]):.2f}")
print(f"Final average invested: {np.mean(history['total_invested'][-100:]):.2f}")

# Evaluate (to be implemented)
# metrics = trainer.evaluate(num_episodes=100)
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
| **Unit Tests - BudgetEnv** | ✅ Complete | `tests/test_budget_env.py` | Comprehensive tests for BudgetEnv |
| **Unit Tests - RewardEngine** | ✅ Complete | `tests/test_reward_engine.py` | Comprehensive tests for RewardEngine with all reward components |
| **Unit Tests - BudgetExecutor** | ✅ Complete | `tests/test_budget_executor.py` | Comprehensive tests for BudgetExecutor including action generation, learning, and policy updates |
| **Unit Tests - FinancialStrategist** | ✅ Complete | `tests/test_financial_strategist.py` | Comprehensive tests for FinancialStrategist including goal generation, state aggregation, learning, and policy updates |
| **Examples** | ✅ Complete | `examples/basic_budget_env_usage.py` | Basic usage demonstration |
| **HRLTrainer** | ✅ Complete | `src/training/hrl_trainer.py` | Training orchestrator with complete training loop, policy coordination, and metrics tracking |

### 🚧 In Progress

| Component | Status | Next Steps |
|-----------|--------|------------|
| **Training Orchestrator - Evaluation** | 🚧 In Progress | `src/training/hrl_trainer.py` - Training loop complete, need to integrate AnalyticsModule with evaluate() method |
| **Analytics Module Integration** | 🚧 In Progress | Integrate AnalyticsModule with HRLTrainer for comprehensive metrics tracking |
| **Unit Tests - AnalyticsModule** | Not started | Write comprehensive tests for AnalyticsModule |

### ✅ Recently Completed

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **AnalyticsModule** | ✅ Complete | `src/utils/analytics.py` | Performance metrics tracking with 5 key metrics: cumulative wealth growth, cash stability index, Sharpe-like ratio, goal adherence, and policy stability |

### 📋 Next Immediate Tasks

1. **Complete Training Orchestrator** (Task 7)
   - ✅ HRLTrainer class structure created
   - ✅ Implement main training loop (train method)
   - ✅ Implement policy update coordination
   - ⏳ Implement evaluation method with AnalyticsModule integration

2. **Analytics Module** (Task 8)
   - ✅ AnalyticsModule class implementation
   - ✅ Data recording (record_step method)
   - ✅ Metric computation (compute_episode_metrics method)
   - ✅ Reset functionality
   - ⏳ Write unit tests

3. **Integration** (Task 9)
   - ⏳ Integrate AnalyticsModule with HRLTrainer
   - ⏳ Update training loop to record analytics data
   - ⏳ Update evaluation method to use AnalyticsModule

## 9. Future Extensions
- Multi-agent simulation (family / household)
- Integration with real macroeconomic data
- Multi-objective reward (comfort + wealth)
- Export results to dashboard
- Integration with forecasting models (RL + LSTM)
