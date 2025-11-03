# Requirements Document

## Introduction

The Personal Finance Optimization HRL System is a hierarchical reinforcement learning agent that simulates and learns to optimally allocate monthly salary among investments, savings, and discretionary spending. The system aims to maximize long-term investments while maintaining financial stability and avoiding negative cash balance through realistic monthly economic simulation.

## Glossary

- **HRL System**: The Hierarchical Reinforcement Learning System consisting of high-level and low-level agents
- **BudgetEnv**: The custom Gymnasium environment that simulates the financial environment
- **High-Level Agent (Strategist)**: The agent that defines medium-term financial strategy (6-12 months)
- **Low-Level Agent (Executor)**: The agent that executes concrete monthly allocation actions
- **Reward Engine**: The component that computes rewards for both short-term and long-term objectives
- **Training Orchestrator**: The component that coordinates PPO/HIRO training for the HRL system
- **Analytics Module**: The component that evaluates performance and stability metrics
- **Action Vector**: A three-element vector [invest, save, consume] representing allocation percentages that sum to 1
- **Goal Vector**: A strategic directive from the High-Level Agent containing [target_invest_ratio, safety_buffer, aggressiveness]
- **Cash Balance**: The current available liquid funds in the simulation
- **Safety Threshold**: The minimum cash balance required to avoid penalties

## Requirements

### Requirement 1

**User Story:** As a financial researcher, I want the system to simulate realistic monthly financial scenarios, so that the HRL agent can learn from diverse economic conditions.

#### Acceptance Criteria

1. WHEN the BudgetEnv is initialized, THE BudgetEnv SHALL accept configuration parameters for income, fixed expenses, variable expense mean, inflation rate, and safety threshold
2. WHEN a simulation step occurs, THE BudgetEnv SHALL generate variable expenses using statistical sampling with the configured mean
3. WHEN a simulation step occurs, THE BudgetEnv SHALL apply inflation adjustments to expenses based on the configured inflation rate
4. WHEN the simulation reaches the maximum number of months or cash balance becomes negative, THE BudgetEnv SHALL terminate the episode
5. WHEN the environment is reset, THE BudgetEnv SHALL initialize the cash balance to zero and reset the month counter

### Requirement 2

**User Story:** As a financial researcher, I want the system to support hierarchical decision-making, so that strategic and tactical financial decisions can be separated and optimized independently.

#### Acceptance Criteria

1. THE High-Level Agent SHALL generate a goal vector containing target investment ratio, safety buffer, and aggressiveness level
2. WHEN the High-Level Agent makes a decision, THE High-Level Agent SHALL base the decision on aggregated state information including average cash, investment return, and spending trends
3. THE High-Level Agent SHALL make strategic decisions at intervals of 6 to 12 simulation steps
4. THE Low-Level Agent SHALL receive both the current financial state and the goal vector from the High-Level Agent as input
5. WHEN the Low-Level Agent acts, THE Low-Level Agent SHALL produce an action vector with three continuous values representing investment, saving, and consumption allocations that sum to 1

### Requirement 3

**User Story:** As a financial researcher, I want the system to compute meaningful rewards, so that the agents learn to balance long-term wealth growth with short-term financial stability.

#### Acceptance Criteria

1. WHEN computing the low-level reward, THE Reward Engine SHALL apply a positive coefficient to the investment allocation amount
2. WHEN the cash balance falls below the safety threshold, THE Reward Engine SHALL apply a penalty proportional to the deficit
3. WHEN consumption exceeds available funds, THE Reward Engine SHALL apply an overspend penalty
4. WHEN the cash balance becomes negative, THE Reward Engine SHALL apply a debt penalty proportional to the absolute value of the negative balance
5. WHEN computing the high-level reward, THE Reward Engine SHALL aggregate rewards over the strategic period and include a term for wealth change

### Requirement 4

**User Story:** As a financial researcher, I want the system to train both agents using reinforcement learning algorithms, so that they learn optimal policies through experience.

#### Acceptance Criteria

1. THE Training Orchestrator SHALL use PPO algorithm for training the Low-Level Agent
2. THE Training Orchestrator SHALL use HIRO or Option-Critic algorithm for training the High-Level Agent
3. WHEN training the Low-Level Agent, THE Training Orchestrator SHALL apply a discount factor of 0.95
4. WHEN training the High-Level Agent, THE Training Orchestrator SHALL apply a discount factor of 0.99
5. WHEN a training episode begins, THE Training Orchestrator SHALL reset the environment and obtain the initial state
6. WHEN the high-level decision period elapses, THE Training Orchestrator SHALL update the High-Level Agent policy and generate a new goal vector

### Requirement 5

**User Story:** As a financial researcher, I want the system to track performance metrics, so that I can evaluate the effectiveness of different policies and configurations.

#### Acceptance Criteria

1. THE Analytics Module SHALL compute cumulative wealth growth as the total invested capital over the simulation period
2. THE Analytics Module SHALL compute a cash stability index as the percentage of months with positive balance
3. THE Analytics Module SHALL compute a Sharpe-like ratio as the return divided by the standard deviation of balance
4. THE Analytics Module SHALL compute goal adherence as the difference between target and realized allocation
5. THE Analytics Module SHALL compute policy stability as the variance of actions over time

### Requirement 6

**User Story:** As a financial researcher, I want the system to support configurable behavioral profiles, so that I can simulate different risk tolerance levels and financial strategies.

#### Acceptance Criteria

1. THE BudgetEnv SHALL accept a risk tolerance parameter in the observation space
2. WHEN a configuration is loaded, THE HRL System SHALL accept parameters for conservative, balanced, or aggressive behavioral profiles
3. WHEN a behavioral profile is set, THE HRL System SHALL adjust reward coefficients according to the profile's risk tolerance
4. THE HRL System SHALL allow configuration of the high-level decision period between 6 and 12 steps
5. THE HRL System SHALL allow configuration of the maximum simulation duration in months

### Requirement 7

**User Story:** As a financial researcher, I want the system to provide a standard Gymnasium interface, so that I can integrate it with existing RL frameworks and tools.

#### Acceptance Criteria

1. THE BudgetEnv SHALL inherit from the Gymnasium Env base class
2. THE BudgetEnv SHALL define an observation space containing income, fixed expenses, variable expenses, cash balance, inflation, risk tolerance, and remaining time
3. THE BudgetEnv SHALL define an action space as a continuous three-dimensional vector
4. WHEN the step method is called, THE BudgetEnv SHALL return a tuple containing next state, reward, done flag, and info dictionary
5. WHEN the reset method is called, THE BudgetEnv SHALL return the initial state observation
