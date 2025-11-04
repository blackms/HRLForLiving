#!/usr/bin/env python3
"""Analyze the learned financial strategy"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.agents.financial_strategist import FinancialStrategist
from src.agents.budget_executor import BudgetExecutor
from src.utils.config_manager import load_config

# Load config and trained models
env_config, training_config, reward_config = load_config('configs/personal_realistic.yaml')

env = BudgetEnv(env_config, reward_config)
high_agent = FinancialStrategist(training_config)
low_agent = BudgetExecutor(training_config)

# Load trained weights
high_agent.load('models/personal_realistic_high_agent.pt')
low_agent.load('models/personal_realistic_low_agent.pt')

print("=" * 70)
print("ANALISI STRATEGIA FINANZIARIA APPRESA")
print("=" * 70)
print(f"\nSituazione Iniziale:")
print(f"  Entrate mensili: {env_config.income} EUR")
print(f"  Spese fisse: {env_config.fixed_expenses} EUR")
print(f"  Spese variabili medie: {env_config.variable_expense_mean} EUR")
print(f"  Disponibile: {env_config.income - env_config.fixed_expenses - env_config.variable_expense_mean} EUR/mese")
print(f"  Cash iniziale: {env_config.initial_cash} EUR")

# Run simulation
state, _ = env.reset(seed=42)
state_history = [state]

# Track metrics
months = []
cash_history = []
invested_history = []
actions_history = []
goals_history = []

aggregated_state = high_agent.aggregate_state(state_history)
goal = high_agent.select_goal(aggregated_state)

month = 0
done = False

while not done and month < 120:
    action = low_agent.act(state, goal, deterministic=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    months.append(month)
    cash_history.append(info['cash_balance'])
    invested_history.append(info['total_invested'])
    actions_history.append(action.copy())
    goals_history.append(goal.copy())
    
    state = next_state
    state_history.append(state)
    
    # Update goal every 6 months
    if (month + 1) % 6 == 0:
        aggregated_state = high_agent.aggregate_state(state_history)
        goal = high_agent.select_goal(aggregated_state)
    
    done = terminated or truncated
    month += 1

print(f"\n" + "=" * 70)
print(f"RISULTATI SIMULAZIONE ({month} mesi)")
print("=" * 70)

# Calculate statistics
actions_array = np.array(actions_history)
avg_invest = np.mean(actions_array[:, 0])
avg_save = np.mean(actions_array[:, 1])
avg_consume = np.mean(actions_array[:, 2])

print(f"\nAllocazione Media:")
print(f"  Investimento: {avg_invest:.1%}")
print(f"  Risparmio: {avg_save:.1%}")
print(f"  Consumo: {avg_consume:.1%}")

print(f"\nRisultati Finali:")
print(f"  Cash finale: {cash_history[-1]:.2f} EUR")
print(f"  Totale investito: {invested_history[-1]:.2f} EUR")
print(f"  Patrimonio totale: {cash_history[-1] + invested_history[-1]:.2f} EUR")

# Monthly investment
monthly_investment = env_config.income * avg_invest
print(f"\nInvestimento Mensile Medio: {monthly_investment:.2f} EUR")

# Recommendations
print(f"\n" + "=" * 70)
print("RACCOMANDAZIONI PRATICHE")
print("=" * 70)

print(f"\n1. ALLOCAZIONE MENSILE CONSIGLIATA:")
print(f"   Con {env_config.income} EUR/mese:")
print(f"   - Investi: {env_config.income * avg_invest:.2f} EUR ({avg_invest:.1%})")
print(f"   - Risparmia: {env_config.income * avg_save:.2f} EUR ({avg_save:.1%})")
print(f"   - Spese discrezionali: {env_config.income * avg_consume:.2f} EUR ({avg_consume:.1%})")

print(f"\n2. BUFFER DI SICUREZZA:")
goals_array = np.array(goals_history)
avg_safety_buffer = np.mean(goals_array[:, 1])
print(f"   Mantieni almeno {avg_safety_buffer:.2f} EUR di riserva")

print(f"\n3. STRATEGIA DI INVESTIMENTO:")
avg_aggressiveness = np.mean(goals_array[:, 2])
if avg_aggressiveness > 0.6:
    risk_profile = "Aggressiva"
elif avg_aggressiveness > 0.4:
    risk_profile = "Moderata"
else:
    risk_profile = "Conservativa"
print(f"   Profilo di rischio: {risk_profile} (aggressività: {avg_aggressiveness:.2f})")

print(f"\n4. ORIZZONTE TEMPORALE:")
print(f"   L'agente è riuscito a gestire {month} mesi")
if month >= 120:
    print(f"   ✓ Strategia sostenibile a lungo termine!")
else:
    print(f"   ⚠ Considera di ridurre le spese o aumentare le entrate")

print(f"\n" + "=" * 70)
print("ANALISI COMPLETATA")
print("=" * 70)
