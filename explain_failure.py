#!/usr/bin/env python3
"""
Explainable AI Analysis - Detailed month-by-month breakdown
Shows WHY the agent fails and WHERE the problem occurs
"""

import sys
import os
import numpy as np

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

print("=" * 80)
print("EXPLAINABLE AI - ANALISI DETTAGLIATA DEL FALLIMENTO")
print("=" * 80)

print(f"\n📊 SITUAZIONE INIZIALE:")
print(f"  💰 Entrate mensili: {env_config.income:,.2f} EUR")
print(f"  🏠 Spese fisse: {env_config.fixed_expenses:,.2f} EUR")
print(f"  🛒 Spese variabili medie: {env_config.variable_expense_mean:,.2f} EUR (±{env_config.variable_expense_std} EUR)")
print(f"  💵 Cash iniziale (buffer): {env_config.initial_cash:,.2f} EUR")
print(f"  ⚠️  Soglia sicurezza: {env_config.safety_threshold:,.2f} EUR")
print(f"  📈 Inflazione annua: {env_config.inflation:.1%}")

available = env_config.income - env_config.fixed_expenses - env_config.variable_expense_mean
print(f"\n  ✅ Disponibile teorico: {available:,.2f} EUR/mese")

# Run detailed simulation
state, _ = env.reset(seed=42)
state_history = [state]

aggregated_state = high_agent.aggregate_state(state_history)
goal = high_agent.select_goal(aggregated_state)

print(f"\n🎯 OBIETTIVO STRATEGICO INIZIALE (High-Level Agent):")
print(f"  Target investimento: {goal[0]:.1%}")
print(f"  Buffer sicurezza: {goal[1]:,.2f} EUR")
print(f"  Aggressività: {goal[2]:.2f}")

print(f"\n" + "=" * 80)
print("SIMULAZIONE MESE PER MESE")
print("=" * 80)

month = 0
done = False
failure_reason = None

while not done and month < 120:
    # Get action from trained agent
    action = low_agent.act(state, goal, deterministic=True)
    
    # Calculate amounts
    invest_amount = action[0] * env_config.income
    save_amount = action[1] * env_config.income
    consume_amount = action[2] * env_config.income
    
    # Store cash before step
    cash_before = env.cash_balance
    
    # Execute step
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Calculate actual expenses this month
    actual_variable = state[2]  # Variable expense from state
    total_expenses = env_config.fixed_expenses + actual_variable
    
    # Calculate cash flow
    cash_in = env_config.income
    cash_out = total_expenses + invest_amount
    net_flow = cash_in - cash_out
    
    print(f"\n📅 MESE {month + 1}:")
    print(f"  💰 Cash iniziale: {cash_before:,.2f} EUR")
    
    print(f"\n  📥 ENTRATE:")
    print(f"    Stipendio: +{env_config.income:,.2f} EUR")
    
    print(f"\n  📤 USCITE:")
    print(f"    Spese fisse: -{env_config.fixed_expenses:,.2f} EUR")
    print(f"    Spese variabili: -{actual_variable:,.2f} EUR")
    print(f"    Investimento: -{invest_amount:,.2f} EUR ({action[0]:.1%})")
    print(f"    TOTALE USCITE: -{cash_out:,.2f} EUR")
    
    print(f"\n  💸 FLUSSO NETTO: {net_flow:+,.2f} EUR")
    print(f"  💵 Cash finale: {info['cash_balance']:,.2f} EUR")
    print(f"  📊 Totale investito: {info['total_invested']:,.2f} EUR")
    print(f"  🎁 Reward: {reward:+.2f}")
    
    # Check for problems
    if info['cash_balance'] < 0:
        print(f"\n  ❌ FALLIMENTO: Cash negativo!")
        failure_reason = "cash_negativo"
        print(f"  🔍 CAUSA: Hai speso {-net_flow:,.2f} EUR più di quanto guadagnato")
        print(f"  💡 Il buffer di {env_config.initial_cash:,.2f} EUR si è esaurito")
    elif info['cash_balance'] < env_config.safety_threshold:
        print(f"  ⚠️  WARNING: Cash sotto soglia sicurezza!")
    
    # Update goal every 6 months
    if (month + 1) % 6 == 0:
        aggregated_state = high_agent.aggregate_state(state_history)
        new_goal = high_agent.select_goal(aggregated_state)
        if not np.array_equal(goal, new_goal):
            print(f"\n  🔄 CAMBIO STRATEGIA:")
            print(f"    Vecchio target investimento: {goal[0]:.1%}")
            print(f"    Nuovo target investimento: {new_goal[0]:.1%}")
            goal = new_goal
    
    state = next_state
    state_history.append(state)
    done = terminated or truncated
    month += 1
    
    if done:
        break

print(f"\n" + "=" * 80)
print("ANALISI DEL FALLIMENTO")
print("=" * 80)

print(f"\n⏱️  Durata: {month} mesi")
print(f"💰 Cash finale: {info['cash_balance']:,.2f} EUR")
print(f"📊 Totale investito: {info['total_invested']:,.2f} EUR")
print(f"💎 Patrimonio totale: {info['cash_balance'] + info['total_invested']:,.2f} EUR")

print(f"\n🔍 PERCHÉ È FALLITO?")
print(f"\n1. PROBLEMA STRUTTURALE:")
print(f"   Disponibile reale: ~{available:,.2f} EUR/mese")
print(f"   Investimento medio: {info['total_invested']/month:,.2f} EUR/mese")
print(f"   Deficit mensile: {(info['total_invested']/month) - available:,.2f} EUR")

print(f"\n2. CONSUMO DEL BUFFER:")
print(f"   Buffer iniziale: {env_config.initial_cash:,.2f} EUR")
print(f"   Buffer consumato: {env_config.initial_cash - info['cash_balance']:,.2f} EUR")
print(f"   Consumo mensile medio: {(env_config.initial_cash - info['cash_balance'])/month:,.2f} EUR")

print(f"\n3. EFFETTO INFLAZIONE:")
inflation_impact = env_config.fixed_expenses * (1 + env_config.inflation)**(month/12) - env_config.fixed_expenses
print(f"   Aumento spese fisse: +{inflation_impact:,.2f} EUR")
print(f"   Questo riduce ulteriormente il disponibile")

print(f"\n💡 CONCLUSIONE:")
print(f"   L'agente ha imparato a investire {action[0]:.1%} del reddito (~{env_config.income * action[0]:,.2f} EUR/mese)")
print(f"   Ma con solo {available:,.2f} EUR disponibili, questo è INSOSTENIBILE")
print(f"   Il buffer di {env_config.initial_cash:,.2f} EUR copre solo ~{month} mesi")

print(f"\n📈 STRATEGIA SOSTENIBILE:")
max_sustainable_invest = available * 0.5  # 50% del disponibile per sicurezza
max_sustainable_pct = max_sustainable_invest / env_config.income
print(f"   Investimento massimo sostenibile: {max_sustainable_invest:,.2f} EUR/mese ({max_sustainable_pct:.1%})")
print(f"   Questo lascerebbe {available - max_sustainable_invest:,.2f} EUR/mese per imprevisti")

print(f"\n" + "=" * 80)
print("RACCOMANDAZIONI")
print("=" * 80)

print(f"\n1. 🎯 OBIETTIVO REALISTICO:")
print(f"   Investi MAX {max_sustainable_invest:,.2f} EUR/mese ({max_sustainable_pct:.1%} del reddito)")

print(f"\n2. 💰 AUMENTA IL DISPONIBILE:")
print(f"   Opzione A: Riduci spese variabili da {env_config.variable_expense_mean} a 300 EUR (+100 EUR/mese)")
print(f"   Opzione B: Aumenta entrate a 3400 EUR (+200 EUR/mese)")
print(f"   Opzione C: Entrambe (+300 EUR/mese disponibili)")

print(f"\n3. 🛡️  MANTIENI IL BUFFER:")
print(f"   Non scendere mai sotto {env_config.safety_threshold:,.2f} EUR")
print(f"   Ricostruisci il buffer quando possibile")

print(f"\n" + "=" * 80)
