#!/usr/bin/env python3
"""Test sustainable investment strategies with your real budget"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.utils.config_manager import load_config

# Load config
env_config, training_config, reward_config = load_config('configs/personal_realistic.yaml')

print("=" * 70)
print("TEST STRATEGIE SOSTENIBILI")
print("=" * 70)
print(f"\nSituazione Reale:")
print(f"  Entrate: {env_config.income} EUR/mese")
print(f"  Spese fisse: {env_config.fixed_expenses} EUR")
print(f"  Spese variabili: {env_config.variable_expense_mean} EUR")
print(f"  Disponibile: {env_config.income - env_config.fixed_expenses - env_config.variable_expense_mean} EUR/mese")
print(f"  Buffer iniziale: {env_config.initial_cash} EUR")

# Test different strategies
strategies = [
    ("Conservativa", [0.10, 0.50, 0.40]),  # 10% invest, 50% save, 40% consume
    ("Moderata", [0.20, 0.40, 0.40]),      # 20% invest, 40% save, 40% consume
    ("Bilanciata", [0.25, 0.35, 0.40]),    # 25% invest, 35% save, 40% consume
    ("Aggressiva", [0.30, 0.30, 0.40]),    # 30% invest, 30% save, 40% consume
]

results = []

for strategy_name, action in strategies:
    env = BudgetEnv(env_config, reward_config)
    state, _ = env.reset(seed=42)
    
    total_reward = 0
    months_survived = 0
    total_invested = 0
    final_cash = env_config.initial_cash
    
    done = False
    while not done and months_survived < 120:
        next_state, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
        
        total_reward += reward
        months_survived += 1
        total_invested = info['total_invested']
        final_cash = info['cash_balance']
        
        done = terminated or truncated
        state = next_state
    
    # Calculate monthly investment
    monthly_investment = env_config.income * action[0]
    monthly_savings = env_config.income * action[1]
    
    results.append({
        'name': strategy_name,
        'action': action,
        'months': months_survived,
        'reward': total_reward,
        'invested': total_invested,
        'final_cash': final_cash,
        'monthly_investment': monthly_investment,
        'monthly_savings': monthly_savings,
        'total_wealth': final_cash + total_invested
    })

# Print results
print(f"\n" + "=" * 70)
print("RISULTATI COMPARATIVI")
print("=" * 70)

for r in results:
    print(f"\n{r['name'].upper()}:")
    print(f"  Allocazione: {r['action'][0]:.0%} invest, {r['action'][1]:.0%} save, {r['action'][2]:.0%} consume")
    print(f"  Investimento mensile: {r['monthly_investment']:.2f} EUR")
    print(f"  Risparmio mensile: {r['monthly_savings']:.2f} EUR")
    print(f"  Mesi sopravvissuti: {r['months']}")
    print(f"  Totale investito: {r['invested']:.2f} EUR")
    print(f"  Cash finale: {r['final_cash']:.2f} EUR")
    print(f"  Patrimonio totale: {r['total_wealth']:.2f} EUR")
    print(f"  Reward totale: {r['reward']:.2f}")
    
    if r['months'] >= 120:
        print(f"  ✓ STRATEGIA SOSTENIBILE!")
    else:
        print(f"  ✗ Fallimento dopo {r['months']} mesi")

# Find best sustainable strategy
sustainable = [r for r in results if r['months'] >= 120]
if sustainable:
    best = max(sustainable, key=lambda x: x['total_wealth'])
    print(f"\n" + "=" * 70)
    print("STRATEGIA CONSIGLIATA")
    print("=" * 70)
    print(f"\n{best['name'].upper()} è la migliore strategia sostenibile!")
    print(f"\nCosa fare ogni mese con i tuoi 3200 EUR:")
    print(f"  1. Paga spese fisse: 2160 EUR")
    print(f"  2. Paga spese variabili: ~400 EUR")
    print(f"  3. Investi: {best['monthly_investment']:.2f} EUR")
    print(f"  4. Risparmia: {best['monthly_savings']:.2f} EUR")
    print(f"  5. Spese discrezionali: {env_config.income * best['action'][2]:.2f} EUR")
    print(f"\nDopo 10 anni (120 mesi):")
    print(f"  Totale investito: {best['invested']:.2f} EUR")
    print(f"  Patrimonio finale: {best['total_wealth']:.2f} EUR")
else:
    print(f"\n" + "=" * 70)
    print("⚠️ ATTENZIONE")
    print("=" * 70)
    print(f"\nNessuna strategia è sostenibile a lungo termine!")
    print(f"Con {env_config.income - env_config.fixed_expenses - env_config.variable_expense_mean} EUR/mese disponibili,")
    print(f"è difficile investire significativamente.")
    print(f"\nConsiderazioni:")
    print(f"  - Riduci le spese variabili")
    print(f"  - Cerca entrate aggiuntive")
    print(f"  - Rivedi le spese fisse (affitto, auto)")

print(f"\n" + "=" * 70)
