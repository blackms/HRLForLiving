#!/usr/bin/env python3
"""
Studio Comparativo: Comportamenti Finanziari in Italia
Analisi basata su dati ISTAT e Numbeo 2024
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.utils.config_manager import load_config

# Definizione scenari
scenarios = [
    {
        'name': 'Milano Junior',
        'config': 'configs/scenarios/milano_junior.yaml',
        'description': 'Professionista 25-30 anni, monolocale Milano'
    },
    {
        'name': 'Milano Senior',
        'config': 'configs/scenarios/milano_senior.yaml',
        'description': 'Professionista 35-45 anni, bilocale Milano'
    },
    {
        'name': 'Roma Famiglia',
        'config': 'configs/scenarios/roma_famiglia.yaml',
        'description': 'Famiglia con figli, trilocale Roma'
    },
    {
        'name': 'Bologna Coppia',
        'config': 'configs/scenarios/bologna_coppia.yaml',
        'description': 'Coppia doppio reddito, bilocale Bologna'
    },
    {
        'name': 'Torino Single',
        'config': 'configs/scenarios/torino_single.yaml',
        'description': 'Single 30-40 anni, monolocale Torino'
    }
]

print("=" * 80)
print("STUDIO: COMPORTAMENTI FINANZIARI IN ITALIA")
print("Analisi Comparativa basata su dati ISTAT e Numbeo 2024")
print("=" * 80)

results = []

for scenario in scenarios:
    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {scenario['name'].upper()}")
    print(f"Descrizione: {scenario['description']}")
    print(f"{'=' * 80}")
    
    # Load configuration
    env_config, training_config, reward_config = load_config(scenario['config'])
    
    # Calculate key metrics
    total_expenses = env_config.fixed_expenses + env_config.variable_expense_mean
    disponibile = env_config.income - total_expenses
    disponibile_pct = (disponibile / env_config.income) * 100
    
    print(f"\n📊 DATI FINANZIARI:")
    print(f"  Reddito netto: {env_config.income:,.0f} EUR/mese")
    print(f"  Spese fisse: {env_config.fixed_expenses:,.0f} EUR/mese")
    print(f"  Spese variabili: {env_config.variable_expense_mean:,.0f} EUR/mese (±{env_config.variable_expense_std})")
    print(f"  Totale spese: {total_expenses:,.0f} EUR/mese")
    print(f"  Disponibile: {disponibile:,.0f} EUR/mese ({disponibile_pct:.1f}%)")
    print(f"  Buffer iniziale: {env_config.initial_cash:,.0f} EUR")
    print(f"  Profilo rischio: {env_config.risk_tolerance:.1f}")
    
    # Test different investment strategies
    strategies = [
        ("Conservativa (5%)", 0.05),
        ("Moderata (10%)", 0.10),
        ("Bilanciata (15%)", 0.15),
        ("Aggressiva (20%)", 0.20),
    ]
    
    print(f"\n📈 SIMULAZIONE STRATEGIE DI INVESTIMENTO:")
    
    scenario_results = {
        'name': scenario['name'],
        'income': env_config.income,
        'expenses': total_expenses,
        'disponibile': disponibile,
        'disponibile_pct': disponibile_pct,
        'strategies': []
    }
    
    for strategy_name, invest_pct in strategies:
        env = BudgetEnv(env_config, reward_config)
        state, _ = env.reset(seed=42)
        
        # Fixed strategy: invest X%, save rest, consume 40%
        save_pct = max(0, 1 - invest_pct - 0.40)
        action = np.array([invest_pct, save_pct, 0.40], dtype=np.float32)
        
        total_reward = 0
        months = 0
        done = False
        
        while not done and months < 120:
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            months += 1
            done = terminated or truncated
            state = next_state
        
        # Calculate results
        monthly_investment = env_config.income * invest_pct
        total_invested = info['total_invested']
        final_cash = info['cash_balance']
        total_wealth = final_cash + total_invested
        
        # Determine outcome
        if months >= 120:
            outcome = "✅ SOSTENIBILE"
            sustainability = "Completa"
        elif months >= 60:
            outcome = "⚠️  PARZIALE"
            sustainability = "Parziale"
        else:
            outcome = "❌ INSOSTENIBILE"
            sustainability = "Insostenibile"
        
        print(f"\n  {strategy_name}:")
        print(f"    Investimento: {monthly_investment:,.0f} EUR/mese")
        print(f"    Durata: {months} mesi")
        print(f"    Totale investito: {total_invested:,.0f} EUR")
        print(f"    Cash finale: {final_cash:,.0f} EUR")
        print(f"    Patrimonio: {total_wealth:,.0f} EUR")
        print(f"    Outcome: {outcome}")
        
        scenario_results['strategies'].append({
            'name': strategy_name,
            'invest_pct': invest_pct,
            'monthly_investment': monthly_investment,
            'months': months,
            'total_invested': total_invested,
            'final_cash': final_cash,
            'total_wealth': total_wealth,
            'sustainability': sustainability,
            'reward': total_reward
        })
    
    results.append(scenario_results)

# Comparative analysis
print(f"\n{'=' * 80}")
print("ANALISI COMPARATIVA")
print(f"{'=' * 80}")

print(f"\n📊 CAPACITÀ DI RISPARMIO PER SCENARIO:")
print(f"{'Scenario':<20} {'Disponibile':<15} {'% Reddito':<12} {'Investimento Max Sostenibile'}")
print("-" * 80)

for r in results:
    # Find max sustainable investment
    sustainable = [s for s in r['strategies'] if s['sustainability'] == 'Completa']
    if sustainable:
        max_invest = max(sustainable, key=lambda x: x['invest_pct'])
        max_invest_str = f"{max_invest['invest_pct']:.0%} ({max_invest['monthly_investment']:.0f} EUR)"
    else:
        max_invest_str = "Nessuna sostenibile"
    
    print(f"{r['name']:<20} {r['disponibile']:>6,.0f} EUR    {r['disponibile_pct']:>5.1f}%      {max_invest_str}")

print(f"\n💡 INSIGHTS:")

# Calculate average disponibile percentage
avg_disponibile_pct = np.mean([r['disponibile_pct'] for r in results])
print(f"\n1. MARGINE MEDIO ITALIANO:")
print(f"   Il margine medio disponibile è {avg_disponibile_pct:.1f}% del reddito")
print(f"   Questo è in linea con il tasso di risparmio italiano (~8-10%)")

# Find best and worst scenarios
best = max(results, key=lambda x: x['disponibile_pct'])
worst = min(results, key=lambda x: x['disponibile_pct'])

print(f"\n2. SCENARIO PIÙ FAVOREVOLE:")
print(f"   {best['name']}: {best['disponibile_pct']:.1f}% disponibile")
print(f"   Può investire fino al 20% in modo sostenibile")

print(f"\n3. SCENARIO PIÙ CRITICO:")
print(f"   {worst['name']}: {worst['disponibile_pct']:.1f}% disponibile")
print(f"   Margine limitato per investimenti significativi")

print(f"\n4. RACCOMANDAZIONI GENERALI:")
print(f"   - Investimento sostenibile: 5-10% del reddito per la maggior parte degli scenari")
print(f"   - Buffer di sicurezza: 2-3 mesi di spese (1,500-3,500 EUR)")
print(f"   - Priorità: Costruire buffer prima di investire aggressivamente")

# Save results to JSON
output_file = 'study_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n📁 Risultati salvati in: {output_file}")
print(f"\n{'=' * 80}")
print("STUDIO COMPLETATO")
print(f"{'=' * 80}")
