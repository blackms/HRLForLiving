#!/usr/bin/env python3
"""
Test Investment Returns Impact
Confronto tra scenario con e senza rendimenti degli investimenti
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.utils.config_manager import load_config

print("=" * 80)
print("TEST: IMPATTO DEI RENDIMENTI DEGLI INVESTIMENTI")
print("=" * 80)

# Test scenario: Bologna Coppia
scenarios = [
    {
        'name': 'SENZA Rendimenti (0%)',
        'config': 'configs/scenarios/bologna_coppia.yaml',
        'description': 'Scenario base senza rendimenti'
    },
    {
        'name': 'CON Rendimenti (~6% annuo)',
        'config': 'configs/scenarios/bologna_coppia_with_returns.yaml',
        'description': 'Scenario con rendimenti realistici'
    }
]

# Test strategy: Moderata (10% investimento)
invest_pct = 0.10
action = np.array([invest_pct, 0.50, 0.40], dtype=np.float32)

print(f"\nStrategia testata: {invest_pct:.0%} investimento mensile")
print(f"Orizzonte: 120 mesi (10 anni)")

results = []

for scenario in scenarios:
    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"{'=' * 80}")
    
    # Load config
    env_config, _, reward_config = load_config(scenario['config'])
    
    print(f"\nParametri investimento:")
    print(f"  Tipo rendimento: {env_config.investment_return_type}")
    if env_config.investment_return_type != "none":
        print(f"  Rendimento medio: {env_config.investment_return_mean:.2%} mensile")
        print(f"  Rendimento annuo: {(1 + env_config.investment_return_mean)**12 - 1:.2%}")
        print(f"  Volatilità: {env_config.investment_return_std:.2%} mensile")
    
    # Run simulation
    env = BudgetEnv(env_config, reward_config)
    state, _ = env.reset(seed=42)
    
    monthly_investment = env_config.income * invest_pct
    
    total_reward = 0
    months = 0
    done = False
    
    monthly_returns = []
    portfolio_values = []
    
    while not done and months < 120:
        next_state, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        months += 1
        done = terminated or truncated
        
        monthly_returns.append(info.get('investment_return', 0))
        portfolio_values.append(info['investment_value'])
        
        state = next_state
    
    # Calculate results
    total_invested = info['total_invested']
    portfolio_value = info['investment_value']
    investment_gains = portfolio_value - total_invested
    final_cash = info['cash_balance']
    total_wealth = final_cash + portfolio_value
    
    # Calculate effective annual return
    if months > 0 and total_invested > 0:
        total_return_pct = (portfolio_value / total_invested - 1) * 100
        years = months / 12
        annual_return = ((portfolio_value / total_invested) ** (1/years) - 1) * 100 if years > 0 else 0
    else:
        total_return_pct = 0
        annual_return = 0
    
    print(f"\n📊 RISULTATI ({months} mesi):")
    print(f"  Investimento mensile: {monthly_investment:,.0f} EUR")
    print(f"  Totale investito (principale): {total_invested:,.0f} EUR")
    print(f"  Valore portafoglio: {portfolio_value:,.0f} EUR")
    print(f"  Guadagni da rendimenti: {investment_gains:,.0f} EUR")
    print(f"  Rendimento totale: {total_return_pct:+.2f}%")
    print(f"  Rendimento annuo effettivo: {annual_return:.2f}%")
    print(f"  Cash finale: {final_cash:,.0f} EUR")
    print(f"  Patrimonio totale: {total_wealth:,.0f} EUR")
    
    if months >= 120:
        print(f"  ✅ SOSTENIBILE per 10 anni!")
    else:
        print(f"  ❌ Insostenibile (durata: {months} mesi)")
    
    results.append({
        'name': scenario['name'],
        'months': months,
        'total_invested': total_invested,
        'portfolio_value': portfolio_value,
        'investment_gains': investment_gains,
        'final_cash': final_cash,
        'total_wealth': total_wealth,
        'annual_return': annual_return
    })

# Comparative analysis
print(f"\n{'=' * 80}")
print("ANALISI COMPARATIVA")
print(f"{'=' * 80}")

without_returns = results[0]
with_returns = results[1]

print(f"\n💰 IMPATTO DEI RENDIMENTI:")
print(f"\n  Durata:")
print(f"    Senza rendimenti: {without_returns['months']} mesi")
print(f"    Con rendimenti: {with_returns['months']} mesi")
print(f"    Differenza: {with_returns['months'] - without_returns['months']:+d} mesi")

print(f"\n  Valore Portafoglio:")
print(f"    Senza rendimenti: {without_returns['portfolio_value']:,.0f} EUR")
print(f"    Con rendimenti: {with_returns['portfolio_value']:,.0f} EUR")
print(f"    Guadagno extra: {with_returns['investment_gains']:,.0f} EUR")

print(f"\n  Patrimonio Totale:")
print(f"    Senza rendimenti: {without_returns['total_wealth']:,.0f} EUR")
print(f"    Con rendimenti: {with_returns['total_wealth']:,.0f} EUR")
print(f"    Differenza: {with_returns['total_wealth'] - without_returns['total_wealth']:+,.0f} EUR")
print(f"    Incremento: {(with_returns['total_wealth'] / without_returns['total_wealth'] - 1) * 100:+.1f}%")

print(f"\n💡 CONCLUSIONI:")

if with_returns['months'] >= 120 and without_returns['months'] < 120:
    print(f"  ✅ I rendimenti rendono la strategia SOSTENIBILE!")
    print(f"  Con ~6% annuo, la strategia diventa viable a lungo termine")
elif with_returns['months'] > without_returns['months']:
    improvement = with_returns['months'] - without_returns['months']
    print(f"  ⚠️  I rendimenti migliorano la durata di {improvement} mesi")
    print(f"  Ma la strategia rimane insostenibile a lungo termine")
else:
    print(f"  ❌ Anche con rendimenti, la strategia non è sostenibile")

gains_pct = (with_returns['investment_gains'] / with_returns['total_invested']) * 100
print(f"\n  📈 Guadagni da rendimenti: {with_returns['investment_gains']:,.0f} EUR ({gains_pct:.1f}%)")
print(f"  Questo rappresenta il potere del compound interest!")

print(f"\n{'=' * 80}")
print("TEST COMPLETATO")
print(f"{'=' * 80}")
