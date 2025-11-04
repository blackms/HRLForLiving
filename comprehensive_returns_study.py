#!/usr/bin/env python3
"""
Studio Completo: Impatto Rendimenti Investimenti
Analisi su diversi orizzonti temporali e strategie
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.utils.config_manager import load_config

print("=" * 80)
print("STUDIO COMPLETO: IMPATTO RENDIMENTI INVESTIMENTI")
print("Scenario: Bologna Coppia (3200 EUR/mese, 600 EUR disponibili)")
print("=" * 80)

# Test different investment percentages
strategies = [
    ("Conservativa", 0.05),
    ("Moderata", 0.10),
    ("Bilanciata", 0.15),
]

# Test with and without returns
return_scenarios = [
    ("Senza Rendimenti", "configs/scenarios/bologna_coppia.yaml"),
    ("Con Rendimenti 6%", "configs/scenarios/bologna_coppia_with_returns.yaml"),
]

all_results = []

for return_name, config_path in return_scenarios:
    print(f"\n{'=' * 80}")
    print(f"SCENARIO: {return_name}")
    print(f"{'=' * 80}")
    
    env_config, _, reward_config = load_config(config_path)
    
    for strategy_name, invest_pct in strategies:
        action = np.array([invest_pct, 1-invest_pct-0.40, 0.40], dtype=np.float32)
        
        # Run 10 simulations with different seeds
        durations = []
        final_wealths = []
        investment_gains_list = []
        
        for seed in range(10):
            env = BudgetEnv(env_config, reward_config)
            state, _ = env.reset(seed=seed)
            
            months = 0
            done = False
            
            while not done and months < 120:
                next_state, reward, terminated, truncated, info = env.step(action)
                months += 1
                done = terminated or truncated
                state = next_state
            
            durations.append(months)
            final_wealths.append(info['cash_balance'] + info['investment_value'])
            investment_gains_list.append(info['investment_value'] - info['total_invested'])
        
        # Calculate statistics
        avg_duration = np.mean(durations)
        avg_wealth = np.mean(final_wealths)
        avg_gains = np.mean(investment_gains_list)
        
        all_results.append({
            'return_scenario': return_name,
            'strategy': strategy_name,
            'invest_pct': invest_pct,
            'avg_duration': avg_duration,
            'avg_wealth': avg_wealth,
            'avg_gains': avg_gains,
            'sustainable': avg_duration >= 120
        })
        
        print(f"\n  {strategy_name} ({invest_pct:.0%}):")
        print(f"    Durata media: {avg_duration:.1f} mesi")
        print(f"    Patrimonio medio: {avg_wealth:,.0f} EUR")
        print(f"    Guadagni medi: {avg_gains:+,.0f} EUR")
        if avg_duration >= 120:
            print(f"    ✅ SOSTENIBILE")
        else:
            print(f"    ❌ Insostenibile")

# Comparative analysis
print(f"\n{'=' * 80}")
print("ANALISI COMPARATIVA: IMPATTO DEI RENDIMENTI")
print(f"{'=' * 80}")

print(f"\n{'Strategia':<20} {'Senza Rend.':<15} {'Con Rend. 6%':<15} {'Differenza':<15}")
print("-" * 80)

for strategy_name, invest_pct in strategies:
    without = [r for r in all_results if r['strategy'] == strategy_name and 'Senza' in r['return_scenario']][0]
    with_ret = [r for r in all_results if r['strategy'] == strategy_name and 'Con' in r['return_scenario']][0]
    
    diff_months = with_ret['avg_duration'] - without['avg_duration']
    diff_wealth = with_ret['avg_wealth'] - without['avg_wealth']
    
    print(f"{strategy_name:<20} {without['avg_duration']:>6.1f} mesi   {with_ret['avg_duration']:>6.1f} mesi   {diff_months:+6.1f} mesi")
    print(f"{'  Patrimonio':<20} {without['avg_wealth']:>10,.0f} EUR {with_ret['avg_wealth']:>10,.0f} EUR {diff_wealth:+10,.0f} EUR")
    print(f"{'  Guadagni':<20} {without['avg_gains']:>10,.0f} EUR {with_ret['avg_gains']:>10,.0f} EUR")
    print()

print(f"\n💡 CONCLUSIONI:")

# Check if any strategy becomes sustainable with returns
without_sustainable = [r for r in all_results if 'Senza' in r['return_scenario'] and r['sustainable']]
with_sustainable = [r for r in all_results if 'Con' in r['return_scenario'] and r['sustainable']]

if len(with_sustainable) > len(without_sustainable):
    print(f"  ✅ I rendimenti rendono {len(with_sustainable) - len(without_sustainable)} strategie sostenibili!")
elif len(with_sustainable) > 0:
    print(f"  ✅ {len(with_sustainable)} strategie sono sostenibili con rendimenti")
else:
    print(f"  ⚠️  Nessuna strategia è sostenibile anche con rendimenti del 6%")
    print(f"  Il problema è strutturale: margine disponibile troppo basso")

# Calculate average improvement
avg_duration_improvement = np.mean([
    r['avg_duration'] for r in all_results if 'Con' in r['return_scenario']
]) - np.mean([
    r['avg_duration'] for r in all_results if 'Senza' in r['return_scenario']
])

avg_wealth_improvement = np.mean([
    r['avg_wealth'] for r in all_results if 'Con' in r['return_scenario']
]) - np.mean([
    r['avg_wealth'] for r in all_results if 'Senza' in r['return_scenario']
])

print(f"\n  📊 Miglioramento medio con rendimenti 6%:")
print(f"    Durata: +{avg_duration_improvement:.1f} mesi")
print(f"    Patrimonio: +{avg_wealth_improvement:,.0f} EUR")

print(f"\n  📈 Il compound interest ha un impatto significativo sul patrimonio finale")
print(f"  Ma non risolve il problema della sostenibilità con margini stretti")

print(f"\n{'=' * 80}")
print("STUDIO COMPLETATO")
print(f"{'=' * 80}")
