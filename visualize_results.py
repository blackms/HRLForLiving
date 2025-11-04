#!/usr/bin/env python3
"""
Visualizzazione Grafiche per Studio HRL Finance
Crea grafici per analisi comparativa e evoluzione temporale
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.environment.budget_env import BudgetEnv
from src.utils.config_manager import load_config

# Create output directory
os.makedirs('figures', exist_ok=True)

print("=" * 80)
print("GENERAZIONE VISUALIZZAZIONI")
print("=" * 80)

# ============================================================================
# GRAFICO 1: Evoluzione Portafoglio con e senza Rendimenti
# ============================================================================
print("\n1. Evoluzione Portafoglio nel Tempo...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Evoluzione Finanziaria: Con vs Senza Rendimenti\nScenario: Bologna Coppia, Strategia Moderata (10%)', 
             fontsize=14, fontweight='bold')

scenarios = [
    ("Senza Rendimenti", "configs/scenarios/bologna_coppia.yaml"),
    ("Con Rendimenti 6%", "configs/scenarios/bologna_coppia_with_returns.yaml"),
]

colors = ['#e74c3c', '#3498db']
action = np.array([0.10, 0.50, 0.40], dtype=np.float32)

for idx, (scenario_name, config_path) in enumerate(scenarios):
    env_config, _, reward_config = load_config(config_path)
    env = BudgetEnv(env_config, reward_config)
    state, _ = env.reset(seed=42)
    
    months = []
    cash_history = []
    invested_history = []
    portfolio_history = []
    wealth_history = []
    
    month = 0
    done = False
    
    while not done and month < 120:
        next_state, reward, terminated, truncated, info = env.step(action)
        
        months.append(month)
        cash_history.append(info['cash_balance'])
        invested_history.append(info['total_invested'])
        portfolio_history.append(info['investment_value'])
        wealth_history.append(info['cash_balance'] + info['investment_value'])
        
        done = terminated or truncated
        month += 1
        state = next_state
    
    color = colors[idx]
    
    # Cash Balance
    axes[0, 0].plot(months, cash_history, label=scenario_name, color=color, linewidth=2)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(y=env_config.safety_threshold, color='orange', linestyle='--', alpha=0.5, label='Soglia sicurezza')
    axes[0, 0].set_xlabel('Mesi')
    axes[0, 0].set_ylabel('Cash Balance (EUR)')
    axes[0, 0].set_title('A) Evoluzione Cash Balance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Portfolio Value
    axes[0, 1].plot(months, portfolio_history, label=scenario_name, color=color, linewidth=2)
    axes[0, 1].plot(months, invested_history, label=f'{scenario_name} (Principale)', 
                    color=color, linestyle='--', alpha=0.6)
    axes[0, 1].set_xlabel('Mesi')
    axes[0, 1].set_ylabel('Valore Portafoglio (EUR)')
    axes[0, 1].set_title('B) Valore Portafoglio vs Capitale Investito')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Total Wealth
    axes[1, 0].plot(months, wealth_history, label=scenario_name, color=color, linewidth=2)
    axes[1, 0].set_xlabel('Mesi')
    axes[1, 0].set_ylabel('Patrimonio Totale (EUR)')
    axes[1, 0].set_title('C) Patrimonio Totale (Cash + Investimenti)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Investment Returns (only for scenario with returns)
    if idx == 1:
        returns = np.array(portfolio_history) - np.array(invested_history)
        axes[1, 1].plot(months, returns, label='Guadagni da Rendimenti', color=color, linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].fill_between(months, 0, returns, alpha=0.3, color=color)
        axes[1, 1].set_xlabel('Mesi')
        axes[1, 1].set_ylabel('Guadagni Cumulativi (EUR)')
        axes[1, 1].set_title('D) Guadagni da Rendimenti nel Tempo')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/portfolio_evolution.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvato: figures/portfolio_evolution.png")

# ============================================================================
# GRAFICO 2: Confronto Strategie
# ============================================================================
print("\n2. Confronto Strategie di Investimento...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confronto Strategie: Impatto dei Rendimenti\nScenario: Bologna Coppia', 
             fontsize=14, fontweight='bold')

strategies = [
    ("Conservativa", 0.05),
    ("Moderata", 0.10),
    ("Bilanciata", 0.15),
]

# Run simulations
results_without = []
results_with = []

for strategy_name, invest_pct in strategies:
    action = np.array([invest_pct, 1-invest_pct-0.40, 0.40], dtype=np.float32)
    
    # Without returns
    env_config, _, reward_config = load_config("configs/scenarios/bologna_coppia.yaml")
    env = BudgetEnv(env_config, reward_config)
    state, _ = env.reset(seed=42)
    
    month = 0
    done = False
    while not done and month < 120:
        next_state, reward, terminated, truncated, info = env.step(action)
        month += 1
        done = terminated or truncated
        state = next_state
    
    results_without.append({
        'strategy': strategy_name,
        'months': month,
        'wealth': info['cash_balance'] + info['investment_value']
    })
    
    # With returns
    env_config, _, reward_config = load_config("configs/scenarios/bologna_coppia_with_returns.yaml")
    env = BudgetEnv(env_config, reward_config)
    state, _ = env.reset(seed=42)
    
    month = 0
    done = False
    while not done and month < 120:
        next_state, reward, terminated, truncated, info = env.step(action)
        month += 1
        done = terminated or truncated
        state = next_state
    
    results_with.append({
        'strategy': strategy_name,
        'months': month,
        'wealth': info['cash_balance'] + info['investment_value'],
        'gains': info['investment_value'] - info['total_invested']
    })

# Plot durations
x = np.arange(len(strategies))
width = 0.35

durations_without = [r['months'] for r in results_without]
durations_with = [r['months'] for r in results_with]

bars1 = axes[0].bar(x - width/2, durations_without, width, label='Senza Rendimenti', color='#e74c3c', alpha=0.8)
bars2 = axes[0].bar(x + width/2, durations_with, width, label='Con Rendimenti 6%', color='#3498db', alpha=0.8)

axes[0].set_xlabel('Strategia')
axes[0].set_ylabel('Durata (mesi)')
axes[0].set_title('A) Durata Sostenibilità per Strategia')
axes[0].set_xticks(x)
axes[0].set_xticklabels([s[0] for s in strategies])
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=120, color='green', linestyle='--', alpha=0.5, label='Target 10 anni')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

# Plot wealth
wealth_without = [r['wealth'] for r in results_without]
wealth_with = [r['wealth'] for r in results_with]

bars3 = axes[1].bar(x - width/2, wealth_without, width, label='Senza Rendimenti', color='#e74c3c', alpha=0.8)
bars4 = axes[1].bar(x + width/2, wealth_with, width, label='Con Rendimenti 6%', color='#3498db', alpha=0.8)

axes[1].set_xlabel('Strategia')
axes[1].set_ylabel('Patrimonio Finale (EUR)')
axes[1].set_title('B) Patrimonio Finale per Strategia')
axes[1].set_xticks(x)
axes[1].set_xticklabels([s[0] for s in strategies])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figures/strategy_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvato: figures/strategy_comparison.png")

# ============================================================================
# GRAFICO 3: Distribuzione Rendimenti Mensili
# ============================================================================
print("\n3. Distribuzione Rendimenti Mensili...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Analisi Rendimenti Mensili\nScenario: Bologna Coppia con Rendimenti 6%, Strategia Moderata', 
             fontsize=14, fontweight='bold')

# Run simulation and collect monthly returns
env_config, _, reward_config = load_config("configs/scenarios/bologna_coppia_with_returns.yaml")
env = BudgetEnv(env_config, reward_config)
state, _ = env.reset(seed=42)

action = np.array([0.10, 0.50, 0.40], dtype=np.float32)

monthly_returns = []
monthly_return_pcts = []
portfolio_values = []

month = 0
done = False

while not done and month < 120:
    next_state, reward, terminated, truncated, info = env.step(action)
    
    if info['investment_value'] > 0:
        monthly_returns.append(info['investment_return'])
        if month > 0:
            return_pct = (info['investment_return'] / portfolio_values[-1]) * 100 if portfolio_values[-1] > 0 else 0
            monthly_return_pcts.append(return_pct)
    
    portfolio_values.append(info['investment_value'])
    
    month += 1
    done = terminated or truncated
    state = next_state

# Histogram of monthly returns
axes[0].hist(monthly_return_pcts, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
axes[0].axvline(x=np.mean(monthly_return_pcts), color='red', linestyle='--', linewidth=2, label=f'Media: {np.mean(monthly_return_pcts):.2f}%')
axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
axes[0].set_xlabel('Rendimento Mensile (%)')
axes[0].set_ylabel('Frequenza')
axes[0].set_title('A) Distribuzione Rendimenti Mensili')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Cumulative returns over time
cumulative_returns = np.cumsum(monthly_returns)
axes[1].plot(range(len(cumulative_returns)), cumulative_returns, color='#2ecc71', linewidth=2)
axes[1].fill_between(range(len(cumulative_returns)), 0, cumulative_returns, alpha=0.3, color='#2ecc71')
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[1].set_xlabel('Mesi')
axes[1].set_ylabel('Guadagni Cumulativi (EUR)')
axes[1].set_title('B) Guadagni Cumulativi nel Tempo')
axes[1].grid(True, alpha=0.3)

# Add statistics text
stats_text = f'Media: {np.mean(monthly_return_pcts):.2f}%\nStd Dev: {np.std(monthly_return_pcts):.2f}%\nMin: {np.min(monthly_return_pcts):.2f}%\nMax: {np.max(monthly_return_pcts):.2f}%'
axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

plt.tight_layout()
plt.savefig('figures/returns_distribution.png', dpi=300, bbox_inches='tight')
print("   ✓ Salvato: figures/returns_distribution.png")

print(f"\n{'=' * 80}")
print("VISUALIZZAZIONI COMPLETATE")
print(f"{'=' * 80}")
print("\nGrafici salvati in:")
print("  - figures/portfolio_evolution.png")
print("  - figures/strategy_comparison.png")
print("  - figures/returns_distribution.png")
print("\nUsa questi grafici per il paper tecnico e le presentazioni!")
