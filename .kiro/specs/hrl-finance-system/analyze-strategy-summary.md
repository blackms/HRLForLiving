# Strategy Analysis Script Addition

## Status: ✅ COMPLETE

## Summary

Added `analyze_strategy.py` - a utility script that loads trained models and analyzes the learned financial strategy to provide practical, actionable recommendations for real-world application.

## Implementation Details

### File Created
- `analyze_strategy.py` (134 lines)

### Key Features

1. **Model Loading**
   - Loads configuration from YAML file (`configs/personal_realistic.yaml`)
   - Loads trained high-level agent (`models/personal_realistic_high_agent.pt`)
   - Loads trained low-level agent (`models/personal_realistic_low_agent.pt`)

2. **Deterministic Simulation**
   - Runs a complete episode (up to 120 months) with trained policy
   - Uses deterministic action selection for consistent results
   - Updates strategic goals every 6 months (matching training behavior)
   - Tracks all actions, goals, cash balances, and investments

3. **Analysis Output**
   - **Initial Situation**: Income, expenses, available funds, initial cash
   - **Simulation Results**: Number of months survived, allocation statistics
   - **Average Allocation**: Mean invest/save/consume ratios over the episode
   - **Financial Outcomes**: Final cash, total invested, total wealth
   - **Monthly Investment**: Average investment amount in currency

4. **Practical Recommendations**
   - **Monthly Allocation Breakdown**: Specific amounts in EUR/USD for each category
   - **Safety Buffer**: Recommended minimum cash reserve based on learned goals
   - **Risk Profile**: Assessment as Conservative/Moderate/Aggressive based on aggressiveness parameter
   - **Sustainability**: Evaluation of long-term viability (did agent survive 120 months?)

### Output Format

The script outputs in Italian (for personal finance context) with clear sections:
- `ANALISI STRATEGIA FINANZIARIA APPRESA` (Learned Financial Strategy Analysis)
- `RISULTATI SIMULAZIONE` (Simulation Results)
- `RACCOMANDAZIONI PRATICHE` (Practical Recommendations)

### Example Output Structure

```
======================================================================
ANALISI STRATEGIA FINANZIARIA APPRESA
======================================================================

Situazione Iniziale:
  Entrate mensili: 3200 EUR
  Spese fisse: 1400 EUR
  Spese variabili medie: 700 EUR
  Disponibile: 1100 EUR/mese
  Cash iniziale: 0 EUR

======================================================================
RISULTATI SIMULAZIONE (120 mesi)
======================================================================

Allocazione Media:
  Investimento: 32.5%
  Risparmio: 45.0%
  Consumo: 22.5%

Risultati Finali:
  Cash finale: 15234.56 EUR
  Totale investito: 124800.00 EUR
  Patrimonio totale: 140034.56 EUR

Investimento Mensile Medio: 1040.00 EUR

======================================================================
RACCOMANDAZIONI PRATICHE
======================================================================

1. ALLOCAZIONE MENSILE CONSIGLIATA:
   Con 3200 EUR/mese:
   - Investi: 1040.00 EUR (32.5%)
   - Risparmia: 1440.00 EUR (45.0%)
   - Spese discrezionali: 720.00 EUR (22.5%)

2. BUFFER DI SICUREZZA:
   Mantieni almeno 1200.00 EUR di riserva

3. STRATEGIA DI INVESTIMENTO:
   Profilo di rischio: Moderata (aggressività: 0.52)

4. ORIZZONTE TEMPORALE:
   L'agente è riuscito a gestire 120 mesi
   ✓ Strategia sostenibile a lungo termine!
```

## Use Cases

1. **Understanding Learned Policies**: See what the agent actually learned during training
2. **Extracting Actionable Advice**: Get specific recommendations for real-world application
3. **Validating Training**: Verify that the learned strategy makes sense
4. **Comparing Profiles**: Analyze different behavioral profiles to see their differences
5. **Personal Finance Planning**: Use the recommendations as a starting point for personal budgeting

## Configuration

The script is currently configured for:
- Configuration: `configs/personal_realistic.yaml`
- High-level agent: `models/personal_realistic_high_agent.pt`
- Low-level agent: `models/personal_realistic_low_agent.pt`

To analyze different models, modify these paths in the script:
```python
env_config, training_config, reward_config = load_config('configs/your_config.yaml')
high_agent.load('models/your_high_agent.pt')
low_agent.load('models/your_low_agent.pt')
```

## Documentation Updates

Updated the following documentation files:

1. **README.md**
   - Added "Analyzing Learned Strategy" section with detailed usage
   - Updated project structure to include `analyze_strategy.py`
   - Added to "Utility Scripts" section
   - Updated "Development Status" to mark as completed

2. **QUICK_START.md**
   - Added step 3.5 "Analyze Your Learned Strategy"
   - Added to "What's Next?" section under "Analyze Your Strategy"

3. **CHANGELOG.md**
   - Added entry for strategy analysis script with feature list

4. **.kiro/specs/hrl-finance-system/analyze-strategy-summary.md**
   - Created this summary document

## Benefits

1. **Bridges Theory and Practice**: Translates learned RL policies into actionable financial advice
2. **Interpretability**: Makes the agent's decision-making transparent and understandable
3. **Validation**: Helps verify that training produced sensible strategies
4. **User-Friendly**: Provides recommendations in clear, practical terms
5. **Customizable**: Easy to modify for different models and configurations

## Technical Details

### Dependencies
- Uses existing system components (no new dependencies)
- Imports: `BudgetEnv`, `FinancialStrategist`, `BudgetExecutor`, `load_config`
- Standard libraries: `sys`, `os`, `numpy`, `matplotlib.pyplot`

### Execution Flow
1. Load configuration and initialize environment
2. Load trained agent models
3. Reset environment with fixed seed (42) for reproducibility
4. Run deterministic simulation:
   - Generate initial goal from high-level agent
   - Execute monthly actions from low-level agent
   - Update goal every 6 months
   - Track all metrics
5. Compute statistics (averages, totals)
6. Display results and recommendations

### Key Implementation Choices
- **Deterministic Actions**: Uses `deterministic=True` for consistent results
- **Fixed Seed**: Uses `seed=42` for reproducibility
- **Goal Update Interval**: Matches training (every 6 months)
- **Episode Length**: Up to 120 months (10 years) for long-term analysis
- **Italian Output**: Appropriate for personal finance context

## Future Enhancements

Potential improvements for future versions:
- [ ] Command-line arguments for model paths and configuration
- [ ] Support for multiple simulations with different seeds
- [ ] Comparison mode (analyze multiple profiles side-by-side)
- [ ] Export recommendations to PDF or JSON
- [ ] Visualization of allocation over time
- [ ] Sensitivity analysis (how recommendations change with different parameters)
- [ ] Multi-language support (English, Spanish, etc.)

## Conclusion

The `analyze_strategy.py` script successfully bridges the gap between trained RL models and practical financial advice. It provides clear, actionable recommendations that users can apply to their personal finance planning, making the HRL system's learned strategies accessible and useful for real-world application.

