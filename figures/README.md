# Visualizations - HRL Finance Study

This directory contains publication-quality visualizations generated for the technical study and research papers.

## Available Charts

### 1. portfolio_evolution.png
**Portfolio Evolution Over Time**

Detailed comparison between scenarios with and without investment returns (6% annual).

**Panels:**
- **Panel A: Cash Balance** - Shows liquid funds evolution with safety threshold indicator
- **Panel B: Portfolio Value vs Invested Capital** - Compares portfolio value (with returns) to principal invested
- **Panel C: Total Wealth** - Combined cash + investments showing total net worth
- **Panel D: Investment Gains** - Cumulative gains from returns over time

**Scenario**: Bologna Coppia (Dual-income couple in Bologna), Moderate Strategy (10% investment)

**Key Insights:**
- Demonstrates the impact of 6% annual returns on portfolio growth
- Shows how returns accumulate over time while maintaining cash stability
- Illustrates the difference between invested capital and portfolio value

### 2. strategy_comparison.png
**Investment Strategy Comparison**

Comparative analysis of three investment strategies (Conservative 5%, Moderate 10%, Balanced 15%) with and without returns.

**Panels:**
- **Panel A: Sustainability Duration** - How many months each strategy remains viable
- **Panel B: Final Wealth** - Total wealth accumulated at the end of simulation

**Key Insights:**
- Returns improve final wealth but don't necessarily extend sustainability
- More aggressive strategies may fail earlier despite higher potential returns
- Conservative strategies provide longer sustainability but lower wealth accumulation

**Strategies Analyzed:**
- **Conservative**: 5% investment, 55% savings, 40% consumption
- **Moderate**: 10% investment, 50% savings, 40% consumption
- **Balanced**: 15% investment, 45% savings, 40% consumption

### 3. returns_distribution.png
**Monthly Returns Distribution**

Statistical analysis of stochastic investment returns.

**Panels:**
- **Panel A: Monthly Returns Histogram** - Distribution of monthly return percentages with mean indicator
- **Panel B: Cumulative Gains Over Time** - Growth of investment gains throughout the simulation

**Parameters**: 
- Mean monthly return (μ): 0.5% (~6% annual)
- Volatility (σ): 2% monthly standard deviation
- Distribution: Normal (Gaussian)

**Key Insights:**
- Shows realistic market volatility with both positive and negative months
- Demonstrates how small monthly returns compound into significant long-term gains
- Provides statistical summary (mean, std dev, min, max) for validation

## Generating Charts

To regenerate all visualizations:

```bash
python3 visualize_results.py
```

**Requirements:**
- Configurations: `configs/scenarios/bologna_coppia.yaml` and `bologna_coppia_with_returns.yaml`
- Python packages: matplotlib, numpy
- Automatically creates `figures/` directory if it doesn't exist

**Output:**
```
✓ Saved: figures/portfolio_evolution.png
✓ Saved: figures/strategy_comparison.png
✓ Saved: figures/returns_distribution.png
```

## Usage

These charts are referenced in the technical paper (`TECHNICAL_PAPER.md`) and can be used for:

- **Research Publications**: High-resolution figures suitable for academic papers
- **Presentations**: Clear, professional visualizations for stakeholder meetings
- **Technical Documentation**: Visual aids for explaining system behavior
- **Financial Reports**: Demonstrating investment strategy outcomes
- **Educational Materials**: Teaching concepts of investment returns and portfolio management

## Technical Specifications

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with transparency support
- **Dimensions**: 
  - Portfolio Evolution: 14×10 inches (4200×3000 pixels)
  - Strategy Comparison: 14×6 inches (4200×1800 pixels)
  - Returns Distribution: 14×6 inches (4200×1800 pixels)
- **Color Palette**: Professional color scheme
  - Red (#e74c3c): Without returns / Conservative
  - Blue (#3498db): With returns / Moderate
  - Green (#2ecc71): Gains / Balanced
- **Font**: System default with clear labels and legends
- **Grid**: Semi-transparent for readability

## Customization

To customize visualizations, edit `visualize_results.py`:

- **Change scenario**: Modify config paths to use different scenarios
- **Adjust strategies**: Change investment percentages in the strategies list
- **Modify colors**: Update the `colors` list with different hex codes
- **Change simulation length**: Adjust the `while not done and month < 120` condition
- **Add more panels**: Extend the subplot grid and add new analyses
