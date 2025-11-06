# HRL Finance System - User Guide

Complete guide to using the Personal Finance Optimization HRL System web interface.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard](#dashboard)
3. [Creating Scenarios](#creating-scenarios)
4. [Training Models](#training-models)
5. [Running Simulations](#running-simulations)
6. [Viewing Results](#viewing-results)
7. [Comparing Scenarios](#comparing-scenarios)
8. [Generating Reports](#generating-reports)
9. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### System Requirements

- **Backend**: Python 3.10+, 4GB RAM minimum
- **Frontend**: Modern web browser (Chrome, Firefox, Safari, Edge)
- **Network**: Backend and frontend must be able to communicate (default: localhost)

### First-Time Setup

1. **Start the Backend Server**:
   ```bash
   cd backend
   uvicorn main:socket_app --reload --port 8000
   ```
   
   The API will be available at http://localhost:8000

2. **Start the Frontend Application**:
   ```bash
   cd frontend
   npm run dev
   ```
   
   The web interface will be available at http://localhost:5173

3. **Open Your Browser**:
   Navigate to http://localhost:5173 to access the application

### Interface Overview

The application consists of:
- **Navigation Sidebar** (desktop) or **Hamburger Menu** (mobile)
- **Main Content Area** with page-specific content
- **Theme Toggle** for light/dark mode
- **Toast Notifications** for feedback messages

## Dashboard

The Dashboard is your home base, providing an overview of your financial scenarios, trained models, and recent activity.

### What You'll See

**Statistics Cards**:
- Total number of scenarios created
- Total number of trained models
- Total number of simulations run

**Quick Actions**:
- **New Scenario**: Create a new financial scenario
- **Start Training**: Train a model on a scenario
- **Run Simulation**: Evaluate a trained model
- **Compare Results**: Compare multiple simulations

**Recent Scenarios**:
- Shows your 3 most recent scenarios
- Displays income, available income percentage, and risk tolerance
- Click to edit a scenario

**Recent Models**:
- Shows your 3 most recent trained models
- Displays episode count, training date, and final reward
- Click to run a simulation with the model

**Recent Activity**:
- Timeline of recent actions (training, simulations, reports)
- Relative timestamps (e.g., "5 minutes ago")

### Getting Started from Dashboard

1. **First Time**: Click "New Scenario" to create your first financial scenario
2. **After Creating Scenarios**: Click "Start Training" to train a model
3. **After Training**: Click "Run Simulation" to evaluate your model
4. **After Simulations**: Click "Compare Results" to analyze different scenarios

## Creating Scenarios

Scenarios define the financial parameters for your simulation. Think of them as different "what-if" situations you want to explore.

### Step-by-Step Guide

1. **Navigate to Scenario Builder**:
   - Click "New Scenario" from Dashboard, or
   - Click "Scenarios" in the sidebar

2. **Choose a Starting Point**:
   - **Use a Template**: Select from 5 preset templates:
     - **Conservative**: Low risk, high savings buffer
     - **Balanced**: Moderate risk, balanced approach
     - **Aggressive**: High risk, maximum investment
     - **Young Professional**: Single professional with owned home
     - **Young Couple**: Dual income couple with rental
   - **Start from Scratch**: Leave template as "None" and fill in all fields

3. **Configure Basic Information**:
   - **Name**: Give your scenario a descriptive name (e.g., "Bologna Couple 2024")
   - **Description**: Optional description of the scenario

4. **Set Environment Parameters**:
   - **Monthly Income**: Your total monthly income in EUR
   - **Fixed Expenses**: Rent, utilities, insurance, etc.
   - **Variable Expenses**: Groceries, entertainment (mean and std dev)
   - **Inflation Rate**: Expected monthly inflation (e.g., 2% = 0.02)
   - **Safety Threshold**: Minimum cash buffer before penalties
   - **Max Months**: Simulation duration (e.g., 60 = 5 years)
   - **Initial Cash**: Starting cash balance
   - **Risk Tolerance**: 0 (conservative) to 1 (aggressive)

5. **Configure Investment Returns**:
   - **Return Type**: Stochastic (realistic) or Fixed (deterministic)
   - **Mean Return**: Expected monthly return (e.g., 0.5% = 0.005)
   - **Std Deviation**: Return volatility (e.g., 2% = 0.02)

6. **Review the Preview Panel**:
   - Check monthly cash flow breakdown
   - Verify available income percentage
   - Confirm risk profile matches your intent

7. **Save Your Scenario**:
   - Click "Save Scenario" to create
   - You'll be redirected to the scenarios list

### Understanding the Preview

The preview panel shows:
- **Income**: Your monthly income
- **Fixed**: Fixed monthly expenses
- **Variable**: Average variable expenses
- **Available**: Income minus expenses (what's left to allocate)
- **Available %**: Percentage of income available for investment/savings
- **Risk Profile**: Conservative (<30%), Balanced (30-60%), Aggressive (>60%)

### Editing Scenarios

1. Navigate to the scenario you want to edit
2. Click the edit button or scenario card
3. Modify any parameters (except the name)
4. Click "Save Scenario" to update

### Example Scenarios

**Young Professional (Owned Home)**:
- Income: €2,000/month
- Fixed Expenses: €770 (utilities, insurance, food)
- Variable Expenses: €500 ± €120
- Available: €730 (36.5%)
- Risk Tolerance: 0.65 (moderate-high)

**Young Couple (Rental)**:
- Income: €3,200/month
- Fixed Expenses: €1,800 (rent, utilities, insurance)
- Variable Expenses: €800 ± €150
- Available: €600 (18.8%)
- Risk Tolerance: 0.55 (moderate)

## Training Models

Training teaches the AI system to make optimal financial decisions for your scenario. This process uses reinforcement learning to learn from experience.

### Step-by-Step Guide

1. **Navigate to Training Monitor**:
   - Click "Start Training" from Dashboard, or
   - Click "Training" in the sidebar

2. **Select a Scenario**:
   - Choose the scenario you want to train on from the dropdown
   - The scenario must exist before you can train

3. **Configure Training Parameters**:
   - **Number of Episodes**: How many simulations to run (default: 1000)
     - More episodes = better learning but longer training time
     - Recommended: 1000-5000 for most scenarios
   - **Save Interval**: Save checkpoint every N episodes (default: 100)
   - **Evaluation Episodes**: Episodes to run for evaluation (default: 10)
   - **Random Seed**: Optional seed for reproducibility

4. **Start Training**:
   - Click "Start Training"
   - The interface will switch to real-time monitoring mode

5. **Monitor Progress**:
   - **Progress Bar**: Shows completion percentage
   - **Current Metrics**: Real-time performance indicators
     - Average Reward: Higher is better
     - Duration: Average episode length in months
     - Stability: Percentage of time with positive cash
     - Average Cash: Current cash balance
     - Average Invested: Total invested amount
     - Goal Adherence: How well executor follows strategist
   - **Live Charts**: Four charts updating in real-time
     - Average Reward Over Time
     - Average Duration Over Time
     - Stability Over Time
     - Cash vs Investment Over Time

6. **Wait for Completion**:
   - Training typically takes 5-30 minutes depending on episodes
   - You can stop training early if needed (model will be saved)
   - The model is automatically saved when training completes

### Understanding Training Metrics

**Average Reward**:
- Measures overall performance
- Should increase over time as the agent learns
- Higher values indicate better financial outcomes

**Duration**:
- Average number of months the agent survives
- Should increase as the agent learns to avoid bankruptcy
- Target: Reach max_months (e.g., 60 months)

**Stability**:
- Percentage of time with positive cash balance
- Should approach 100% as training progresses
- Values >95% indicate good financial stability

**Goal Adherence**:
- How well the low-level agent follows high-level strategy
- Lower values are better (less deviation)
- Values <0.05 indicate good coordination

### Training Tips

**For Faster Training**:
- Start with 1000 episodes for initial experiments
- Use smaller max_months in scenario (e.g., 30 instead of 60)
- Increase episodes to 5000+ for production models

**For Better Results**:
- Train for at least 2000 episodes
- Monitor stability - should reach >90%
- Check that reward is increasing consistently
- Ensure duration reaches max_months

**If Training Fails**:
- Agent keeps going bankrupt: Increase safety_threshold or use conservative profile
- Reward not increasing: Try different learning rates or more episodes
- Training too slow: Reduce number of episodes or max_months

## Running Simulations

Simulations evaluate your trained models to see how they perform on financial scenarios. Unlike training, simulations use deterministic policies (no exploration).

### Step-by-Step Guide

1. **Navigate to Simulation Runner**:
   - Click "Run Simulation" from Dashboard, or
   - Click "Simulation" in the sidebar

2. **Select a Trained Model**:
   - Choose from the dropdown (shows model name and scenario)
   - Only models that have completed training will appear

3. **Select a Scenario**:
   - Choose the scenario to simulate
   - Can be the same scenario used for training or a different one

4. **Configure Simulation**:
   - **Number of Episodes**: How many runs to simulate (default: 10)
     - More episodes = more reliable statistics
     - Recommended: 10-50 episodes
   - **Random Seed**: Optional seed for reproducibility

5. **Run Simulation**:
   - Click "Run Simulation"
   - Progress indicator shows while simulation runs
   - Results appear automatically when complete

6. **Review Results**:
   - **Summary Statistics**: Duration, wealth, gains, portfolio
   - **Strategy Breakdown**: Invest/Save/Consume percentages
   - **Action Buttons**:
     - View Detailed Results: See charts and episode data
     - Compare Scenarios: Add to comparison view
     - Generate Report: Create PDF/HTML report

### Understanding Simulation Results

**Duration**:
- Mean ± standard deviation in months
- Shows how long the agent survives on average
- Lower std dev indicates more consistent performance

**Total Wealth**:
- Final wealth (cash + portfolio value)
- Mean ± std dev across all episodes
- Higher is better

**Investment Gains**:
- Profit/loss from investments
- Shown with +/- prefix and color coding
- Return percentage calculated from invested amount

**Final Portfolio**:
- Value of investment portfolio at end
- Includes both invested amount and returns
- Cash breakdown shown separately

**Strategy Breakdown**:
- **Invest %**: Percentage allocated to investments
- **Save %**: Percentage kept as cash savings
- **Consume %**: Percentage used for discretionary spending
- Should sum to 100%

### Simulation Tips

**For Reliable Statistics**:
- Run at least 10 episodes
- Use 50+ episodes for production analysis
- Check std dev - lower is more consistent

**For Comparison**:
- Run same model on different scenarios
- Run different models on same scenario
- Use consistent episode counts for fair comparison

**For Reproducibility**:
- Set a random seed
- Document the seed used
- Use same seed for repeated runs

## Viewing Results

The Results Viewer provides detailed analysis and visualization of simulation results.

### Step-by-Step Guide

1. **Navigate to Results Viewer**:
   - Click "View Detailed Results" from Simulation Runner, or
   - Access via URL with simulation ID

2. **Review Summary Statistics**:
   - Four metric cards at the top
   - Duration, Total Wealth, Investment Gains, Final Portfolio
   - All values show mean ± std dev

3. **Select an Episode**:
   - Use the dropdown to select individual episodes
   - View episode-specific data and charts
   - Compare different episodes to see variation

4. **Explore Charts** (4 tabs):
   - **Cash Balance**: Track cash over time
   - **Portfolio Evolution**: See investment growth
   - **Wealth Accumulation**: Total wealth progression
   - **Action Distribution**: Pie chart of strategy

5. **Analyze Strategy**:
   - Strategy Learned section shows aggregate behavior
   - Three progress bars for Invest/Save/Consume
   - Percentages averaged across all episodes

6. **Take Action**:
   - **Compare Scenarios**: Add to comparison view
   - **Export Data**: Download results as JSON
   - **Generate Report**: Create formatted report

### Understanding the Charts

**Cash Balance Over Time**:
- Blue line showing cash progression
- Should stay positive (above 0)
- Dips indicate spending, rises indicate saving
- Pattern shows cash management strategy

**Portfolio Evolution**:
- Green line: Amount invested (cumulative)
- Purple line: Portfolio value (with returns)
- Gap between lines shows investment gains/losses
- Upward trend indicates successful investing

**Wealth Accumulation**:
- Blue: Cash balance
- Green: Portfolio value
- Orange: Total wealth (cash + portfolio)
- Shows overall wealth building trajectory

**Action Distribution**:
- Pie chart with 3 segments
- Blue: Invest, Green: Save, Purple: Consume
- Shows learned allocation strategy
- Balanced distribution indicates adaptive behavior

### Analysis Tips

**Look for These Patterns**:
- Cash balance staying positive throughout
- Portfolio value growing over time
- Total wealth increasing consistently
- Stable action distribution (not erratic)

**Red Flags**:
- Cash going negative (bankruptcy)
- Portfolio value declining
- Erratic action distribution
- High standard deviations

**Comparing Episodes**:
- Select different episodes from dropdown
- Look for consistency across episodes
- High variation suggests unstable policy
- Low variation indicates reliable strategy

## Comparing Scenarios

The Comparison view enables side-by-side analysis of multiple simulations to identify the best performing scenarios.

### Step-by-Step Guide

1. **Navigate to Comparison**:
   - Click "Compare Results" from Dashboard, or
   - Click "Compare Scenarios" from Results Viewer, or
   - Click "Comparison" in the sidebar

2. **Select Simulations** (up to 4):
   - Click on simulation cards to select/deselect
   - Selected cards show blue border and background
   - Selection counter shows "X / 4 selected"
   - Click "Clear All" to deselect all

3. **Review Metrics Table**:
   - Side-by-side comparison of 8 key metrics
   - Color-coded columns for each simulation
   - Difference percentages shown below values
   - Green (+) for improvements, Red (-) for declines

4. **Analyze Charts**:
   - **Total Wealth Comparison**: Bar chart
   - **Duration Comparison**: Bar chart
   - **Investment Gains Comparison**: Bar chart
   - **Strategy Distribution**: Grouped bar chart
   - **Wealth Over Time**: Line chart comparing trajectories

5. **Review Key Insights**:
   - Automatically calculated highlights:
     - 🏆 Highest Total Wealth
     - 📈 Best Investment Returns
     - ⚡ Shortest Duration
   - Color-coded insight cards

6. **Export Data**:
   - **Export CSV**: Download comparison table
   - **Export JSON**: Download full results data
   - Files include timestamp in filename

### Understanding the Comparison

**Metrics Table**:
- Each column represents one simulation
- Rows show different metrics
- Percentage differences relative to first simulation
- Helps identify which scenario performs best

**Bar Charts**:
- Visual comparison of key metrics
- Color-coded by simulation
- Easy to spot differences at a glance
- Hover for exact values

**Wealth Over Time Chart**:
- Shows wealth trajectories for all simulations
- Multiple colored lines (one per simulation)
- Reveals which scenario builds wealth faster
- Shows consistency vs volatility

**Key Insights**:
- Automatically identifies best performers
- Saves time analyzing results
- Highlights most important findings
- Helps make decisions quickly

### Comparison Tips

**Selecting Simulations**:
- Compare similar scenarios (e.g., different risk levels)
- Compare same scenario with different models
- Limit to 2-3 for clearer visualization
- Use 4 only when necessary

**Interpreting Results**:
- Look for consistent winners across metrics
- Consider trade-offs (e.g., wealth vs stability)
- Check standard deviations for reliability
- Review strategy differences

**Making Decisions**:
- Highest wealth isn't always best
- Consider your risk tolerance
- Look at stability and consistency
- Choose strategy that matches your goals

## Generating Reports

Reports provide professional documentation of your simulation results for sharing, archiving, or presentation.

### Step-by-Step Guide

1. **Open Report Modal**:
   - Click "Generate Report" from Results Viewer, or
   - Click "Generate Report" from Simulation Runner

2. **Configure Report**:
   - **Report Title**: Customize the title (default provided)
   - **Report Format**: Choose HTML or PDF
     - HTML: Always available, viewable in browser
     - PDF: Requires WeasyPrint, professional document format
   - **Sections to Include**: Select which sections to include
     - ✅ Summary Statistics
     - ✅ Scenario Configuration
     - ✅ Training Configuration
     - ✅ Detailed Results
     - ✅ Strategy Learned
     - ✅ Charts & Visualizations

3. **Generate Report**:
   - Click "Generate Report"
   - Wait for generation (usually <5 seconds)
   - Success banner appears when ready

4. **Download Report**:
   - Click "Download Report" button
   - Report opens in new browser tab
   - Save or print as needed

5. **Close Modal**:
   - Click "Close" or X button
   - Modal can be reopened to generate another report

### Report Sections Explained

**Summary Statistics**:
- Duration, total wealth, investment gains
- Final cash and portfolio values
- Mean and standard deviation for all metrics

**Scenario Configuration**:
- Income, expenses, inflation parameters
- Investment return settings
- Risk tolerance and safety thresholds

**Training Configuration**:
- Number of episodes trained
- Learning rates and batch size
- High-level planning period

**Detailed Results**:
- Wealth breakdown (cash, portfolio, total)
- Investment performance analysis
- Episode-by-episode statistics

**Strategy Learned**:
- Investment, saving, consumption percentages
- Visual progress bars
- Strategy interpretation

**Charts & Visualizations**:
- Episode data tables
- First 10 episodes shown
- Cash, invested, and portfolio values

### Report Tips

**Choosing Format**:
- **HTML**: Best for quick viewing, sharing via web
- **PDF**: Best for formal documents, presentations, archiving

**Selecting Sections**:
- Include all sections for comprehensive reports
- Exclude sections for focused reports
- Summary + Strategy for executive summaries
- All sections for technical documentation

**Customizing Title**:
- Use descriptive titles (e.g., "Q4 2024 Financial Analysis")
- Include scenario name for clarity
- Add date or version if tracking over time

**Sharing Reports**:
- HTML reports can be emailed or hosted
- PDF reports are universally compatible
- Include simulation ID for reference
- Document any assumptions or notes separately

## Tips and Best Practices

### Getting Started

**Start Simple**:
1. Use a template scenario (Balanced recommended)
2. Train with 1000 episodes first
3. Run 10-episode simulation
4. Review results before customizing

**Learn the System**:
1. Try all three templates (Conservative, Balanced, Aggressive)
2. Compare their results
3. Understand how parameters affect outcomes
4. Then create custom scenarios

### Creating Effective Scenarios

**Realistic Parameters**:
- Use actual income and expense data when possible
- Set safety threshold to 1-2 months of expenses
- Use realistic investment returns (0.5% monthly ≈ 6% annual)
- Match risk tolerance to your actual risk appetite

**Scenario Naming**:
- Use descriptive names (e.g., "Milan_Senior_Professional_2024")
- Include key characteristics (location, profile, year)
- Avoid generic names like "test" or "scenario1"

**Testing Variations**:
- Create multiple scenarios with small variations
- Test different risk tolerance levels
- Try different income/expense ratios
- Compare results to find optimal parameters

### Training Best Practices

**Episode Count**:
- 1000 episodes: Quick experiments, initial testing
- 2000-3000 episodes: Good balance of time and quality
- 5000+ episodes: Production models, best performance

**Monitoring Training**:
- Watch for reward increasing consistently
- Ensure stability reaches >90%
- Check that duration reaches max_months
- Stop early if metrics plateau

**When to Retrain**:
- Scenario parameters changed significantly
- Model performance is poor
- Want to try different training settings
- Need better results for production use

### Simulation Best Practices

**Episode Count**:
- 10 episodes: Quick checks, initial evaluation
- 20-30 episodes: Standard analysis
- 50+ episodes: Statistical significance, production use

**Reproducibility**:
- Always set random seed for important simulations
- Document seed value
- Use same seed for comparisons
- Record all parameters used

**Validation**:
- Run multiple simulations with different seeds
- Check consistency across runs
- Verify results make sense
- Compare with expected outcomes

### Analysis Best Practices

**Interpreting Results**:
- Don't focus only on total wealth
- Consider stability and consistency
- Look at strategy learned
- Check standard deviations

**Making Decisions**:
- Compare multiple scenarios
- Consider your risk tolerance
- Think about real-world applicability
- Don't over-optimize for single metric

**Documentation**:
- Generate reports for important simulations
- Export data for further analysis
- Keep notes on parameter choices
- Track changes over time

### Performance Optimization

**Faster Training**:
- Reduce max_months in scenario
- Use fewer episodes for experiments
- Increase episodes only for final models

**Better Results**:
- Train longer (5000+ episodes)
- Use appropriate risk tolerance
- Set realistic safety thresholds
- Validate with multiple simulations

**System Resources**:
- Close unused browser tabs
- Run one training session at a time
- Monitor system memory usage
- Restart backend if performance degrades

### Troubleshooting

**Training Issues**:
- Agent goes bankrupt: Increase safety_threshold
- Reward not increasing: Train longer or adjust parameters
- Training too slow: Reduce episodes or max_months

**Simulation Issues**:
- Poor results: Retrain with more episodes
- Inconsistent results: Run more simulation episodes
- Unexpected behavior: Check scenario parameters

**Interface Issues**:
- Page not loading: Check backend is running
- WebSocket disconnected: Refresh page
- Charts not updating: Check browser console for errors

### Advanced Tips

**Scenario Design**:
- Model real-world situations accurately
- Test edge cases (high expenses, low income)
- Create scenario families (conservative/balanced/aggressive versions)
- Document assumptions and rationale

**Model Evaluation**:
- Test models on different scenarios
- Compare with baseline strategies
- Validate against domain knowledge
- Consider multiple evaluation metrics

**Workflow Optimization**:
- Create template scenarios for common cases
- Batch train multiple models
- Automate repetitive tasks
- Keep organized naming conventions

**Data Management**:
- Export important results regularly
- Generate reports for key findings
- Back up trained models
- Maintain scenario library

---

## Need More Help?

- **API Documentation**: http://localhost:8000/docs
- **Backend README**: See `backend/README.md`
- **Frontend README**: See `frontend/README.md`
- **Troubleshooting Guide**: See `TROUBLESHOOTING.md`
- **Quick Start**: See `QUICK_START.md`
- **Full Documentation**: See `README.md`

## Keyboard Shortcuts

- **Tab**: Navigate between form fields
- **Enter**: Submit forms, activate buttons
- **Escape**: Close modals
- **Arrow Keys**: Navigate dropdowns and selectors

## Accessibility Features

- Full keyboard navigation support
- Screen reader compatible
- High contrast mode available
- Focus indicators on all interactive elements
- ARIA labels for assistive technologies

---

**Last Updated**: November 2024
**Version**: 1.0
**For**: HRL Finance System Web Interface
