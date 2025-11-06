# HRL Finance System Frontend

React + TypeScript frontend for the HRL Finance System.

## Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Tech Stack

- React 19
- TypeScript
- Vite
- Tailwind CSS
- Recharts (data visualization)
- React Router (navigation)
- Axios (HTTP client)
- Socket.IO Client (WebSocket)

## Development

The app will be available at http://localhost:5173

## Project Structure

```
frontend/
├── src/
│   ├── components/      # Reusable UI components
│   │   └── Layout.tsx   # Main layout with navigation
│   ├── contexts/        # React contexts
│   │   └── ThemeContext.tsx  # Theme provider (light/dark mode)
│   ├── pages/           # Page components
│   │   ├── Dashboard.tsx
│   │   ├── ScenarioBuilder.tsx
│   │   ├── TrainingMonitor.tsx
│   │   ├── SimulationRunner.tsx
│   │   ├── ResultsViewer.tsx
│   │   └── Comparison.tsx
│   ├── services/        # API and WebSocket clients
│   │   ├── api.ts       # REST API client
│   │   └── websocket.ts # WebSocket client
│   ├── types/           # TypeScript type definitions
│   │   └── index.ts     # Core types matching backend models
│   ├── utils/           # Utility functions
│   ├── App.tsx          # Main app component with routing
│   └── main.tsx         # App entry point
├── public/              # Static assets
└── index.html           # HTML template
```

## Type Definitions

The `src/types/index.ts` file contains TypeScript interfaces that match the backend Pydantic models:

### Core Configuration Types

- **EnvironmentConfig**: Financial simulation environment parameters
  - Income, expenses, inflation, risk tolerance
  - Investment return parameters
  - Safety thresholds and initial conditions

- **TrainingConfig**: HRL training parameters
  - Number of episodes, discount factors
  - Learning rates, batch size
  - High-level planning period

- **RewardConfig**: Reward function coefficients
  - Alpha, beta, gamma, delta, lambda, mu parameters

### Scenario Types

- **Scenario**: Complete scenario configuration
  - Name, description, timestamps
  - Environment, training, and reward configs

- **ScenarioSummary**: Lightweight scenario info for lists
  - Key metrics: income, expenses, risk tolerance
  - Available income percentage

### Model Types

- **ModelSummary**: Trained model summary for lists
  - Name, scenario, file size, training date
  - Final metrics: reward, duration, cash, invested

- **ModelDetail**: Complete model information
  - Agent file paths, metadata, history
  - Environment, training, and reward configs
  - Processed training history and final metrics

### Training Types

- **TrainingRequest**: Request to start training
  - Scenario name, episodes, save interval
  - Evaluation episodes, random seed

- **TrainingProgress**: Real-time training updates
  - Current episode, average metrics
  - Reward, duration, cash, invested
  - Stability, goal adherence, elapsed time

- **TrainingStatus**: Current training state
  - Is training active, scenario name
  - Current/total episodes, latest progress

### Simulation Types

- **SimulationRequest**: Request to run simulation
  - Model name, scenario name
  - Number of episodes, random seed

- **SimulationResult**: Aggregated simulation results
  - Summary statistics (mean, std dev)
  - Wealth breakdown, strategy metrics
  - Individual episode data

- **EpisodeData**: Single episode results
  - Duration, final balances, wealth
  - Time series: cash, invested, portfolio
  - Action history

### Report Types

- **ReportRequest**: Request to generate report
  - Simulation ID, report type (pdf/html)
  - Sections to include, custom title

- **Report**: Generated report metadata
  - Report ID, file path, file size
  - Generation timestamp, sections included

## API Integration

The frontend communicates with the backend via:

1. **REST API** (`src/services/api.ts`)
   - Scenarios CRUD operations
   - Training control (start/stop/status)
   - Simulation execution
   - Model management
   - Report generation

2. **WebSocket** (`src/services/websocket.ts`)
   - Real-time training progress updates
   - Live metric streaming during training

### Example API Usage

```typescript
import { api } from './services/api';

// List scenarios
const scenarios = await api.getScenarios();

// Start training
const response = await api.startTraining({
  scenario_name: 'bologna_coppia',
  num_episodes: 1000,
  save_interval: 100,
  eval_episodes: 10
});

// Run simulation
const results = await api.runSimulation({
  model_name: 'bologna_coppia',
  scenario_name: 'bologna_coppia',
  num_episodes: 10
});

// Generate report
const report = await api.generateReport({
  simulation_id: results.simulation_id,
  report_type: 'html',
  title: 'My Financial Report'
});
```

### Example WebSocket Usage

```typescript
import { websocketService } from './services/websocket';

// Connect to WebSocket
websocketService.connect();

// Listen for training progress
websocketService.on('training_progress', (data) => {
  console.log(`Episode ${data.episode}/${data.total_episodes}`);
  console.log(`Reward: ${data.avg_reward}`);
});

// Listen for training completion
websocketService.on('training_completed', (data) => {
  console.log('Training completed!');
});

// Disconnect when done
websocketService.disconnect();
```

## Pages

### Dashboard ✅ **IMPLEMENTED**

The Dashboard is the main landing page providing an overview of the entire system.

**Features:**
- **Statistics Summary Cards**: Display counts for scenarios, models, and simulations with icons
- **Quick Actions**: Four action buttons for common workflows:
  - New Scenario → Navigate to Scenario Builder
  - Start Training → Navigate to Training Monitor
  - Run Simulation → Navigate to Simulation Runner
  - Compare Results → Navigate to Comparison View
- **Recent Scenarios**: Shows up to 3 most recent scenarios with:
  - Scenario name and description
  - Income and available income percentage
  - Risk tolerance badge (Low/Medium/High with color coding)
  - Empty state with "Create your first scenario" CTA
- **Recent Models**: Shows up to 3 most recent trained models with:
  - Model name and associated scenario
  - Episode count and training date (relative time)
  - Final reward metric
  - Empty state with "Train your first model" CTA
- **Recent Activity Feed**: Timeline of recent actions including:
  - Simulation runs with episode counts
  - Model training completions
  - Relative timestamps (e.g., "5 minutes ago", "2 hours ago")
  - Activity type icons (🎯 training, 🔬 simulation, 📝 scenario, 🤖 model)
- **Loading States**: Animated spinner during data fetch
- **Error Handling**: Error banner with retry button
- **Refresh Button**: Manual data reload capability
- **Responsive Design**: Adapts from 1 to 4 columns based on screen size
- **Dark Mode**: Full support for light/dark themes

**API Integration:**
- `api.listScenarios()` - Fetches all scenarios
- `api.listModels()` - Fetches all trained models
- `api.getSimulationHistory()` - Fetches simulation history

**Navigation:**
- Clicking scenario cards → Scenario Builder page
- Clicking model cards → Simulation Runner page
- Quick action buttons → Respective feature pages

### Scenario Builder ✅ **IMPLEMENTED**

The Scenario Builder allows users to create and edit financial scenarios with comprehensive configuration options.

**Features:**
- **Basic Information Section**:
  - Scenario name input (required, min 3 characters)
  - Description textarea (optional)
  - Template selector dropdown with 5 presets (conservative, balanced, aggressive, young_professional, young_couple)
  - Name field disabled in edit mode (immutable identifier)
- **Environment Configuration** (9 parameters):
  - Monthly income (EUR)
  - Fixed expenses (EUR)
  - Variable expense mean and std dev (EUR)
  - Inflation rate (%)
  - Safety threshold (EUR)
  - Max months (simulation horizon)
  - Initial cash (EUR)
  - Risk tolerance (0-1 scale)
- **Investment Returns Configuration**:
  - Return type selector (stochastic/fixed)
  - Mean return (% per month)
  - Standard deviation (% per month)
- **Training Configuration**:
  - Number of episodes
  - High-level period (months)
  - Batch size
  - Learning rate (low-level agent)
- **Preview Panel** (sticky sidebar):
  - Monthly cash flow breakdown (income, expenses, available)
  - Available income percentage
  - Risk profile badge (Conservative/Balanced/Aggressive with color coding)
  - Key metrics summary (safety buffer, initial cash, time horizon, inflation)
  - Investment returns summary (type, expected return, volatility)
  - Updates in real-time as form values change
- **Form Validation**:
  - 15+ validation rules for all required fields
  - Field-level error messages (red text below inputs)
  - Red borders on invalid fields
  - Submit-level error banner for API errors
  - Prevents invalid data submission
- **Edit Mode Support**:
  - Load existing scenario via `?edit=scenario_name` query parameter
  - Pre-populate all form fields
  - Update instead of create on save
  - Different page title and button text
- **User Experience**:
  - Loading spinner when loading scenario
  - Save button shows "Saving..." with spinner
  - Cancel button returns to scenarios list
  - Back button in header
  - Auto-redirect to scenarios list on successful save
- **Responsive Design**: 3-column layout on desktop (2 cols form + 1 col preview), adapts to mobile
- **Dark Mode**: Full support for light/dark themes

**API Integration:**
- `api.getScenarioTemplates()` - Loads preset templates on mount
- `api.getScenario(name)` - Loads scenario for editing
- `api.createScenario(scenario)` - Creates new scenario
- `api.updateScenario(name, scenario)` - Updates existing scenario

**Navigation:**
- Back button → Previous page
- Cancel button → `/scenarios` list
- Save success → `/scenarios` list

**Validation Rules:**
- Name: Required, min 3 characters
- Income: Must be positive
- Fixed expenses: Cannot be negative
- Variable expenses: Cannot be negative
- Variable expense std dev: Cannot be negative
- Inflation: Must be between -100% and 100%
- Safety threshold: Cannot be negative
- Max months: Must be positive
- Initial cash: Cannot be negative
- Risk tolerance: Must be between 0 and 1

### Training Monitor ✅ **IMPLEMENTED**

The Training Monitor provides a comprehensive interface for training AI models on financial scenarios with real-time progress tracking.

**Features:**
- **WebSocket Connection Status**: Visual indicator (green/red dot) showing connection state
- **Training Configuration Form** (shown when not training):
  - Scenario selector dropdown (loads from API)
  - Number of episodes input (default: 1000)
  - Save interval input (default: 100 episodes)
  - Evaluation episodes input (default: 10)
  - Random seed input (optional, for reproducibility)
  - Start Training button (disabled if no scenario selected)
  - Form validation with error messages
- **Training Status Bar** (shown during training):
  - Current scenario name
  - Episode progress (e.g., "Episode 45 / 1000")
  - Elapsed time formatted (hours/minutes/seconds)
  - Animated pulsing indicator
  - Animated progress bar with percentage
  - Stop Training button
- **Current Metrics Cards** (6 real-time metrics):
  - Average Reward (2 decimal places)
  - Duration (months with 1 decimal)
  - Stability (percentage with 1 decimal)
  - Average Cash (EUR, no decimals)
  - Average Invested (EUR, no decimals)
  - Goal Adherence (percentage with 1 decimal)
- **Real-Time Charts** (4 interactive visualizations):
  - **Average Reward Over Time**: Line chart tracking reward progression
  - **Average Duration Over Time**: Line chart showing episode duration in months
  - **Stability Over Time**: Line chart displaying stability percentage (0-100%)
  - **Cash vs Investment Over Time**: Dual-line chart comparing cash and invested amounts
  - All charts feature:
    - Responsive containers (100% width, 300px height)
    - Custom dark-themed tooltips
    - Axis labels and legends
    - Grid lines for readability
    - No data points (smooth lines)
    - Color-coded lines (blue, green, purple, orange, red)
- **WebSocket Event Handling**:
  - `training_started`: Resets UI, clears previous data
  - `training_progress`: Updates metrics and appends to charts
  - `training_completed`: Shows completion state
  - `training_stopped`: Handles manual stop
  - `training_error`: Displays error message
- **Status Polling**: Fallback polling every 5 seconds for status updates
- **Loading States**: Spinner and "Starting..." / "Stopping..." button text
- **Error Handling**: Red error banner with descriptive messages
- **Conditional Rendering**: Shows form when idle, metrics/charts when training
- **Responsive Design**: Grid layouts adapt from 1 to 3 columns
- **Dark Mode**: Full support for light/dark themes

**API Integration:**
- `api.listScenarios()` - Loads available scenarios on mount
- `api.startTraining(request)` - Initiates training session
- `api.stopTraining()` - Stops active training
- `api.getTrainingStatus()` - Polls current training state every 5 seconds

**WebSocket Integration:**
- `websocket.connect()` - Establishes WebSocket connection
- `websocket.onConnect()` - Handles connection events
- `websocket.onDisconnect()` - Handles disconnection events
- `websocket.on('training_started')` - Listens for training start
- `websocket.on('training_progress')` - Receives real-time updates
- `websocket.on('training_completed')` - Handles completion
- `websocket.on('training_stopped')` - Handles manual stop
- `websocket.on('training_error')` - Handles errors

**User Experience:**
- Real-time chart updates as training progresses (no page refresh needed)
- Smooth progress bar animation
- Time formatting (e.g., "2h 15m 30s", "45m 12s", "30s")
- Percentage formatting with 1 decimal place
- Currency formatting with EUR symbol
- Disabled states prevent duplicate training sessions
- Clear visual feedback for all actions

### Simulation Runner ✅ **IMPLEMENTED**

The Simulation Runner allows users to evaluate trained models on financial scenarios and view comprehensive results.

**Features:**
- **Configuration Form** (shown before running):
  - **Trained Model Selector**: Dropdown showing all available models with scenario names
    - Format: "model_name (scenario_name)"
    - Empty state message if no models available
    - Auto-selects first model on load
  - **Scenario Selector**: Dropdown showing all available scenarios
    - Auto-selects first scenario on load
  - **Number of Episodes**: Input field (1-100 range)
    - Default: 10 episodes
    - Helper text: "Recommended: 10-50 episodes for reliable statistics"
  - **Random Seed**: Optional input for reproducibility
    - Placeholder: "Leave empty for random"
    - Helper text: "Set a seed for reproducible results"
  - **Run Simulation Button**: 
    - Disabled when no model/scenario selected
    - Shows spinner and "Running Simulation..." when active
    - Disabled during simulation execution
- **Progress Indicator** (shown during simulation):
  - Animated spinner (blue, 8x8 size)
  - Status text: "Running simulation..."
  - Details: "Evaluating X episodes with model_name"
  - Centered layout with white/dark background card
- **Summary Statistics** (4 metric cards after completion):
  - **Duration**: Mean ± std dev in months
  - **Total Wealth**: Mean ± std dev in EUR
  - **Investment Gains**: 
    - Mean value in EUR with +/- prefix
    - Color-coded (green for positive, red for negative)
    - Return percentage calculated from invested amount
  - **Final Portfolio**: 
    - Mean portfolio value in EUR
    - Cash breakdown displayed below
- **Strategy Breakdown Section**:
  - **Invest**: Horizontal progress bar showing percentage (blue)
  - **Save**: Horizontal progress bar showing percentage (green)
  - **Consume**: Horizontal progress bar showing percentage (purple)
  - Each bar shows label, percentage value, and visual indicator
- **Simulation Metadata Display**:
  - Model name, scenario name, and episode count
  - "Run Another" button to reset and configure new simulation
- **Action Buttons** (after completion):
  - **View Detailed Results**: Navigate to Results Viewer with simulation ID
  - **Compare Scenarios**: Navigate to Comparison page
- **Loading States**: Spinner and disabled states during API calls
- **Error Handling**: Red error banner with descriptive messages
- **Conditional Rendering**: Shows form → progress → results flow
- **Responsive Design**: Grid layouts adapt from 1 to 4 columns
- **Dark Mode**: Full support for light/dark themes

**API Integration:**
- `api.listModels()` - Loads available trained models on mount
- `api.listScenarios()` - Loads available scenarios on mount
- `api.runSimulation(request)` - Executes simulation and returns results
- `api.getSimulationResults(id)` - Fetches detailed results by simulation ID

**Navigation:**
- "View Detailed Results" button → `/results?id={simulationId}`
- "Compare Scenarios" button → `/comparison`

**Data Flow:**
1. User selects model and scenario from dropdowns
2. User configures episodes and optional seed
3. User clicks "Run Simulation"
4. Progress indicator shows while simulation runs
5. Results are fetched and displayed automatically
6. User can view detailed results or run another simulation

**Metrics Displayed:**
- Duration statistics (mean, std dev)
- Wealth statistics (total, portfolio, cash, invested)
- Investment performance (gains, return percentage)
- Strategy breakdown (invest/save/consume percentages)
- All monetary values formatted with EUR symbol
- All percentages formatted with 1 decimal place

### Results Viewer ✅ **IMPLEMENTED**

The Results Viewer provides detailed analysis and visualization of simulation results with interactive charts and episode-level data exploration.

**Features:**
- **Header Section**:
  - Back button to Simulation Runner
  - Results title with scenario name
  - Metadata: Model name and episode count
- **Summary Statistics** (4 metric cards):
  - **Duration**: Mean ± std dev in months
  - **Total Wealth**: Mean ± std dev in EUR
  - **Investment Gains**: 
    - Mean value in EUR with +/- prefix
    - Color-coded (green for positive, red for negative)
    - Return percentage calculated from invested amount
  - **Final Portfolio**: 
    - Mean portfolio value in EUR
    - Cash breakdown displayed below
- **Episode Selector**:
  - Dropdown to select individual episodes for detailed view
  - Shows episode number and duration for each episode
  - Updates all charts when episode changes
- **Tab Navigation** (4 chart views):
  - **Cash Balance**: Line chart showing cash over time
  - **Portfolio Evolution**: Dual-line chart (invested amount vs portfolio value)
  - **Wealth Accumulation**: Triple-line chart (cash, portfolio, total wealth)
  - **Action Distribution**: Pie chart showing invest/save/consume percentages
- **Interactive Charts** (Recharts):
  - **Cash Balance Over Time**:
    - Blue line showing cash balance progression
    - X-axis: Month, Y-axis: Cash (EUR)
    - Tooltip with formatted EUR values
  - **Portfolio Evolution**:
    - Green line: Amount invested
    - Purple line: Portfolio value (with returns)
    - Shows investment performance over time
  - **Wealth Accumulation**:
    - Blue line: Cash balance
    - Green line: Portfolio value
    - Orange line: Total wealth (cash + portfolio)
    - Comprehensive view of wealth building
  - **Action Distribution**:
    - Pie chart with 3 segments (Invest, Save, Consume)
    - Color-coded: Blue (Invest), Green (Save), Purple (Consume)
    - Labels show name and percentage
    - Custom tooltips with percentage formatting
  - All charts feature:
    - Responsive containers (100% width, 400px height)
    - Dark-themed tooltips with custom styling
    - Grid lines for readability
    - Axis labels and legends
    - Smooth lines without data points
    - Hover interactions
- **Strategy Learned Section**:
  - **Invest**: Horizontal progress bar showing percentage (blue)
  - **Save**: Horizontal progress bar showing percentage (green)
  - **Consume**: Horizontal progress bar showing percentage (purple)
  - Each bar shows label, percentage value, and visual indicator
  - Aggregated across all episodes
- **Action Buttons**:
  - **Compare Scenarios**: Navigate to Comparison page
  - **Export Data**: Download simulation results as JSON file
    - Filename format: `simulation_{id}_results.json`
    - Includes all episode data and statistics
- **Loading States**: 
  - Full-screen spinner with "Loading results..." message
  - Centered layout with animation
- **Error Handling**: 
  - Red error banner with descriptive message
  - "Back to Simulation Runner" button for recovery
  - Handles missing simulation ID
  - Handles API errors gracefully
- **Responsive Design**: Grid layouts adapt from 1 to 4 columns
- **Dark Mode**: Full support for light/dark themes throughout

**API Integration:**
- `api.getSimulationResults(id)` - Loads simulation results by ID from URL query parameter

**Navigation:**
- Back button → `/simulation` (Simulation Runner)
- "Compare Scenarios" button → `/comparison`
- URL parameter: `?id={simulationId}` (required)

**Data Visualization:**
- Episode-level time series data for all metrics
- Aggregated statistics across all episodes
- Strategy breakdown showing learned behavior
- Investment performance analysis
- Wealth accumulation tracking

**Chart Interactions:**
- Tab switching between different chart views
- Episode selector to explore individual runs
- Hover tooltips showing exact values
- Responsive resizing based on screen size
- Smooth animations and transitions

**Export Functionality:**
- JSON export includes:
  - All episode data (cash, invested, portfolio, actions)
  - Summary statistics (means, std devs)
  - Strategy metrics (invest/save/consume percentages)
  - Metadata (model name, scenario name, episode count)
- Browser download with descriptive filename
- Formatted JSON with 2-space indentation

### Comparison ✅ **IMPLEMENTED**

The Comparison view enables side-by-side analysis of multiple simulation results to identify the best performing scenarios and strategies.

**Features:**
- **Simulation Selector Section**:
  - Grid of selectable simulation cards (up to 4 at once)
  - Each card shows:
    - Scenario name (bold header)
    - Model name
    - Episode count and creation date
    - Checkbox indicator
  - Visual feedback: Blue border and background when selected
  - Selection counter: "X / 4 selected"
  - "Clear All" button to deselect all simulations
  - Empty state with "Go to Simulation Runner" CTA
  - Maximum 4 simulations can be compared simultaneously
  - Click anywhere on card to toggle selection
- **Metrics Comparison Table**:
  - Side-by-side comparison of 8 key metrics:
    - Duration (months)
    - Total Wealth (EUR)
    - Investment Gains (EUR)
    - Final Cash (EUR)
    - Final Portfolio (EUR)
    - Invest %
    - Save %
    - Consume %
  - **Column Headers**:
    - Color-coded dots matching chart colors
    - Scenario name (bold)
    - Model name (smaller text below)
  - **Difference Highlighting**:
    - Shows percentage change from first simulation
    - Green text for positive differences (+X%)
    - Red text for negative differences (-X%)
    - Displayed below main value
  - **Export Buttons**:
    - "Export CSV" button (green) - Downloads comparison as CSV file
    - "Export JSON" button (blue) - Downloads comparison as JSON file
  - Responsive table with horizontal scroll on mobile
- **Comparative Bar Charts** (4 visualizations):
  - **Total Wealth Comparison**: Bar chart comparing final wealth across scenarios
  - **Duration Comparison**: Bar chart comparing simulation duration in months
  - **Investment Gains Comparison**: Bar chart comparing investment returns
  - **Strategy Distribution Comparison**: Grouped bar chart showing invest/save/consume percentages
  - All charts feature:
    - Color-coded bars matching scenario colors (blue, green, orange, purple)
    - Dark-themed tooltips with formatted values
    - Axis labels and legends
    - Grid lines for readability
    - Responsive containers (100% width, 300px height)
- **Wealth Over Time Comparison**:
  - Line chart showing wealth accumulation for all selected scenarios
  - Multiple colored lines (one per scenario)
  - X-axis: Month, Y-axis: Total Wealth (EUR)
  - Allows visual comparison of wealth trajectories
  - Smooth lines without data points
  - Responsive container (100% width, 400px height)
- **Key Insights Section**:
  - Automatically calculated insights with emoji icons:
    - 🏆 **Highest Total Wealth**: Shows which scenario achieved best wealth
    - 📈 **Best Investment Returns**: Highlights scenario with highest gains
    - ⚡ **Shortest Duration**: Identifies fastest completion
  - Color-coded insight cards (green, blue, purple backgrounds)
  - Only shown when 2+ simulations selected
- **Loading States**:
  - Full-screen spinner when loading simulation list
  - Inline spinner when loading selected simulation results
  - "Loading simulation results..." message
- **Error Handling**:
  - Yellow warning banner for errors
  - Maximum selection limit warning
  - API error messages
  - Graceful handling of missing data
- **Export Functionality**:
  - **CSV Export**:
    - Rows: Metrics (Duration, Wealth, Gains, etc.)
    - Columns: Selected scenarios
    - Filename: `comparison_{timestamp}.csv`
    - Comma-separated format
  - **JSON Export**:
    - Array of objects with simulation metadata and full results
    - Filename: `comparison_{timestamp}.json`
    - Formatted with 2-space indentation
    - Includes all episode data
- **Responsive Design**: 
  - Grid layouts adapt from 1 to 3 columns
  - Table scrolls horizontally on mobile
  - Charts resize based on screen width
- **Dark Mode**: Full support for light/dark themes

**API Integration:**
- `api.getSimulationHistory()` - Loads list of available simulations on mount
- `api.getSimulationResults(id)` - Loads detailed results for each selected simulation
- Efficient result caching using Map-based storage
- Automatic result loading when selections change

**Navigation:**
- Back button → `/dashboard`
- "Go to Simulation Runner" button (empty state) → `/simulation`

**Data Analysis:**
- Automatic calculation of percentage differences
- Identification of best performing scenarios
- Visual comparison across multiple dimensions
- Strategy pattern analysis
- Wealth trajectory comparison

**Color Scheme:**
- Scenario 1: Blue (#3B82F6)
- Scenario 2: Green (#10B981)
- Scenario 3: Orange (#F59E0B)
- Scenario 4: Purple (#8B5CF6)

**User Experience:**
- Click-to-select interface for simulations
- Real-time chart updates as selections change
- Clear visual feedback for all interactions
- Automatic insights generation
- One-click export to CSV or JSON
- Maximum 4 simulations to prevent overcrowding
- Responsive tooltips on all charts
- Smooth animations and transitions

**Implementation Details:**
- 783 lines of TypeScript React code
- Uses Recharts for all visualizations (BarChart, LineChart)
- State management with React hooks (useState, useEffect)
- Efficient data transformation for chart rendering
- Dynamic color assignment based on selection order
- Percentage difference calculations relative to first simulation
- Browser-based file downloads for exports

## Styling

The app uses Tailwind CSS for styling with:
- Responsive design (mobile-first)
- Dark/light theme support via ThemeContext
- Consistent color palette
- Accessible components (ARIA labels, keyboard navigation)

## Backend API

The frontend expects the backend API to be running at `http://localhost:8000` by default.

See [backend/README.md](../backend/README.md) for backend setup and API documentation.

## Type Safety

All API responses are typed using the interfaces in `src/types/index.ts`, which match the backend Pydantic models. This ensures type safety across the full stack:

- Frontend TypeScript types ↔️ Backend Pydantic models
- Compile-time type checking
- IntelliSense support in IDEs
- Reduced runtime errors
