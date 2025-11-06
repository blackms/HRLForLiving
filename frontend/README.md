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

### Scenario Builder
- Create and edit financial scenarios
- Template selector with presets
- Form validation
- Preview panel with monthly cash flow

### Training Monitor
- Start/stop training
- Real-time progress updates via WebSocket
- Live charts for metrics (reward, duration, stability)
- Current episode statistics

### Simulation Runner
- Select model and scenario
- Configure simulation parameters
- Run evaluation episodes
- View summary statistics

### Results Viewer
- Interactive charts (Recharts)
- Cash balance over time
- Portfolio evolution
- Wealth accumulation
- Action distribution
- Strategy learned visualization

### Comparison
- Compare multiple scenarios/models
- Side-by-side metrics
- Comparative charts
- Difference highlighting
- Export to CSV/JSON

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
