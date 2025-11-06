import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { ToastProvider } from './contexts/ToastContext';
import ErrorBoundary from './components/ErrorBoundary';
import ToastContainer from './components/ToastContainer';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import ScenarioBuilder from './pages/ScenarioBuilder';
import TrainingMonitor from './pages/TrainingMonitor';
import SimulationRunner from './pages/SimulationRunner';
import ResultsViewer from './pages/ResultsViewer';
import Comparison from './pages/Comparison';

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <ToastProvider>
          <BrowserRouter>
            <a href="#main-content" className="skip-to-main">
              Skip to main content
            </a>
            <ToastContainer />
            <Routes>
              <Route path="/" element={<Layout />}>
                <Route index element={<Dashboard />} />
                <Route path="scenarios" element={<ScenarioBuilder />} />
                <Route path="scenarios/:name" element={<ScenarioBuilder />} />
                <Route path="training" element={<TrainingMonitor />} />
                <Route path="simulation" element={<SimulationRunner />} />
                <Route path="results" element={<ResultsViewer />} />
                <Route path="results/:id" element={<ResultsViewer />} />
                <Route path="comparison" element={<Comparison />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </ToastProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
