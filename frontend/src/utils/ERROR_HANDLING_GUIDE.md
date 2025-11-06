# Error Handling and Loading States Guide

This guide explains how to use the error handling and loading state utilities in the HRL Finance UI.

## Table of Contents

1. [Toast Notifications](#toast-notifications)
2. [Error Boundaries](#error-boundaries)
3. [Loading Spinners](#loading-spinners)
4. [Error Messages](#error-messages)
5. [Async Hook](#async-hook)
6. [Error Logging](#error-logging)
7. [Graceful Degradation](#graceful-degradation)
8. [Best Practices](#best-practices)

## Toast Notifications

Use toast notifications to show temporary success, error, warning, or info messages to users.

### Basic Usage

```typescript
import { useToast } from '../contexts/ToastContext';

function MyComponent() {
  const { success, error, warning, info } = useToast();

  const handleSave = async () => {
    try {
      await api.createScenario(data);
      success('Scenario created successfully!');
    } catch (err) {
      error('Failed to create scenario. Please try again.');
    }
  };

  return <button onClick={handleSave}>Save</button>;
}
```

### Custom Duration

```typescript
// Show toast for 10 seconds
success('Operation completed!', 10000);

// Show toast indefinitely (until manually dismissed)
error('Critical error occurred', 0);

// Default duration is 5000ms (5 seconds)
info('Data loaded'); // Auto-dismisses after 5 seconds
```

### All Available Methods

```typescript
const { success, error, warning, info, showToast, removeToast } = useToast();

// Convenience methods (recommended)
success('Success message', 5000);
error('Error message', 5000);
warning('Warning message', 5000);
info('Info message', 5000);

// Generic method
showToast('success', 'Custom message', 5000);

// Manual removal (if you have the toast ID)
removeToast('toast-id');
```

### Toast Types

- **success**: Green background, checkmark icon - for successful operations
- **error**: Red background, X icon - for errors and failures
- **warning**: Yellow background, warning icon - for warnings and cautions
- **info**: Blue background, info icon - for informational messages

## Error Boundaries

Error boundaries catch React errors and display a fallback UI.

### App-Level Error Boundary

Already configured in `App.tsx`. All pages are wrapped automatically.

### Component-Level Error Boundary

```typescript
import ErrorBoundary from '../components/ErrorBoundary';

function MyPage() {
  return (
    <ErrorBoundary
      fallback={<div>Something went wrong in this section</div>}
      onError={(error, errorInfo) => {
        // Custom error handling
        console.log('Error caught:', error);
      }}
    >
      <MyComponent />
    </ErrorBoundary>
  );
}
```

## Loading Spinners

### Full Screen Loading

```typescript
import LoadingSpinner from '../components/LoadingSpinner';

function MyComponent() {
  const [loading, setLoading] = useState(true);

  if (loading) {
    return <LoadingSpinner fullScreen message="Loading data..." />;
  }

  return <div>Content</div>;
}
```

### Inline Loading

```typescript
import LoadingSpinner from '../components/LoadingSpinner';

function MyComponent() {
  return (
    <div>
      {loading ? (
        <LoadingSpinner size="md" message="Processing..." />
      ) : (
        <div>Content</div>
      )}
    </div>
  );
}
```

### Button Loading

```typescript
import { ButtonSpinner } from '../components/LoadingSpinner';

function MyComponent() {
  const [saving, setSaving] = useState(false);

  return (
    <button disabled={saving}>
      {saving ? (
        <>
          <ButtonSpinner />
          <span className="ml-2">Saving...</span>
        </>
      ) : (
        'Save'
      )}
    </button>
  );
}
```

### Skeleton Loader

```typescript
import { SkeletonLoader } from '../components/LoadingSpinner';

function MyComponent() {
  return (
    <div>
      {loading ? (
        <SkeletonLoader lines={5} />
      ) : (
        <div>Actual content</div>
      )}
    </div>
  );
}
```

## Error Messages

### Display Error with Retry

```typescript
import ErrorMessage from '../components/ErrorMessage';

function MyComponent() {
  const [error, setError] = useState<string | null>(null);

  return (
    <div>
      {error && (
        <ErrorMessage
          message={error}
          onRetry={loadData}
          onDismiss={() => setError(null)}
        />
      )}
    </div>
  );
}
```

### Empty State

```typescript
import { EmptyState } from '../components/ErrorMessage';

function MyComponent() {
  if (items.length === 0) {
    return (
      <EmptyState
        icon="📭"
        title="No scenarios found"
        message="Create your first scenario to get started"
        action={() => navigate('/scenarios')}
        actionLabel="Create Scenario"
      />
    );
  }

  return <div>Items list</div>;
}
```

## Async Hook

The `useAsync` hook simplifies async operations with built-in loading, error, and retry logic.

### Basic Usage

```typescript
import { useAsync } from '../hooks/useAsync';
import { api } from '../services/api';

function MyComponent() {
  const { data, error, loading, execute } = useAsync(
    () => api.listScenarios(),
    {
      immediate: true, // Execute on mount
      onSuccess: (data) => console.log('Loaded:', data),
      onError: (error) => console.error('Failed:', error),
    }
  );

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error.message} />;

  return <div>{/* Render data */}</div>;
}
```

### With Parameters

```typescript
const { data, loading, execute } = useAsync(
  (scenarioName: string) => api.getScenario(scenarioName)
);

// Call with parameters
const handleLoad = () => {
  execute('my-scenario');
};
```

### With Automatic Retry

```typescript
const { data, loading, execute, retry } = useAsync(
  () => api.listModels(),
  {
    retryCount: 3, // Retry up to 3 times
    retryDelay: 1000, // Wait 1 second between retries
  }
);

// Manual retry
<button onClick={retry}>Retry</button>
```

### Batch Operations

```typescript
import { useAsyncBatch } from '../hooks/useAsync';

function MyComponent() {
  const { results, errors, loading, progress, execute } = useAsyncBatch(
    [
      () => api.listScenarios(),
      () => api.listModels(),
      () => api.getSimulationHistory(),
    ],
    {
      onComplete: (results) => console.log('All loaded:', results),
      onError: (error) => console.error('One failed:', error),
    }
  );

  return (
    <div>
      {loading && <div>Progress: {progress}%</div>}
      <button onClick={execute}>Load All</button>
    </div>
  );
}
```

## Error Logging

Errors are automatically logged to the console (dev) and localStorage (all environments).

### Manual Logging

```typescript
import { logError } from '../utils/errorLogger';

try {
  // Some operation
} catch (err) {
  logError(err, {
    component: 'MyComponent',
    action: 'saveData',
    userId: user.id,
  });
}
```

### View Logs

```typescript
import { errorLogger } from '../utils/errorLogger';

// Get all logs
const logs = errorLogger.getLogs();

// Export logs as JSON
const json = errorLogger.exportLogs();

// Download logs as file
errorLogger.downloadLogs();

// Clear logs
errorLogger.clearLogs();
```

## Graceful Degradation

Use these utilities to handle missing or invalid data gracefully.

### Safe Formatting

```typescript
import {
  formatNumber,
  formatCurrency,
  formatPercentage,
  formatDate,
  formatRelativeTime,
} from '../utils/gracefulDegradation';

// Safe number formatting
formatNumber(value, 'N/A', 2); // "123.45" or "N/A"

// Safe currency formatting
formatCurrency(1234.56, 'EUR'); // "€1,235" or "N/A"

// Safe percentage formatting
formatPercentage(45.678, 'N/A', 1); // "45.7%" or "N/A"

// Safe date formatting
formatDate(dateString, 'Unknown'); // "Jan 15, 2024" or "Unknown"

// Relative time
formatRelativeTime(dateString); // "2 hours ago" or "Unknown"
```

### Safe Data Access

```typescript
import {
  safeGet,
  safeArrayAccess,
  isValidData,
  withDefaults,
} from '../utils/gracefulDegradation';

// Safe object property access
const name = safeGet(user, 'profile.name', 'Anonymous');

// Safe array access
const firstItem = safeArrayAccess(items, 0, defaultItem);

// Check if data is valid
if (isValidData(scenario, ['name', 'environment'])) {
  // All required fields are present
}

// Provide defaults for missing data
const config = withDefaults(partialConfig, {
  episodes: 1000,
  saveInterval: 100,
  seed: 42,
});
```

### Safe Statistics

```typescript
import { calculateStats } from '../utils/gracefulDegradation';

const stats = calculateStats(values);
// Returns: { mean, std, min, max, count }
// All values are null if array is empty or invalid
```

## Best Practices

### 1. Always Show Loading States

```typescript
// ✅ Good
if (loading) return <LoadingSpinner />;
if (error) return <ErrorMessage message={error.message} />;
return <Content data={data} />;

// ❌ Bad
return <Content data={data} />; // No loading or error handling
```

### 2. Use Toast for User Actions

```typescript
// ✅ Good - User gets immediate feedback
const handleSave = async () => {
  try {
    await api.createScenario(data);
    success('Scenario created!');
    navigate('/scenarios');
  } catch (err) {
    error('Failed to create scenario');
  }
};

// ❌ Bad - Silent failure
const handleSave = async () => {
  await api.createScenario(data);
  navigate('/scenarios');
};
```

### 3. Provide Retry Options

```typescript
// ✅ Good
<ErrorMessage
  message={error.message}
  onRetry={loadData}
  onDismiss={() => setError(null)}
/>

// ❌ Bad - No way to recover
<div>Error: {error.message}</div>
```

### 4. Use Graceful Degradation

```typescript
// ✅ Good - Handles missing data
<div>
  <p>Name: {scenario?.name || 'Unnamed'}</p>
  <p>Income: {formatCurrency(scenario?.environment?.income)}</p>
</div>

// ❌ Bad - Will crash if data is missing
<div>
  <p>Name: {scenario.name}</p>
  <p>Income: {scenario.environment.income}</p>
</div>
```

### 5. Log Errors for Debugging

```typescript
// ✅ Good - Errors are logged
try {
  await riskyOperation();
} catch (err) {
  logError(err, { context: 'important-operation' });
  error('Operation failed');
}

// ❌ Bad - Error is lost
try {
  await riskyOperation();
} catch (err) {
  // Silent failure
}
```

### 6. Use Error Boundaries for Component Errors

```typescript
// ✅ Good - Errors are caught
<ErrorBoundary>
  <ComplexComponent />
</ErrorBoundary>

// ❌ Bad - Errors crash the app
<ComplexComponent />
```

### 7. Combine Multiple Utilities

```typescript
// ✅ Excellent - Complete error handling
function MyComponent() {
  const { success, error } = useToast();
  const { data, loading, error: asyncError, retry } = useAsync(
    () => api.listScenarios(),
    {
      immediate: true,
      retryCount: 2,
      onSuccess: () => success('Data loaded'),
      onError: (err) => {
        logError(err, { component: 'MyComponent' });
        error('Failed to load data');
      },
    }
  );

  if (loading) return <LoadingSpinner message="Loading scenarios..." />;
  
  if (asyncError) {
    return (
      <ErrorMessage
        message={getErrorMessage(asyncError)}
        onRetry={retry}
      />
    );
  }

  if (!data || data.length === 0) {
    return (
      <EmptyState
        title="No scenarios"
        message="Create your first scenario"
        action={() => navigate('/scenarios')}
        actionLabel="Create Scenario"
      />
    );
  }

  return <div>{/* Render data */}</div>;
}
```

## Complete Example

Here's a complete example showing all utilities working together:

```typescript
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useToast } from '../contexts/ToastContext';
import { useAsync } from '../hooks/useAsync';
import LoadingSpinner, { ButtonSpinner } from '../components/LoadingSpinner';
import ErrorMessage, { EmptyState } from '../components/ErrorMessage';
import { formatCurrency, formatRelativeTime } from '../utils/gracefulDegradation';
import { getErrorMessage } from '../utils/apiWrapper';
import { api } from '../services/api';

export default function ScenarioList() {
  const navigate = useNavigate();
  const { success, error: showError } = useToast();
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Load scenarios with automatic retry
  const {
    data: scenarios,
    loading,
    error,
    retry,
  } = useAsync(() => api.listScenarios(), {
    immediate: true,
    retryCount: 2,
    onError: (err) => {
      showError(getErrorMessage(err));
    },
  });

  // Delete scenario
  const handleDelete = async (name: string) => {
    setDeletingId(name);
    try {
      await api.deleteScenario(name);
      success('Scenario deleted successfully');
      retry(); // Reload list
    } catch (err) {
      showError('Failed to delete scenario');
    } finally {
      setDeletingId(null);
    }
  };

  // Loading state
  if (loading) {
    return <LoadingSpinner fullScreen message="Loading scenarios..." />;
  }

  // Error state
  if (error) {
    return (
      <ErrorMessage
        message={getErrorMessage(error)}
        onRetry={retry}
      />
    );
  }

  // Empty state
  if (!scenarios || scenarios.length === 0) {
    return (
      <EmptyState
        icon="📝"
        title="No scenarios yet"
        message="Create your first financial scenario to get started"
        action={() => navigate('/scenarios')}
        actionLabel="Create Scenario"
      />
    );
  }

  // Success state
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Scenarios</h1>
      {scenarios.map((scenario) => (
        <div key={scenario.name} className="p-4 bg-white rounded-lg shadow">
          <h3 className="font-medium">{scenario.name}</h3>
          <p className="text-sm text-gray-600">
            {scenario.description || 'No description'}
          </p>
          <p className="text-xs text-gray-400">
            Updated {formatRelativeTime(scenario.updated_at)}
          </p>
          <button
            onClick={() => handleDelete(scenario.name)}
            disabled={deletingId === scenario.name}
            className="mt-2 px-3 py-1 text-sm bg-red-600 text-white rounded"
          >
            {deletingId === scenario.name ? (
              <>
                <ButtonSpinner />
                <span className="ml-2">Deleting...</span>
              </>
            ) : (
              'Delete'
            )}
          </button>
        </div>
      ))}
    </div>
  );
}
```

