# Error Handling and Loading States System

This document provides an overview of the comprehensive error handling and loading states system implemented in the HRL Finance UI.

## Overview

The error handling system provides:

1. **Toast Notifications** - User-friendly temporary messages
2. **Error Boundaries** - Catch and handle React component errors
3. **Loading Spinners** - Visual feedback during async operations
4. **Error Messages** - Inline error display with retry options
5. **Async Hook** - Simplified async state management
6. **Error Logging** - Automatic error tracking and debugging
7. **Graceful Degradation** - Safe handling of missing/invalid data
8. **API Wrapper** - Centralized API error handling

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Application                         │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────┐  │
│  │           Error Boundary (Top Level)             │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │         Toast Provider                     │  │  │
│  │  │  ┌──────────────────────────────────────┐  │  │  │
│  │  │  │        Page Components               │  │  │  │
│  │  │  │  ┌────────────────────────────────┐  │  │  │  │
│  │  │  │  │  useAsync Hook                 │  │  │  │  │
│  │  │  │  │  - Loading States              │  │  │  │  │
│  │  │  │  │  - Error Handling              │  │  │  │  │
│  │  │  │  │  - Retry Logic                 │  │  │  │  │
│  │  │  │  └────────────────────────────────┘  │  │  │  │
│  │  │  │                                      │  │  │  │
│  │  │  │  ┌────────────────────────────────┐  │  │  │  │
│  │  │  │  │  API Calls                     │  │  │  │  │
│  │  │  │  │  - Automatic Retry             │  │  │  │  │
│  │  │  │  │  - Error Logging               │  │  │  │  │
│  │  │  │  └────────────────────────────────┘  │  │  │  │
│  │  │  └──────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Error Logger (Global)                  │  │
│  │  - Console Logging (Dev)                         │  │
│  │  - LocalStorage (All)                            │  │
│  │  - Remote Service (Production - Optional)        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. Toast System

**Location:** `src/contexts/ToastContext.tsx`, `src/components/ToastContainer.tsx`

**Features:**
- Success, error, warning, and info toast types
- Auto-dismiss with configurable duration (default: 5000ms)
- Manual dismiss option
- Slide-in animation
- Dark mode support
- Accessible (ARIA labels, live regions)
- Queue management for multiple toasts
- Unique ID generation for each toast
- Automatic cleanup with setTimeout

**Context API:**
```typescript
interface ToastContextType {
  toasts: Toast[];
  showToast: (type: ToastType, message: string, duration?: number) => void;
  removeToast: (id: string) => void;
  success: (message: string, duration?: number) => void;
  error: (message: string, duration?: number) => void;
  warning: (message: string, duration?: number) => void;
  info: (message: string, duration?: number) => void;
}
```

**Usage:**
```typescript
const { success, error, warning, info } = useToast();
success('Operation completed!');
error('Something went wrong');
warning('Please review your input', 10000); // Custom duration
info('New feature available', 0); // No auto-dismiss
```

### 2. Error Boundary

**Location:** `src/components/ErrorBoundary.tsx`

**Features:**
- Catches React component errors
- Displays user-friendly error UI
- Logs errors automatically
- Provides "Try Again" and "Go Home" actions
- Customizable fallback UI
- Optional error callback

**Usage:**
```typescript
<ErrorBoundary onError={(error, info) => console.log(error)}>
  <MyComponent />
</ErrorBoundary>
```

### 3. Loading Spinners

**Location:** `src/components/LoadingSpinner.tsx`

**Features:**
- Multiple sizes (sm, md, lg, xl)
- Full-screen mode
- Optional loading message
- Button spinner variant
- Skeleton loader for content
- Accessible (role="status", aria-label)

**Usage:**
```typescript
<LoadingSpinner size="lg" message="Loading..." fullScreen />
<ButtonSpinner />
<SkeletonLoader lines={5} />
```

### 4. Error Messages

**Location:** `src/components/ErrorMessage.tsx`

**Features:**
- Error and warning variants
- Retry button
- Dismiss button
- Empty state component
- Dark mode support
- Accessible

**Usage:**
```typescript
<ErrorMessage
  message="Failed to load data"
  onRetry={loadData}
  onDismiss={() => setError(null)}
/>

<EmptyState
  title="No data"
  message="Get started by creating something"
  action={onCreate}
  actionLabel="Create"
/>
```

### 5. Async Hook

**Location:** `src/hooks/useAsync.ts`

**Features:**
- Automatic loading state management
- Error handling
- Retry logic with exponential backoff
- Success/error callbacks
- Immediate execution option
- Batch operations support

**Usage:**
```typescript
const { data, loading, error, execute, retry } = useAsync(
  () => api.getData(),
  {
    immediate: true,
    retryCount: 3,
    onSuccess: (data) => console.log(data),
    onError: (error) => console.error(error),
  }
);
```

### 6. Error Logger

**Location:** `src/utils/errorLogger.ts`

**Features:**
- Automatic error logging
- Console logging in development
- LocalStorage persistence
- Export logs as JSON
- Download logs as file
- Global error handlers (window.onerror, unhandledrejection)
- Context tracking

**Usage:**
```typescript
import { logError, errorLogger } from '../utils/errorLogger';

logError(error, { component: 'MyComponent', action: 'save' });

// View logs
const logs = errorLogger.getLogs();

// Download logs
errorLogger.downloadLogs();
```

### 7. Graceful Degradation

**Location:** `src/utils/gracefulDegradation.ts`

**Features:**
- Safe number/currency/percentage formatting
- Safe date formatting
- Relative time formatting
- Safe object/array access
- Data validation
- Default value provision
- Statistics calculation with fallbacks

**Usage:**
```typescript
import {
  formatCurrency,
  formatRelativeTime,
  safeGet,
  isValidData,
} from '../utils/gracefulDegradation';

const amount = formatCurrency(value, 'EUR', 'N/A');
const time = formatRelativeTime(date, 'Unknown');
const name = safeGet(user, 'profile.name', 'Anonymous');
```

### 8. API Wrapper

**Location:** `src/utils/apiWrapper.ts`

**Features:**
- Centralized error handling
- User-friendly error messages
- Retry with exponential backoff
- Health check utility
- Batch API calls
- Error message extraction

**Usage:**
```typescript
import { apiCall, getErrorMessage, retryApiCall } from '../utils/apiWrapper';

const { data, error } = await apiCall(() => api.getData());

const message = getErrorMessage(error);

const result = await retryApiCall(() => api.getData(), 3, 1000);
```

## Integration

The system is integrated into the application at multiple levels:

1. **App.tsx** - Wraps entire app with ErrorBoundary and ToastProvider
2. **API Client** - Built-in retry logic and error handling
3. **Components** - Use hooks and utilities for consistent error handling
4. **Global Handlers** - Catch unhandled errors and promise rejections

## Best Practices

### 1. Always Handle Loading States

```typescript
if (loading) return <LoadingSpinner />;
if (error) return <ErrorMessage message={error.message} onRetry={retry} />;
return <Content data={data} />;
```

### 2. Use Toast for User Feedback

```typescript
try {
  await api.save(data);
  success('Saved successfully!');
} catch (err) {
  error('Failed to save');
}
```

### 3. Provide Retry Options

```typescript
<ErrorMessage message={error} onRetry={loadData} />
```

### 4. Use Graceful Degradation

```typescript
<p>Amount: {formatCurrency(amount, 'EUR', 'N/A')}</p>
<p>Updated: {formatRelativeTime(date, 'Unknown')}</p>
```

### 5. Log Errors for Debugging

```typescript
try {
  await riskyOperation();
} catch (err) {
  logError(err, { context: 'operation-name' });
  error('Operation failed');
}
```

### 6. Use Empty States

```typescript
if (items.length === 0) {
  return (
    <EmptyState
      title="No items"
      message="Create your first item"
      action={onCreate}
      actionLabel="Create"
    />
  );
}
```

## Testing

### Unit Tests

Test individual utilities:

```typescript
import { formatCurrency, safeGet } from '../utils/gracefulDegradation';

test('formatCurrency handles null', () => {
  expect(formatCurrency(null)).toBe('N/A');
});

test('safeGet returns fallback', () => {
  expect(safeGet({}, 'missing.path', 'default')).toBe('default');
});
```

### Integration Tests

Test error handling flows:

```typescript
test('shows error message on API failure', async () => {
  // Mock API to fail
  jest.spyOn(api, 'getData').mockRejectedValue(new Error('API Error'));
  
  render(<MyComponent />);
  
  await waitFor(() => {
    expect(screen.getByText(/API Error/i)).toBeInTheDocument();
  });
});
```

### E2E Tests

Test complete user flows with error scenarios:

```typescript
test('user can retry after error', async () => {
  // Simulate network error
  await page.route('**/api/data', route => route.abort());
  
  await page.goto('/');
  
  // Should show error
  await expect(page.locator('text=Failed to load')).toBeVisible();
  
  // Click retry
  await page.click('button:has-text("Try Again")');
  
  // Should load successfully
  await expect(page.locator('text=Data loaded')).toBeVisible();
});
```

## Accessibility

All components follow WCAG 2.1 AA guidelines:

- **Toast Notifications**: `role="alert"`, `aria-live="polite"`
- **Loading Spinners**: `role="status"`, `aria-label="Loading"`
- **Error Messages**: `role="alert"`
- **Buttons**: Proper focus states, keyboard navigation
- **Empty States**: Semantic HTML, clear messaging

## Performance

- **Toast Notifications**: Auto-cleanup prevents memory leaks
- **Error Logging**: Limited to 100 in-memory logs, 50 in localStorage
- **Retry Logic**: Exponential backoff prevents server overload
- **Loading States**: Conditional rendering minimizes re-renders

## Future Enhancements

1. **Remote Error Logging**: Send errors to Sentry, LogRocket, etc.
2. **Error Analytics**: Track error patterns and frequencies
3. **Offline Support**: Queue failed requests for retry when online
4. **Custom Error Pages**: 404, 500, etc.
5. **Error Recovery Strategies**: Automatic fallback to cached data
6. **Performance Monitoring**: Track loading times and slow operations

## Documentation

- **User Guide**: See `ERROR_HANDLING_GUIDE.md` for detailed usage examples
- **API Reference**: See inline JSDoc comments in source files
- **Examples**: See `ERROR_HANDLING_GUIDE.md` for complete examples

## Support

For questions or issues:
1. Check the `ERROR_HANDLING_GUIDE.md` for examples
2. Review inline code comments
3. Check error logs: `errorLogger.downloadLogs()`
4. Open an issue with error details and reproduction steps

