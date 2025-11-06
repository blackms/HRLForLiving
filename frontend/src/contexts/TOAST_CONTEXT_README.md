# ToastContext Implementation

## Overview

The ToastContext provides a centralized toast notification system for the HRL Finance UI, enabling user-friendly temporary messages throughout the application.

## File Location

`src/contexts/ToastContext.tsx` (82 lines)

## Architecture

The implementation uses React Context API with custom hooks for state management:

```
ToastProvider (Context Provider)
    ↓
  Toast State Management
    ↓
  useToast Hook (Consumer)
    ↓
  Components (success, error, warning, info)
    ↓
  ToastContainer (Display)
```

## Type Definitions

```typescript
export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;           // Unique identifier (timestamp + random)
  type: ToastType;      // Toast variant
  message: string;      // Message to display
  duration?: number;    // Auto-dismiss duration in ms (0 = no auto-dismiss)
}

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

## Implementation Details

### ToastProvider Component

- Manages toast state using `useState<Toast[]>`
- Provides context value to child components
- Wraps the entire application in `App.tsx`

### State Management

**Adding Toasts:**
- Generates unique ID: `toast-${Date.now()}-${Math.random()}`
- Appends new toast to state array
- Sets up auto-dismiss timer if duration > 0

**Removing Toasts:**
- Filters toast array by ID
- Called automatically after duration expires
- Can be called manually for immediate dismissal

### Auto-Dismiss Logic

```typescript
if (duration > 0) {
  setTimeout(() => {
    removeToast(id);
  }, duration);
}
```

- Default duration: 5000ms (5 seconds)
- Duration of 0: No auto-dismiss (persistent)
- Custom duration: Any positive number in milliseconds

### Convenience Methods

Four convenience methods wrap `showToast` for common use cases:

```typescript
const success = useCallback(
  (message: string, duration?: number) => showToast('success', message, duration),
  [showToast]
);
```

All methods use `useCallback` for performance optimization.

## Usage

### Setup (App.tsx)

```typescript
import { ToastProvider } from './contexts/ToastContext';
import ToastContainer from './components/ToastContainer';

function App() {
  return (
    <ToastProvider>
      <ToastContainer />
      {/* Rest of app */}
    </ToastProvider>
  );
}
```

### Basic Usage

```typescript
import { useToast } from '../contexts/ToastContext';

function MyComponent() {
  const { success, error, warning, info } = useToast();

  const handleSave = async () => {
    try {
      await api.save(data);
      success('Saved successfully!');
    } catch (err) {
      error('Failed to save');
    }
  };

  return <button onClick={handleSave}>Save</button>;
}
```

### Custom Duration

```typescript
// Show for 10 seconds
success('Operation completed!', 10000);

// Show indefinitely (until manually dismissed)
error('Critical error', 0);

// Default (5 seconds)
info('Data loaded');
```

### Manual Removal

```typescript
const { showToast, removeToast } = useToast();

// Add toast and get ID
const id = `toast-${Date.now()}-${Math.random()}`;
showToast('info', 'Processing...', 0);

// Remove manually later
setTimeout(() => removeToast(id), 3000);
```

## Toast Types

| Type | Color | Use Case |
|------|-------|----------|
| `success` | Green | Successful operations, confirmations |
| `error` | Red | Errors, failures, critical issues |
| `warning` | Yellow | Warnings, cautions, important notices |
| `info` | Blue | Informational messages, tips, updates |

## Integration with ToastContainer

The ToastContext manages state, while ToastContainer handles display:

- ToastContext: State management, add/remove logic
- ToastContainer: Visual rendering, animations, styling

See `src/components/ToastContainer.tsx` for display implementation.

## Error Handling

The `useToast` hook includes error checking:

```typescript
export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}
```

This ensures the hook is only used within a ToastProvider.

## Performance Considerations

1. **useCallback**: All methods use `useCallback` to prevent unnecessary re-renders
2. **Auto-cleanup**: setTimeout automatically removes toasts, preventing memory leaks
3. **Unique IDs**: Timestamp + random ensures no ID collisions
4. **Minimal re-renders**: State updates only affect ToastContainer, not entire app

## Accessibility

- Toast notifications should be rendered with `role="alert"` in ToastContainer
- Use `aria-live="polite"` for non-critical messages
- Use `aria-live="assertive"` for critical errors
- Provide dismiss buttons for keyboard navigation

## Testing

### Unit Tests

```typescript
import { renderHook, act } from '@testing-library/react';
import { ToastProvider, useToast } from './ToastContext';

test('adds toast on success call', () => {
  const wrapper = ({ children }) => <ToastProvider>{children}</ToastProvider>;
  const { result } = renderHook(() => useToast(), { wrapper });

  act(() => {
    result.current.success('Test message');
  });

  expect(result.current.toasts).toHaveLength(1);
  expect(result.current.toasts[0].type).toBe('success');
  expect(result.current.toasts[0].message).toBe('Test message');
});
```

### Integration Tests

```typescript
test('toast auto-dismisses after duration', async () => {
  const wrapper = ({ children }) => <ToastProvider>{children}</ToastProvider>;
  const { result } = renderHook(() => useToast(), { wrapper });

  act(() => {
    result.current.success('Test', 1000);
  });

  expect(result.current.toasts).toHaveLength(1);

  await waitFor(() => {
    expect(result.current.toasts).toHaveLength(0);
  }, { timeout: 1500 });
});
```

## Future Enhancements

Potential improvements:

1. **Position Control**: Allow toasts at different screen positions
2. **Stacking Limit**: Maximum number of visible toasts
3. **Priority Queue**: Higher priority toasts shown first
4. **Action Buttons**: Add action buttons to toasts (e.g., "Undo")
5. **Progress Bar**: Visual countdown for auto-dismiss
6. **Sound Effects**: Optional audio feedback
7. **Animations**: More animation options (fade, bounce, etc.)
8. **Persistence**: Save toasts to localStorage for page refresh

## Related Files

- `src/components/ToastContainer.tsx` - Display component
- `src/contexts/ThemeContext.tsx` - Similar context pattern
- `frontend/ERROR_HANDLING_README.md` - System overview
- `frontend/src/utils/ERROR_HANDLING_GUIDE.md` - Usage guide

## References

- [React Context API](https://react.dev/reference/react/useContext)
- [React useCallback](https://react.dev/reference/react/useCallback)
- [WCAG 2.1 - Status Messages](https://www.w3.org/WAI/WCAG21/Understanding/status-messages.html)
