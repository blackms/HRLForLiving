# Frontend Tests

This directory contains comprehensive tests for the HRL Finance UI frontend application.

## Test Structure

```
tests/
├── setup.ts                      # Test setup and configuration
├── components/                   # Component tests
│   ├── ErrorBoundary.test.tsx
│   ├── LoadingSpinner.test.tsx
│   └── ErrorMessage.test.tsx
├── contexts/                     # Context tests
│   └── ToastContext.test.tsx
├── hooks/                        # Custom hooks tests
│   └── useAsync.test.ts
├── services/                     # API service tests
│   └── api.test.ts
├── utils/                        # Utility function tests
│   └── gracefulDegradation.test.ts
├── validation/                   # Form validation tests
│   └── formValidation.test.ts
└── integration/                  # Integration tests
    └── Dashboard.test.tsx
```

## Running Tests

### Run all tests
```bash
cd frontend
npm test
```

### Run tests in watch mode
```bash
npm run test:watch
```

### Run tests with UI
```bash
npm run test:ui
```

### Run specific test file
```bash
npm test -- ErrorBoundary.test.tsx
```

### Run with coverage
```bash
npm test -- --coverage
```

## Test Categories

### Component Tests
Test individual React components in isolation:
- Rendering behavior
- User interactions
- Props handling
- State management
- Error boundaries

### Hook Tests
Test custom React hooks:
- State updates
- Side effects
- Return values
- Error handling

### Service Tests
Test API service functions:
- HTTP requests
- Response handling
- Error handling
- Data transformation

### Utility Tests
Test utility functions:
- Data formatting
- Validation
- Calculations
- Safe data access

### Integration Tests
Test complete user workflows:
- Page rendering
- API integration
- User interactions
- Error states

## Testing Tools

- **Vitest**: Fast unit test framework
- **React Testing Library**: Component testing utilities
- **@testing-library/user-event**: User interaction simulation
- **@testing-library/jest-dom**: Custom matchers

## Writing New Tests

### Component Test Example
```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import MyComponent from './MyComponent';

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />);
    expect(screen.getByText('Hello')).toBeInTheDocument();
  });
});
```

### Hook Test Example
```typescript
import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useMyHook } from './useMyHook';

describe('useMyHook', () => {
  it('returns expected value', () => {
    const { result } = renderHook(() => useMyHook());
    expect(result.current).toBe('value');
  });
});
```

### API Test Example
```typescript
import { describe, it, expect, vi } from 'vitest';
import axios from 'axios';
import * as api from './api';

vi.mock('axios');

describe('API', () => {
  it('fetches data', async () => {
    (axios.get as any).mockResolvedValue({ data: 'test' });
    const result = await api.getData();
    expect(result).toBe('test');
  });
});
```

## Best Practices

1. **Test Behavior, Not Implementation**: Focus on what the component does, not how it does it
2. **Use Semantic Queries**: Prefer `getByRole`, `getByLabelText` over `getByTestId`
3. **Mock External Dependencies**: Mock API calls, WebSocket connections, etc.
4. **Clean Up**: Use `afterEach(cleanup)` to clean up after tests
5. **Descriptive Names**: Use clear, descriptive test names
6. **Arrange-Act-Assert**: Structure tests with clear setup, action, and assertion phases

## Test Coverage Goals

- ✅ All major components
- ✅ All custom hooks
- ✅ All API service functions
- ✅ All utility functions
- ✅ Form validation logic
- ✅ Error handling paths
- ✅ Critical user workflows

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Pre-deployment checks

## Troubleshooting

### Tests fail with "Cannot find module"
- Check import paths
- Ensure dependencies are installed: `npm install`

### Tests timeout
- Increase timeout in vitest.config.ts
- Check for unresolved promises

### Mock not working
- Ensure mock is defined before import
- Use `vi.clearAllMocks()` in `beforeEach`
