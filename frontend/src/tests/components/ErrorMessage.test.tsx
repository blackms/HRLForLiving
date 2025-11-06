import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ErrorMessage from '../../components/ErrorMessage';

describe('ErrorMessage', () => {
  it('renders error message', () => {
    render(<ErrorMessage message="Test error message" />);
    
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });
  
  it('calls onRetry when retry button is clicked', () => {
    const onRetry = vi.fn();
    render(<ErrorMessage message="Error" onRetry={onRetry} />);
    
    const retryButton = screen.getByText(/try again/i);
    fireEvent.click(retryButton);
    
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
  
  it('does not render retry button when onRetry is not provided', () => {
    render(<ErrorMessage message="Error" />);
    
    expect(screen.queryByText(/try again/i)).not.toBeInTheDocument();
  });
  
  it('renders with different variants', () => {
    const { rerender, container } = render(
      <ErrorMessage message="Error" variant="error" />
    );
    
    let errorDiv = container.querySelector('.bg-red-50');
    expect(errorDiv).toBeInTheDocument();
    
    rerender(<ErrorMessage message="Warning" variant="warning" />);
    errorDiv = container.querySelector('.bg-yellow-50');
    expect(errorDiv).toBeInTheDocument();
  });
});
