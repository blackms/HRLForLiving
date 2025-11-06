import { describe, it, expect, vi } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useAsync } from '../../hooks/useAsync';

describe('useAsync hook', () => {
  it('handles successful async operation', async () => {
    const asyncFn = vi.fn().mockResolvedValue('success');
    
    const { result } = renderHook(() => useAsync(asyncFn));
    
    // Initially loading
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
    
    // Wait for completion
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    
    expect(result.current.data).toBe('success');
    expect(result.current.error).toBeNull();
    expect(asyncFn).toHaveBeenCalledTimes(1);
  });
  
  it('handles async operation errors', async () => {
    const error = new Error('Test error');
    const asyncFn = vi.fn().mockRejectedValue(error);
    
    const { result } = renderHook(() => useAsync(asyncFn));
    
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe(error);
  });
  
  it('supports retry functionality', async () => {
    let callCount = 0;
    const asyncFn = vi.fn().mockImplementation(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.reject(new Error('First attempt failed'));
      }
      return Promise.resolve('success on retry');
    });
    
    const { result } = renderHook(() => useAsync(asyncFn, { retryCount: 1 }));
    
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    
    // Should have retried once
    expect(asyncFn).toHaveBeenCalledTimes(2);
    expect(result.current.data).toBe('success on retry');
  });
  
  it('allows manual retry', async () => {
    const asyncFn = vi.fn()
      .mockRejectedValueOnce(new Error('First error'))
      .mockResolvedValueOnce('success');
    
    const { result } = renderHook(() => useAsync(asyncFn, { retryCount: 0 }));
    
    // Wait for initial error
    await waitFor(() => {
      expect(result.current.error).toBeTruthy();
    });
    
    // Manual retry
    result.current.retry();
    
    await waitFor(() => {
      expect(result.current.data).toBe('success');
    });
    
    expect(asyncFn).toHaveBeenCalledTimes(2);
  });
});
