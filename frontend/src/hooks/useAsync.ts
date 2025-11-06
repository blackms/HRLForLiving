import { useState, useCallback, useEffect, useRef } from 'react';
import { logError } from '../utils/errorLogger';

interface UseAsyncOptions<T> {
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
  immediate?: boolean;
  retryCount?: number;
  retryDelay?: number;
}

interface UseAsyncReturn<T, Args extends any[]> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  execute: (...args: Args) => Promise<T | null>;
  reset: () => void;
  retry: () => Promise<T | null>;
}

export function useAsync<T, Args extends any[] = []>(
  asyncFunction: (...args: Args) => Promise<T>,
  options: UseAsyncOptions<T> = {}
): UseAsyncReturn<T, Args> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  
  const lastArgsRef = useRef<Args | null>(null);
  const mountedRef = useRef(true);
  const retryCountRef = useRef(0);

  const {
    onSuccess,
    onError,
    immediate = false,
    retryCount = 0,
    retryDelay = 1000,
  } = options;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const execute = useCallback(
    async (...args: Args): Promise<T | null> => {
      lastArgsRef.current = args;
      setLoading(true);
      setError(null);

      try {
        const result = await asyncFunction(...args);
        
        if (mountedRef.current) {
          setData(result);
          setLoading(false);
          retryCountRef.current = 0;
          
          if (onSuccess) {
            onSuccess(result);
          }
        }
        
        return result;
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        
        // Log error
        logError(error, {
          function: asyncFunction.name,
          args: args,
          retryAttempt: retryCountRef.current,
        });

        // Retry logic
        if (retryCountRef.current < retryCount) {
          retryCountRef.current++;
          await new Promise(resolve => setTimeout(resolve, retryDelay * retryCountRef.current));
          return execute(...args);
        }

        if (mountedRef.current) {
          setError(error);
          setLoading(false);
          
          if (onError) {
            onError(error);
          }
        }
        
        return null;
      }
    },
    [asyncFunction, onSuccess, onError, retryCount, retryDelay]
  );

  const reset = useCallback(() => {
    setData(null);
    setError(null);
    setLoading(false);
    retryCountRef.current = 0;
  }, []);

  const retry = useCallback(async (): Promise<T | null> => {
    if (lastArgsRef.current) {
      retryCountRef.current = 0;
      return execute(...lastArgsRef.current);
    }
    return null;
  }, [execute]);

  // Execute immediately if requested
  useEffect(() => {
    if (immediate) {
      execute(...([] as unknown as Args));
    }
  }, [immediate]); // Only run on mount

  return { data, error, loading, execute, reset, retry };
}

// Hook for handling multiple async operations
export function useAsyncBatch<T>(
  asyncFunctions: (() => Promise<T>)[],
  options: { onComplete?: (results: T[]) => void; onError?: (error: Error) => void } = {}
) {
  const [results, setResults] = useState<T[]>([]);
  const [errors, setErrors] = useState<Error[]>([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const execute = useCallback(async () => {
    setLoading(true);
    setErrors([]);
    setResults([]);
    setProgress(0);

    const newResults: T[] = [];
    const newErrors: Error[] = [];

    for (let i = 0; i < asyncFunctions.length; i++) {
      try {
        const result = await asyncFunctions[i]();
        newResults.push(result);
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        newErrors.push(error);
        logError(error, { batchIndex: i });
        
        if (options.onError) {
          options.onError(error);
        }
      }
      
      setProgress(((i + 1) / asyncFunctions.length) * 100);
    }

    setResults(newResults);
    setErrors(newErrors);
    setLoading(false);

    if (options.onComplete && newErrors.length === 0) {
      options.onComplete(newResults);
    }
  }, [asyncFunctions, options]);

  return { results, errors, loading, progress, execute };
}
