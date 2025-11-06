/**
 * API wrapper with automatic error handling and toast notifications
 * Use this wrapper for API calls that should show user-friendly error messages
 */

import { api } from '../services/api';
import { logError } from './errorLogger';

export interface ApiCallOptions {
  silent?: boolean;
}

/**
 * Wrapper for API calls with automatic error handling
 * Note: Toast notifications should be triggered by the calling component using useToast
 */
export async function apiCall<T>(
  apiFunction: () => Promise<T>,
  options: ApiCallOptions = {}
): Promise<{ data: T | null; error: Error | null }> {
  const {
    silent = false,
  } = options;

  try {
    const data = await apiFunction();
    
    // Return success result with metadata for toast handling
    return {
      data,
      error: null,
    };
  } catch (err) {
    const error = err instanceof Error ? err : new Error(String(err));
    
    // Log error
    if (!silent) {
      logError(error, {
        apiFunction: apiFunction.name,
        options,
      });
    }

    // Return error result with metadata for toast handling
    return {
      data: null,
      error,
    };
  }
}

/**
 * Get user-friendly error message from error object
 */
export function getErrorMessage(error: Error | null | undefined): string {
  if (!error) return 'An unknown error occurred';
  
  // Check for common error patterns
  if (error.message.includes('Network Error') || error.message.includes('No response from server')) {
    return 'Unable to connect to the server. Please check your internet connection.';
  }
  
  if (error.message.includes('timeout')) {
    return 'The request took too long. Please try again.';
  }
  
  if (error.message.includes('404')) {
    return 'The requested resource was not found.';
  }
  
  if (error.message.includes('403')) {
    return 'You do not have permission to perform this action.';
  }
  
  if (error.message.includes('500')) {
    return 'A server error occurred. Please try again later.';
  }
  
  // Return the original message if it's user-friendly
  return error.message || 'An unexpected error occurred';
}

/**
 * Retry an API call with exponential backoff
 */
export async function retryApiCall<T>(
  apiFunction: () => Promise<T>,
  maxRetries: number = 3,
  initialDelay: number = 1000
): Promise<T> {
  let lastError: Error | null = null;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await apiFunction();
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      
      if (attempt < maxRetries) {
        const delay = initialDelay * Math.pow(2, attempt);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  throw lastError || new Error('Max retries exceeded');
}

/**
 * Check if API is available
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    await api.healthCheck();
    return true;
  } catch {
    return false;
  }
}

/**
 * Batch API calls with error handling
 */
export async function batchApiCalls<T>(
  apiCalls: (() => Promise<T>)[],
  options: { continueOnError?: boolean } = {}
): Promise<{ results: T[]; errors: Error[] }> {
  const { continueOnError = true } = options;
  const results: T[] = [];
  const errors: Error[] = [];

  for (const apiCall of apiCalls) {
    try {
      const result = await apiCall();
      results.push(result);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      errors.push(error);
      logError(error, { batchCall: true });
      
      if (!continueOnError) {
        break;
      }
    }
  }

  return { results, errors };
}
