/**
 * Utilities for graceful degradation when data is missing or invalid
 */

// Safe number formatting
export function formatNumber(
  value: number | null | undefined,
  fallback: string = 'N/A',
  decimals: number = 2
): string {
  if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
    return fallback;
  }
  return value.toFixed(decimals);
}

// Safe currency formatting
export function formatCurrency(
  value: number | null | undefined,
  currency: string = 'EUR',
  fallback: string = 'N/A'
): string {
  if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
    return fallback;
  }
  try {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  } catch {
    return `${formatNumber(value, fallback, 0)} ${currency}`;
  }
}

// Safe percentage formatting
export function formatPercentage(
  value: number | null | undefined,
  fallback: string = 'N/A',
  decimals: number = 1
): string {
  if (value === null || value === undefined || isNaN(value) || !isFinite(value)) {
    return fallback;
  }
  return `${value.toFixed(decimals)}%`;
}

// Safe date formatting
export function formatDate(
  date: string | Date | null | undefined,
  fallback: string = 'Unknown'
): string {
  if (!date) return fallback;
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(dateObj.getTime())) return fallback;
    
    return dateObj.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return fallback;
  }
}

// Safe relative time formatting
export function formatRelativeTime(
  date: string | Date | null | undefined,
  fallback: string = 'Unknown'
): string {
  if (!date) return fallback;
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    if (isNaN(dateObj.getTime())) return fallback;
    
    const now = new Date();
    const diffMs = now.getTime() - dateObj.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
    
    return formatDate(dateObj, fallback);
  } catch {
    return fallback;
  }
}

// Safe array access with fallback
export function safeArrayAccess<T>(
  array: T[] | null | undefined,
  index: number,
  fallback: T
): T {
  if (!array || !Array.isArray(array) || index < 0 || index >= array.length) {
    return fallback;
  }
  return array[index];
}

// Safe object property access
export function safeGet<T>(
  obj: any,
  path: string,
  fallback: T
): T {
  if (!obj) return fallback;
  
  const keys = path.split('.');
  let current = obj;
  
  for (const key of keys) {
    if (current === null || current === undefined || !(key in current)) {
      return fallback;
    }
    current = current[key];
  }
  
  return current === null || current === undefined ? fallback : current;
}

// Validate and sanitize numeric input
export function sanitizeNumber(
  value: any,
  min?: number,
  max?: number,
  fallback: number = 0
): number {
  const num = Number(value);
  
  if (isNaN(num) || !isFinite(num)) {
    return fallback;
  }
  
  if (min !== undefined && num < min) return min;
  if (max !== undefined && num > max) return max;
  
  return num;
}

// Check if data is valid and complete
export function isValidData<T>(
  data: T | null | undefined,
  requiredFields?: (keyof T)[]
): data is T {
  if (data === null || data === undefined) return false;
  
  if (requiredFields && typeof data === 'object') {
    return requiredFields.every(field => {
      const value = (data as any)[field];
      return value !== null && value !== undefined;
    });
  }
  
  return true;
}

// Provide default values for missing data
export function withDefaults<T extends Record<string, any>>(
  data: Partial<T> | null | undefined,
  defaults: T
): T {
  if (!data) return defaults;
  
  return {
    ...defaults,
    ...data,
  };
}

// Safe JSON parse with fallback
export function safeJsonParse<T>(
  json: string | null | undefined,
  fallback: T
): T {
  if (!json) return fallback;
  
  try {
    return JSON.parse(json);
  } catch {
    return fallback;
  }
}

// Truncate text with ellipsis
export function truncate(
  text: string | null | undefined,
  maxLength: number,
  ellipsis: string = '...'
): string {
  if (!text) return '';
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - ellipsis.length) + ellipsis;
}

// Calculate statistics with fallback for empty arrays
export function calculateStats(
  values: number[] | null | undefined
): {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  count: number;
} {
  if (!values || values.length === 0) {
    return { mean: null, std: null, min: null, max: null, count: 0 };
  }
  
  const validValues = values.filter(v => isFinite(v) && !isNaN(v));
  
  if (validValues.length === 0) {
    return { mean: null, std: null, min: null, max: null, count: 0 };
  }
  
  const mean = validValues.reduce((a, b) => a + b, 0) / validValues.length;
  const variance = validValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / validValues.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...validValues);
  const max = Math.max(...validValues);
  
  return { mean, std, min, max, count: validValues.length };
}
