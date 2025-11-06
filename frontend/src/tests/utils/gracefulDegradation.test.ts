import { describe, it, expect } from 'vitest';
import {
  formatCurrency,
  formatDate,
  formatPercentage,
  safeGet,
  safeArrayAccess,
  isValidNumber,
  calculateMean,
  calculateStdDev,
} from '../../utils/gracefulDegradation';

describe('gracefulDegradation utilities', () => {
  describe('formatCurrency', () => {
    it('formats valid numbers correctly', () => {
      expect(formatCurrency(1000)).toBe('€1,000.00');
      expect(formatCurrency(1234.56)).toBe('€1,234.56');
    });
    
    it('handles invalid values gracefully', () => {
      expect(formatCurrency(NaN)).toBe('€0.00');
      expect(formatCurrency(Infinity)).toBe('€0.00');
      expect(formatCurrency(null as any)).toBe('€0.00');
      expect(formatCurrency(undefined as any)).toBe('€0.00');
    });
  });
  
  describe('formatDate', () => {
    it('formats valid dates correctly', () => {
      const date = '2024-01-15T10:30:00';
      const formatted = formatDate(date);
      expect(formatted).toContain('2024');
    });
    
    it('handles invalid dates gracefully', () => {
      expect(formatDate('invalid-date')).toBe('Invalid date');
      expect(formatDate(null as any)).toBe('Invalid date');
    });
  });
  
  describe('formatPercentage', () => {
    it('formats valid numbers correctly', () => {
      expect(formatPercentage(0.5)).toBe('50.0%');
      expect(formatPercentage(0.123)).toBe('12.3%');
    });
    
    it('handles invalid values gracefully', () => {
      expect(formatPercentage(NaN)).toBe('0.0%');
      expect(formatPercentage(Infinity)).toBe('0.0%');
    });
  });
  
  describe('safeGet', () => {
    it('retrieves nested values correctly', () => {
      const obj = { a: { b: { c: 'value' } } };
      expect(safeGet(obj, 'a.b.c')).toBe('value');
    });
    
    it('returns default value for missing paths', () => {
      const obj = { a: { b: 'value' } };
      expect(safeGet(obj, 'a.x.y', 'default')).toBe('default');
    });
    
    it('handles null and undefined objects', () => {
      expect(safeGet(null, 'a.b', 'default')).toBe('default');
      expect(safeGet(undefined, 'a.b', 'default')).toBe('default');
    });
  });
  
  describe('safeArrayAccess', () => {
    it('retrieves array elements correctly', () => {
      const arr = [1, 2, 3];
      expect(safeArrayAccess(arr, 1)).toBe(2);
    });
    
    it('returns default value for out of bounds', () => {
      const arr = [1, 2, 3];
      expect(safeArrayAccess(arr, 10, 'default')).toBe('default');
    });
    
    it('handles null and undefined arrays', () => {
      expect(safeArrayAccess(null, 0, 'default')).toBe('default');
      expect(safeArrayAccess(undefined, 0, 'default')).toBe('default');
    });
  });
  
  describe('isValidNumber', () => {
    it('identifies valid numbers', () => {
      expect(isValidNumber(42)).toBe(true);
      expect(isValidNumber(0)).toBe(true);
      expect(isValidNumber(-10)).toBe(true);
    });
    
    it('identifies invalid numbers', () => {
      expect(isValidNumber(NaN)).toBe(false);
      expect(isValidNumber(Infinity)).toBe(false);
      expect(isValidNumber(-Infinity)).toBe(false);
      expect(isValidNumber(null as any)).toBe(false);
      expect(isValidNumber(undefined as any)).toBe(false);
    });
  });
  
  describe('calculateMean', () => {
    it('calculates mean correctly', () => {
      expect(calculateMean([1, 2, 3, 4, 5])).toBe(3);
      expect(calculateMean([10, 20, 30])).toBe(20);
    });
    
    it('handles empty arrays', () => {
      expect(calculateMean([])).toBe(0);
    });
    
    it('filters out invalid numbers', () => {
      expect(calculateMean([1, NaN, 3, Infinity, 5])).toBe(3);
    });
  });
  
  describe('calculateStdDev', () => {
    it('calculates standard deviation correctly', () => {
      const result = calculateStdDev([2, 4, 4, 4, 5, 5, 7, 9]);
      expect(result).toBeCloseTo(2, 0);
    });
    
    it('handles empty arrays', () => {
      expect(calculateStdDev([])).toBe(0);
    });
    
    it('filters out invalid numbers', () => {
      const result = calculateStdDev([1, NaN, 3, Infinity, 5]);
      expect(result).toBeGreaterThan(0);
    });
  });
});
