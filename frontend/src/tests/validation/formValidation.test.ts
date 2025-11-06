import { describe, it, expect } from 'vitest';

// Form validation helpers
const validateScenarioName = (name: string): string | null => {
  if (!name || name.trim().length === 0) {
    return 'Scenario name is required';
  }
  if (name.length < 3) {
    return 'Scenario name must be at least 3 characters';
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
    return 'Scenario name can only contain letters, numbers, hyphens, and underscores';
  }
  return null;
};

const validateIncome = (income: number): string | null => {
  if (income <= 0) {
    return 'Income must be greater than 0';
  }
  if (income > 1000000) {
    return 'Income seems unreasonably high';
  }
  return null;
};

const validateExpenses = (expenses: number, income: number): string | null => {
  if (expenses < 0) {
    return 'Expenses cannot be negative';
  }
  if (expenses > income) {
    return 'Expenses cannot exceed income';
  }
  return null;
};

const validateEpisodes = (episodes: number): string | null => {
  if (episodes <= 0) {
    return 'Number of episodes must be greater than 0';
  }
  if (episodes > 10000) {
    return 'Number of episodes is too high (max 10000)';
  }
  return null;
};

describe('Form Validation', () => {
  describe('validateScenarioName', () => {
    it('accepts valid scenario names', () => {
      expect(validateScenarioName('test_scenario')).toBeNull();
      expect(validateScenarioName('scenario-123')).toBeNull();
      expect(validateScenarioName('MyScenario')).toBeNull();
    });
    
    it('rejects empty names', () => {
      expect(validateScenarioName('')).toBeTruthy();
      expect(validateScenarioName('   ')).toBeTruthy();
    });
    
    it('rejects short names', () => {
      expect(validateScenarioName('ab')).toBeTruthy();
    });
    
    it('rejects names with invalid characters', () => {
      expect(validateScenarioName('test scenario')).toBeTruthy();
      expect(validateScenarioName('test@scenario')).toBeTruthy();
      expect(validateScenarioName('test/scenario')).toBeTruthy();
    });
  });
  
  describe('validateIncome', () => {
    it('accepts valid income values', () => {
      expect(validateIncome(1000)).toBeNull();
      expect(validateIncome(50000)).toBeNull();
      expect(validateIncome(100000)).toBeNull();
    });
    
    it('rejects zero or negative income', () => {
      expect(validateIncome(0)).toBeTruthy();
      expect(validateIncome(-1000)).toBeTruthy();
    });
    
    it('rejects unreasonably high income', () => {
      expect(validateIncome(2000000)).toBeTruthy();
    });
  });
  
  describe('validateExpenses', () => {
    it('accepts valid expense values', () => {
      expect(validateExpenses(500, 2000)).toBeNull();
      expect(validateExpenses(0, 2000)).toBeNull();
      expect(validateExpenses(2000, 2000)).toBeNull();
    });
    
    it('rejects negative expenses', () => {
      expect(validateExpenses(-100, 2000)).toBeTruthy();
    });
    
    it('rejects expenses exceeding income', () => {
      expect(validateExpenses(3000, 2000)).toBeTruthy();
    });
  });
  
  describe('validateEpisodes', () => {
    it('accepts valid episode counts', () => {
      expect(validateEpisodes(10)).toBeNull();
      expect(validateEpisodes(100)).toBeNull();
      expect(validateEpisodes(1000)).toBeNull();
    });
    
    it('rejects zero or negative episodes', () => {
      expect(validateEpisodes(0)).toBeTruthy();
      expect(validateEpisodes(-10)).toBeTruthy();
    });
    
    it('rejects excessively high episode counts', () => {
      expect(validateEpisodes(20000)).toBeTruthy();
    });
  });
});
