interface ErrorLogEntry {
  timestamp: string;
  message: string;
  stack?: string;
  context?: Record<string, any>;
  userAgent: string;
  url: string;
}

class ErrorLogger {
  private logs: ErrorLogEntry[] = [];
  private maxLogs = 100;
  private isDevelopment = import.meta.env.DEV;

  log(error: Error | string, context?: Record<string, any>) {
    const entry: ErrorLogEntry = {
      timestamp: new Date().toISOString(),
      message: typeof error === 'string' ? error : error.message,
      stack: typeof error === 'string' ? undefined : error.stack,
      context,
      userAgent: navigator.userAgent,
      url: window.location.href,
    };

    // Add to in-memory logs
    this.logs.push(entry);
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }

    // Log to console in development
    if (this.isDevelopment) {
      console.error('[Error Logger]', entry);
    }

    // In production, you would send this to a logging service
    // Example: this.sendToLoggingService(entry);

    // Store in localStorage for debugging
    try {
      const storedLogs = this.getStoredLogs();
      storedLogs.push(entry);
      // Keep only last 50 logs in localStorage
      const recentLogs = storedLogs.slice(-50);
      localStorage.setItem('error_logs', JSON.stringify(recentLogs));
    } catch (e) {
      // Ignore localStorage errors
      console.warn('Failed to store error log:', e);
    }
  }

  getLogs(): ErrorLogEntry[] {
    return [...this.logs];
  }

  getStoredLogs(): ErrorLogEntry[] {
    try {
      const stored = localStorage.getItem('error_logs');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  clearLogs() {
    this.logs = [];
    try {
      localStorage.removeItem('error_logs');
    } catch {
      // Ignore
    }
  }

  // Method to send logs to a remote service (placeholder)
  // Uncomment and implement in production
  // private async sendToLoggingService(entry: ErrorLogEntry) {
  //   // In production, implement sending to services like:
  //   // - Sentry
  //   // - LogRocket
  //   // - Custom logging endpoint
  //   // Example:
  //   await fetch('/api/logs', {
  //     method: 'POST',
  //     headers: { 'Content-Type': 'application/json' },
  //     body: JSON.stringify(entry),
  //   });
  // }

  // Export logs for debugging
  exportLogs(): string {
    const allLogs = [...this.getStoredLogs(), ...this.logs];
    return JSON.stringify(allLogs, null, 2);
  }

  // Download logs as a file
  downloadLogs() {
    const logs = this.exportLogs();
    const blob = new Blob([logs], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `error-logs-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}

// Singleton instance
const errorLogger = new ErrorLogger();

// Convenience function
export function logError(error: Error | string, context?: Record<string, any>) {
  errorLogger.log(error, context);
}

// Export logger instance for advanced usage
export { errorLogger };

// Global error handler
if (typeof window !== 'undefined') {
  window.addEventListener('error', (event) => {
    logError(event.error || event.message, {
      type: 'Unhandled Error',
      filename: event.filename,
      lineno: event.lineno,
      colno: event.colno,
    });
  });

  window.addEventListener('unhandledrejection', (event) => {
    logError(
      event.reason instanceof Error ? event.reason : String(event.reason),
      {
        type: 'Unhandled Promise Rejection',
      }
    );
  });
}
