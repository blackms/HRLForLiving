/**
 * Demo page to showcase error handling and loading states
 * This is for testing purposes and can be removed in production
 */

import { useState } from 'react';
import { useToast } from '../contexts/ToastContext';
import { useAsync } from '../hooks/useAsync';
import LoadingSpinner, { ButtonSpinner, SkeletonLoader } from '../components/LoadingSpinner';
import ErrorMessage, { EmptyState } from '../components/ErrorMessage';
import { formatCurrency, formatPercentage, formatRelativeTime } from '../utils/gracefulDegradation';
import { logError, errorLogger } from '../utils/errorLogger';

export default function ErrorHandlingDemo() {
  const { success, error, warning, info } = useToast();
  const [showLoading, setShowLoading] = useState(false);
  const [showError, setShowError] = useState(false);

  // Demo async operation
  const { data, loading, error: asyncError, execute, retry } = useAsync(
    async () => {
      await new Promise(resolve => setTimeout(resolve, 2000));
      if (Math.random() > 0.5) {
        throw new Error('Random error occurred!');
      }
      return { value: 1234.56, date: new Date().toISOString() };
    },
    {
      onSuccess: () => success('Data loaded successfully!'),
      onError: (err) => error(err.message),
    }
  );

  const handleTestError = () => {
    const testError = new Error('Test error for logging');
    logError(testError, { component: 'ErrorHandlingDemo', action: 'test' });
    error('Error logged! Check console or download logs.');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Error Handling Demo
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Test all error handling and loading state components
        </p>
      </div>

      {/* Toast Notifications */}
      <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Toast Notifications
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => success('Success! Operation completed.')}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
          >
            Show Success
          </button>
          <button
            onClick={() => error('Error! Something went wrong.')}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Show Error
          </button>
          <button
            onClick={() => warning('Warning! Please review this.')}
            className="px-4 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700"
          >
            Show Warning
          </button>
          <button
            onClick={() => info('Info: Here is some information.')}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Show Info
          </button>
        </div>
      </section>

      {/* Loading Spinners */}
      <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Loading Spinners
        </h2>
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <LoadingSpinner size="sm" />
            <LoadingSpinner size="md" />
            <LoadingSpinner size="lg" />
            <LoadingSpinner size="xl" />
          </div>
          <div>
            <LoadingSpinner size="md" message="Loading data..." />
          </div>
          <div>
            <button
              onClick={() => {
                setShowLoading(true);
                setTimeout(() => setShowLoading(false), 3000);
              }}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Show Full Screen Loading
            </button>
            {showLoading && <LoadingSpinner fullScreen message="Loading..." />}
          </div>
          <div>
            <SkeletonLoader lines={3} />
          </div>
        </div>
      </section>

      {/* Error Messages */}
      <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Error Messages
        </h2>
        <div className="space-y-4">
          <button
            onClick={() => setShowError(!showError)}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Toggle Error Message
          </button>
          {showError && (
            <ErrorMessage
              message="Failed to load data. Please try again."
              onRetry={() => {
                info('Retrying...');
                setShowError(false);
              }}
              onDismiss={() => setShowError(false)}
            />
          )}
          <EmptyState
            icon="📭"
            title="No items found"
            message="Get started by creating your first item"
            action={() => success('Action clicked!')}
            actionLabel="Create Item"
          />
        </div>
      </section>

      {/* Async Hook Demo */}
      <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Async Hook with Retry
        </h2>
        <div className="space-y-4">
          <button
            onClick={() => execute()}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
          >
            {loading && <ButtonSpinner />}
            {loading ? 'Loading...' : 'Load Data (50% fail rate)'}
          </button>
          {loading && <LoadingSpinner message="Fetching data..." />}
          {asyncError && (
            <ErrorMessage
              message={asyncError.message}
              onRetry={retry}
            />
          )}
          {data && (
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded border border-green-200 dark:border-green-800">
              <p className="text-green-800 dark:text-green-200">
                Data loaded successfully!
              </p>
              <p className="text-sm text-green-600 dark:text-green-400 mt-2">
                Value: {formatCurrency(data.value)}
              </p>
              <p className="text-sm text-green-600 dark:text-green-400">
                Time: {formatRelativeTime(data.date)}
              </p>
            </div>
          )}
        </div>
      </section>

      {/* Graceful Degradation */}
      <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Graceful Degradation
        </h2>
        <div className="space-y-2 text-gray-700 dark:text-gray-300">
          <p>Valid number: {formatCurrency(1234.56)}</p>
          <p>Null number: {formatCurrency(null)}</p>
          <p>NaN number: {formatCurrency(NaN)}</p>
          <p>Valid percentage: {formatPercentage(45.678)}</p>
          <p>Null percentage: {formatPercentage(null)}</p>
          <p>Valid date: {formatRelativeTime(new Date().toISOString())}</p>
          <p>Invalid date: {formatRelativeTime(null)}</p>
        </div>
      </section>

      {/* Error Logging */}
      <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Error Logging
        </h2>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={handleTestError}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
          >
            Log Test Error
          </button>
          <button
            onClick={() => {
              const logs = errorLogger.getLogs();
              console.log('Error logs:', logs);
              info(`Found ${logs.length} error logs in memory`);
            }}
            className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
          >
            View Logs (Console)
          </button>
          <button
            onClick={() => {
              errorLogger.downloadLogs();
              success('Logs downloaded!');
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Download Logs
          </button>
          <button
            onClick={() => {
              errorLogger.clearLogs();
              success('Logs cleared!');
            }}
            className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Clear Logs
          </button>
        </div>
      </section>

      {/* Trigger Error Boundary */}
      <section className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
          Error Boundary Test
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          Note: This will crash the component. The error boundary will catch it.
        </p>
        <button
          onClick={() => {
            throw new Error('Test error boundary!');
          }}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Trigger Error Boundary
        </button>
      </section>
    </div>
  );
}
