interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
  onDismiss?: () => void;
  type?: 'error' | 'warning';
}

export default function ErrorMessage({
  message,
  onRetry,
  onDismiss,
  type = 'error',
}: ErrorMessageProps) {
  const bgColor = type === 'error'
    ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
    : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';

  const textColor = type === 'error'
    ? 'text-red-800 dark:text-red-200'
    : 'text-yellow-800 dark:text-yellow-200';

  const icon = type === 'error' ? '⚠️' : '⚠';

  return (
    <div
      className={`${bgColor} border rounded-lg p-4 ${textColor}`}
      role="alert"
    >
      <div className="flex items-start space-x-3">
        <span className="text-xl flex-shrink-0" role="img" aria-label={type}>
          {icon}
        </span>
        <div className="flex-1">
          <p className="text-sm font-medium">{message}</p>
          {(onRetry || onDismiss) && (
            <div className="mt-3 flex space-x-2">
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="px-3 py-1 text-sm font-medium bg-white dark:bg-gray-800 border border-current rounded hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-current focus:ring-offset-2"
                >
                  Try Again
                </button>
              )}
              {onDismiss && (
                <button
                  onClick={onDismiss}
                  className="px-3 py-1 text-sm font-medium hover:underline focus:outline-none focus:ring-2 focus:ring-current focus:ring-offset-2 rounded"
                >
                  Dismiss
                </button>
              )}
            </div>
          )}
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="flex-shrink-0 hover:opacity-70 transition-opacity focus:outline-none focus:ring-2 focus:ring-current focus:ring-offset-2 rounded"
            aria-label="Close"
          >
            <span className="text-lg" aria-hidden="true">×</span>
          </button>
        )}
      </div>
    </div>
  );
}

// Empty state component for when there's no data
export function EmptyState({
  icon = '📭',
  title,
  message,
  action,
  actionLabel,
}: {
  icon?: string;
  title: string;
  message: string;
  action?: () => void;
  actionLabel?: string;
}) {
  return (
    <div className="text-center py-12">
      <span className="text-6xl mb-4 block" role="img" aria-hidden="true">
        {icon}
      </span>
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
        {title}
      </h3>
      <p className="text-gray-600 dark:text-gray-400 mb-6 max-w-md mx-auto">
        {message}
      </p>
      {action && actionLabel && (
        <button
          onClick={action}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
        >
          {actionLabel}
        </button>
      )}
    </div>
  );
}
