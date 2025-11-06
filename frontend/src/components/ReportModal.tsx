import { useState } from 'react';
import { api } from '../services/api';
import type { ReportRequest } from '../types';

interface ReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  simulationId: string;
  scenarioName: string;
  modelName: string;
}

export default function ReportModal({
  isOpen,
  onClose,
  simulationId,
  scenarioName,
  modelName,
}: ReportModalProps) {
  // Form state
  const [reportType, setReportType] = useState<'html' | 'pdf'>('html');
  const [title, setTitle] = useState(`Financial Report: ${scenarioName}`);
  const [sections, setSections] = useState({
    summary: true,
    scenario: true,
    training: true,
    results: true,
    strategy: true,
    charts: true,
  });

  // UI state
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reportId, setReportId] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

  const handleSectionToggle = (section: keyof typeof sections) => {
    setSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    setError(null);
    setReportId(null);
    setDownloadUrl(null);

    try {
      // Build request
      const includeSections = Object.entries(sections)
        .filter(([_, enabled]) => enabled)
        .map(([section, _]) => section);

      const request: ReportRequest = {
        simulation_id: simulationId,
        report_type: reportType,
        include_sections: includeSections,
        title: title || undefined,
      };

      // Generate report
      const response = await api.generateReport(request);
      
      setReportId(response.report_id);
      
      // Set download URL
      const baseUrl = 'http://localhost:8000';
      setDownloadUrl(`${baseUrl}/api/reports/${response.report_id}`);
    } catch (err: any) {
      setError(err.message || 'Failed to generate report');
      console.error(err);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (downloadUrl) {
      window.open(downloadUrl, '_blank');
    }
  };

  const handleClose = () => {
    setReportId(null);
    setDownloadUrl(null);
    setError(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Generate Report</h2>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Simulation Info */}
          <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-medium">Scenario:</span> {scenarioName}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-medium">Model:</span> {modelName}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-medium">Simulation ID:</span> {simulationId}
            </p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
              <p className="text-red-800 dark:text-red-200">{error}</p>
            </div>
          )}

          {/* Success Message */}
          {reportId && downloadUrl && (
            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
              <p className="text-green-800 dark:text-green-200 font-medium mb-2">
                Report generated successfully!
              </p>
              <button
                onClick={handleDownload}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium transition-colors"
              >
                Download Report
              </button>
            </div>
          )}

          {/* Report Title */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Report Title
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              disabled={isGenerating || !!reportId}
              className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
              placeholder="Enter report title"
            />
          </div>

          {/* Report Type */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Report Format
            </label>
            <div className="flex gap-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="html"
                  checked={reportType === 'html'}
                  onChange={(e) => setReportType(e.target.value as 'html' | 'pdf')}
                  disabled={isGenerating || !!reportId}
                  className="mr-2"
                />
                <span className="text-gray-900 dark:text-white">HTML</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="pdf"
                  checked={reportType === 'pdf'}
                  onChange={(e) => setReportType(e.target.value as 'html' | 'pdf')}
                  disabled={isGenerating || !!reportId}
                  className="mr-2"
                />
                <span className="text-gray-900 dark:text-white">PDF</span>
              </label>
            </div>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {reportType === 'pdf' 
                ? 'PDF format requires WeasyPrint. Falls back to HTML if not available.'
                : 'HTML format can be viewed in any browser.'}
            </p>
          </div>

          {/* Section Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
              Include Sections
            </label>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={sections.summary}
                  onChange={() => handleSectionToggle('summary')}
                  disabled={isGenerating || !!reportId}
                  className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <div>
                  <span className="text-gray-900 dark:text-white font-medium">Summary Statistics</span>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Duration, wealth, investment gains, and key metrics
                  </p>
                </div>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={sections.scenario}
                  onChange={() => handleSectionToggle('scenario')}
                  disabled={isGenerating || !!reportId}
                  className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <div>
                  <span className="text-gray-900 dark:text-white font-medium">Scenario Configuration</span>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Income, expenses, inflation, and investment parameters
                  </p>
                </div>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={sections.training}
                  onChange={() => handleSectionToggle('training')}
                  disabled={isGenerating || !!reportId}
                  className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <div>
                  <span className="text-gray-900 dark:text-white font-medium">Training Configuration</span>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Episodes, learning rates, and training parameters
                  </p>
                </div>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={sections.results}
                  onChange={() => handleSectionToggle('results')}
                  disabled={isGenerating || !!reportId}
                  className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <div>
                  <span className="text-gray-900 dark:text-white font-medium">Detailed Results</span>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Wealth breakdown and portfolio analysis
                  </p>
                </div>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={sections.strategy}
                  onChange={() => handleSectionToggle('strategy')}
                  disabled={isGenerating || !!reportId}
                  className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <div>
                  <span className="text-gray-900 dark:text-white font-medium">Strategy Learned</span>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Investment, saving, and consumption patterns
                  </p>
                </div>
              </label>

              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={sections.charts}
                  onChange={() => handleSectionToggle('charts')}
                  disabled={isGenerating || !!reportId}
                  className="mr-3 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <div>
                  <span className="text-gray-900 dark:text-white font-medium">Charts & Visualizations</span>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Episode data tables and visual summaries
                  </p>
                </div>
              </label>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={handleClose}
            disabled={isGenerating}
            className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
          >
            {reportId ? 'Close' : 'Cancel'}
          </button>
          {!reportId && (
            <button
              onClick={handleGenerate}
              disabled={isGenerating || Object.values(sections).every(v => !v)}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-colors"
            >
              {isGenerating ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Generating...
                </span>
              ) : (
                'Generate Report'
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
