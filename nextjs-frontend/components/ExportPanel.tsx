'use client';

import React, { useState } from 'react';
import { Download, FileText, FileSpreadsheet, Calendar, Database, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { toast } from 'react-toastify';

export default function ExportPanel() {
  const [selectedDate, setSelectedDate] = useState<string>(
    new Date().toISOString().split('T')[0]
  );
  const [format, setFormat] = useState<'excel' | 'csv' | 'json'>('excel');
  const [loading, setLoading] = useState(false);
  const [backupLoading, setBackupLoading] = useState(false);

  const API_BASE_URL = 'http://localhost:5000';

  const handleExport = async () => {
    if (!selectedDate) {
      toast.error('Please select a date');
      return;
    }

    setLoading(true);
    const toastId = toast.info('Preparing export...', { autoClose: false });

    try {
      const url = `${API_BASE_URL}/api/export?date=${selectedDate}&format=${format}`;
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/octet-stream',
        },
      });

      // Check if response is OK
      if (!response.ok) {
        // Try to parse error message
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json();
          
          // Check for pandas/dependency error
          if (errorData.suggestion) {
            throw new Error(`${errorData.error}\n${errorData.suggestion}`);
          }
          
          throw new Error(errorData.error || errorData.details || 'Export failed');
        } else {
          throw new Error(`Export failed with status: ${response.status}`);
        }
      }

      // Get the blob
      const blob = await response.blob();

      // Check if blob is empty
      if (blob.size === 0) {
        throw new Error('No data available for the selected date');
      }

      // Get filename from Content-Disposition header or create default
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `attendance_${selectedDate}.${format === 'excel' ? 'xlsx' : format}`;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1];
        }
      }

      // Create download link
      const url_blob = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url_blob;
      link.download = filename;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url_blob);
      }, 100);

      toast.update(toastId, {
        render: `✅ Successfully exported ${filename}!`,
        type: 'success',
        autoClose: 3000,
      });

    } catch (error: any) {
      console.error('Export error:', error);
      
      let errorMessage = 'Export failed. Please try again.';
      let showAlternative = false;
      
      if (error.message.includes('No data') || error.message.includes('not found')) {
        errorMessage = `No attendance data found for ${selectedDate}`;
      } else if (error.message.includes('Missing required package') || error.message.includes('pandas')) {
        errorMessage = 'Excel export requires pandas. Try CSV or JSON instead!';
        showAlternative = true;
      } else if (error.message.includes('network') || error.message.includes('fetch')) {
        errorMessage = 'Cannot connect to server. Is it running?';
      } else if (error.message) {
        errorMessage = error.message;
      }

      toast.update(toastId, {
        render: (
          <div>
            <div className="font-medium">❌ {errorMessage}</div>
            {showAlternative && (
              <div className="text-sm mt-1">
                Install: <code className="bg-slate-700 px-1 rounded">pip install pandas openpyxl</code>
              </div>
            )}
          </div>
        ),
        type: 'error',
        autoClose: showAlternative ? 10000 : 5000,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleBackup = async () => {
    setBackupLoading(true);
    const toastId = toast.info('Creating backup...', { autoClose: false });

    try {
      const response = await fetch(`${API_BASE_URL}/api/backup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Backup failed');
      }

      const data = await response.json();
      
      toast.update(toastId, {
        render: `✅ ${data.message || 'Backup created successfully!'}`,
        type: 'success',
        autoClose: 3000,
      });

    } catch (error: any) {
      console.error('Backup error:', error);
      
      let errorMessage = 'Backup failed. Please try again.';
      
      if (error.message.includes('network') || error.message.includes('fetch')) {
        errorMessage = 'Cannot connect to server. Is it running?';
      } else if (error.message) {
        errorMessage = error.message;
      }

      toast.update(toastId, {
        render: `❌ ${errorMessage}`,
        type: 'error',
        autoClose: 5000,
      });
    } finally {
      setBackupLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Export Section */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Select Date
            </label>
            <div className="relative">
              <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400 pointer-events-none" />
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                max={new Date().toISOString().split('T')[0]}
                className="w-full pl-10 pr-4 py-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
              />
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Selected: {new Date(selectedDate).toLocaleDateString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Export Format
            </label>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setFormat('excel')}
                type="button"
                disabled={loading}
                className={`p-3 rounded-lg border-2 transition-all ${
                  format === 'excel'
                    ? 'border-blue-500 bg-blue-50 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <FileSpreadsheet
                  className={`w-6 h-6 mx-auto mb-1 ${
                    format === 'excel' ? 'text-blue-500' : 'text-slate-400'
                  }`}
                />
                <p className="text-xs font-medium text-center">Excel</p>
              </button>

              <button
                onClick={() => setFormat('csv')}
                type="button"
                disabled={loading}
                className={`p-3 rounded-lg border-2 transition-all ${
                  format === 'csv'
                    ? 'border-green-500 bg-green-50 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <FileText
                  className={`w-6 h-6 mx-auto mb-1 ${
                    format === 'csv' ? 'text-green-500' : 'text-slate-400'
                  }`}
                />
                <p className="text-xs font-medium text-center">CSV</p>
              </button>

              <button
                onClick={() => setFormat('json')}
                type="button"
                disabled={loading}
                className={`p-3 rounded-lg border-2 transition-all ${
                  format === 'json'
                    ? 'border-purple-500 bg-purple-50 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <FileText
                  className={`w-6 h-6 mx-auto mb-1 ${
                    format === 'json' ? 'text-purple-500' : 'text-slate-400'
                  }`}
                />
                <p className="text-xs font-medium text-center">JSON</p>
              </button>
            </div>
          </div>

          <button
            onClick={handleExport}
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white py-3 rounded-lg font-medium hover:from-blue-600 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-sm hover:shadow-md"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Exporting...</span>
              </>
            ) : (
              <>
                <Download className="w-5 h-5" />
                <span>Export Data</span>
              </>
            )}
          </button>
        </div>

        {/* Backup Section */}
        <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg p-6 border border-slate-200">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg flex items-center justify-center shadow-sm">
              <Database className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-900">Database Backup</h3>
              <p className="text-sm text-slate-500">Create a full backup</p>
            </div>
          </div>

          <div className="bg-white rounded-lg p-4 mb-4 border border-slate-200">
            <div className="flex items-start space-x-2">
              <AlertCircle className="w-5 h-5 text-blue-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm text-slate-700 font-medium mb-1">Backup includes:</p>
                <ul className="text-xs text-slate-600 space-y-1">
                  <li>• All attendance records</li>
                  <li>• User information</li>
                  <li>• System settings</li>
                  <li>• Historical data</li>
                </ul>
              </div>
            </div>
          </div>

          <button
            onClick={handleBackup}
            disabled={backupLoading}
            className="w-full bg-gradient-to-r from-purple-500 to-purple-600 text-white py-3 rounded-lg font-medium hover:from-purple-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 shadow-sm hover:shadow-md"
          >
            {backupLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Creating Backup...</span>
              </>
            ) : (
              <>
                <Database className="w-5 h-5" />
                <span>Create Backup</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 hover:shadow-sm transition-shadow">
          <div className="flex items-center space-x-2 mb-2">
            <FileSpreadsheet className="w-5 h-5 text-blue-600" />
            <h4 className="font-medium text-blue-900">Excel Export</h4>
          </div>
          <p className="text-sm text-blue-700">
            Full-featured spreadsheet with formatting and formulas
          </p>
        </div>
        
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 hover:shadow-sm transition-shadow">
          <div className="flex items-center space-x-2 mb-2">
            <FileText className="w-5 h-5 text-green-600" />
            <h4 className="font-medium text-green-900">CSV Export</h4>
          </div>
          <p className="text-sm text-green-700">
            Universal format compatible with all spreadsheet software
          </p>
        </div>
        
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 hover:shadow-sm transition-shadow">
          <div className="flex items-center space-x-2 mb-2">
            <FileText className="w-5 h-5 text-purple-600" />
            <h4 className="font-medium text-purple-900">JSON Export</h4>
          </div>
          <p className="text-sm text-purple-700">
            Structured data format ideal for developers and APIs
          </p>
        </div>
      </div>

      {/* Quick Tips */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <div className="flex items-start space-x-2">
          <CheckCircle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
          <div>
            <h4 className="font-medium text-amber-900 mb-1">Quick Tips</h4>
            <ul className="text-sm text-amber-800 space-y-1">
              <li>• Export data is saved to <code className="bg-amber-100 px-1 rounded">attendance_data/exports/</code></li>
              <li>• Backups are stored in <code className="bg-amber-100 px-1 rounded">attendance_data/backups/</code></li>
              <li>• You can export data for any date that has attendance records</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}