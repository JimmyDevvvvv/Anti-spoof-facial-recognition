'use client';

import React, { useState } from 'react';
import { Download, FileText, FileSpreadsheet, Calendar, Database } from 'lucide-react';
import { toast } from 'react-toastify';

export default function ExportPanel() {
  const [selectedDate, setSelectedDate] = useState<string>(
    new Date().toISOString().split('T')[0]
  );
  const [format, setFormat] = useState<'excel' | 'csv' | 'json'>('excel');
  const [loading, setLoading] = useState(false);

  const handleExport = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:5000/api/export?date=${selectedDate}&format=${format}`,
        {
          method: 'GET',
        }
      );

      if (!response.ok) {
        throw new Error('Export failed');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `attendance_${selectedDate}.${format === 'excel' ? 'xlsx' : format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast.success('Export completed successfully!');
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Export failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleBackup = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/backup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Backup failed');
      }

      const data = await response.json();
      toast.success(data.message || 'Backup created successfully!');
    } catch (error) {
      console.error('Backup error:', error);
      toast.error('Backup failed. Please try again.');
    } finally {
      setLoading(false);
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
              <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              Export Format
            </label>
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setFormat('excel')}
                className={`p-3 rounded-lg border-2 transition-all ${
                  format === 'excel'
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-slate-200 hover:border-slate-300'
                }`}
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
                className={`p-3 rounded-lg border-2 transition-all ${
                  format === 'csv'
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-slate-200 hover:border-slate-300'
                }`}
              >
                <FileText
                  className={`w-6 h-6 mx-auto mb-1 ${
                    format === 'csv' ? 'text-blue-500' : 'text-slate-400'
                  }`}
                />
                <p className="text-xs font-medium text-center">CSV</p>
              </button>

              <button
                onClick={() => setFormat('json')}
                className={`p-3 rounded-lg border-2 transition-all ${
                  format === 'json'
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-slate-200 hover:border-slate-300'
                }`}
              >
                <FileText
                  className={`w-6 h-6 mx-auto mb-1 ${
                    format === 'json' ? 'text-blue-500' : 'text-slate-400'
                  }`}
                />
                <p className="text-xs font-medium text-center">JSON</p>
              </button>
            </div>
          </div>

          <button
            onClick={handleExport}
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white py-3 rounded-lg font-medium hover:from-blue-600 hover:to-blue-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            <Download className="w-5 h-5" />
            <span>{loading ? 'Exporting...' : 'Export Data'}</span>
          </button>
        </div>

        {/* Backup Section */}
        <div className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-lg p-6 border border-slate-200">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Database className="w-5 h-5 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-slate-900">Database Backup</h3>
              <p className="text-sm text-slate-500">Create a full backup</p>
            </div>
          </div>

          <p className="text-sm text-slate-600 mb-4">
            Create a complete backup of your attendance database. This includes all records, users, and system data.
          </p>

          <button
            onClick={handleBackup}
            disabled={loading}
            className="w-full bg-gradient-to-r from-purple-500 to-purple-600 text-white py-3 rounded-lg font-medium hover:from-purple-600 hover:to-purple-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            <Database className="w-5 h-5" />
            <span>{loading ? 'Creating Backup...' : 'Create Backup'}</span>
          </button>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 mb-1">Excel Export</h4>
          <p className="text-sm text-blue-700">
            Full-featured spreadsheet with formatting and formulas
          </p>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h4 className="font-medium text-green-900 mb-1">CSV Export</h4>
          <p className="text-sm text-green-700">
            Universal format compatible with all spreadsheet software
          </p>
        </div>
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <h4 className="font-medium text-purple-900 mb-1">JSON Export</h4>
          <p className="text-sm text-purple-700">
            Structured data format ideal for developers and APIs
          </p>
        </div>
      </div>
    </div>
  );
}