'use client';

import React, { useState } from 'react';
import VideoFeed from '@/components/VideoFeed';
import StatsPanel from '@/components/StatsPanel';
import RecordsTable from '@/components/RecordsTable';
import ModeToggle from '@/components/ModeToggle';
import NotificationToast from '@/components/NotificationToast';
import ExportPanel from '@/components/ExportPanel';
import { Activity, Users, Clock, TrendingUp } from 'lucide-react';

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<'live' | 'records' | 'users' | 'reports'>('live');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <NotificationToast />
      
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-900">Attendance System</h1>
                <p className="text-sm text-slate-500">Real-time face recognition</p>
              </div>
            </div>
            <ModeToggle />
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-6">
        <div className="bg-white rounded-lg shadow-sm p-1 inline-flex space-x-1">
          <button
            onClick={() => setActiveTab('live')}
            className={`px-4 py-2 rounded-md font-medium transition-all flex items-center space-x-2 ${
              activeTab === 'live'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <Activity className="w-4 h-4" />
            <span>Live Feed</span>
          </button>
          <button
            onClick={() => setActiveTab('records')}
            className={`px-4 py-2 rounded-md font-medium transition-all flex items-center space-x-2 ${
              activeTab === 'records'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <Clock className="w-4 h-4" />
            <span>Records</span>
          </button>
          <button
            onClick={() => setActiveTab('users')}
            className={`px-4 py-2 rounded-md font-medium transition-all flex items-center space-x-2 ${
              activeTab === 'users'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <Users className="w-4 h-4" />
            <span>Users</span>
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            className={`px-4 py-2 rounded-md font-medium transition-all flex items-center space-x-2 ${
              activeTab === 'reports'
                ? 'bg-blue-500 text-white shadow-sm'
                : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            <span>Reports</span>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'live' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <div className="bg-white rounded-xl shadow-sm overflow-hidden">
                <div className="p-4 bg-gradient-to-r from-blue-500 to-blue-600">
                  <h2 className="text-lg font-semibold text-white flex items-center space-x-2">
                    <Activity className="w-5 h-5" />
                    <span>Live Camera Feed</span>
                  </h2>
                </div>
                <VideoFeed />
              </div>
            </div>
            <div className="space-y-6">
              <StatsPanel />
              <div className="bg-white rounded-xl shadow-sm p-6">
                <h3 className="font-semibold text-lg mb-4 text-slate-900">Recent Activity</h3>
                <RecordsTable limit={5} />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'records' && (
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h2 className="text-lg font-semibold text-slate-900 mb-4">Attendance Records</h2>
            <RecordsTable limit={20} showFilters />
          </div>
        )}

        {activeTab === 'reports' && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-4">Export Data</h2>
              <ExportPanel />
            </div>
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-4">Statistics Overview</h2>
              <StatsPanel detailed />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}