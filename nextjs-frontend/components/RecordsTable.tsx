'use client';

import React, { useEffect, useState } from 'react';
import api from '@/lib/api';
import { socket } from '@/lib/socket';
import { Clock, User, CheckCircle, XCircle, Filter, Download } from 'lucide-react';

interface Record {
  name: string;
  check_type: string;
  timestamp: string;
  confidence: number;
  quality: number;
  is_live: boolean;
  date: string;
  time: string;
}

interface RecordsTableProps {
  limit?: number;
  showFilters?: boolean;
}

export default function RecordsTable({ limit = 10, showFilters = false }: RecordsTableProps) {
  const [records, setRecords] = useState<Record[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'ALL' | 'IN' | 'OUT'>('ALL');

  const fetchRecords = async () => {
    try {
      const res = await api.get(`/records/latest?n=${limit}`);
      setRecords(res.data.records);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching records:', error);
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();

    const handleNewRecord = (record: Record) => {
      setRecords((prev) => [record, ...prev.slice(0, limit - 1)]);
    };

    socket.on('new_record', handleNewRecord);

    return () => {
      socket.off('new_record', handleNewRecord);
    };
  }, [limit]);

  const filteredRecords = filter === 'ALL' 
    ? records 
    : records.filter(r => r.check_type === filter);

  if (loading) {
    return (
      <div className="space-y-3">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="animate-pulse flex items-center space-x-4 p-4 bg-slate-50 rounded-lg">
            <div className="w-10 h-10 bg-slate-200 rounded-full"></div>
            <div className="flex-1 space-y-2">
              <div className="h-4 bg-slate-200 rounded w-1/4"></div>
              <div className="h-3 bg-slate-200 rounded w-1/3"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div>
      {showFilters && (
        <div className="mb-4 flex items-center justify-between">
          <div className="flex space-x-2">
            <button
              onClick={() => setFilter('ALL')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                filter === 'ALL'
                  ? 'bg-blue-500 text-white shadow-sm'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilter('IN')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                filter === 'IN'
                  ? 'bg-green-500 text-white shadow-sm'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Check-In
            </button>
            <button
              onClick={() => setFilter('OUT')}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                filter === 'OUT'
                  ? 'bg-orange-500 text-white shadow-sm'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              Check-Out
            </button>
          </div>
          <button
            onClick={fetchRecords}
            className="px-4 py-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 transition-all font-medium"
          >
            Refresh
          </button>
        </div>
      )}

      {filteredRecords.length === 0 ? (
        <div className="text-center py-12 text-slate-500">
          <Clock className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No records found</p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-slate-200">
          <table className="w-full">
            <thead className="bg-slate-50 border-b border-slate-200">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                  Person
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                  Time
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-slate-200">
              {filteredRecords.map((record, index) => (
                <tr key={index} className="hover:bg-slate-50 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                        <User className="w-4 h-4 text-white" />
                      </div>
                      <span className="font-medium text-slate-900">{record.name}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        record.check_type === 'IN'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-orange-100 text-orange-800'
                      }`}
                    >
                      {record.check_type === 'IN' ? (
                        <>
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Check-In
                        </>
                      ) : (
                        <>
                          <XCircle className="w-3 h-3 mr-1" />
                          Check-Out
                        </>
                      )}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center space-x-2 text-sm text-slate-600">
                      <Clock className="w-4 h-4" />
                      <span>{new Date(record.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-slate-200 rounded-full h-2 max-w-[100px]">
                        <div
                          className={`h-2 rounded-full ${
                            record.confidence >= 80
                              ? 'bg-green-500'
                              : record.confidence >= 60
                              ? 'bg-yellow-500'
                              : 'bg-red-500'
                          }`}
                          style={{ width: `${record.confidence}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-slate-700">
                        {record.confidence?.toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    {record.is_live ? (
                      <span className="inline-flex items-center text-green-600 text-sm font-medium">
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Live
                      </span>
                    ) : (
                      <span className="inline-flex items-center text-red-600 text-sm font-medium">
                        <XCircle className="w-4 h-4 mr-1" />
                        Spoof
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}