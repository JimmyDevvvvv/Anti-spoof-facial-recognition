'use client';
import React, { useEffect, useState } from 'react';
import api from '@/lib/api';
import { socket } from '@/lib/socket';

// ✅ 1️⃣ Define a prop type for StatsPanel
interface StatsPanelProps {
  detailed?: boolean; // Optional flag to show more data
}

// ✅ 2️⃣ Accept and destructure the prop
export default function StatsPanel({ detailed = false }: StatsPanelProps) {
  const [stats, setStats] = useState<any>({});

  useEffect(() => {
    const fetchStats = async () => {
      const res = await api.get('/statistics/quick');
      setStats(res.data);
    };

    fetchStats();

    const handleStatsUpdate = (data: any) => setStats(data);
    socket.on('stats_update', handleStatsUpdate);

    return () => {
      socket.off('stats_update', handleStatsUpdate);
    };
  }, []);

  return (
    <div className="p-4 bg-gray-50 border rounded-md">
      <h2 className="font-semibold text-lg mb-2">
        📈 {detailed ? 'Detailed Attendance Statistics' : 'Quick Statistics'}
      </h2>
      <ul>
        <li>Total Records: {stats.total_records ?? 0}</li>
        <li>Unique People: {stats.unique_people ?? 0}</li>
        <li>Check-ins: {stats.check_ins ?? 0}</li>
        <li>Check-outs: {stats.check_outs ?? 0}</li>

        {/* ✅ 3️⃣ Show extra info only when detailed mode is on */}
        {detailed && (
          <>
            <li>Current Mode: {stats.current_mode}</li>
            <li>Last Updated: {new Date().toLocaleTimeString()}</li>
          </>
        )}
      </ul>
    </div>
  );
}
