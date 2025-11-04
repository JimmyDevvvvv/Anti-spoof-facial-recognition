'use client';
import React, { useState, useEffect } from 'react';
import api from '@/lib/api';
import { socket } from '@/lib/socket';

export default function ModeToggle() {
  const [mode, setMode] = useState('IN');

  useEffect(() => {
    const handleModeChange = (data: any) => setMode(data.mode);

    socket.on('mode_changed', handleModeChange);

    // ✅ proper cleanup function
    return () => {
      socket.off('mode_changed', handleModeChange);
    };
  }, []);

  const toggleMode = async () => {
    const newMode = mode === 'IN' ? 'OUT' : 'IN';
    await api.post('/mode', { mode: newMode });
  };

  return (
    <button
      onClick={toggleMode}
      className={`px-4 py-2 rounded-md font-semibold text-white ${
        mode === 'IN' ? 'bg-green-600' : 'bg-orange-500'
      }`}
    >
      Current Mode: {mode} (Click to toggle)
    </button>
  );
}
