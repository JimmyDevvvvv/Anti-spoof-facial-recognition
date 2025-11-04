'use client';

import React, { useState } from 'react';
import { Video, VideoOff } from 'lucide-react';

export default function VideoFeed() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);

  return (
    <div className="relative bg-black aspect-video">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-white font-medium">Loading camera feed...</p>
          </div>
        </div>
      )}
      
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900">
          <div className="text-center">
            <VideoOff className="w-16 h-16 text-red-500 mx-auto mb-4" />
            <p className="text-white font-medium">Failed to load camera feed</p>
            <p className="text-slate-400 text-sm mt-2">Please check your camera connection</p>
          </div>
        </div>
      )}

      <img
        src="http://localhost:5000/api/video/feed"
        alt="Live Attendance Feed"
        className="w-full h-full object-cover"
        onLoad={() => setIsLoading(false)}
        onError={() => {
          setIsLoading(false);
          setError(true);
        }}
      />

      {/* Live Indicator */}
      {!isLoading && !error && (
        <div className="absolute top-4 right-4 flex items-center space-x-2 bg-red-500 px-3 py-1.5 rounded-full">
          <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
          <span className="text-white text-sm font-medium">LIVE</span>
        </div>
      )}
    </div>
  );
}