'use client';
import { useEffect } from 'react';
import { socket } from '@/lib/socket';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

export default function NotificationToast() {
  useEffect(() => {
    const handleNotification = (notif: any) => {
      if (notif.type === 'success') toast.success(notif.title);
      else toast.error(notif.title);
    };

    socket.on('notification', handleNotification);

    return () => {
      socket.off('notification', handleNotification);
    };
  }, []);

  return (
    <ToastContainer
      position="bottom-right"
      aria-label="Notification messages" // ✅ Fix for TS + accessibility
    />
  );
}
