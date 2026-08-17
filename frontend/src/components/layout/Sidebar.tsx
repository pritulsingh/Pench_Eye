import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  BarChart3,
  BellRing,
  Camera,
  Eye,
  Filter,
  Image as ImageIcon,
  LayoutDashboard,
  Map as MapIcon,
  PawPrint,
  Upload,
  UserCheck,
} from 'lucide-react';

import { alertsApi, dashboardApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';

const SECTIONS: Array<{
  heading: string;
  links: Array<{ to: string; label: string; icon: JSX.Element; badge?: 'alerts' | 'reviews' }>;
}> = [
  {
    heading: 'Monitor',
    links: [
      { to: '/', label: 'Command Center', icon: <LayoutDashboard size={18} /> },
      { to: '/map', label: 'Reserve Map', icon: <MapIcon size={18} /> },
      { to: '/alerts', label: 'Alerts', icon: <BellRing size={18} />, badge: 'alerts' },
      { to: '/analytics', label: 'Analytics', icon: <BarChart3 size={18} /> },
    ],
  },
  {
    heading: 'Wildlife',
    links: [
      { to: '/tigers', label: 'Tigers', icon: <PawPrint size={18} /> },
      { to: '/observations', label: 'Detections', icon: <Eye size={18} /> },
      { to: '/gallery', label: 'Gallery', icon: <ImageIcon size={18} /> },
    ],
  },
  {
    heading: 'Network',
    links: [
      { to: '/cameras', label: 'Camera Traps', icon: <Camera size={18} /> },
      { to: '/upload', label: 'Ingest Images', icon: <Upload size={18} /> },
      { to: '/triage', label: 'Triage', icon: <Filter size={18} /> },
      { to: '/reviews', label: 'Human Review', icon: <UserCheck size={18} />, badge: 'reviews' },
    ],
  },
];

export function Sidebar() {
  const { data: alertSummary } = useApi(() => alertsApi.summary());
  const { data: stats } = useApi(() => dashboardApi.getStats());

  const badgeValue = (kind?: 'alerts' | 'reviews'): number => {
    if (kind === 'alerts') return alertSummary?.open ?? 0;
    if (kind === 'reviews') return stats?.pending_reviews ?? 0;
    return 0;
  };

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-16 lg:w-60 bg-white/95 border-r border-border flex flex-col backdrop-blur-xl z-20 shadow-sm">
      <div className="p-4 lg:p-6">
        <div className="flex items-center gap-3 mb-1">
          <PawPrint className="text-tiger-600 w-6 h-6 shrink-0" aria-hidden="true" />
          <h1 className="font-bold text-xl tracking-tight hidden lg:block">PENCH EYE</h1>
        </div>
        <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider ml-9 hidden lg:block">
          Wildlife Intelligence
        </p>
      </div>

      <nav className="flex-1 px-2 lg:px-3 space-y-5 mt-2 overflow-y-auto" aria-label="Main navigation">
        {SECTIONS.map((section) => (
          <div key={section.heading}>
            <p className="px-3 mb-1.5 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/70 hidden lg:block">
              {section.heading}
            </p>
            <div className="space-y-1">
              {section.links.map((link) => {
                const count = badgeValue(link.badge);
                return (
                  <NavLink
                    key={link.to}
                    to={link.to}
                    end={link.to === '/'}
                    title={link.label}
                    className={({ isActive }) =>
                      `nav-link justify-center lg:justify-start ${isActive ? 'active' : ''}`
                    }
                  >
                    {link.icon}
                    <span className="flex-1 hidden lg:inline">{link.label}</span>
                    {count > 0 && (
                      <span className="bg-tiger-600 text-white text-[10px] font-bold px-2 py-0.5 rounded-full hidden lg:inline">
                        {count}
                      </span>
                    )}
                  </NavLink>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="p-4 mt-auto hidden lg:block">
        <div className="text-center text-[10px] text-muted-foreground font-mono opacity-60">
          Pench Eye v1.0
          <br />
          {stats ? `${stats.total_cameras} cameras • ${stats.total_tigers} tigers` : 'connecting…'}
        </div>
      </div>
    </aside>
  );
}
