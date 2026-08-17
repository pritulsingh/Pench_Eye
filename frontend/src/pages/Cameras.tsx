import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, Camera as CameraIcon, Search } from 'lucide-react';

import { camerasApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { MARKER_COLORS, MARKER_LABELS } from '@/features/map/mapConfig';
import { relativeTime, titleCase } from '@/lib/utils';

export default function Cameras() {
  const [zone, setZone] = useState('');
  const [status, setStatus] = useState('');
  const [search, setSearch] = useState('');

  const { data, loading, error, reload } = useApi(
    () =>
      camerasApi.list({
        limit: 200,
        ...(zone ? { zone } : {}),
        ...(status ? { status } : {}),
        ...(search ? { search } : {}),
      }),
    [zone, status, search]
  );

  const cameras = data?.items ?? [];

  if (error) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Cameras unavailable"
        description={error}
        action={
          <button className="btn-primary" onClick={reload}>
            Retry
          </button>
        }
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="card p-4 flex flex-wrap gap-3 items-center">
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search camera ID or name…"
            aria-label="Search cameras"
            className="filter-input w-full pl-9"
          />
        </div>
        <select
          value={zone}
          onChange={(e) => setZone(e.target.value)}
          aria-label="Filter by zone"
          className="filter-input"
        >
          <option value="">All zones</option>
          <option value="core">Core</option>
          <option value="buffer">Buffer</option>
          <option value="village_adjacent">Village adjacent</option>
        </select>
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          aria-label="Filter by status"
          className="filter-input"
        >
          <option value="">Any status</option>
          <option value="active">Active</option>
          <option value="inactive">Inactive</option>
          <option value="maintenance">Maintenance</option>
        </select>

        <div className="ml-auto flex flex-wrap gap-3 text-xs text-muted-foreground">
          {Object.entries(MARKER_LABELS).map(([key, label]) => (
            <span key={key} className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: MARKER_COLORS[key] }}
              />
              {label}
            </span>
          ))}
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Camera</th>
                <th>Name</th>
                <th>Zone</th>
                <th>Coordinates</th>
                <th>State</th>
                <th>Battery</th>
                <th>Last seen</th>
                <th>Last detection</th>
                <th className="text-right">Detections</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={9} className="py-20 text-center">
                    <LoadingSpinner label="Loading cameras…" />
                  </td>
                </tr>
              ) : cameras.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-16 text-center text-muted-foreground">
                    No cameras match these filters.
                  </td>
                </tr>
              ) : (
                cameras.map((cam) => (
                  <tr key={cam.camera_id}>
                    <td className="font-bold">
                      <Link to={`/cameras/${cam.camera_id}`} className="hover:text-tiger-700">
                        {cam.camera_id}
                      </Link>
                    </td>
                    <td>{cam.name}</td>
                    <td>
                      <Badge
                        variant={
                          cam.zone === 'core'
                            ? 'success'
                            : cam.zone === 'buffer'
                            ? 'review'
                            : 'error'
                        }
                      >
                        {titleCase(cam.zone)}
                      </Badge>
                    </td>
                    <td className="font-mono text-xs text-muted-foreground">
                      {cam.latitude !== null && cam.longitude !== null
                        ? `${cam.latitude.toFixed(4)}, ${cam.longitude.toFixed(4)}`
                        : '—'}
                    </td>
                    <td>
                      <span className="inline-flex items-center gap-2 text-xs">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{ backgroundColor: MARKER_COLORS[cam.marker_state] }}
                        />
                        {MARKER_LABELS[cam.marker_state] ?? titleCase(cam.marker_state)}
                      </span>
                    </td>
                    <td className="text-xs">
                      {cam.battery_percent === null ? (
                        '—'
                      ) : (
                        <span
                          className={
                            cam.battery_percent < 25 ? 'text-red-700' : 'text-muted-foreground'
                          }
                        >
                          {cam.battery_percent}%
                        </span>
                      )}
                    </td>
                    <td className="text-muted-foreground text-xs">
                      {relativeTime(cam.last_active_at)}
                    </td>
                    <td className="text-muted-foreground text-xs">
                      {relativeTime(cam.last_detection_at)}
                    </td>
                    <td className="text-right font-medium text-tiger-700">
                      {cam.observation_count}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {!loading && cameras.length > 0 && (
        <p className="text-xs text-muted-foreground flex items-center gap-2">
          <CameraIcon className="w-3.5 h-3.5" />
          {cameras.length} of {data?.total ?? cameras.length} camera stations
        </p>
      )}
    </div>
  );
}
