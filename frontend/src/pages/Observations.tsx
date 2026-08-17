import React, { useEffect, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { AlertCircle, Eye } from 'lucide-react';

import { camerasApi, observationsApi, tigersApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatDateTime, formatPercent, titleCase } from '@/lib/utils';

const PAGE_SIZE = 50;

export default function Observations() {
  const [searchParams] = useSearchParams();
  const [cameraId, setCameraId] = useState(searchParams.get('camera_id') ?? '');
  const [tigerCode, setTigerCode] = useState(searchParams.get('tiger_code') ?? '');
  const [zone, setZone] = useState('');
  const [species, setSpecies] = useState('');
  const [minConfidence, setMinConfidence] = useState('');
  const [days, setDays] = useState('');
  const [page, setPage] = useState(0);

  const { data: cameras } = useApi(() => camerasApi.list({ limit: 200 }), []);
  const { data: tigers } = useApi(() => tigersApi.list({ limit: 200 }), []);

  const { data, loading, error, reload } = useApi(
    () =>
      observationsApi.list({
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        ...(cameraId ? { camera_id: cameraId } : {}),
        ...(tigerCode ? { tiger_code: tigerCode } : {}),
        ...(zone ? { zone } : {}),
        ...(species ? { species } : {}),
        ...(minConfidence ? { min_confidence: Number(minConfidence) } : {}),
        ...(days ? { days: Number(days) } : {}),
      }),
    [page, cameraId, tigerCode, zone, species, minConfidence, days]
  );

  const observations = data?.items ?? [];
  const totalPages = data?.pages ?? 1;

  const onFilterChange = (setter: (v: string) => void) => (value: string) => {
    setPage(0);
    setter(value);
  };

  if (error) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Detections unavailable"
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
      <div className="card p-4 flex flex-wrap gap-3">
        <select
          value={cameraId}
          onChange={(e) => onFilterChange(setCameraId)(e.target.value)}
          aria-label="Filter by camera"
          className="filter-input"
        >
          <option value="">All cameras</option>
          {cameras?.items.map((c) => (
            <option key={c.camera_id} value={c.camera_id}>
              {c.camera_id} — {c.name}
            </option>
          ))}
        </select>

        <select
          value={tigerCode}
          onChange={(e) => onFilterChange(setTigerCode)(e.target.value)}
          aria-label="Filter by tiger"
          className="filter-input"
        >
          <option value="">All tigers</option>
          {tigers?.items.map((t) => (
            <option key={t.tiger_id} value={t.tiger_id}>
              {t.tiger_id}
              {t.name ? ` — ${t.name}` : ''}
            </option>
          ))}
        </select>

        <select
          value={zone}
          onChange={(e) => onFilterChange(setZone)(e.target.value)}
          aria-label="Filter by zone"
          className="filter-input"
        >
          <option value="">All zones</option>
          <option value="core">Core</option>
          <option value="buffer">Buffer</option>
          <option value="village_adjacent">Village adjacent</option>
        </select>

        <select
          value={species}
          onChange={(e) => onFilterChange(setSpecies)(e.target.value)}
          aria-label="Filter by species"
          className="filter-input"
        >
          <option value="">All species</option>
          {['tiger', 'leopard', 'sambar', 'chital', 'wild_dog', 'gaur', 'sloth_bear'].map((s) => (
            <option key={s} value={s}>
              {titleCase(s)}
            </option>
          ))}
        </select>

        <select
          value={minConfidence}
          onChange={(e) => onFilterChange(setMinConfidence)(e.target.value)}
          aria-label="Minimum identity confidence"
          className="filter-input"
        >
          <option value="">Any confidence</option>
          <option value="0.75">≥ 75%</option>
          <option value="0.85">≥ 85%</option>
          <option value="0.9">≥ 90%</option>
        </select>

        <select
          value={days}
          onChange={(e) => onFilterChange(setDays)(e.target.value)}
          aria-label="Filter by period"
          className="filter-input"
        >
          <option value="">All time</option>
          <option value="7">Last 7 days</option>
          <option value="30">Last 30 days</option>
          <option value="90">Last 90 days</option>
        </select>
      </div>

      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Observation</th>
                <th>Species</th>
                <th>Tiger</th>
                <th>Camera</th>
                <th>Zone</th>
                <th>When</th>
                <th>Detection</th>
                <th>Identity</th>
                <th>Match</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={9} className="py-20 text-center">
                    <LoadingSpinner label="Loading detections…" />
                  </td>
                </tr>
              ) : observations.length === 0 ? (
                <tr>
                  <td colSpan={9} className="py-16 text-center text-muted-foreground">
                    No detections match these filters.
                  </td>
                </tr>
              ) : (
                observations.map((obs) => (
                  <tr key={obs.id}>
                    <td className="font-mono text-xs text-muted-foreground">
                      {obs.observation_id}
                    </td>
                    <td>{titleCase(obs.species)}</td>
                    <td>
                      {obs.tiger_code ? (
                        <Link to={`/tigers/${obs.tiger_code}`} className="hover:text-tiger-700">
                          <Badge variant="tiger">{obs.tiger_code}</Badge>
                        </Link>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td>
                      {obs.camera_id ? (
                        <Link to={`/cameras/${obs.camera_id}`}>
                          <Badge>{obs.camera_id}</Badge>
                        </Link>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="text-muted-foreground text-xs">{titleCase(obs.zone)}</td>
                    <td className="text-muted-foreground text-xs">{formatDateTime(obs.timestamp)}</td>
                    <td>{formatPercent(obs.detection_confidence)}</td>
                    <td>{formatPercent(obs.identity_confidence)}</td>
                    <td>
                      {obs.match_type ? <StatusBadge status={obs.match_type} /> : <Badge>—</Badge>}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {!loading && observations.length > 0 && (
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground flex items-center gap-2">
            <Eye className="w-4 h-4" />
            {data?.total ?? 0} detections • page {(data?.page ?? 1)} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              className="btn-secondary"
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
            >
              Previous
            </button>
            <button
              className="btn-secondary"
              onClick={() => setPage((p) => p + 1)}
              disabled={(data?.page ?? 1) >= totalPages}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
