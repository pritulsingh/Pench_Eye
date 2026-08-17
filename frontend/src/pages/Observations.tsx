import { useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { Eye, LayoutGrid, List } from 'lucide-react';

import { StatusBadge } from '@/components/ui/StatusBadge';
import { Badge } from '@/components/ui/Badge';
import { useMonitoringStore } from '@/features/monitoring/MonitoringContext';
import { formatDateTime, titleCase } from '@/lib/utils';

/**
 * Detections page.
 *
 * Populated from the SHARED monitoring store — the same detection records the
 * Reserve Map, Tiger Profiles and Dashboard use. A camera upload therefore
 * appears here immediately. Detections are no longer trapped on the map.
 */
export default function Observations() {
  const [searchParams] = useSearchParams();
  const { detections, tigers, cameras } = useMonitoringStore();

  const [cameraId, setCameraId] = useState(searchParams.get('camera_id') ?? '');
  const [tigerCode, setTigerCode] = useState(searchParams.get('tiger_code') ?? '');
  const [source, setSource] = useState('');
  const [minConfidence, setMinConfidence] = useState('');
  const [view, setView] = useState<'grid' | 'table'>('grid');

  const tigerById = useMemo(() => new Map(tigers.map((t) => [t.id, t])), [tigers]);
  const cameraById = useMemo(() => new Map(cameras.map((c) => [c.id, c])), [cameras]);

  const filtered = useMemo(() => {
    return detections
      .filter((d) => {
        if (cameraId && d.cameraId !== cameraId) return false;
        if (tigerCode && d.tigerId !== tigerCode) return false;
        if (source && d.source !== source) return false;
        if (minConfidence && d.confidence < Number(minConfidence)) return false;
        return true;
      })
      .sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp));
  }, [detections, cameraId, tigerCode, source, minConfidence]);

  return (
    <div className="space-y-6">
      <div className="card p-4 flex flex-wrap gap-3 items-center">
        <select
          value={cameraId}
          onChange={(e) => setCameraId(e.target.value)}
          aria-label="Filter by camera"
          className="filter-input"
        >
          <option value="">All cameras</option>
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>
              {c.id} — {c.name}
            </option>
          ))}
        </select>

        <select
          value={tigerCode}
          onChange={(e) => setTigerCode(e.target.value)}
          aria-label="Filter by tiger"
          className="filter-input"
        >
          <option value="">All tigers</option>
          {tigers.map((t) => (
            <option key={t.id} value={t.id}>
              {t.id} — {t.name}
            </option>
          ))}
        </select>

        <select
          value={source}
          onChange={(e) => setSource(e.target.value)}
          aria-label="Filter by source"
          className="filter-input"
        >
          <option value="">Any source</option>
          <option value="simulated">Simulated</option>
          <option value="manual">Manual</option>
          <option value="ai">AI</option>
        </select>

        <select
          value={minConfidence}
          onChange={(e) => setMinConfidence(e.target.value)}
          aria-label="Minimum confidence"
          className="filter-input"
        >
          <option value="">Any confidence</option>
          <option value="0.75">≥ 75%</option>
          <option value="0.85">≥ 85%</option>
          <option value="0.9">≥ 90%</option>
        </select>

        <div className="ml-auto flex gap-1">
          <button
            onClick={() => setView('grid')}
            aria-pressed={view === 'grid'}
            aria-label="Grid view"
            className={view === 'grid' ? 'btn-primary !px-2.5' : 'btn-secondary !px-2.5'}
          >
            <LayoutGrid className="w-4 h-4" />
          </button>
          <button
            onClick={() => setView('table')}
            aria-pressed={view === 'table'}
            aria-label="Table view"
            className={view === 'table' ? 'btn-primary !px-2.5' : 'btn-secondary !px-2.5'}
          >
            <List className="w-4 h-4" />
          </button>
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="card p-16 text-center text-muted-foreground">
          No detections match these filters.
        </div>
      ) : view === 'grid' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map((det) => {
            const tiger = det.tigerId ? tigerById.get(det.tigerId) : undefined;
            const cam = cameraById.get(det.cameraId);
            const img = det.imagePath ?? tiger?.referenceImage;
            return (
              <div key={det.id} className="card overflow-hidden">
                <div className="h-36 bg-secondary/40">
                  {img ? (
                    <img
                      src={img}
                      alt={`Detection ${det.id}`}
                      loading="lazy"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full grid place-items-center text-muted-foreground">
                      <Eye className="w-6 h-6" />
                    </div>
                  )}
                </div>
                <div className="p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    {det.tigerId ? (
                      <Link to={`/tigers/${det.tigerId}`}>
                        <Badge variant="tiger">{det.tigerId}</Badge>
                      </Link>
                    ) : (
                      <Badge>Unidentified</Badge>
                    )}
                    <Badge variant="default">{titleCase(det.source)}</Badge>
                  </div>
                  <div className="text-sm">
                    <Link to={`/cameras/${det.cameraId}`} className="font-medium hover:text-tiger-700">
                      {det.cameraId}
                    </Link>
                    {cam && <span className="text-muted-foreground"> — {cam.name}</span>}
                  </div>
                  <div className="text-xs text-muted-foreground">{formatDateTime(det.timestamp)}</div>
                  <div className="flex justify-between text-xs pt-1 border-t border-border/60">
                    <span className="text-muted-foreground">
                      Detection <strong className="text-foreground">{Math.round(det.confidence * 100)}%</strong>
                    </span>
                    <span className="text-muted-foreground">
                      ~{det.estimatedDistanceFromCameraKm} km
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Detection</th>
                  <th>Tiger</th>
                  <th>Camera</th>
                  <th>When</th>
                  <th>Confidence</th>
                  <th>Location</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((det) => {
                  const cam = cameraById.get(det.cameraId);
                  return (
                    <tr key={det.id}>
                      <td className="font-mono text-xs text-muted-foreground">{det.id}</td>
                      <td>
                        {det.tigerId ? (
                          <Link to={`/tigers/${det.tigerId}`}>
                            <Badge variant="tiger">{det.tigerId}</Badge>
                          </Link>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </td>
                      <td>
                        <Link to={`/cameras/${det.cameraId}`}>
                          <Badge>{det.cameraId}</Badge>
                        </Link>
                        {cam && <span className="text-muted-foreground text-xs"> {cam.name}</span>}
                      </td>
                      <td className="text-muted-foreground text-xs">{formatDateTime(det.timestamp)}</td>
                      <td>{Math.round(det.confidence * 100)}%</td>
                      <td className="text-muted-foreground text-xs">
                        {det.latitude.toFixed(3)}, {det.longitude.toFixed(3)}
                      </td>
                      <td>
                        <StatusBadge status={det.source} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Eye className="w-4 h-4" />
        {filtered.length} of {detections.length} detections · shared monitoring data
      </div>
    </div>
  );
}
