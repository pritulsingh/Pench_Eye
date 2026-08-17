import { useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertTriangle, Play, Radio } from 'lucide-react';

import { assetUrl, camerasApi, demoApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { formatDateTime, formatPercent, titleCase } from '@/lib/utils';
import type { SimulationEvent } from '@/types';

export default function DemoMode() {
  const [cameraId, setCameraId] = useState('');
  const [count, setCount] = useState(1);
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<SimulationEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  const { data: status } = useApi(() => demoApi.status());
  const { data: cameras } = useApi(() => camerasApi.list({ limit: 200 }));

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await demoApi.simulate({ count, camera_id: cameraId || undefined });
      setEvents((prev) => [...res.data.events, ...prev].slice(0, 20));
    } catch (err) {
      const e = err as { userMessage?: string };
      setError(e.userMessage ?? 'Simulation failed.');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="demo-banner">
        <AlertTriangle className="w-4 h-4 shrink-0" />
        <span>
          {status?.simulation_disclaimer ??
            'Simulated captures are generated locally and are not live camera-trap data.'}
        </span>
      </div>

      <div className="card p-6 space-y-5">
        <div className="flex flex-wrap items-center gap-6">
          <div>
            <div className="section-title mb-1">Inference mode</div>
            <div className="flex items-center gap-2">
              <Badge variant={status?.is_demo_inference ? 'demo' : 'success'}>
                {status?.ml_mode ?? '—'}
              </Badge>
              <span className="text-sm text-muted-foreground">{status?.model_version}</span>
            </div>
          </div>
          <div>
            <div className="section-title mb-1">Tiger Re-ID</div>
            <div className="flex items-center gap-2">
              <Badge
                variant={
                  status?.reid_is_demo ? 'demo' : status?.reid_available ? 'success' : 'error'
                }
              >
                {status?.reid_is_demo
                  ? 'simulated'
                  : status?.reid_available
                  ? status?.reid_validated
                    ? 'trained + validated'
                    : 'trained (unvalidated)'
                  : 'unavailable'}
              </Badge>
              {status?.reid_model_version && !status.reid_is_demo && (
                <span className="text-sm text-muted-foreground">{status.reid_model_version}</span>
              )}
              {status?.reid_known_identities ? (
                <span className="text-xs text-muted-foreground">
                  {status.reid_known_identities} enrolled
                </span>
              ) : null}
            </div>
          </div>
          <div>
            <div className="section-title mb-1">Geo data</div>
            <Badge variant={status?.geo_data_source === 'demo' ? 'demo' : 'success'}>
              {status?.geo_data_source ?? '—'}
            </Badge>
          </div>
        </div>

        {status?.reid_error && (
          <div className="badge-error px-4 py-2 rounded-md text-sm">{status.reid_error}</div>
        )}

        {status?.disclaimer && (
          <p className="text-xs text-muted-foreground max-w-2xl leading-relaxed">
            {status.disclaimer}
          </p>
        )}

        <div className="flex flex-wrap items-end gap-3 pt-2 border-t border-border">
          <label className="text-sm text-muted-foreground flex flex-col gap-1">
            Camera
            <select
              value={cameraId}
              onChange={(e) => setCameraId(e.target.value)}
              className="filter-input"
            >
              <option value="">Random active camera</option>
              {cameras?.items.map((c) => (
                <option key={c.camera_id} value={c.camera_id}>
                  {c.camera_id} — {c.name}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm text-muted-foreground flex flex-col gap-1">
            Captures
            <select
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
              className="filter-input"
            >
              {[1, 3, 5].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>

          <button onClick={run} disabled={running} className="btn-primary flex items-center gap-2">
            <Play className={`w-4 h-4 ${running ? 'animate-pulse' : ''}`} />
            {running ? 'Running pipeline…' : 'Simulate capture'}
          </button>
        </div>

        {error && <div className="badge-error px-4 py-2 rounded-md text-sm">{error}</div>}
      </div>

      {events.length === 0 ? (
        <EmptyState
          icon={<Radio />}
          title="No simulated events yet"
          description="Trigger a capture to watch a frame flow through triage, detection, identification, alerting, and onto the map."
        />
      ) : (
        <div className="space-y-4">
          {events.map((event, i) => (
            <article key={`${event.image_id}-${i}`} className="card p-5">
              <div className="flex flex-wrap items-center gap-3 mb-4">
                <Badge variant="demo">Simulated</Badge>
                <h3 className="font-semibold">
                  {event.camera_id} — {event.camera_name}
                </h3>
                <span className="text-xs text-muted-foreground">
                  {formatDateTime(event.captured_at)}
                </span>
                <Link
                  to={`/cameras/${event.camera_id}`}
                  className="text-xs text-tiger-700 hover:underline ml-auto"
                >
                  Open camera →
                </Link>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <img
                  src={assetUrl(`/api/v1/images/${event.image_id}/file`)}
                  alt={`Simulated capture ${event.image_id}`}
                  className="w-full h-40 object-cover rounded-lg border border-border bg-secondary/40"
                />

                <ol className="md:col-span-2 space-y-2 text-sm">
                  {[
                    { step: 'Capture', detail: `${event.image_id} stored` },
                    {
                      step: 'Triage',
                      detail: event.is_blank
                        ? `Blank frame (${formatPercent(event.blank_probability)}) — quarantined`
                        : `Subject frame (${formatPercent(event.blank_probability)} blank score)`,
                    },
                    {
                      step: 'Detection',
                      detail: event.species ? titleCase(event.species) : 'No animal detected',
                    },
                    {
                      step: 'Identification',
                      detail:
                        event.decision === 'identity_unavailable'
                          ? 'Unavailable — no trained Re-ID model installed; queued for review'
                          : event.tiger_code
                          ? `${event.tiger_code} at ${formatPercent(event.identity_confidence)} (${titleCase(
                              event.decision
                            )})`
                          : event.decision
                          ? titleCase(event.decision)
                          : 'Not applicable',
                    },
                    {
                      step: 'Sighting',
                      detail: event.observation_id
                        ? `${event.observation_id} recorded in ${titleCase(event.zone)}`
                        : 'No sighting created',
                    },
                    {
                      step: 'Alerts',
                      detail:
                        event.alerts_created > 0
                          ? `${event.alerts_created} alert(s) raised`
                          : 'No alert thresholds crossed',
                    },
                  ].map((row) => (
                    <li key={row.step} className="flex gap-3">
                      <span className="w-28 shrink-0 text-xs uppercase tracking-wider text-muted-foreground pt-0.5">
                        {row.step}
                      </span>
                      <span>{row.detail}</span>
                    </li>
                  ))}
                </ol>
              </div>

              <div className="flex flex-wrap gap-3 mt-5 pt-4 border-t border-border">
                <Link to="/map" className="btn-secondary">
                  See it on the map
                </Link>
                {event.tiger_code && (
                  <Link to={`/tigers/${event.tiger_code}`} className="btn-secondary">
                    Tiger profile
                  </Link>
                )}
                <Link to="/alerts" className="btn-secondary">
                  Alerts
                </Link>
                <Link to="/analytics" className="btn-secondary">
                  Analytics
                </Link>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
