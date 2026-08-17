import { useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { MapContainer, Marker, Polyline, Popup, TileLayer, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { AlertCircle, ArrowLeft, Clock, ImageIcon } from 'lucide-react';

import { StatusBadge } from '@/components/ui/StatusBadge';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import {
  SATELLITE_ATTRIBUTION,
  SATELLITE_TILES,
  cameraIcon,
  sightingIcon,
} from '@/features/map/mapConfig';
import { useMonitoringStore } from '@/features/monitoring/MonitoringContext';
import { nearbyTigers } from '@/features/monitoring/analysis';
import { formatDateTime, titleCase } from '@/lib/utils';

type Tab = 'gallery' | 'timeline' | 'movement' | 'conflict';

const SEX_LABEL: Record<string, string> = {
  male: '♂ Male',
  female: '♀ Female',
  unknown: 'Unknown',
};

/**
 * Tiger Profile — derived entirely from the shared monitoring store so the
 * identity image, gallery, detection history, movement path and conflict
 * status always agree with the Reserve Map and every other view.
 */
export default function TigerProfile() {
  const { id } = useParams<{ id: string }>();
  const [tab, setTab] = useState<Tab>('gallery');
  const { tigers, cameras, detections, territories, conflicts, overlaps, coDetections } =
    useMonitoringStore();

  const tiger = useMemo(() => tigers.find((t) => t.id === id), [tigers, id]);

  const tigerDetections = useMemo(
    () =>
      detections
        .filter((d) => d.tigerId === id)
        .sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp)),
    [detections, id]
  );

  const cameraById = useMemo(() => new Map(cameras.map((c) => [c.id, c])), [cameras]);

  const territory = useMemo(
    () => territories.find((t) => t.tigerId === id),
    [territories, id]
  );

  const nearby = useMemo(
    () => (tiger ? nearbyTigers(tiger, tigers) : []),
    [tiger, tigers]
  );

  const relatedConflicts = useMemo(
    () => conflicts.filter((c) => c.tigerA === id || c.tigerB === id),
    [conflicts, id]
  );
  const relatedOverlaps = useMemo(
    () => overlaps.filter((o) => o.tigerA === id || o.tigerB === id),
    [overlaps, id]
  );
  const relatedCoDetections = useMemo(
    () => coDetections.filter((cd) => cd.tigerIds.includes(id as string)),
    [coDetections, id]
  );

  if (!tiger) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Tiger not found"
        description={`No individual matches ${id}.`}
        action={
          <Link to="/tigers" className="btn-primary">
            Back to catalog
          </Link>
        }
      />
    );
  }

  const lastCamera = tiger.lastDetectedCamera ? cameraById.get(tiger.lastDetectedCamera) : undefined;
  const center: [number, number] = tiger.currentLocation;

  const movementCameras = Array.from(
    new Set(tiger.movementHistory.map((m) => m.cameraId))
  )
    .map((cid) => cameraById.get(cid))
    .filter(Boolean);

  return (
    <div className="space-y-6">
      <Link
        to="/tigers"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="w-4 h-4" /> Tiger catalog
      </Link>

      {/* Identity header */}
      <div className="card p-6 bg-gradient-to-br from-tiger-100 to-white border-tiger-500/20">
        <div className="flex flex-col md:flex-row gap-6">
          <div className="w-full md:w-56 shrink-0">
            <div className="rounded-xl overflow-hidden border border-border bg-secondary/40 h-56">
              {tiger.referenceImage ? (
                <img
                  src={tiger.referenceImage}
                  alt={`${tiger.id} reference`}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                  <ImageIcon />
                </div>
              )}
            </div>
          </div>

          <div className="flex-1">
            <div className="flex items-center gap-3">
              <h1 className="text-4xl font-bold">{tiger.id}</h1>
              <StatusBadge status={tiger.status} />
              {relatedConflicts.length > 0 && <Badge variant="error">In conflict</Badge>}
            </div>
            <h2 className="text-xl text-tiger-700 mt-1">{tiger.name}</h2>

            <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mt-4">
              <span>
                Sex: <strong className="text-foreground">{SEX_LABEL[tiger.sex]}</strong>
              </span>
              <span className="border-l border-border pl-4">
                Age: <strong className="text-foreground">{titleCase(tiger.ageClass)}</strong>
              </span>
              <span className="border-l border-border pl-4">
                Territory: <strong className="text-foreground">{tiger.territoryId}</strong>
                {territory && (
                  <span className="text-foreground"> · ~{territory.areaLabelKm2} km²</span>
                )}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
              {[
                { label: 'Detections', value: tiger.detectionIds.length },
                {
                  label: 'Cameras',
                  value: new Set(tiger.movementHistory.map((m) => m.cameraId)).size,
                },
                {
                  label: 'Confidence',
                  value: tiger.confidence != null ? `${Math.round(tiger.confidence * 100)}%` : '—',
                },
                { label: 'Nearby tigers', value: nearby.length },
              ].map((item) => (
                <div
                  key={item.label}
                  className="bg-secondary/50 p-3 rounded-lg border border-border"
                >
                  <div className="text-xl font-bold text-tiger-500">{item.value}</div>
                  <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
                    {item.label}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Current status */}
      <div className="card p-6">
        <h3 className="font-semibold mb-4">Current Status</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-xs text-muted-foreground">Estimated location</div>
            <div className="font-medium">
              {center[0].toFixed(4)}, {center[1].toFixed(4)}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Last camera</div>
            <div className="font-medium">
              {lastCamera ? (
                <Link to={`/cameras/${lastCamera.id}`} className="hover:text-tiger-700">
                  {lastCamera.id} — {lastCamera.name}
                </Link>
              ) : (
                '—'
              )}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Last detection</div>
            <div className="font-medium">
              {tiger.lastDetectionTime ? formatDateTime(tiger.lastDetectionTime) : '—'}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Territory</div>
            <div className="font-medium">{tiger.territoryId}</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-border flex gap-6">
        {(['gallery', 'timeline', 'movement', 'conflict'] as Tab[]).map((key) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            aria-pressed={tab === key}
            className={`pb-3 text-sm font-medium transition-colors ${
              tab === key
                ? 'text-tiger-500 border-b-2 border-tiger-500'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {titleCase(key)}
          </button>
        ))}
      </div>

      {tab === 'gallery' && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {(tiger.gallery ?? []).map((src, i) => (
            <figure key={src} className="card overflow-hidden group">
              <img
                src={src}
                alt={`${tiger.id} image ${i + 1}`}
                loading="lazy"
                className="w-full h-44 object-cover bg-secondary/40 group-hover:scale-105 transition-transform"
              />
              <figcaption className="p-2 text-[11px] text-muted-foreground flex justify-between">
                <span>{tiger.id}</span>
                <span>{i === 0 ? 'Reference' : `Variant ${i}`}</span>
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      {tab === 'timeline' && (
        <div className="space-y-3 max-w-3xl">
          {tigerDetections.length === 0 ? (
            <EmptyState
              icon={<Clock />}
              title="No detections yet"
              description="Detections appear here chronologically."
            />
          ) : (
            tigerDetections.map((det) => {
              const cam = cameraById.get(det.cameraId);
              return (
                <div key={det.id} className="flex gap-4 p-4 card items-center">
                  <img
                    src={det.imagePath ?? tiger.referenceImage ?? ''}
                    alt={`Detection ${det.id}`}
                    loading="lazy"
                    className="w-16 h-16 rounded-lg object-cover bg-secondary/40 shrink-0"
                  />
                  <div className="flex-1">
                    <div className="font-medium">{formatDateTime(det.timestamp)}</div>
                    <div className="text-sm text-muted-foreground">
                      {det.cameraId}
                      {cam && ` — ${cam.name}`} · ~{det.estimatedDistanceFromCameraKm} km away
                    </div>
                  </div>
                  <div className="text-right text-sm">
                    <div className="font-medium">{Math.round(det.confidence * 100)}%</div>
                    <div className="text-[11px] text-muted-foreground">{titleCase(det.source)}</div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {tab === 'movement' && (
        <div className="space-y-4">
          <div className="demo-banner">
            Estimated Movement Path — straight-line links between consecutive camera detections, not
            observed travel routes or GPS tracking.
          </div>
          <div className="h-[480px] rounded-xl overflow-hidden border border-border relative z-0">
            <MapContainer
              center={center}
              zoom={12}
              style={{ height: '100%', width: '100%', background: '#f3f8f4' }}
            >
              <TileLayer url={SATELLITE_TILES} attribution={SATELLITE_ATTRIBUTION} />

              {movementCameras.map((cam) => (
                <Marker
                  key={cam!.id}
                  position={[cam!.latitude, cam!.longitude]}
                  icon={cameraIcon('active')}
                >
                  <Popup>
                    <div className="text-xs text-slate-900">
                      <strong>{cam!.id}</strong>
                      <br />
                      {cam!.name}
                    </div>
                  </Popup>
                </Marker>
              ))}

              {tiger.movementHistory.map((m) => (
                <Marker
                  key={m.detectionId}
                  position={[m.latitude, m.longitude]}
                  icon={sightingIcon(true)}
                >
                  <Popup>
                    <div className="text-xs text-slate-900">
                      <strong>{formatDateTime(m.timestamp)}</strong>
                      <br />
                      {m.cameraId}
                    </div>
                  </Popup>
                </Marker>
              ))}

              {tiger.movementHistory.length > 1 && (
                <Polyline
                  positions={tiger.movementHistory.map((m) => [m.latitude, m.longitude])}
                  pathOptions={{ color: '#f59e0b', weight: 2 }}
                >
                  <Tooltip sticky>Estimated Movement Path</Tooltip>
                </Polyline>
              )}
            </MapContainer>
          </div>
        </div>
      )}

      {tab === 'conflict' && (
        <div className="space-y-4">
          <div className="card p-6">
            <h3 className="font-semibold mb-3">Nearby tigers (within conflict radius)</h3>
            {nearby.length === 0 ? (
              <p className="text-sm text-muted-foreground">No tigers within the conflict radius.</p>
            ) : (
              <ul className="space-y-2 text-sm">
                {nearby.map((n) => (
                  <li key={n.id} className="flex items-center justify-between">
                    <Link to={`/tigers/${n.id}`} className="hover:text-tiger-700 font-medium">
                      {n.id} — {n.name}
                    </Link>
                    <Badge variant="error">{n.distanceKm} km</Badge>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="card p-6">
            <h3 className="font-semibold mb-3">Conflict / proximity events</h3>
            {relatedConflicts.length === 0 &&
            relatedOverlaps.length === 0 &&
            relatedCoDetections.length === 0 ? (
              <p className="text-sm text-muted-foreground">No conflict events for this tiger.</p>
            ) : (
              <ul className="space-y-2 text-sm">
                {relatedConflicts.map((c) => {
                  const other = c.tigerA === id ? c.tigerB : c.tigerA;
                  return (
                    <li key={c.id} className="flex items-center gap-2">
                      <Badge variant="error">Proximity</Badge>
                      <span>
                        {c.distanceKm} km from{' '}
                        <Link to={`/tigers/${other}`} className="hover:text-tiger-700">
                          {other}
                        </Link>
                      </span>
                    </li>
                  );
                })}
                {relatedOverlaps.map((o) => {
                  const other = o.tigerA === id ? o.tigerB : o.tigerA;
                  return (
                    <li key={o.id} className="flex items-center gap-2">
                      <Badge variant="warning">Territory overlap</Badge>
                      <span>
                        with{' '}
                        <Link to={`/tigers/${other}`} className="hover:text-tiger-700">
                          {other}
                        </Link>
                      </span>
                    </li>
                  );
                })}
                {relatedCoDetections.map((cd) => (
                  <li key={cd.id} className="flex items-center gap-2">
                    <Badge variant="review">Co-detection</Badge>
                    <span>
                      with {cd.tigerIds.filter((t) => t !== id).join(', ')} at {cd.cameraId}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
