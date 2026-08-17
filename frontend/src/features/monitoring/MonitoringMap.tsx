import { useMemo, useState } from 'react';
import {
  Circle,
  MapContainer,
  Marker,
  Polygon,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
  useMap,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Layers, Crosshair } from 'lucide-react';


import { formatDateTime, titleCase } from '@/lib/utils';
import { Badge } from '@/components/ui/Badge';
import {
  PLACES_ATTRIBUTION,
  PLACES_TILES,
  SATELLITE_ATTRIBUTION,
  SATELLITE_TILES,
} from '../map/mapConfig';
import { MONITORING_COLORS, RESERVE_BOUNDS, RESERVE_CENTER } from './config';
import { cameraMarker, tigerMarker } from './markers';
import { CameraUploadPanel } from './CameraUploadPanel';
import { findCameraById, nearbyTigers } from './analysis';
import { useMonitoringStore } from './MonitoringContext';
import type { CameraTrap, Detection, TrackedTiger } from './types';
import type { UploadResult } from './useMonitoring';


type LayerKey = 'cameras' | 'tigers' | 'territories' | 'coverage' | 'conflicts' | 'movement';

const LAYER_LABELS: Record<LayerKey, string> = {
  cameras: 'Cameras',
  tigers: 'Tigers',
  territories: 'Territories',
  coverage: 'Camera Coverage',
  conflicts: 'Conflict Zones',
  movement: 'Movement Paths',
};

/** Imperatively fly/pan the map when the user focuses a tiger. */
function MapFocus({ target }: { target: { center: [number, number]; zoom: number } | null }) {
  const map = useMap();
  useMemo(() => {
    if (target) map.flyTo(target.center, target.zoom, { duration: 0.6 });
  }, [target, map]);
  return null;
}

export default function MonitoringMap() {
  const store = useMonitoringStore();
  const {
    cameras,
    tigers,
    territories,
    detections,
    conflicts,
    overlaps,
    coDetections,
    alerts,
    uploadDetection,
    identificationKind,
  } = store;

  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    cameras: true,
    tigers: true,
    territories: true,
    coverage: false,
    conflicts: true,
    movement: false,
  });
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);
  const [selectedTiger, setSelectedTiger] = useState<string | null>(null);
  const [focus, setFocus] = useState<{ center: [number, number]; zoom: number } | null>(null);
  const [movementTiger, setMovementTiger] = useState<string | null>(null);
  // Tiger filter: '' = show all. When set, that tiger becomes visually dominant
  // and unrelated map objects are subdued (rather than removed).
  const [tigerFilter, setTigerFilter] = useState<string>('');
  // Conflict focus: emphasise conflict zones and subdue normal territory.
  const [conflictFocus, setConflictFocus] = useState(false);

  const toggle = (k: LayerKey) => setLayers((p) => ({ ...p, [k]: !p[k] }));

  /** Cameras that have detected the filtered tiger (for highlighting). */
  const filterTigerCameras = useMemo(() => {
    if (!tigerFilter) return new Set<string>();
    return new Set(
      cameras.filter((c) => c.detectedTigerIds.includes(tigerFilter)).map((c) => c.id)
    );
  }, [cameras, tigerFilter]);

  const applyTigerFilter = (value: string) => {

    setTigerFilter(value);
    if (value) {
      setSelectedTiger(value);
      setMovementTiger(value);
      setLayers((p) => ({ ...p, movement: true, tigers: true, territories: true }));
      const t = tigerById.get(value);
      if (t) setFocus({ center: t.currentLocation, zoom: 13 });
    } else {
      setMovementTiger(null);
    }
  };


  const tigerById = useMemo(() => new Map(tigers.map((t) => [t.id, t])), [tigers]);
  const territoryByTiger = useMemo(
    () => new Map(territories.map((t) => [t.tigerId, t])),
    [territories]
  );
  const overlapTigerIds = useMemo(() => {
    const s = new Set<string>();
    overlaps.forEach((o) => {
      s.add(o.tigerA);
      s.add(o.tigerB);
    });
    return s;
  }, [overlaps]);

  /** Tigers/cameras involved in a proximity conflict (for the Conflict filter). */
  const conflictTigerIds = useMemo(() => {
    const s = new Set<string>();
    conflicts.forEach((c) => {
      s.add(c.tigerA);
      s.add(c.tigerB);
    });
    return s;
  }, [conflicts]);
  const conflictCameraIds = useMemo(() => {
    const s = new Set<string>();
    conflicts.forEach((c) => c.nearbyCameraIds.forEach((id) => s.add(id)));
    return s;
  }, [conflicts]);

  // Dimming helpers — when a filter is active, unrelated objects are subdued
  // rather than removed so the map stays legible and the focus is obvious.
  const dimTerritory = (tigerId: string) => {
    if (tigerFilter && tigerId !== tigerFilter) return true;
    if (conflictFocus && !conflictTigerIds.has(tigerId)) return true;
    return false;
  };
  const dimTiger = (tigerId: string) => {
    if (tigerFilter && tigerId !== tigerFilter) return true;
    if (conflictFocus && !conflictTigerIds.has(tigerId)) return true;
    return false;
  };
  const dimCamera = (cameraId: string) => {
    if (tigerFilter && !filterTigerCameras.has(cameraId)) return true;
    if (conflictFocus && !conflictCameraIds.has(cameraId)) return true;
    return false;
  };


  const activeTiger = selectedTiger ? tigerById.get(selectedTiger) : undefined;
  const activeCamera = selectedCamera ? findCameraById(cameras, selectedCamera) : undefined;

  const viewMovement = (t: TrackedTiger) => {
    setMovementTiger(t.id);
    setLayers((p) => ({ ...p, movement: true }));
    setFocus({ center: t.currentLocation, zoom: 13 });
  };

  return (
    <div className="space-y-4">
      <div className="demo-banner">
        <Crosshair className="w-4 h-4 shrink-0" />
        <span>
          Simulated camera-trap monitoring. Identification is a <strong>{identificationKind}</strong>{' '}
          service — replace it with the real Tiger Re-ID model without changing this map.
        </span>
      </div>

      <div className="card p-4 flex flex-wrap items-center gap-3">
        <span className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <Layers className="w-4 h-4" /> Layers
        </span>
        {(Object.keys(LAYER_LABELS) as LayerKey[]).map((key) => (
          <button
            key={key}
            onClick={() => toggle(key)}
            aria-pressed={layers[key]}
            className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
              layers[key]
                ? 'bg-tiger-100 border-tiger-300 text-tiger-800'
                : 'bg-secondary/40 border-border text-muted-foreground hover:text-foreground'
            }`}
          >
            {LAYER_LABELS[key]}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="error">{conflicts.length} proximity</Badge>

          <Badge variant="warning">{overlaps.length} overlap</Badge>
          <Badge variant="default">{alerts.length} alerts</Badge>
        </div>
      </div>

      {/* Focus controls: isolate a single tiger, or emphasise conflict zones. */}
      <div className="card p-4 flex flex-wrap items-center gap-3">
        <label className="text-sm font-medium text-muted-foreground">Tiger filter</label>
        <select
          value={tigerFilter}
          onChange={(e) => applyTigerFilter(e.target.value)}
          aria-label="Focus a single tiger"
          className="filter-input"
        >
          <option value="">All Tigers</option>
          {tigers.map((t) => (
            <option key={t.id} value={t.id}>
              {t.id} · {t.name}
            </option>
          ))}
        </select>

        <button
          onClick={() => {
            setConflictFocus((v) => !v);
            setLayers((p) => ({ ...p, conflicts: true }));
          }}
          aria-pressed={conflictFocus}
          className={`px-3 py-1.5 rounded-md text-xs font-medium border transition-colors ${
            conflictFocus
              ? 'bg-red-100 border-red-300 text-red-800'
              : 'bg-secondary/40 border-border text-muted-foreground hover:text-foreground'
          }`}
        >
          Conflict focus
        </button>

        {(tigerFilter || conflictFocus) && (
          <button
            onClick={() => {
              setTigerFilter('');
              setConflictFocus(false);
              setMovementTiger(null);
            }}
            className="text-xs text-tiger-700 hover:underline"
          >
            Reset focus
          </button>
        )}
      </div>


      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        <div className="xl:col-span-3 h-[640px] rounded-xl overflow-hidden border border-border relative z-0">
          <MapContainer
            center={RESERVE_CENTER}
            bounds={RESERVE_BOUNDS}
            boundsOptions={{ padding: [18, 18] }}
            minZoom={9}
            maxBounds={[
              [21.35, 78.9],
              [22.15, 79.75],
            ]}
            maxBoundsViscosity={0.75}
            scrollWheelZoom
            zoomAnimation={false}
            style={{ height: '100%', width: '100%', background: '#f3f8f4' }}
          >
            <TileLayer url={SATELLITE_TILES} attribution={SATELLITE_ATTRIBUTION} maxZoom={19} />
            <TileLayer url={PLACES_TILES} attribution={PLACES_ATTRIBUTION} pane="overlayPane" />
            <MapFocus target={focus} />

            {/* Territories */}
            {layers.territories &&
              territories.map((terr) => {
                const overlapping = overlapTigerIds.has(terr.tigerId);
                const dimmed = dimTerritory(terr.tigerId);
                const emphasised =
                  (tigerFilter && terr.tigerId === tigerFilter) ||
                  (conflictFocus && conflictTigerIds.has(terr.tigerId));
                return (
                  <Polygon
                    key={terr.id}
                    positions={terr.ring}
                    pathOptions={{
                      color: overlapping ? MONITORING_COLORS.overlap : MONITORING_COLORS.territoryBorder,
                      weight: emphasised ? 3 : overlapping ? 2 : 1.5,
                      fillColor: overlapping ? MONITORING_COLORS.overlap : MONITORING_COLORS.territoryFill,
                      fillOpacity: dimmed ? 0.02 : emphasised ? 0.18 : 0.1,
                      opacity: dimmed ? 0.25 : 1,
                      dashArray: overlapping ? '4 4' : undefined,
                    }}
                    eventHandlers={{ click: () => setSelectedTiger(terr.tigerId) }}
                  >

                    <Tooltip sticky>
                      <div className="text-xs">
                        <strong>{terr.tigerId}</strong> territory
                        {overlapping && <> • overlapping</>}
                        <br />~{terr.areaLabelKm2} km²
                      </div>
                    </Tooltip>
                  </Polygon>
                );
              })}

            {/* Camera coverage (subtle detection radius) */}
            {layers.coverage &&
              cameras.map((c) => (
                <Circle
                  key={`cov-${c.id}`}
                  center={[c.latitude, c.longitude]}
                  radius={c.detectionRadiusKm * 1000}
                  pathOptions={{
                    color: MONITORING_COLORS.cameraCoverage,
                    weight: 1,
                    fillColor: MONITORING_COLORS.cameraCoverage,
                    fillOpacity: 0.05,
                  }}
                />
              ))}

            {/* Selected camera coverage even when the layer is off */}
            {!layers.coverage && activeCamera && (
              <Circle
                center={[activeCamera.latitude, activeCamera.longitude]}
                radius={activeCamera.detectionRadiusKm * 1000}
                pathOptions={{
                  color: MONITORING_COLORS.cameraCoverage,
                  weight: 1,
                  fillColor: MONITORING_COLORS.cameraCoverage,
                  fillOpacity: 0.07,
                }}
              />
            )}

            {/* Movement path (estimated) for the focused tiger */}
            {layers.movement &&
              tigers
                .filter((t) => !movementTiger || t.id === movementTiger)
                .map((t) => {
                  const pts = t.movementHistory.map(
                    (m) => [m.latitude, m.longitude] as [number, number]
                  );
                  if (pts.length < 2) return null;
                  return (
                    <Polyline
                      key={`mv-${t.id}`}
                      positions={pts}
                      pathOptions={{
                        color: MONITORING_COLORS.movementPath,
                        weight: 2.5,
                        opacity: 0.9,
                        dashArray: '2 8',
                        lineCap: 'round',
                      }}
                    >
                      <Tooltip sticky>
                        <div className="text-xs">
                          <strong>{t.id}</strong> — Estimated Movement Path
                          <br />
                          {pts.length} detections (discrete observations)
                        </div>
                      </Tooltip>
                    </Polyline>
                  );
                })}

            {/* Conflict zones (dynamic) */}
            {layers.conflicts &&
              conflicts.map((c) => (
                <div key={c.id} style={{ display: 'contents' }}>
                  <Circle
                    center={c.midpoint}
                    radius={(c.distanceKm / 2) * 1000 + 400}
                    pathOptions={{
                      color: MONITORING_COLORS.conflictZone,
                      weight: 1.5,
                      fillColor: MONITORING_COLORS.conflictZone,
                      fillOpacity: 0.12,
                      dashArray: '3 5',
                    }}
                  />
                  <Polyline
                    positions={[c.positionA, c.positionB]}
                    pathOptions={{ color: MONITORING_COLORS.conflict, weight: 2, opacity: 0.85 }}
                  >
                    <Tooltip permanent direction="center" className="pench-conflict-label">
                      <span>⚠ {c.distanceKm} km</span>
                    </Tooltip>
                  </Polyline>
                </div>
              ))}

            {/* Tigers */}
            {layers.tigers &&
              tigers.map((t) => (
                <Marker
                  key={t.id}
                  position={t.currentLocation}
                  icon={tigerMarker(t, t.id === selectedTiger || t.id === tigerFilter)}
                  opacity={dimTiger(t.id) ? 0.3 : 1}
                  eventHandlers={{ click: () => setSelectedTiger(t.id) }}
                >
                  <Tooltip direction="top">
                    <span className="text-xs font-semibold">
                      {t.id} · {t.name}
                    </span>
                  </Tooltip>
                </Marker>
              ))}

            {/* Cameras */}
            {layers.cameras &&
              cameras.map((c) => (
                <Marker
                  key={c.id}
                  position={[c.latitude, c.longitude]}
                  icon={cameraMarker(c, c.id === selectedCamera)}
                  opacity={dimCamera(c.id) ? 0.35 : 1}
                  eventHandlers={{ click: () => setSelectedCamera(c.id) }}
                >

                  <Popup>
                    <CameraPopup
                      camera={c}
                      detections={detections.filter((d) => d.cameraId === c.id)}
                      onUpload={uploadDetection}
                    />

                  </Popup>
                </Marker>
              ))}
          </MapContainer>
        </div>

        {/* Side panels */}
        <div className="space-y-4">
          {activeTiger ? (
            <TigerDetailPanel
              tiger={activeTiger}
              tigers={tigers}
              territoryName={territoryByTiger.get(activeTiger.id)?.id ?? '—'}
              detectionCount={activeTiger.detectionIds.length}
              inConflict={conflicts.some(
                (c) => c.tigerA === activeTiger.id || c.tigerB === activeTiger.id
              )}
              onClose={() => setSelectedTiger(null)}
              onViewMovement={() => viewMovement(activeTiger)}
            />
          ) : (
            <div className="card p-4">
              <h3 className="font-semibold text-sm mb-1">Tiger detail</h3>
              <p className="text-xs text-muted-foreground">
                Click a tiger marker or its territory to see status, conflicts and movement.
              </p>
            </div>
          )}

          <div className="card p-4">
            <h3 className="font-semibold text-sm mb-3">Legend</h3>
            <ul className="space-y-2 text-xs text-muted-foreground">
              <li className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-[3px]" style={{ background: MONITORING_COLORS.camera }} />
                Camera trap ({cameras.length})
              </li>
              <li className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full" style={{ background: MONITORING_COLORS.tigerMale }} />
                Tiger (male)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full" style={{ background: MONITORING_COLORS.tigerFemale }} />
                Tiger (female)
              </li>
              <li className="flex items-center gap-2">
                <span className="w-3 h-3 border" style={{ borderColor: MONITORING_COLORS.overlap }} />
                Overlapping territory
              </li>
              <li className="flex items-center gap-2">
                <span className="w-3 h-3 rounded-full border" style={{ borderColor: MONITORING_COLORS.conflict }} />
                Proximity / conflict zone
              </li>
            </ul>
          </div>

          <div className="card p-4">
            <h3 className="font-semibold text-sm mb-3">Active events</h3>
            <dl className="space-y-2 text-xs">
              <Row label="Tigers tracked" value={tigers.length} />
              <Row label="Cameras" value={cameras.length} />
              <Row label="Proximity conflicts" value={conflicts.length} />
              <Row label="Territory overlaps" value={overlaps.length} />
              <Row label="Camera co-detections" value={coDetections.length} />
              <Row label="Alerts" value={alerts.length} />
            </dl>
          </div>

          {conflicts.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold text-sm mb-3">Proximity conflicts</h3>
              <ul className="space-y-2 text-xs">
                {conflicts.map((c) => (
                  <li key={c.id} className="flex items-center justify-between gap-2">
                    <button
                      className="text-tiger-700 hover:underline font-medium"
                      onClick={() => setFocus({ center: c.midpoint, zoom: 13 })}
                    >
                      {c.tigerA} ↔ {c.tigerB}
                    </button>
                    <Badge variant={c.distanceKm <= 2 ? 'error' : 'warning'}>{c.distanceKm} km</Badge>

                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="flex justify-between">
      <dt className="text-muted-foreground">{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function CameraPopup({
  camera,
  detections,
  onUpload,
}: {
  camera: CameraTrap;
  detections: Detection[];
  onUpload: (cameraId: string, image: File) => Promise<UploadResult>;
}) {

  const recent = [...detections]
    .sort((a, b) => Date.parse(b.timestamp) - Date.parse(a.timestamp))
    .slice(0, 4);
  return (
    <div className="text-xs text-slate-900 space-y-0.5 min-w-[210px]">
      <div className="font-bold text-sm">{camera.id}</div>
      <div>{camera.name}</div>
      <div>Status: {titleCase(camera.status)}</div>
      <div>
        {camera.latitude.toFixed(4)}, {camera.longitude.toFixed(4)}
      </div>
      <div>Detection radius: {camera.detectionRadiusKm} km</div>
      <div>Last detection: {formatDateTime(camera.lastDetection)}</div>
      <div>
        Detected tigers:{' '}
        {camera.detectedTigerIds.length ? camera.detectedTigerIds.join(', ') : '—'}
      </div>
      {recent.length > 0 && (
        <div className="pt-1">
          <div className="font-medium">Recent detections</div>
          {recent.map((d) => (
            <div key={d.id}>
              {d.tigerId ?? 'unidentified'} · {formatDateTime(d.timestamp)} ·{' '}
              {d.estimatedDistanceFromCameraKm} km
            </div>
          ))}
        </div>
      )}
      <CameraUploadPanel cameraId={camera.id} onUpload={onUpload} />
      <p className="text-[10px] text-slate-500 pt-1">
        Manual upload is only available here, from the camera.
      </p>
    </div>

  );
}

function TigerDetailPanel({
  tiger,
  tigers,
  territoryName,
  detectionCount,
  inConflict,
  onClose,
  onViewMovement,
}: {
  tiger: TrackedTiger;
  tigers: TrackedTiger[];
  territoryName: string;
  detectionCount: number;
  inConflict: boolean;
  onClose: () => void;
  onViewMovement: () => void;
}) {
  const near = nearbyTigers(tiger, tigers);
  return (
    <div className="card p-4">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-semibold text-sm">
            {tiger.id} · {tiger.name}
          </h3>
          <p className="text-xs text-muted-foreground">
            {titleCase(tiger.sex)} · {titleCase(tiger.ageClass)} · {titleCase(tiger.status)}
          </p>
        </div>
        <button onClick={onClose} className="text-xs text-muted-foreground hover:text-foreground">
          ✕
        </button>
      </div>

      <dl className="space-y-1.5 text-xs mt-3">
        <Row
          label="Estimated location"
          value={`${tiger.currentLocation[0].toFixed(4)}, ${tiger.currentLocation[1].toFixed(4)}`}
        />
        <Row label="Territory" value={territoryName} />
        <Row label="Last camera" value={tiger.lastDetectedCamera ?? '—'} />
        <Row label="Last detection" value={formatDateTime(tiger.lastDetectionTime)} />
        <Row
          label="Confidence"
          value={tiger.confidence != null ? `${Math.round(tiger.confidence * 100)}%` : '—'}
        />
        <Row label="Detections" value={detectionCount} />
        <div className="flex justify-between">
          <dt className="text-muted-foreground">Conflict status</dt>
          <dd>
            {inConflict ? (
              <Badge variant="error">In proximity conflict</Badge>

            ) : (
              <Badge variant="success">Clear</Badge>
            )}
          </dd>
        </div>
      </dl>

      <div className="mt-3">
        <div className="text-xs font-medium mb-1">Nearby tigers (&lt; 5 km)</div>
        {near.length ? (
          <ul className="text-xs text-muted-foreground space-y-0.5">
            {near.map((n) => (
              <li key={n.id} className="flex justify-between">
                <span>
                  {n.id} · {n.name}
                </span>
                <span>{n.distanceKm} km</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs text-muted-foreground">None within 5 km.</p>
        )}
      </div>

      <button onClick={onViewMovement} className="btn-secondary w-full mt-3 !py-1.5 text-xs">
        View Movement
      </button>
    </div>
  );
}
