import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  MapContainer,
  Marker,
  Polygon,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { AlertTriangle, Layers, RefreshCw } from 'lucide-react';

import { assetUrl, mapApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Badge } from '@/components/ui/Badge';
import { formatDateTime, formatPercent, titleCase } from '@/lib/utils';
import {
  PLACES_ATTRIBUTION,
  PLACES_TILES,
  SATELLITE_ATTRIBUTION,
  SATELLITE_TILES,
  MARKER_COLORS,
  MARKER_LABELS,
  cameraIcon,
  convexHull,
  curvedPath,
  gateIcon,
  geometryToLatLngs,
  sightingIcon,
} from './mapConfig';

type LayerKey = 'zones' | 'gates' | 'cameras' | 'sightings' | 'movement';

const LAYER_LABELS: Record<LayerKey, string> = {
  zones: 'Zones & boundary',
  gates: 'Gates',
  cameras: 'Camera traps',
  sightings: 'Sightings',
  movement: 'Movement paths',
};

const TRACK_COLORS = ['#f59e0b', '#38bdf8', '#a855f7', '#22c55e', '#ef4444', '#eab308'];

export default function PenchMap() {
  const [layers, setLayers] = useState<Record<LayerKey, boolean>>({
    zones: true,
    gates: true,
    cameras: true,
    sightings: true,
    movement: false,
  });
  const [tigerFilter, setTigerFilter] = useState('');
  const [dayFilter, setDayFilter] = useState('');

  const { data, loading, error, reload } = useApi(
    () =>
      mapApi.overview({
        sighting_limit: 400,
        ...(tigerFilter ? { tiger_code: tigerFilter } : {}),
        ...(dayFilter ? { days: Number(dayFilter) } : {}),
      }),
    [tigerFilter, dayFilter]
  );

  const tigerCodes = useMemo(() => {
    const codes = new Set<string>();
    data?.sightings.forEach((s) => s.tiger_code && codes.add(s.tiger_code));
    data?.tracks.forEach((t) => codes.add(t.tiger_code));
    return Array.from(codes).sort();
  }, [data]);

  const toggle = (key: LayerKey) => setLayers((prev) => ({ ...prev, [key]: !prev[key] }));

  if (loading && !data) {
    return (
      <div className="py-24">
        <LoadingSpinner label="Loading reserve map…" />
      </div>
    );
  }

  if (error) {
    return (
      <EmptyState
        icon={<AlertTriangle />}
        title="Map data unavailable"
        description={error}
        action={
          <button className="btn-primary" onClick={reload}>
            Retry
          </button>
        }
      />
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-4">
      <div className="demo-banner">
        <AlertTriangle className="w-4 h-4 shrink-0" />
        <span>{data.disclaimer}</span>
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

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <label className="text-xs text-muted-foreground" htmlFor="map-tiger">
            Tiger
          </label>
          <select
            id="map-tiger"
            value={tigerFilter}
            onChange={(e) => setTigerFilter(e.target.value)}
            className="bg-secondary/50 border border-border rounded-md px-2 py-1.5 text-xs focus:outline-none focus:border-tiger-500"
          >
            <option value="">All tigers</option>
            {tigerCodes.map((code) => (
              <option key={code} value={code}>
                {code}
              </option>
            ))}
          </select>

          <label className="text-xs text-muted-foreground" htmlFor="map-days">
            Period
          </label>
          <select
            id="map-days"
            value={dayFilter}
            onChange={(e) => setDayFilter(e.target.value)}
            className="bg-secondary/50 border border-border rounded-md px-2 py-1.5 text-xs focus:outline-none focus:border-tiger-500"
          >
            <option value="">All time</option>
            <option value="7">Last 7 days</option>
            <option value="30">Last 30 days</option>
            <option value="90">Last 90 days</option>
          </select>

          <button
            onClick={reload}
            className="btn-secondary flex items-center gap-2 !py-1.5 !px-3 text-xs"
            title="Reload map data"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
        <div className="xl:col-span-3 h-[640px] rounded-xl overflow-hidden border border-border relative z-0">
          <MapContainer
            center={data.center}
            bounds={data.bounds}
            boundsOptions={{ padding: [18, 18] }}
            minZoom={9}
            maxBounds={[
              [21.35, 78.9],
              [22.15, 79.75],
            ]}
            maxBoundsViscosity={0.75}
            zoomAnimation={false}
            style={{ height: '100%', width: '100%', background: '#f3f8f4' }}
          >
            <TileLayer url={SATELLITE_TILES} attribution={SATELLITE_ATTRIBUTION} maxZoom={19} />
            <TileLayer url={PLACES_TILES} attribution={PLACES_ATTRIBUTION} pane="overlayPane" />

            {layers.zones &&
              data.zones.map((zone) => {
                const rings = geometryToLatLngs(zone.geometry_json);
                if (!rings.length) return null;
                const isBoundary = zone.zone_type === 'reserve_boundary';
                return (
                  <Polygon
                    key={zone.zone_code}
                    positions={rings}
                    pathOptions={{
                      color: zone.style_color ?? '#f59e0b',
                      weight: isBoundary ? 3 : 1.5,
                      dashArray: isBoundary ? '6 6' : undefined,
                      fillOpacity: isBoundary ? 0 : 0.12,
                    }}
                  >
                    <Tooltip sticky>
                      <div className="text-xs">
                        <strong>{zone.name}</strong>
                        <br />
                        {titleCase(zone.zone_type)}
                        {zone.area_km2 ? ` • ${zone.area_km2} km²` : ''}
                        <br />
                        {zone.camera_count} cameras • {zone.observation_count} detections
                      </div>
                    </Tooltip>
                  </Polygon>
                );
              })}

            {layers.gates &&
              data.gates.map((gate) => (
                <Marker
                  key={gate.code}
                  position={[gate.latitude, gate.longitude]}
                  icon={gateIcon()}
                >
                  <Popup>
                    <div className="text-xs text-slate-900">
                      <strong>{gate.name}</strong>
                      <br />
                      {titleCase(gate.gate_type)} gate
                    </div>
                  </Popup>
                </Marker>
              ))}

            {layers.cameras &&
              data.cameras
                .filter((c) => c.latitude !== null && c.longitude !== null)
                .map((cam) => (
                  <Marker
                    key={cam.camera_id}
                    position={[cam.latitude as number, cam.longitude as number]}
                    icon={cameraIcon(cam.marker_state)}
                  >
                    <Popup>
                      <div className="text-xs text-slate-900 space-y-0.5 min-w-[190px]">
                        <div className="font-bold text-sm">{cam.camera_id}</div>
                        <div>{cam.name}</div>
                        <div>Zone: {titleCase(cam.zone)}</div>
                        <div>Status: {titleCase(cam.marker_state)}</div>
                        <div>Last active: {formatDateTime(cam.last_active_at)}</div>
                        <div>Last detection: {formatDateTime(cam.last_detection_at)}</div>
                        <div>Detections: {cam.observation_count}</div>
                        {cam.open_alert_count > 0 && (
                          <div className="text-red-600 font-medium">
                            {cam.open_alert_count} open alert(s)
                          </div>
                        )}
                        <Link
                          to={`/cameras/${cam.camera_id}`}
                          className="text-amber-700 underline font-medium inline-block pt-1"
                        >
                          Open camera detail →
                        </Link>
                      </div>
                    </Popup>
                  </Marker>
                ))}

            {layers.sightings &&
              data.sightings
                .filter((s) => s.latitude !== null && s.longitude !== null)
                .map((s) => (
                  <Marker
                    key={s.observation_id}
                    position={[s.latitude as number, s.longitude as number]}
                    icon={sightingIcon(s.species === 'tiger')}
                  >
                    <Popup>
                      <div className="text-xs text-slate-900 space-y-0.5 min-w-[190px]">
                        <div className="font-bold text-sm">
                          {s.tiger_code ?? titleCase(s.species)}
                        </div>
                        {s.tiger_name && <div>{s.tiger_name}</div>}
                        <div>{formatDateTime(s.timestamp)}</div>
                        <div>
                          {s.camera_id} — {s.camera_name}
                        </div>
                        <div>
                          {(s.latitude as number).toFixed(4)}, {(s.longitude as number).toFixed(4)}
                        </div>
                        <div>Identity: {formatPercent(s.identity_confidence)}</div>
                        <div>Detection: {formatPercent(s.detection_confidence)}</div>
                        {s.image_url && (
                          <img
                            src={assetUrl(s.image_url)}
                            alt={`Capture ${s.observation_id}`}
                            className="w-full rounded mt-1"
                          />
                        )}
                        {s.tiger_code && (
                          <Link
                            to={`/tigers/${s.tiger_code}`}
                            className="text-amber-700 underline font-medium inline-block pt-1"
                          >
                            Open tiger profile →
                          </Link>
                        )}
                      </div>
                    </Popup>
                  </Marker>
                ))}

            {layers.movement &&
              data.tracks.map((track, ti) => {
                const color = TRACK_COLORS[ti % TRACK_COLORS.length];
                // Home-range region: convex hull of every observed point once a
                // tiger has enough sightings to enclose an area.
                const points = (track.observations ?? [])
                  .filter((o) => o.latitude !== null && o.longitude !== null)
                  .map((o) => [o.latitude as number, o.longitude as number] as [number, number]);
                const hull = convexHull(points);
                return (
                  <div key={track.tiger_code} style={{ display: 'contents' }}>
                    {hull.length >= 3 && (
                      <Polygon
                        positions={hull}
                        pathOptions={{
                          color,
                          weight: 1.5,
                          fillOpacity: 0.08,
                          dashArray: '2 8',
                        }}
                      >
                        <Tooltip sticky>
                          <div className="text-xs">
                            <strong>{track.tiger_code}</strong> home range
                            <br />
                            {track.sighting_count} sightings
                          </div>
                        </Tooltip>
                      </Polygon>
                    )}
                    {track.legs.map((leg, li) => {
                      if (
                        leg.from_latitude === null ||
                        leg.from_longitude === null ||
                        leg.to_latitude === null ||
                        leg.to_longitude === null
                      )
                        return null;
                      // Alternate the bow direction so consecutive legs don't
                      // overlap, giving a natural meandering path.
                      const curve = curvedPath(
                        [leg.from_latitude, leg.from_longitude],
                        [leg.to_latitude, leg.to_longitude],
                        0.18,
                        li % 2 === 0 ? 1 : -1
                      );
                      return (
                        <Polyline
                          key={`${track.tiger_code}-${li}`}
                          positions={curve}
                          smoothFactor={0}
                          pathOptions={{
                            color,
                            weight: 2.5,
                            opacity: 0.9,
                            dashArray: '1 8',
                            lineCap: 'round',
                          }}
                        >
                          <Tooltip sticky>
                            <div className="text-xs">
                              <strong>{track.tiger_code}</strong>
                              <br />
                              {leg.from_camera_id} → {leg.to_camera_id}
                              <br />
                              {leg.distance_km} km in {leg.hours_elapsed} h
                              <br />
                              {formatDateTime(leg.to_timestamp)}
                            </div>
                          </Tooltip>
                        </Polyline>
                      );
                    })}
                  </div>
                );
              })}
          </MapContainer>
        </div>

        <div className="space-y-4">
          <div className="card p-4">
            <h3 className="font-semibold text-sm mb-3">Camera marker legend</h3>
            <ul className="space-y-2 text-xs text-muted-foreground">
              {Object.entries(MARKER_LABELS).map(([state, label]) => (
                <li key={state} className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full border border-white/20"
                    style={{ backgroundColor: MARKER_COLORS[state] }}
                  />
                  {label}
                </li>
              ))}
              <li className="flex items-center gap-2 pt-1 border-t border-border/60 mt-2">
                <span className="w-3 h-3 rotate-45 bg-tiger-500" /> Tiger sighting
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-slate-400" /> Other wildlife
              </li>
            </ul>
          </div>

          <div className="card p-4">
            <h3 className="font-semibold text-sm mb-3">On this map</h3>
            <dl className="space-y-2 text-xs">
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Zones</dt>
                <dd>{data.zones.length}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Cameras</dt>
                <dd>{data.cameras.length}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Sightings shown</dt>
                <dd>{data.sightings.length}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Movement tracks</dt>
                <dd>{data.tracks.length}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted-foreground">Geo source</dt>
                <dd>
                  <Badge variant="success">
                    {data.data_source}
                  </Badge>
                </dd>
              </div>
            </dl>
          </div>

          {data.tracks.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold text-sm mb-3">Recent movement</h3>
              <ul className="space-y-3 text-xs max-h-72 overflow-y-auto pr-1">
                {data.tracks.map((track, ti) => (
                  <li key={track.tiger_code}>
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: TRACK_COLORS[ti % TRACK_COLORS.length] }}
                      />
                      <Link
                        to={`/tigers/${track.tiger_code}`}
                        className="font-semibold hover:text-tiger-700"
                      >
                        {track.tiger_code}
                      </Link>
                      <span className="text-muted-foreground ml-auto">
                        {track.total_distance_km} km
                      </span>
                    </div>
                    <p className="text-muted-foreground leading-relaxed">
                      {track.legs
                        .slice(-3)
                        .map((l) => l.from_camera_id)
                        .concat(track.legs[track.legs.length - 1]?.to_camera_id ?? '')
                        .filter(Boolean)
                        .join(' → ')}
                    </p>
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
