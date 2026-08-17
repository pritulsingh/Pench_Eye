import { MapContainer, Marker, Polygon, Popup, TileLayer, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { Link } from 'react-router-dom';
import { AlertTriangle } from 'lucide-react';

import { mapApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { EmptyState } from '@/components/ui/EmptyState';
import { formatDateTime, titleCase } from '@/lib/utils';
import {
  PLACES_ATTRIBUTION,
  PLACES_TILES,
  SATELLITE_ATTRIBUTION,
  SATELLITE_TILES,
  cameraIcon,
  geometryToLatLngs,
  sightingIcon,
} from './mapConfig';

/**
 * Compact, self-contained reserve map for embedding on the dashboard.
 * Renders the Pench territory border (reserve_boundary zone), the other
 * zone polygons, camera traps and recent sightings. Leaflet zoom controls
 * are enabled so the user can zoom in/out and pan within the reserve.
 */
export default function ReserveMap() {
  const { data, loading, error, reload } = useApi(() => mapApi.overview({ sighting_limit: 400 }));

  if (loading && !data) {
    return (
      <div className="h-full grid place-items-center">
        <LoadingSpinner label="Loading reserve map…" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full grid place-items-center p-4">
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
      </div>
    );
  }

  if (!data) return null;

  return (
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
      scrollWheelZoom
      zoomAnimation={false}
      style={{ height: '100%', width: '100%', background: '#f3f8f4' }}
    >
      <TileLayer url={SATELLITE_TILES} attribution={SATELLITE_ATTRIBUTION} maxZoom={19} />
      <TileLayer url={PLACES_TILES} attribution={PLACES_ATTRIBUTION} pane="overlayPane" />

      {data.zones.map((zone) => {
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
              </div>
            </Tooltip>
          </Polygon>
        );
      })}

      {data.cameras
        .filter((c) => c.latitude !== null && c.longitude !== null)
        .map((cam) => (
          <Marker
            key={cam.camera_id}
            position={[cam.latitude as number, cam.longitude as number]}
            icon={cameraIcon(cam.marker_state)}
          >
            <Popup>
              <div className="text-xs text-slate-900 space-y-0.5 min-w-[180px]">
                <div className="font-bold text-sm">{cam.camera_id}</div>
                <div>{cam.name}</div>
                <div>Zone: {titleCase(cam.zone ?? '')}</div>
                <div>Status: {titleCase(cam.marker_state)}</div>
                <div>Detections: {cam.observation_count}</div>
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

      {data.sightings
        .filter((s) => s.latitude !== null && s.longitude !== null)
        .map((s) => (
          <Marker
            key={s.observation_id}
            position={[s.latitude as number, s.longitude as number]}
            icon={sightingIcon(s.species === 'tiger')}
          >
            <Popup>
              <div className="text-xs text-slate-900 space-y-0.5 min-w-[180px]">
                <div className="font-bold text-sm">{s.tiger_code ?? titleCase(s.species ?? '')}</div>
                {s.tiger_name && <div>{s.tiger_name}</div>}
                <div>{formatDateTime(s.timestamp)}</div>
                <div>
                  {s.camera_id} — {s.camera_name}
                </div>
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
    </MapContainer>
  );
}
