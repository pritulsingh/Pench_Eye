import { Link, useParams } from 'react-router-dom';
import { MapContainer, Marker, Popup, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { AlertCircle, ArrowLeft, Battery, Camera, Clock, ImageIcon, MapPin } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { assetUrl, camerasApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import {
  MARKER_COLORS,
  MARKER_LABELS,
  SATELLITE_ATTRIBUTION,
  SATELLITE_TILES,
  cameraIcon,
} from '@/features/map/mapConfig';
import { formatDateTime, formatPercent, relativeTime, titleCase } from '@/lib/utils';

export default function CameraDetail() {
  const { cameraId } = useParams<{ cameraId: string }>();
  const { data: camera, loading, error, reload } = useApi(
    () => camerasApi.get(cameraId as string),
    [cameraId]
  );

  if (loading) {
    return (
      <div className="py-24">
        <LoadingSpinner label="Loading camera…" />
      </div>
    );
  }

  if (error || !camera) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Camera not found"
        description={error ?? `No camera station matches ${cameraId}.`}
        action={
          <div className="flex gap-3">
            <button className="btn-secondary" onClick={reload}>
              Retry
            </button>
            <Link to="/cameras" className="btn-primary">
              Back to cameras
            </Link>
          </div>
        }
      />
    );
  }

  const hasCoords = camera.latitude !== null && camera.longitude !== null;

  return (
    <div className="space-y-6">
      <Link
        to="/cameras"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="w-4 h-4" /> All cameras
      </Link>

      <div className="card p-6 flex flex-col lg:flex-row gap-6 lg:items-center">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-1">
            <h2 className="text-2xl font-bold">{camera.camera_id}</h2>
            <span className="inline-flex items-center gap-2 text-xs">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: MARKER_COLORS[camera.marker_state] }}
              />
              {MARKER_LABELS[camera.marker_state] ?? titleCase(camera.marker_state)}
            </span>
          </div>
          <p className="text-muted-foreground">{camera.name}</p>
          {camera.description && (
            <p className="text-xs text-muted-foreground mt-2 max-w-xl">{camera.description}</p>
          )}
          <div className="flex flex-wrap gap-x-6 gap-y-2 mt-4 text-sm">
            <span className="flex items-center gap-2 text-muted-foreground">
              <MapPin className="w-4 h-4" />
              {hasCoords
                ? `${(camera.latitude as number).toFixed(4)}, ${(camera.longitude as number).toFixed(4)}`
                : 'No coordinates'}
            </span>
            <span className="flex items-center gap-2 text-muted-foreground">
              <Camera className="w-4 h-4" /> Zone: {titleCase(camera.zone)}
              {camera.zone_code ? ` (${camera.zone_code})` : ''}
            </span>
            <span className="flex items-center gap-2 text-muted-foreground">
              <Battery className="w-4 h-4" />
              {camera.battery_percent === null ? 'Battery unknown' : `${camera.battery_percent}%`}
            </span>
            <span className="flex items-center gap-2 text-muted-foreground">
              <Clock className="w-4 h-4" /> Last seen {relativeTime(camera.last_active_at)}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 min-w-[260px]">
          {[
            { label: 'Detections', value: camera.observation_count },
            { label: 'Images', value: camera.image_count },
            { label: 'Unique tigers', value: camera.unique_tigers },
            { label: 'Open alerts', value: camera.open_alert_count },
          ].map((item) => (
            <div key={item.label} className="bg-secondary/50 border border-border rounded-lg p-3">
              <div className="text-2xl font-bold text-tiger-500">{item.value}</div>
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
                {item.label}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-6">
          <h3 className="font-semibold mb-4">Detection timeline</h3>
          {camera.detection_timeline.length === 0 ? (
            <p className="text-sm text-muted-foreground">No detections recorded at this station.</p>
          ) : (
            <div className="h-[240px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={camera.detection_timeline}>
                  <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                  <XAxis
                    dataKey="date"
                    stroke="hsl(140 9% 38%)"
                    fontSize={10}
                    tickFormatter={(d: string) => d.slice(5)}
                    tickLine={false}
                  />
                  <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(0 0% 100%)',
                      border: '1px solid hsl(136 18% 84%)',
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="detections" fill="hsl(145 55% 34%)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="card overflow-hidden">
          <div className="p-4 border-b border-border">
            <h3 className="font-semibold">Location</h3>
          </div>
          {hasCoords ? (
            <div className="h-[280px] relative z-0">
              <MapContainer
                center={[camera.latitude as number, camera.longitude as number]}
                zoom={12}
                style={{ height: '100%', width: '100%', background: '#f3f8f4' }}
              >
                <TileLayer url={SATELLITE_TILES} attribution={SATELLITE_ATTRIBUTION} />
                <Marker
                  position={[camera.latitude as number, camera.longitude as number]}
                  icon={cameraIcon(camera.marker_state)}
                >
                  <Popup>
                    <div className="text-xs text-slate-900">
                      <strong>{camera.camera_id}</strong>
                      <br />
                      {camera.name}
                    </div>
                  </Popup>
                </Marker>
              </MapContainer>
            </div>
          ) : (
            <p className="p-6 text-sm text-muted-foreground">
              This station has no coordinates recorded.
            </p>
          )}
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <h3 className="font-semibold">Recent detections</h3>
          <Link
            to={`/observations?camera_id=${camera.camera_id}`}
            className="text-xs text-tiger-700 hover:underline"
          >
            View all detections
          </Link>
        </div>
        {camera.recent_detections.length === 0 ? (
          <p className="p-6 text-sm text-muted-foreground">Nothing detected here yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Observation</th>
                  <th>When</th>
                  <th>Species</th>
                  <th>Tiger</th>
                  <th>Detection</th>
                  <th>Identity</th>
                </tr>
              </thead>
              <tbody>
                {camera.recent_detections.map((det) => (
                  <tr key={det.observation_id}>
                    <td className="font-mono text-xs">{det.observation_id}</td>
                    <td className="text-muted-foreground">{formatDateTime(det.timestamp)}</td>
                    <td>{titleCase(det.species)}</td>
                    <td>
                      {det.tiger_code ? (
                        <Link to={`/tigers/${det.tiger_code}`} className="hover:text-tiger-700">
                          <Badge variant="tiger">{det.tiger_code}</Badge>
                        </Link>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td>{formatPercent(det.detection_confidence)}</td>
                    <td>{formatPercent(det.identity_confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card p-6">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <ImageIcon className="w-4 h-4" /> Recent images
        </h3>
        {camera.recent_images.length === 0 ? (
          <p className="text-sm text-muted-foreground">No stored frames for this camera.</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {camera.recent_images.map((img) => (
              <figure key={img.image_id} className="rounded-lg overflow-hidden border border-border">
                <img
                  src={assetUrl(img.url)}
                  alt={`Capture ${img.image_id}`}
                  loading="lazy"
                  className="w-full h-24 object-cover bg-secondary/40"
                />
                <figcaption className="p-2 text-[11px] text-muted-foreground">
                  <div>{relativeTime(img.timestamp)}</div>
                  <div>{titleCase(img.status)}</div>
                </figcaption>
              </figure>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
