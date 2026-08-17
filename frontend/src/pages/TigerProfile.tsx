import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { MapContainer, Marker, Polyline, Popup, TileLayer, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { AlertCircle, ArrowLeft, Clock, ImageIcon } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from 'recharts';

import { assetUrl, mapApi, tigersApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import {
  SATELLITE_ATTRIBUTION,
  SATELLITE_TILES,
  cameraIcon,
  sightingIcon,
} from '@/features/map/mapConfig';
import { formatDate, formatDateTime, formatPercent, titleCase } from '@/lib/utils';

type Tab = 'gallery' | 'timeline' | 'movement';

export default function TigerProfile() {
  const { id } = useParams<{ id: string }>();
  const [tab, setTab] = useState<Tab>('gallery');

  const { data: tiger, loading, error, reload } = useApi(
    () => tigersApi.get(id as string),
    [id]
  );
  const { data: gallery } = useApi(() => tigersApi.getGallery(id as string), [id]);
  const { data: tracks } = useApi(() => mapApi.movement({ tiger_code: id }), [id]);

  if (loading) {
    return (
      <div className="py-24">
        <LoadingSpinner label="Loading profile…" />
      </div>
    );
  }

  if (error || !tiger) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Tiger not found"
        description={error ?? `No individual matches ${id}.`}
        action={
          <div className="flex gap-3">
            <button className="btn-secondary" onClick={reload}>
              Retry
            </button>
            <Link to="/tigers" className="btn-primary">
              Back to catalog
            </Link>
          </div>
        }
      />
    );
  }

  const track = tracks?.[0];
  const mapPoints = tiger.recent_observations.filter(
    (o) => o.latitude !== null && o.longitude !== null
  );
  const center: [number, number] | null = mapPoints.length
    ? [mapPoints[0].latitude as number, mapPoints[0].longitude as number]
    : null;

  return (
    <div className="space-y-6">
      <Link
        to="/tigers"
        className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="w-4 h-4" /> Tiger catalog
      </Link>

      <div className="card p-8 bg-gradient-to-br from-tiger-100 to-white border-tiger-500/20">
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-4xl font-bold">{tiger.tiger_id}</h1>
              {tiger.is_demo && <Badge variant="demo">Demo profile</Badge>}
            </div>
            {tiger.name && <h2 className="text-xl text-tiger-700 mt-1">{tiger.name}</h2>}
            <div className="flex flex-wrap gap-4 text-sm text-muted-foreground mt-4">
              <StatusBadge status={tiger.status ?? 'unknown'} />
              <span className="border-l border-border pl-4">
                Sex: <strong className="text-foreground">{titleCase(tiger.sex)}</strong>
              </span>
              <span className="border-l border-border pl-4">
                First detected: <strong className="text-foreground">{formatDate(tiger.first_seen)}</strong>
              </span>
              <span className="border-l border-border pl-4">
                Last detected: <strong className="text-foreground">{formatDate(tiger.last_seen)}</strong>
              </span>
            </div>
            {tiger.notes && (
              <p className="text-xs text-muted-foreground mt-4 max-w-2xl">{tiger.notes}</p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 min-w-[260px]">
            {[
              { label: 'Sightings', value: tiger.total_observations },
              { label: 'Cameras', value: tiger.camera_count },
              { label: 'Mean similarity', value: formatPercent(tiger.mean_confidence) },
              { label: 'Distance tracked', value: `${track?.total_distance_km ?? 0} km` },
            ].map((item) => (
              <div key={item.label} className="bg-secondary/50 p-3 rounded-lg border border-border">
                <div className="text-xl font-bold text-tiger-500">{item.value}</div>
                <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
                  {item.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-6">
          <h3 className="font-semibold mb-4">Frequently detected cameras</h3>
          {tiger.frequent_cameras.length === 0 ? (
            <p className="text-sm text-muted-foreground">No camera detections recorded.</p>
          ) : (
            <ul className="space-y-3">
              {tiger.frequent_cameras.map((cam) => {
                const max = tiger.frequent_cameras[0].detections || 1;
                return (
                  <li key={cam.camera_id}>
                    <div className="flex justify-between text-sm mb-1">
                      <Link to={`/cameras/${cam.camera_id}`} className="hover:text-tiger-700">
                        {cam.camera_id} — {cam.camera_name}
                      </Link>
                      <span className="font-medium">{cam.detections}</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-tiger-500"
                        style={{ width: `${(cam.detections / max) * 100}%` }}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          )}

          <h3 className="font-semibold mt-8 mb-3">Zone distribution</h3>
          <div className="flex flex-wrap gap-2">
            {tiger.zone_distribution.length === 0 ? (
              <p className="text-sm text-muted-foreground">No zone data.</p>
            ) : (
              tiger.zone_distribution.map((z) => (
                <Badge key={z.label} variant="tiger">
                  {titleCase(z.label)}: {z.count}
                </Badge>
              ))
            )}
          </div>
        </div>

        <div className="card p-6">
          <h3 className="font-semibold mb-4">Detections per month</h3>
          {tiger.detections_by_month.length === 0 ? (
            <p className="text-sm text-muted-foreground">No detections recorded.</p>
          ) : (
            <div className="h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tiger.detections_by_month}>
                  <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                  <XAxis dataKey="label" stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} />
                  <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                  <ChartTooltip
                    contentStyle={{
                      backgroundColor: 'hsl(0 0% 100%)',
                      border: '1px solid hsl(136 18% 84%)',
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="count" name="Detections" fill="hsl(145 55% 34%)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      <div className="border-b border-border flex gap-6">
        {(['gallery', 'timeline', 'movement'] as Tab[]).map((key) => (
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
        <div>
          {(gallery?.length ?? 0) === 0 ? (
            <EmptyState
              icon={<ImageIcon />}
              title="No images yet"
              description="Frames appear here once detections for this individual are stored."
            />
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {gallery?.map((img) => (
                <figure key={img.image_id} className="card overflow-hidden group">
                  <img
                    src={assetUrl(img.url)}
                    alt={`Capture ${img.image_id}`}
                    loading="lazy"
                    className="w-full h-44 object-cover bg-secondary/40 group-hover:scale-105 transition-transform"
                  />
                  <figcaption className="p-3 text-xs text-muted-foreground space-y-1">
                    <div className="flex justify-between">
                      <span>{formatDate(img.timestamp)}</span>
                      <span>{img.camera_id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>{titleCase(img.species)}</span>
                      <span>{formatPercent(img.identity_confidence)}</span>
                    </div>
                  </figcaption>
                </figure>
              ))}
            </div>
          )}
        </div>
      )}

      {tab === 'timeline' && (
        <div className="space-y-3 max-w-3xl">
          {tiger.recent_observations.length === 0 ? (
            <EmptyState
              icon={<Clock />}
              title="No sightings yet"
              description="Detections will be listed here chronologically."
            />
          ) : (
            tiger.recent_observations.map((obs) => (
              <div key={obs.observation_id} className="flex gap-4 p-4 card items-center">
                <div className="bg-tiger-500/10 p-3 rounded-full text-tiger-500">
                  <Clock size={18} />
                </div>
                <div className="flex-1">
                  <div className="font-medium">{formatDateTime(obs.timestamp)}</div>
                  <div className="text-sm text-muted-foreground">
                    {obs.camera_id} — {obs.camera_name} • {titleCase(obs.zone)}
                  </div>
                </div>
                <div className="text-sm font-medium">{formatPercent(obs.identity_confidence)}</div>
              </div>
            ))
          )}
        </div>
      )}

      {tab === 'movement' && (
        <div className="space-y-4">
          <div className="demo-banner">
            Movement paths are straight-line links between consecutive camera detections, not
            observed travel routes.
          </div>
          {center ? (
          <div className="h-[480px] rounded-xl overflow-hidden border border-border relative z-0">
            <MapContainer
              center={center}
              zoom={11}
              style={{ height: '100%', width: '100%', background: '#f3f8f4' }}
            >
              <TileLayer url={SATELLITE_TILES} attribution={SATELLITE_ATTRIBUTION} />

              {tiger.frequent_cameras
                .filter((c) => c.latitude !== null && c.longitude !== null)
                .map((cam) => (
                  <Marker
                    key={cam.camera_id}
                    position={[cam.latitude as number, cam.longitude as number]}
                    icon={cameraIcon('active')}
                  >
                    <Popup>
                      <div className="text-xs text-slate-900">
                        <strong>{cam.camera_id}</strong>
                        <br />
                        {cam.camera_name}
                        <br />
                        {cam.detections} detections
                      </div>
                    </Popup>
                  </Marker>
                ))}

              {mapPoints.map((obs) => (
                <Marker
                  key={obs.observation_id}
                  position={[obs.latitude as number, obs.longitude as number]}
                  icon={sightingIcon(true)}
                >
                  <Popup>
                    <div className="text-xs text-slate-900">
                      <strong>{formatDateTime(obs.timestamp)}</strong>
                      <br />
                      {obs.camera_id} — {obs.camera_name}
                      <br />
                      Identity: {formatPercent(obs.identity_confidence)}
                    </div>
                  </Popup>
                </Marker>
              ))}

              {track?.legs.map((leg, i) =>
                leg.from_latitude !== null &&
                leg.from_longitude !== null &&
                leg.to_latitude !== null &&
                leg.to_longitude !== null ? (
                  <Polyline
                    key={i}
                    positions={[
                      [leg.from_latitude, leg.from_longitude],
                      [leg.to_latitude, leg.to_longitude],
                    ]}
                    pathOptions={{ color: '#f59e0b', weight: 2 }}
                  >
                    <Tooltip sticky>
                      <div className="text-xs">
                        {leg.from_camera_id} → {leg.to_camera_id}
                        <br />
                        {leg.distance_km} km in {leg.hours_elapsed} h
                      </div>
                    </Tooltip>
                  </Polyline>
                ) : null
              )}
            </MapContainer>
          </div>
          ) : (
            <EmptyState
              icon={<ImageIcon />}
              title="No movement data"
              description="No geolocated observations exist for this tiger."
            />
          )}

          {track && track.legs.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold text-sm mb-3">Movement legs</h3>
              <ol className="space-y-2 text-sm">
                {track.legs.map((leg, i) => (
                  <li key={i} className="flex flex-wrap items-center gap-2 text-muted-foreground">
                    <Badge>{leg.from_camera_id}</Badge>
                    <span>→</span>
                    <Badge>{leg.to_camera_id}</Badge>
                    <span className="text-xs">
                      {leg.distance_km} km • {leg.hours_elapsed} h • {formatDateTime(leg.to_timestamp)}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
