import { Link } from 'react-router-dom';
import {
  AlertCircle,
  BellRing,
  Camera,
  Cat,
  Eye,
  HardDrive,
  Image as ImageIcon,
  ShieldCheck,
  UserCheck,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { assetUrl, dashboardApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { KpiCard } from '@/components/ui/KpiCard';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { SeverityBadge } from '@/components/ui/SeverityBadge';
import ReserveMap from '@/features/map/ReserveMap';
import { formatBytes, formatDateTime, formatPercent, relativeTime, titleCase } from '@/lib/utils';

const CHART_TOOLTIP = {
  contentStyle: {
    backgroundColor: 'hsl(0 0% 100%)',
    border: '1px solid hsl(136 18% 84%)',
    borderRadius: '8px',
    fontSize: 12,
  },
  itemStyle: { color: 'hsl(140 18% 9%)' },
  labelStyle: { color: 'hsl(140 9% 38%)' },
};

const HEALTH_COLORS: Record<string, string> = {
  active: '#22c55e',
  warning: '#eab308',
  offline: '#ef4444',
  maintenance: '#38bdf8',
};

export default function Dashboard() {
  const { data: stats, loading, error, reload } = useApi(() => dashboardApi.getStats());

  if (error) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Dashboard unavailable"
        description={error}
        action={
          <button className="btn-primary" onClick={reload}>
            Retry
          </button>
        }
      />
    );
  }

  const healthData = stats
    ? (['active', 'warning', 'offline', 'maintenance'] as const)
        .map((key) => ({ name: titleCase(key), value: stats.camera_health[key], key }))
        .filter((d) => d.value > 0)
    : [];

  const zoneMax = Math.max(1, ...(stats?.detections_by_zone.map((z) => z.count) ?? [1]));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <KpiCard
          title="Camera Traps"
          value={stats?.total_cameras ?? 0}
          subtitle={`${stats?.active_cameras ?? 0} reporting normally`}
          icon={<Camera />}
          loading={loading}
        />
        <KpiCard
          title="Total Detections"
          value={stats?.total_observations ?? 0}
          subtitle={`${stats?.detections_last_7_days ?? 0} in the last 7 days`}
          icon={<Eye />}
          loading={loading}
        />
        <KpiCard
          title="Identified Tigers"
          value={stats?.total_tigers ?? 0}
          subtitle={`${stats?.active_tigers ?? 0} currently active`}
          icon={<Cat />}
          loading={loading}
        />
        <KpiCard
          title="Open Alerts"
          value={stats?.open_alerts ?? 0}
          subtitle={`${stats?.pending_reviews ?? 0} identities awaiting review`}
          icon={<BellRing />}
          loading={loading}
        />

        <KpiCard
          title="Images Ingested"
          value={stats?.total_images ?? 0}
          subtitle={`${stats?.subject_images ?? 0} with subjects`}
          icon={<ImageIcon />}
          loading={loading}
        />
        <KpiCard
          title="Blank Frames Filtered"
          value={stats?.blank_images ?? 0}
          subtitle={`${stats?.quarantined_images ?? 0} quarantined`}
          icon={<ShieldCheck />}
          loading={loading}
        />
        <KpiCard
          title="Storage Recoverable"
          value={formatBytes(stats?.storage_saved_bytes ?? 0)}
          subtitle={`of ${formatBytes(stats?.total_storage_bytes ?? 0)} stored`}
          icon={<HardDrive />}
          loading={loading}
        />
        <KpiCard
          title="Mean ID Confidence"
          value={formatPercent(stats?.mean_identity_confidence)}
          subtitle="Across identified detections"
          icon={<UserCheck />}
          loading={loading}
        />
      </div>

      <div className="card overflow-hidden">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <div>
            <h3 className="font-semibold">Pench Tiger Reserve — Live Map</h3>
            <p className="text-xs text-muted-foreground">
              Reserve territory border, zones, camera traps and recent sightings. Zoom and pan to explore.
            </p>
          </div>
          <Link to="/map" className="text-xs text-tiger-700 hover:underline">
            Open full map
          </Link>
        </div>
        <div className="h-[520px] w-full relative z-0">
          <ReserveMap />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card p-6 lg:col-span-2">
          <div className="flex items-baseline justify-between mb-6">
            <h3 className="font-semibold">Detection Trend — last 14 days</h3>
            <span className="text-xs text-muted-foreground">detections vs blank frames</span>
          </div>
          <div className="h-[260px]">
            {loading ? (
              <div className="w-full h-full animate-pulse bg-secondary/50 rounded" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={stats?.detection_trend ?? []}>
                  <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                  <XAxis
                    dataKey="date"
                    stroke="hsl(140 9% 38%)"
                    fontSize={11}
                    tickFormatter={(d: string) => d.slice(5)}
                    tickLine={false}
                  />
                  <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                  <Tooltip {...CHART_TOOLTIP} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line
                    type="monotone"
                    dataKey="detections"
                    name="Detections"
                    stroke="hsl(145 55% 34%)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="blanks"
                    name="Blank frames"
                    stroke="hsl(140 9% 38%)"
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="card p-6">
          <h3 className="font-semibold mb-6">Camera Health</h3>
          <div className="h-[200px]">
            {loading ? (
              <div className="w-full h-full animate-pulse bg-secondary/50 rounded-full" />
            ) : healthData.length === 0 ? (
              <p className="text-sm text-muted-foreground">No cameras registered yet.</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={healthData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={54}
                    outerRadius={76}
                    paddingAngle={4}
                  >
                    {healthData.map((entry) => (
                      <Cell key={entry.key} fill={HEALTH_COLORS[entry.key]} />
                    ))}
                  </Pie>
                  <Tooltip {...CHART_TOOLTIP} />
                </PieChart>
              </ResponsiveContainer>
            )}
          </div>
          <ul className="mt-4 space-y-1.5 text-xs">
            {(['active', 'warning', 'offline', 'maintenance'] as const).map((key) => (
              <li key={key} className="flex items-center gap-2">
                <span
                  className="w-2.5 h-2.5 rounded-full"
                  style={{ backgroundColor: HEALTH_COLORS[key] }}
                />
                <span className="text-muted-foreground">{titleCase(key)}</span>
                <span className="ml-auto font-medium">{stats?.camera_health[key] ?? 0}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card p-6 lg:col-span-2">
          <h3 className="font-semibold mb-6">Images per Camera</h3>
          <div className="h-[240px]">
            {loading ? (
              <div className="w-full h-full animate-pulse bg-secondary/50 rounded" />
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stats?.images_by_camera ?? []}>
                  <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                  <XAxis dataKey="label" stroke="hsl(140 9% 38%)" fontSize={10} tickLine={false} />
                  <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                  <Tooltip cursor={{ fill: 'hsl(132 22% 94% / 0.85)' }} {...CHART_TOOLTIP} />
                  <Bar dataKey="count" name="Images" fill="hsl(145 55% 34%)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="card p-6">
          <h3 className="font-semibold mb-6">Detections by Zone</h3>
          {loading ? (
            <div className="h-[160px] animate-pulse bg-secondary/50 rounded" />
          ) : (stats?.detections_by_zone.length ?? 0) === 0 ? (
            <p className="text-sm text-muted-foreground">No detections recorded yet.</p>
          ) : (
            <ul className="space-y-4">
              {stats?.detections_by_zone.map((zone) => (
                <li key={zone.label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{titleCase(zone.label)}</span>
                    <span className="font-medium">{zone.count}</span>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-tiger-500"
                      style={{ width: `${(zone.count / zoneMax) * 100}%` }}
                    />
                  </div>
                </li>
              ))}
            </ul>
          )}

          <h3 className="font-semibold mt-8 mb-4">Most Active Tigers</h3>
          {(stats?.most_active_tigers.length ?? 0) === 0 ? (
            <p className="text-sm text-muted-foreground">No identified tigers yet.</p>
          ) : (
            <ul className="space-y-2 text-sm">
              {stats?.most_active_tigers.map((t) => (
                <li key={t.label} className="flex items-center justify-between">
                  <Link to={`/tigers/${t.label}`} className="hover:text-tiger-700 font-medium">
                    {t.label}
                  </Link>
                  <span className="text-muted-foreground">{t.count} detections</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="card overflow-hidden xl:col-span-2">
          <div className="p-4 border-b border-border flex items-center justify-between">
            <h3 className="font-semibold">Recent Identifications</h3>
            <Link to="/observations" className="text-xs text-tiger-700 hover:underline">
              View all
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Tiger</th>
                  <th>Match</th>
                  <th>Camera</th>
                  <th>When</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  Array.from({ length: 4 }).map((_, i) => (
                    <tr key={i}>
                      <td colSpan={5} className="py-4">
                        <div className="h-4 bg-secondary/50 animate-pulse rounded w-full" />
                      </td>
                    </tr>
                  ))
                ) : (stats?.recent_identifications.length ?? 0) === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-8 text-center text-muted-foreground">
                      No detections yet — upload images or run a demo simulation.
                    </td>
                  </tr>
                ) : (
                  stats?.recent_identifications.map((ident) => (
                    <tr key={ident.observation_id}>
                      <td className="font-medium">
                        {ident.tiger_code ? (
                          <Link to={`/tigers/${ident.tiger_code}`} className="hover:text-tiger-700">
                            {ident.tiger_code}
                          </Link>
                        ) : (
                          <span className="text-muted-foreground">Unassigned</span>
                        )}
                      </td>
                      <td>
                        {ident.match_type ? (
                          <StatusBadge status={ident.match_type} />
                        ) : (
                          <Badge>—</Badge>
                        )}
                      </td>
                      <td>
                        {ident.camera_id ? (
                          <Link to={`/cameras/${ident.camera_id}`}>
                            <Badge>{ident.camera_id}</Badge>
                          </Link>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="text-muted-foreground">{relativeTime(ident.timestamp)}</td>
                      <td>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-secondary rounded-full overflow-hidden">
                            <div
                              className="h-full bg-tiger-500"
                              style={{ width: `${(ident.identity_confidence ?? 0) * 100}%` }}
                            />
                          </div>
                          <span className="text-xs">{formatPercent(ident.identity_confidence)}</span>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card overflow-hidden">
          <div className="p-4 border-b border-border flex items-center justify-between">
            <h3 className="font-semibold">Active Alerts</h3>
            <Link to="/alerts" className="text-xs text-tiger-700 hover:underline">
              View all
            </Link>
          </div>
          <div className="divide-y divide-border/60 max-h-[420px] overflow-y-auto">
            {loading ? (
              Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="p-4">
                  <div className="h-4 bg-secondary/50 animate-pulse rounded w-full" />
                </div>
              ))
            ) : (stats?.recent_alerts.length ?? 0) === 0 ? (
              <p className="p-6 text-sm text-muted-foreground">No active alerts.</p>
            ) : (
              stats?.recent_alerts.map((alert) => (
                <div key={alert.alert_id} className="p-4 space-y-1.5">
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-sm font-medium leading-snug">{alert.title}</span>
                    <SeverityBadge severity={alert.severity} />
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">{alert.message}</p>
                  <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                    <span>{titleCase(alert.alert_type)}</span>
                    <span>•</span>
                    <span>{formatDateTime(alert.created_at)}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold">Latest Captures</h3>
          <Link to="/gallery" className="text-xs text-tiger-700 hover:underline">
            Open gallery
          </Link>
        </div>
        {loading ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-32 bg-secondary/50 animate-pulse rounded" />
            ))}
          </div>
        ) : (stats?.recent_images.length ?? 0) === 0 ? (
          <p className="text-sm text-muted-foreground">
            No images ingested yet. Use Upload or run a demo simulation.
          </p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {stats?.recent_images.map((img) => (
              <div key={img.image_id} className="rounded-lg overflow-hidden border border-border">
                <img
                  src={assetUrl(img.url)}
                  alt={`Capture ${img.image_id}`}
                  loading="lazy"
                  className="w-full h-28 object-cover bg-secondary/40"
                />
                <div className="p-2 text-[11px] text-muted-foreground flex justify-between">
                  <span>{img.camera_id ?? '—'}</span>
                  <span>{relativeTime(img.timestamp)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
