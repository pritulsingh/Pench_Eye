import { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { analyticsApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Badge } from '@/components/ui/Badge';
import { formatPercent, titleCase } from '@/lib/utils';

const TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: 'hsl(0 0% 100%)',
    border: '1px solid hsl(136 18% 84%)',
    borderRadius: '8px',
    fontSize: 12,
  },
  itemStyle: { color: 'hsl(140 18% 9%)' },
  labelStyle: { color: 'hsl(140 9% 38%)' },
};

const PIE_COLORS = ['#f59e0b', '#38bdf8', '#22c55e', '#a855f7', '#ef4444', '#eab308', '#94a3b8'];

function ChartCard({
  title,
  subtitle,
  children,
  className,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`card p-6 ${className ?? ''}`}>
      <div className="flex items-baseline justify-between mb-5">
        <h3 className="font-semibold">{title}</h3>
        {subtitle && <span className="text-xs text-muted-foreground">{subtitle}</span>}
      </div>
      {children}
    </div>
  );
}

export default function Analytics() {
  const [days, setDays] = useState(90);
  const { data, loading, error, reload } = useApi(() => analyticsApi.overview(days), [days]);

  if (loading && !data) {
    return (
      <div className="py-24">
        <LoadingSpinner label="Crunching detection analytics…" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Analytics unavailable"
        description={error ?? 'No analytics returned.'}
        action={
          <button className="btn-primary" onClick={reload}>
            Retry
          </button>
        }
      />
    );
  }

  const hasData = data.detections_over_time.some((d) => d.detections > 0);

  return (
    <div className="space-y-6">
      <div className="card p-4 flex flex-wrap items-center gap-3">
        <label htmlFor="range" className="text-sm text-muted-foreground">
          Period
        </label>
        <select
          id="range"
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          className="filter-input"
        >
          <option value={30}>Last 30 days</option>
          <option value={90}>Last 90 days</option>
          <option value={180}>Last 180 days</option>
          <option value={365}>Last 365 days</option>
        </select>
        <span className="text-xs text-muted-foreground">
          Mean identity confidence: {formatPercent(data.mean_identity_confidence)}
        </span>
        {data.is_demo_data && <Badge variant="demo">Demo dataset</Badge>}
      </div>

      {!hasData && (
        <EmptyState
          icon={<AlertCircle />}
          title="No detections in this period"
          description="Widen the period, upload camera-trap images, or run a demo simulation."
        />
      )}

      <ChartCard title="Detections over time" subtitle="tigers vs all detections vs blank frames">
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.detections_over_time}>
              <defs>
                <linearGradient id="detGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(145 55% 34%)" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="hsl(145 55% 34%)" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
              <XAxis
                dataKey="date"
                stroke="hsl(140 9% 38%)"
                fontSize={10}
                tickFormatter={(d: string) => d.slice(5)}
                tickLine={false}
              />
              <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
              <Tooltip {...TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Area
                type="monotone"
                dataKey="detections"
                name="All detections"
                stroke="hsl(145 55% 34%)"
                fill="url(#detGradient)"
                strokeWidth={2}
              />
              <Area
                type="monotone"
                dataKey="tigers"
                name="Tiger detections"
                stroke="#22c55e"
                fill="transparent"
                strokeWidth={2}
              />
              <Area
                type="monotone"
                dataKey="blanks"
                name="Blank frames"
                stroke="hsl(140 9% 38%)"
                fill="transparent"
                strokeDasharray="4 4"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="Detections by camera">
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.detections_by_camera} layout="vertical">
                <CartesianGrid stroke="hsl(136 18% 84%)" horizontal={false} />
                <XAxis type="number" stroke="hsl(140 9% 38%)" fontSize={11} allowDecimals={false} />
                <YAxis
                  type="category"
                  dataKey="label"
                  stroke="hsl(140 9% 38%)"
                  fontSize={10}
                  width={70}
                />
                <Tooltip cursor={{ fill: 'hsl(132 22% 94% / 0.85)' }} {...TOOLTIP_STYLE} />
                <Bar dataKey="count" name="Detections" fill="hsl(145 55% 34%)" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        <ChartCard title="Detections by hour of day" subtitle="activity rhythm">
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.detections_by_hour}>
                <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                <XAxis dataKey="label" stroke="hsl(140 9% 38%)" fontSize={9} tickLine={false} />
                <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                <Tooltip cursor={{ fill: 'hsl(132 22% 94% / 0.85)' }} {...TOOLTIP_STYLE} />
                <Bar dataKey="count" name="Detections" fill="#38bdf8" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        <ChartCard title="Species distribution">
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data.species_distribution.map((s) => ({ ...s, label: titleCase(s.label) }))}
                  dataKey="count"
                  nameKey="label"
                  innerRadius={55}
                  outerRadius={95}
                  paddingAngle={3}
                >
                  {data.species_distribution.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip {...TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        <ChartCard title="Identity confidence distribution">
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.confidence_distribution}>
                <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                <XAxis dataKey="range" stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} />
                <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                <Tooltip cursor={{ fill: 'hsl(132 22% 94% / 0.85)' }} {...TOOLTIP_STYLE} />
                <Bar dataKey="count" name="Detections" fill="#a855f7" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        <ChartCard title="Detections by weekday">
          <div className="h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.detections_by_weekday}>
                <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                <XAxis dataKey="label" stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} />
                <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                <Tooltip cursor={{ fill: 'hsl(132 22% 94% / 0.85)' }} {...TOOLTIP_STYLE} />
                <Bar dataKey="count" name="Detections" fill="#22c55e" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        <ChartCard title="Detections by zone">
          <div className="h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data.detections_by_zone.map((z) => ({ ...z, label: titleCase(z.label) }))}
              >
                <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                <XAxis dataKey="label" stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} />
                <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                <Tooltip cursor={{ fill: 'hsl(132 22% 94% / 0.85)' }} {...TOOLTIP_STYLE} />
                <Bar dataKey="count" name="Detections" fill="#eab308" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card overflow-hidden">
          <div className="p-4 border-b border-border">
            <h3 className="font-semibold">Most frequent camera-to-camera movements</h3>
          </div>
          {data.movement_frequency.length === 0 ? (
            <p className="p-6 text-sm text-muted-foreground">
              Not enough consecutive detections to infer movement.
            </p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>From</th>
                  <th>To</th>
                  <th className="text-right">Transitions</th>
                </tr>
              </thead>
              <tbody>
                {data.movement_frequency.map((m, i) => (
                  <tr key={i}>
                    <td>{m.from_camera}</td>
                    <td>{m.to_camera}</td>
                    <td className="text-right font-medium">{m.transitions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="card overflow-hidden">
          <div className="p-4 border-b border-border">
            <h3 className="font-semibold">Most detected individuals</h3>
          </div>
          {data.top_tigers.length === 0 ? (
            <p className="p-6 text-sm text-muted-foreground">No identified individuals yet.</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Tiger</th>
                  <th className="text-right">Detections</th>
                </tr>
              </thead>
              <tbody>
                {data.top_tigers.map((t) => (
                  <tr key={t.label}>
                    <td className="font-medium">{t.label}</td>
                    <td className="text-right">{t.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
