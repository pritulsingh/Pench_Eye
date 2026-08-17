import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, BellRing, Check, MapPin } from 'lucide-react';

import { EmptyState } from '@/components/ui/EmptyState';
import { Badge } from '@/components/ui/Badge';
import { SeverityBadge } from '@/components/ui/SeverityBadge';
import { KpiCard } from '@/components/ui/KpiCard';
import { useMonitoringStore } from '@/features/monitoring/MonitoringContext';
import { formatDateTime, titleCase } from '@/lib/utils';
import type { AlertSeverity } from '@/features/monitoring/types';

const ALERT_TYPE_LABEL: Record<string, string> = {
  tiger_proximity: 'Tiger Proximity',
  multiple_tiger_detection: 'Multiple Tiger Detection',
  territory_overlap: 'Territory Overlap',
  new_territory_movement: 'New Movement',
  long_detection_gap: 'Long Detection Gap',
  high_risk_conflict_zone: 'High-Risk Conflict Zone',
};

const SEVERITY_ORDER: Record<AlertSeverity, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3,
  info: 4,
};

/**
 * Alerts page.
 *
 * Derived entirely from the shared monitoring store. Proximity, overlap,
 * co-detection, new-movement and detection-gap events are computed by the same
 * conflict engine the map uses, so alerts always agree with the map, dashboard
 * and tiger profiles. A new camera detection re-runs the engine and updates
 * this page automatically.
 */
export default function Alerts() {
  const { alerts } = useMonitoringStore();
  const [severity, setSeverity] = useState('');
  const [alertType, setAlertType] = useState('');

  const filtered = useMemo(() => {
    return alerts
      .filter((a) => {
        if (severity && a.severity !== severity) return false;
        if (alertType && a.type !== alertType) return false;
        return true;
      })
      .sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]);
  }, [alerts, severity, alertType]);

  const summary = useMemo(() => {
    const critical = alerts.filter((a) => a.severity === 'critical').length;
    const high = alerts.filter((a) => a.severity === 'high').length;
    const medium = alerts.filter((a) => a.severity === 'medium').length;
    return { total: alerts.length, critical, high, medium };
  }, [alerts]);

  const alertTypes = useMemo(
    () => Array.from(new Set(alerts.map((a) => a.type))),
    [alerts]
  );

  /** Deep-link an alert to the most relevant existing view. */
  const alertLink = (tigerIds: string[], cameraId: string | null): string => {
    if (tigerIds.length === 1) return `/tigers/${tigerIds[0]}`;
    if (cameraId) return `/cameras/${cameraId}`;
    return '/map';
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <KpiCard title="Total alerts" value={summary.total} icon={<BellRing />} />
        <KpiCard title="Critical" value={summary.critical} icon={<AlertCircle />} />
        <KpiCard title="High" value={summary.high} icon={<AlertCircle />} />
        <KpiCard title="Medium" value={summary.medium} icon={<Check />} />
      </div>

      <div className="card p-4 flex flex-wrap gap-3 items-center">
        <select
          value={severity}
          onChange={(e) => setSeverity(e.target.value)}
          aria-label="Filter by severity"
          className="filter-input"
        >
          <option value="">Any severity</option>
          {(['critical', 'high', 'medium', 'low', 'info'] as const).map((s) => (
            <option key={s} value={s}>
              {titleCase(s)}
            </option>
          ))}
        </select>
        <select
          value={alertType}
          onChange={(e) => setAlertType(e.target.value)}
          aria-label="Filter by alert type"
          className="filter-input"
        >
          <option value="">All types</option>
          {alertTypes.map((t) => (
            <option key={t} value={t}>
              {ALERT_TYPE_LABEL[t] ?? titleCase(t)}
            </option>
          ))}
        </select>
        <span className="ml-auto text-xs text-muted-foreground">
          Live conflict-engine output · shared monitoring data
        </span>
      </div>

      {filtered.length === 0 ? (
        <EmptyState
          icon={<Check />}
          title="No active alerts"
          description="The conflict engine found no proximity, overlap or movement events."
        />
      ) : (
        <div className="space-y-3">
          {filtered.map((alert) => (
            <article key={alert.id} className="card p-5">
              <div className="flex flex-wrap items-start gap-3">
                <div className="flex-1 min-w-[240px]">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="font-semibold">
                      {ALERT_TYPE_LABEL[alert.type] ?? titleCase(alert.type)}
                    </h3>
                    <SeverityBadge severity={alert.severity} />
                    {alert.tigerIds.map((tid) => (
                      <Link key={tid} to={`/tigers/${tid}`}>
                        <Badge variant="tiger">{tid}</Badge>
                      </Link>
                    ))}
                  </div>
                  <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
                    {alert.message}
                  </p>
                  <div className="flex flex-wrap gap-3 mt-3 text-xs text-muted-foreground">
                    {alert.cameraId && (
                      <Link to={`/cameras/${alert.cameraId}`} className="hover:text-tiger-700">
                        {alert.cameraId}
                      </Link>
                    )}
                    {alert.distanceKm != null && <span>{alert.distanceKm} km</span>}
                    {alert.location && (
                      <span className="flex items-center gap-1">
                        <MapPin className="w-3 h-3" />
                        {alert.location[0].toFixed(3)}, {alert.location[1].toFixed(3)}
                      </span>
                    )}
                    <span>{formatDateTime(alert.timestamp)}</span>
                  </div>
                </div>

                <Link
                  to={alertLink(alert.tigerIds, alert.cameraId)}
                  className="btn-secondary self-center"
                >
                  Investigate
                </Link>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
