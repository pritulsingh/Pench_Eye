import { useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, BellRing, Check, RefreshCw } from 'lucide-react';

import { alertsApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Badge } from '@/components/ui/Badge';
import { SeverityBadge } from '@/components/ui/SeverityBadge';
import { KpiCard } from '@/components/ui/KpiCard';
import { formatDateTime, titleCase } from '@/lib/utils';

const ALERT_TYPES = [
  'high_priority_detection',
  'camera_offline',
  'unusual_movement',
  'high_activity',
  'low_confidence',
];

export default function Alerts() {
  const [status, setStatus] = useState('open');
  const [severity, setSeverity] = useState('');
  const [alertType, setAlertType] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const { data: summary, reload: reloadSummary } = useApi(() => alertsApi.summary());
  const { data, loading, error, reload } = useApi(
    () =>
      alertsApi.list({
        limit: 100,
        ...(status ? { status } : {}),
        ...(severity ? { severity } : {}),
        ...(alertType ? { alert_type: alertType } : {}),
      }),
    [status, severity, alertType]
  );

  const alerts = data?.items ?? [];

  const changeStatus = async (alertId: string, next: string) => {
    setBusy(alertId);
    setActionError(null);
    try {
      await alertsApi.updateStatus(alertId, next, 'control-room');
      reload();
      reloadSummary();
    } catch (err) {
      const e = err as { userMessage?: string };
      setActionError(e.userMessage ?? 'Could not update the alert.');
    } finally {
      setBusy(null);
    }
  };

  const runRules = async () => {
    setBusy('evaluate');
    setActionError(null);
    try {
      await alertsApi.evaluate();
      reload();
      reloadSummary();
    } catch (err) {
      const e = err as { userMessage?: string };
      setActionError(e.userMessage ?? 'Rule evaluation failed.');
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <KpiCard title="Open" value={summary?.open ?? 0} icon={<BellRing />} />
        <KpiCard title="Critical" value={summary?.critical ?? 0} icon={<AlertCircle />} />
        <KpiCard title="Acknowledged" value={summary?.acknowledged ?? 0} icon={<Check />} />
        <KpiCard title="Resolved" value={summary?.resolved ?? 0} icon={<Check />} />
      </div>

      <div className="card p-4 flex flex-wrap gap-3 items-center">
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          aria-label="Filter by status"
          className="filter-input"
        >
          <option value="">All statuses</option>
          <option value="open">Open</option>
          <option value="acknowledged">Acknowledged</option>
          <option value="resolved">Resolved</option>
        </select>
        <select
          value={severity}
          onChange={(e) => setSeverity(e.target.value)}
          aria-label="Filter by severity"
          className="filter-input"
        >
          <option value="">Any severity</option>
          {['critical', 'high', 'medium', 'low', 'info'].map((s) => (
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
          {ALERT_TYPES.map((t) => (
            <option key={t} value={t}>
              {titleCase(t)}
            </option>
          ))}
        </select>

        <button
          onClick={runRules}
          disabled={busy === 'evaluate'}
          className="btn-secondary ml-auto flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${busy === 'evaluate' ? 'animate-spin' : ''}`} />
          Re-evaluate rules
        </button>
      </div>

      {actionError && (
        <div className="badge-error px-4 py-2 rounded-md text-sm">{actionError}</div>
      )}

      {loading ? (
        <div className="py-20">
          <LoadingSpinner label="Loading alerts…" />
        </div>
      ) : error ? (
        <EmptyState
          icon={<AlertCircle />}
          title="Alerts unavailable"
          description={error}
          action={
            <button className="btn-primary" onClick={reload}>
              Retry
            </button>
          }
        />
      ) : alerts.length === 0 ? (
        <EmptyState
          icon={<Check />}
          title="Nothing to action"
          description="No alerts match the current filters."
        />
      ) : (
        <div className="space-y-3">
          {alerts.map((alert) => (
            <article key={alert.alert_id} className="card p-5">
              <div className="flex flex-wrap items-start gap-3">
                <div className="flex-1 min-w-[240px]">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h3 className="font-semibold">{alert.title}</h3>
                    <SeverityBadge severity={alert.severity} />
                    <Badge
                      variant={
                        alert.status === 'open'
                          ? 'warning'
                          : alert.status === 'acknowledged'
                          ? 'review'
                          : 'success'
                      }
                    >
                      {titleCase(alert.status)}
                    </Badge>
                    {alert.is_demo && <Badge variant="demo">Demo</Badge>}
                  </div>
                  <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
                    {alert.message}
                  </p>
                  <div className="flex flex-wrap gap-3 mt-3 text-xs text-muted-foreground">
                    <span>{titleCase(alert.alert_type)}</span>
                    {alert.camera_id && (
                      <Link to={`/cameras/${alert.camera_id}`} className="hover:text-tiger-700">
                        {alert.camera_id}
                      </Link>
                    )}
                    {alert.zone_code && <span>{alert.zone_code}</span>}
                    <span>{formatDateTime(alert.created_at)}</span>
                    {alert.acknowledged_by && <span>by {alert.acknowledged_by}</span>}
                  </div>
                </div>

                <div className="flex gap-2">
                  {alert.status === 'open' && (
                    <button
                      className="btn-secondary"
                      disabled={busy === alert.alert_id}
                      onClick={() => changeStatus(alert.alert_id, 'acknowledged')}
                    >
                      Acknowledge
                    </button>
                  )}
                  {alert.status !== 'resolved' && (
                    <button
                      className="btn-primary"
                      disabled={busy === alert.alert_id}
                      onClick={() => changeStatus(alert.alert_id, 'resolved')}
                    >
                      Resolve
                    </button>
                  )}
                  {alert.status === 'resolved' && (
                    <button
                      className="btn-secondary"
                      disabled={busy === alert.alert_id}
                      onClick={() => changeStatus(alert.alert_id, 'open')}
                    >
                      Reopen
                    </button>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
