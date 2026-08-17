import { useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  FileX,
  Image as ImageIcon,
  RotateCcw,
  ShieldAlert,
  Trash2,
} from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { assetUrl, imagesApi, triageApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { KpiCard } from '@/components/ui/KpiCard';
import { EmptyState } from '@/components/ui/EmptyState';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatBytes, formatDateTime, formatPercent } from '@/lib/utils';

export default function Triage() {
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const { data: report, loading, error, reload: reloadReport } = useApi(() => triageApi.getReport());
  const { data: quarantine, reload: reloadQuarantine } = useApi(() =>
    triageApi.getQuarantine({ limit: 60 })
  );

  const act = async (imageId: string, action: 'restore' | 'delete') => {
    if (action === 'delete' && !window.confirm('Mark this image as deleted?')) return;
    setBusy(imageId);
    setActionError(null);
    try {
      if (action === 'restore') await imagesApi.restore(imageId);
      else await imagesApi.delete(imageId);
      reloadQuarantine();
      reloadReport();
    } catch (err) {
      const e = err as { userMessage?: string };
      setActionError(e.userMessage ?? 'The action failed.');
    } finally {
      setBusy(null);
    }
  };

  if (error) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Triage data unavailable"
        description={error}
        action={
          <button className="btn-primary" onClick={reloadReport}>
            Retry
          </button>
        }
      />
    );
  }

  const items = quarantine ?? [];

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Images ingested"
          value={report?.total_images ?? 0}
          icon={<CheckCircle2 />}
          loading={loading}
        />
        <KpiCard
          title="Blank frames"
          value={report?.blank_count ?? 0}
          subtitle={`${report?.subject_count ?? 0} with subjects`}
          icon={<FileX />}
          loading={loading}
        />
        <KpiCard
          title="Quarantined"
          value={report?.quarantined_count ?? 0}
          icon={<ShieldAlert />}
          loading={loading}
        />
        <KpiCard
          title="Storage recoverable"
          value={formatBytes(report?.storage_saved_bytes ?? 0)}
          icon={<FileX />}
          loading={loading}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-6">
          <h3 className="font-semibold mb-5">Image quality distribution</h3>
          {loading ? (
            <div className="h-[240px] animate-pulse bg-secondary/50 rounded" />
          ) : (
            <div className="h-[240px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={report?.quality_distribution ?? []}>
                  <CartesianGrid stroke="hsl(136 18% 84%)" vertical={false} />
                  <XAxis dataKey="range" stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} />
                  <YAxis stroke="hsl(140 9% 38%)" fontSize={11} tickLine={false} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(0 0% 100%)',
                      border: '1px solid hsl(136 18% 84%)',
                      borderRadius: 8,
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="count" name="Images" fill="hsl(145 55% 34%)" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="card p-6">
          <h3 className="font-semibold mb-5">Blank frames by camera</h3>
          {(report?.blanks_by_camera.length ?? 0) === 0 ? (
            <p className="text-sm text-muted-foreground">No blank frames recorded.</p>
          ) : (
            <ul className="space-y-3">
              {report?.blanks_by_camera.map((row) => {
                const max = report.blanks_by_camera[0].count || 1;
                return (
                  <li key={row.label}>
                    <div className="flex justify-between text-sm mb-1">
                      <span>{row.label}</span>
                      <span className="font-medium">{row.count}</span>
                    </div>
                    <div className="h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-muted-foreground"
                        style={{ width: `${(row.count / max) * 100}%` }}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>

      <div>
        <h2 className="text-xl font-bold mb-4">Quarantine review</h2>
        {actionError && (
          <div className="badge-error px-4 py-2 rounded-md text-sm mb-4">{actionError}</div>
        )}
        {loading ? (
          <div className="py-16">
            <LoadingSpinner label="Loading quarantine…" />
          </div>
        ) : items.length === 0 ? (
          <EmptyState
            icon={<ShieldAlert />}
            title="Quarantine empty"
            description="No blank or duplicate frames are currently held for review."
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {items.map((img) => (
              <div key={img.image_id} className="card overflow-hidden flex flex-col">
                <div className="h-40 bg-secondary/30 relative flex items-center justify-center">
                  {img.url ? (
                    <img
                      src={assetUrl(img.url)}
                      alt={`Quarantined capture ${img.image_id}`}
                      loading="lazy"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <ImageIcon className="text-muted-foreground/30" size={40} />
                  )}
                  <div className="absolute top-2 right-2">
                    <StatusBadge status="quarantined" />
                  </div>
                </div>
                <div className="p-4 flex-1 flex flex-col gap-3">
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">
                      Blank probability {formatPercent(img.blank_probability)}
                    </div>
                    <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden">
                      <div
                        className={`h-full ${
                          (img.blank_probability ?? 0) > 0.95 ? 'bg-red-500' : 'bg-amber-500'
                        }`}
                        style={{ width: `${(img.blank_probability ?? 0) * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="text-xs space-y-1 text-muted-foreground flex-1">
                    <div>
                      <span className="font-semibold text-foreground">Camera:</span>{' '}
                      {img.camera_id ?? '—'}
                    </div>
                    <div>
                      <span className="font-semibold text-foreground">Captured:</span>{' '}
                      {formatDateTime(img.timestamp)}
                    </div>
                    <div>
                      <span className="font-semibold text-foreground">Reason:</span>{' '}
                      {img.triage_reason ?? 'unknown'}
                    </div>
                  </div>
                  <div className="flex gap-2 mt-auto">
                    <button
                      onClick={() => act(img.image_id, 'restore')}
                      disabled={busy === img.image_id}
                      className="flex-1 btn-secondary text-xs flex items-center justify-center gap-1"
                    >
                      <RotateCcw size={14} /> Restore
                    </button>
                    <button
                      onClick={() => act(img.image_id, 'delete')}
                      disabled={busy === img.image_id}
                      className="flex-1 btn-danger text-xs flex items-center justify-center gap-1"
                    >
                      <Trash2 size={14} /> Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
