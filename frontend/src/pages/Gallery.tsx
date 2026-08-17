import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, ImageIcon, X } from 'lucide-react';

import { assetUrl, camerasApi, imagesApi, tigersApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { formatDateTime, formatPercent, titleCase } from '@/lib/utils';
import type { ImageRecord } from '@/types';

const PAGE_SIZE = 24;

export default function Gallery() {
  const [cameraId, setCameraId] = useState('');
  const [tigerCode, setTigerCode] = useState('');
  const [species, setSpecies] = useState('');
  const [status, setStatus] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<ImageRecord | null>(null);

  const { data: cameras } = useApi(() => camerasApi.list({ limit: 200 }), []);
  const { data: tigers } = useApi(() => tigersApi.list({ limit: 200 }), []);

  const { data, loading, error, reload } = useApi(
    () =>
      imagesApi.list({
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        ...(cameraId ? { camera_id: cameraId } : {}),
        ...(tigerCode ? { tiger_code: tigerCode } : {}),
        ...(species ? { species } : {}),
        ...(status ? { status } : {}),
        ...(dateFrom ? { date_from: new Date(dateFrom).toISOString() } : {}),
        ...(dateTo ? { date_to: new Date(dateTo).toISOString() } : {}),
      }),
    [page, cameraId, tigerCode, species, status, dateFrom, dateTo]
  );

  const images = data?.items ?? [];
  const update = (setter: (v: string) => void) => (value: string) => {
    setPage(0);
    setter(value);
  };

  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSelected(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected]);

  if (error) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Gallery unavailable"
        description={error}
        action={
          <button className="btn-primary" onClick={reload}>
            Retry
          </button>
        }
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="card p-4 flex flex-wrap gap-3">
        <select
          value={cameraId}
          onChange={(e) => update(setCameraId)(e.target.value)}
          aria-label="Filter by camera"
          className="filter-input"
        >
          <option value="">All cameras</option>
          {cameras?.items.map((c) => (
            <option key={c.camera_id} value={c.camera_id}>
              {c.camera_id}
            </option>
          ))}
        </select>
        <select
          value={tigerCode}
          onChange={(e) => update(setTigerCode)(e.target.value)}
          aria-label="Filter by tiger"
          className="filter-input"
        >
          <option value="">All tigers</option>
          {tigers?.items.map((t) => (
            <option key={t.tiger_id} value={t.tiger_id}>
              {t.tiger_id}
            </option>
          ))}
        </select>
        <select
          value={species}
          onChange={(e) => update(setSpecies)(e.target.value)}
          aria-label="Filter by species"
          className="filter-input"
        >
          <option value="">All species</option>
          {['tiger', 'leopard', 'sambar', 'chital', 'wild_dog', 'gaur', 'sloth_bear'].map((s) => (
            <option key={s} value={s}>
              {titleCase(s)}
            </option>
          ))}
        </select>
        <select
          value={status}
          onChange={(e) => update(setStatus)(e.target.value)}
          aria-label="Filter by status"
          className="filter-input"
        >
          <option value="">Any status</option>
          {['processed', 'triaged', 'quarantined', 'pending', 'deleted'].map((s) => (
            <option key={s} value={s}>
              {titleCase(s)}
            </option>
          ))}
        </select>
        <input
          type="date"
          value={dateFrom}
          onChange={(e) => update(setDateFrom)(e.target.value)}
          aria-label="From date"
          className="filter-input"
        />
        <input
          type="date"
          value={dateTo}
          onChange={(e) => update(setDateTo)(e.target.value)}
          aria-label="To date"
          className="filter-input"
        />
      </div>

      {loading ? (
        <div className="py-20">
          <LoadingSpinner label="Loading captures…" />
        </div>
      ) : images.length === 0 ? (
        <EmptyState
          icon={<ImageIcon />}
          title="No captures found"
          description="Adjust the filters, upload camera-trap images, or run a demo simulation."
        />
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
            {images.map((img) => (
              <button
                key={img.image_id}
                onClick={() => setSelected(img)}
                className="card overflow-hidden text-left group hover:border-tiger-500/50 transition-colors"
              >
                <img
                  src={assetUrl(img.url)}
                  alt={`Capture ${img.image_id}`}
                  loading="lazy"
                  className="w-full h-32 object-cover bg-secondary/40 group-hover:scale-105 transition-transform"
                />
                <div className="p-2.5 space-y-1 text-[11px] text-muted-foreground">
                  <div className="flex justify-between">
                    <span className="font-medium text-foreground">{img.camera_id ?? '—'}</span>
                    <span>{titleCase(img.species)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>{img.tiger_code ?? 'unassigned'}</span>
                    <span>{formatPercent(img.identity_confidence)}</span>
                  </div>
                  <div>{formatDateTime(img.timestamp)}</div>
                </div>
              </button>
            ))}
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">
              {data?.total ?? 0} images • page {data?.page ?? 1} of {data?.pages ?? 1}
            </span>
            <div className="flex gap-2">
              <button
                className="btn-secondary"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
              >
                Previous
              </button>
              <button
                className="btn-secondary"
                onClick={() => setPage((p) => p + 1)}
                disabled={(data?.page ?? 1) >= (data?.pages ?? 1)}
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}

      {selected && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-6"
          role="dialog"
          aria-modal="true"
          aria-label={`Capture ${selected.image_id}`}
          onClick={() => setSelected(null)}
        >
          <div
            className="card max-w-3xl w-full overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="font-semibold font-mono text-sm">{selected.image_id}</h3>
              <button
                onClick={() => setSelected(null)}
                aria-label="Close image details"
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <img
              src={assetUrl(selected.url)}
              alt={`Capture ${selected.image_id}`}
              className="w-full max-h-[55vh] object-contain bg-black"
            />
            <dl className="p-5 grid grid-cols-2 sm:grid-cols-3 gap-4 text-sm">
              <div>
                <dt className="text-xs text-muted-foreground">Camera</dt>
                <dd>
                  {selected.camera_id ? (
                    <Link to={`/cameras/${selected.camera_id}`} className="hover:text-tiger-700">
                      {selected.camera_id}
                    </Link>
                  ) : (
                    '—'
                  )}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Captured</dt>
                <dd>{formatDateTime(selected.timestamp)}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Status</dt>
                <dd>
                  <StatusBadge status={selected.status ?? 'pending'} />
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Species</dt>
                <dd>{titleCase(selected.species)}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Tiger</dt>
                <dd>
                  {selected.tiger_code ? (
                    <Link to={`/tigers/${selected.tiger_code}`} className="hover:text-tiger-700">
                      {selected.tiger_code}
                    </Link>
                  ) : (
                    '—'
                  )}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Similarity</dt>
                <dd>
                  {selected.identity_confidence != null
                    ? formatPercent(selected.identity_confidence)
                    : '—'}
                  <span className="block text-[10px] text-muted-foreground mt-0.5">
                    Cosine similarity (not calibrated confidence)
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Blank score</dt>
                <dd>
                  {selected.blank_probability != null
                    ? formatPercent(selected.blank_probability)
                    : '—'}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Triage reason</dt>
                <dd>{selected.triage_reason ?? '—'}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Source</dt>
                <dd>{selected.is_demo ? <Badge variant="demo">Demo</Badge> : 'Uploaded'}</dd>
              </div>
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}
