import { useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, Cat, Search } from 'lucide-react';

import { tigersApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { StatusBadge } from '@/components/ui/StatusBadge';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Badge } from '@/components/ui/Badge';
import { formatDate, formatPercent } from '@/lib/utils';

const SEX_LABEL: Record<string, string> = {
  male: '♂ Male',
  female: '♀ Female',
  unknown: 'Unknown',
};

export default function Tigers() {
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('');
  const [sex, setSex] = useState('');

  const { data, loading, error, reload } = useApi(
    () =>
      tigersApi.list({
        limit: 200,
        ...(status ? { status } : {}),
        ...(search ? { search } : {}),
      }),
    [status, search]
  );

  const tigers = (data?.items ?? []).filter((t) => (sex ? t.sex === sex : true));

  if (error) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Tiger catalog unavailable"
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
      <div className="flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-center">
        <div className="relative w-full sm:w-72">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="search"
            placeholder="Search by ID or name…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Search tigers"
            className="filter-input w-full pl-9"
          />
        </div>
        <div className="flex gap-2">
          <select
            value={status}
            onChange={(e) => setStatus(e.target.value)}
            aria-label="Filter by status"
            className="filter-input"
          >
            <option value="">All statuses</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="deceased">Deceased</option>
          </select>
          <select
            value={sex}
            onChange={(e) => setSex(e.target.value)}
            aria-label="Filter by sex"
            className="filter-input"
          >
            <option value="">Any sex</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="unknown">Unknown</option>
          </select>
        </div>
      </div>

      {loading ? (
        <div className="py-20">
          <LoadingSpinner label="Loading tigers…" />
        </div>
      ) : tigers.length === 0 ? (
        <EmptyState
          icon={<Cat />}
          title="No tigers found"
          description="Try adjusting your search or filters, or ingest more camera-trap images."
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {tigers.map((tiger) => (
            <Link
              key={tiger.id}
              to={`/tigers/${tiger.tiger_id}`}
              className="card p-5 hover:border-tiger-500/50 transition-all hover:-translate-y-0.5 block"
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="font-bold text-lg text-foreground">{tiger.tiger_id}</h3>
                  {tiger.name && <p className="text-sm text-tiger-700 font-medium">{tiger.name}</p>}
                </div>
                <StatusBadge status={tiger.status ?? 'unknown'} />
              </div>

              <dl className="space-y-2 text-sm text-muted-foreground">
                <div className="flex justify-between">
                  <dt>Sex</dt>
                  <dd className="text-foreground">{SEX_LABEL[tiger.sex ?? 'unknown']}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Detections</dt>
                  <dd className="text-foreground font-medium">{tiger.total_observations}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Cameras</dt>
                  <dd className="text-foreground">{tiger.camera_count}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Mean confidence</dt>
                  <dd className="text-foreground">{formatPercent(tiger.mean_confidence)}</dd>
                </div>
                <div className="flex justify-between">
                  <dt>Last seen</dt>
                  <dd className="text-foreground">{formatDate(tiger.last_seen)}</dd>
                </div>
              </dl>

              {tiger.is_demo && (
                <div className="mt-4">
                  <Badge variant="demo">Demo profile</Badge>
                </div>
              )}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
