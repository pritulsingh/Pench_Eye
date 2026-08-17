import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Cat, Search } from 'lucide-react';

import { StatusBadge } from '@/components/ui/StatusBadge';
import { EmptyState } from '@/components/ui/EmptyState';
import { Badge } from '@/components/ui/Badge';
import { useMonitoringStore } from '@/features/monitoring/MonitoringContext';
import { formatDateTime, titleCase } from '@/lib/utils';

const SEX_LABEL: Record<string, string> = {
  male: '♂ Male',
  female: '♀ Female',
  unknown: 'Unknown',
};

/**
 * Tiger Gallery / Catalog.
 *
 * Reads the 12 tracked tigers from the shared monitoring store — the SAME
 * source of truth the Reserve Map uses — so their reference images (from the
 * `implement/` dataset), latest detection, territory and conflict status are
 * visible outside the map.
 */
export default function Tigers() {
  const { tigers, conflicts } = useMonitoringStore();
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('');
  const [sex, setSex] = useState('');

  const conflictTigerIds = useMemo(() => {
    const s = new Set<string>();
    conflicts.forEach((c) => {
      s.add(c.tigerA);
      s.add(c.tigerB);
    });
    return s;
  }, [conflicts]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return tigers.filter((t) => {
      if (status && t.status !== status) return false;
      if (sex && t.sex !== sex) return false;
      if (q && !`${t.id} ${t.name}`.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [tigers, search, status, sex]);

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
            <option value="unknown">Unknown</option>
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

      {filtered.length === 0 ? (
        <EmptyState
          icon={<Cat />}
          title="No tigers found"
          description="Try adjusting your search or filters."
        />
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map((tiger) => {
            const inConflict = conflictTigerIds.has(tiger.id);
            return (
              <Link
                key={tiger.id}
                to={`/tigers/${tiger.id}`}
                className="card overflow-hidden hover:border-tiger-500/50 transition-all hover:-translate-y-0.5 block"
              >
                <div className="relative h-40 bg-secondary/40">
                  {tiger.referenceImage ? (
                    <img
                      src={tiger.referenceImage}
                      alt={`${tiger.id} reference`}
                      loading="lazy"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                      <Cat className="w-8 h-8" />
                    </div>
                  )}
                  <div className="absolute top-2 left-2">
                    <StatusBadge status={tiger.status} />
                  </div>
                  {inConflict && (
                    <div className="absolute top-2 right-2">
                      <Badge variant="error">Conflict</Badge>
                    </div>
                  )}
                </div>

                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-bold text-lg leading-tight">{tiger.id}</h3>
                      <p className="text-sm text-tiger-700 font-medium">{tiger.name}</p>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {SEX_LABEL[tiger.sex]}
                    </span>
                  </div>

                  <dl className="space-y-1.5 text-xs text-muted-foreground">
                    <div className="flex justify-between">
                      <dt>Last detected</dt>
                      <dd className="text-foreground text-right">
                        {tiger.lastDetectedCamera ?? '—'}
                        {tiger.lastDetectionTime && (
                          <>
                            {' · '}
                            {formatDateTime(tiger.lastDetectionTime)}
                          </>
                        )}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt>Territory</dt>
                      <dd className="text-foreground">{tiger.territoryId}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt>Detections</dt>
                      <dd className="text-foreground font-medium">{tiger.detectionIds.length}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt>Confidence</dt>
                      <dd className="text-foreground">
                        {tiger.confidence != null
                          ? `${Math.round(tiger.confidence * 100)}%`
                          : '—'}
                      </dd>
                    </div>
                  </dl>

                  <span className="btn-secondary w-full !py-1.5 text-xs text-center block">
                    View Profile
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        {filtered.length} of {tigers.length} tracked individuals · {titleCase('shared monitoring data')}.
      </p>
    </div>
  );
}
