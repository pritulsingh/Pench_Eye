import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, Check, ImageIcon, UserCheck } from 'lucide-react';

import { assetUrl, reviewsApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { EmptyState } from '@/components/ui/EmptyState';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Badge } from '@/components/ui/Badge';
import { formatDateTime, formatPercent } from '@/lib/utils';

export default function Reviews() {
  const [reviewer, setReviewer] = useState('control-room');
  const [busy, setBusy] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const { data, loading, error, reload } = useApi(() =>
    reviewsApi.list({ status: 'pending', limit: 50 })
  );

  const reviews = data?.items ?? [];

  const act = async (
    reviewId: string,
    action: 'approve' | 'reject' | 'new',
    tigerId?: string
  ) => {
    setBusy(reviewId);
    setActionError(null);
    try {
      if (action === 'approve' && tigerId) {
        await reviewsApi.approve(reviewId, { tiger_id: tigerId, reviewer });
      } else if (action === 'reject') {
        await reviewsApi.reject(reviewId, { reviewer });
      } else if (action === 'new') {
        await reviewsApi.newTiger(reviewId, { reviewer });
      }
      reload();
    } catch (err) {
      const e = err as { userMessage?: string };
      setActionError(e.userMessage ?? 'The review action failed.');
    } finally {
      setBusy(null);
    }
  };

  if (loading) {
    return (
      <div className="py-20">
        <LoadingSpinner label="Loading review queue…" />
      </div>
    );
  }

  if (error) {
    return (
      <EmptyState
        icon={<AlertCircle />}
        title="Review queue unavailable"
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
      <div className="card p-4 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-3">
          <span className="bg-tiger-500/10 text-tiger-700 p-2 rounded-md">
            <UserCheck className="w-5 h-5" />
          </span>
          <div>
            <div className="text-2xl font-bold text-tiger-500">{data?.total ?? 0}</div>
            <div className="text-xs uppercase tracking-wider text-muted-foreground">
              Pending identity decisions
            </div>
          </div>
        </div>
        <label className="ml-auto flex items-center gap-2 text-sm text-muted-foreground">
          Reviewer
          <input
            type="text"
            value={reviewer}
            onChange={(e) => setReviewer(e.target.value)}
            className="filter-input w-40"
            aria-label="Reviewer name"
          />
        </label>
      </div>

      {actionError && <div className="badge-error px-4 py-2 rounded-md text-sm">{actionError}</div>}

      {reviews.length === 0 ? (
        <EmptyState
          icon={<Check />}
          title="All caught up"
          description="No detections are currently waiting for human identity review."
        />
      ) : (
        <div className="space-y-6">
          {reviews.map((review) => (
            <article key={review.id} className="card p-0 overflow-hidden flex flex-col md:flex-row">
              <div className="w-full md:w-1/3 h-64 bg-secondary/50 border-b md:border-b-0 md:border-r border-border">
                {review.image_url ? (
                  <img
                    src={assetUrl(review.image_url)}
                    alt={`Capture for review ${review.review_id}`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <ImageIcon className="text-muted-foreground/30" size={48} />
                  </div>
                )}
              </div>

              <div className="p-6 flex-1 flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-lg font-bold text-tiger-700">
                      {review.observation_code ?? review.review_id}
                    </h3>
                    <Badge variant="warning">Pending</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-6">
                    {review.camera_id ? (
                      <Link to={`/cameras/${review.camera_id}`} className="hover:text-tiger-700">
                        Camera {review.camera_id}
                      </Link>
                    ) : (
                      'Unknown camera'
                    )}{' '}
                    • {formatDateTime(review.timestamp)}
                  </p>

                  <h4 className="text-sm font-semibold mb-3">Candidate matches</h4>
                  {review.candidates.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      No candidates were suggested — create a new individual if appropriate.
                    </p>
                  ) : (
                    <ul className="space-y-3">
                      {review.candidates.map((cand, i) => (
                        <li key={`${cand.tiger_id}-${i}`} className="flex items-center gap-4">
                          <span className="w-28 font-mono font-bold text-sm">
                            {cand.tiger_code ?? cand.tiger_id}
                          </span>
                          <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
                            <div
                              className={`h-full ${i === 0 ? 'bg-tiger-500' : 'bg-muted-foreground'}`}
                              style={{ width: `${(cand.score ?? 0) * 100}%` }}
                            />
                          </div>
                          <span className="w-14 text-right text-sm">
                            {formatPercent(cand.score)}
                          </span>
                          <button
                            onClick={() => act(review.review_id, 'approve', cand.tiger_id)}
                            disabled={busy === review.review_id}
                            className={`px-3 py-1 text-xs rounded font-medium ${
                              i === 0
                                ? 'bg-tiger-500 text-foreground hover:bg-tiger-400'
                                : 'bg-secondary hover:bg-secondary/80'
                            }`}
                          >
                            Confirm
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>

                <div className="flex flex-wrap gap-3 mt-8 pt-4 border-t border-border">
                  <button
                    onClick={() => act(review.review_id, 'new')}
                    disabled={busy === review.review_id}
                    className="btn-secondary"
                  >
                    Create new tiger
                  </button>
                  <button
                    onClick={() => act(review.review_id, 'reject')}
                    disabled={busy === review.review_id}
                    className="btn-danger ml-auto"
                  >
                    Reject detection
                  </button>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
