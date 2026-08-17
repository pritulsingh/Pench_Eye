import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import { CheckCircle2, FileImage, Loader2, Upload as UploadIcon, X } from 'lucide-react';

import { camerasApi, imagesApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';
import { Badge } from '@/components/ui/Badge';
import { titleCase } from '@/lib/utils';

interface UploadResult {
  image_id: string;
  status: string;
  is_blank: boolean;
  blank_probability: number | null;
  triage_reason: string | null;
  observation_id: string | null;
  tiger_code: string | null;
  identity_confidence: number | null;
  similarity?: number | null;
  candidate_tiger?: string | null;
  decision: string | null;
  species?: string | null;
  alerts_created: number;
  message: string;
}

export default function Upload() {
  const [files, setFiles] = useState<File[]>([]);
  const [cameraId, setCameraId] = useState('');
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<UploadResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const { data: cameras } = useApi(() => camerasApi.list({ limit: 200 }));

  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const images = Array.from(incoming).filter((f) => f.type.startsWith('image/'));
    setFiles((prev) => [...prev, ...images]);
    setError(images.length === 0 ? 'Only JPEG, PNG and WebP images are accepted.' : null);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  }, []);

  const handleUpload = async () => {
    if (!files.length) return;
    setUploading(true);
    setError(null);
    setResults(null);
    try {
      const res = await imagesApi.batch(files, cameraId || undefined);
      setResults(res.data.images as UploadResult[]);
      setFiles([]);
    } catch (err) {
      const e = err as { userMessage?: string };
      setError(e.userMessage ?? 'Upload failed.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div
        className={`border-2 border-dashed rounded-xl p-12 flex flex-col items-center justify-center transition-colors ${
          files.length > 0
            ? 'border-tiger-500 bg-tiger-500/5'
            : 'border-border bg-secondary/20 hover:border-tiger-500/50'
        }`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <UploadIcon className="w-12 h-12 text-muted-foreground mb-4" />
        <h3 className="text-xl font-medium mb-2">Drop camera-trap images here</h3>
        <p className="text-sm text-muted-foreground mb-6">
          JPEG, PNG or WebP, up to 15 MB each. Each image runs through triage → detection →
          identification.
        </p>
        <label className="btn-secondary cursor-pointer">
          Browse files
          <input
            type="file"
            multiple
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(e) => addFiles(e.target.files)}
          />
        </label>
      </div>

      {error && <div className="badge-error px-4 py-2 rounded-md text-sm">{error}</div>}

      {files.length > 0 && (
        <div className="card p-6 space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <h3 className="font-semibold flex items-center gap-2">
              <FileImage className="w-5 h-5" /> Selected files ({files.length})
            </h3>
            <label className="flex items-center gap-3 text-sm text-muted-foreground">
              Camera station
              <select
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                className="filter-input"
                aria-label="Camera station"
              >
                <option value="">Unassigned</option>
                {cameras?.items.map((c) => (
                  <option key={c.camera_id} value={c.camera_id}>
                    {c.camera_id} — {c.name}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <ul className="max-h-48 overflow-y-auto space-y-2 pr-2">
            {files.map((f, i) => (
              <li
                key={`${f.name}-${i}`}
                className="flex items-center justify-between p-2 rounded bg-secondary/50 text-sm"
              >
                <span className="truncate max-w-[240px]">{f.name}</span>
                <span className="flex items-center gap-3 text-muted-foreground">
                  {(f.size / 1024 / 1024).toFixed(2)} MB
                  <button
                    onClick={() => setFiles((prev) => prev.filter((_, idx) => idx !== i))}
                    aria-label={`Remove ${f.name}`}
                    className="hover:text-foreground"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </span>
              </li>
            ))}
          </ul>

          <div className="pt-4 border-t border-border">
            <button
              onClick={handleUpload}
              disabled={uploading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {uploading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Running pipeline…
                </>
              ) : (
                <>
                  <UploadIcon className="w-4 h-4" /> Upload &amp; process
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {results && (
        <div className="card p-6 animate-fade-in">
          <div className="flex items-center gap-3 text-green-400 mb-5">
            <CheckCircle2 className="w-6 h-6" />
            <h3 className="text-lg font-semibold">Pipeline complete</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Image</th>
                  <th>Status</th>
                  <th>Species</th>
                  <th>Tiger</th>
                  <th>Similarity</th>
                  <th>Decision</th>
                  <th>Alerts</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.image_id}>
                    <td className="font-mono text-xs">{r.image_id}</td>
                    <td>
                      {r.status === 'no_tiger_detected'
                        ? 'No tiger detected'
                        : r.status === 'inference_unavailable'
                          ? 'Tiger detector unavailable'
                        : r.status === 'rejected'
                          ? 'Rejected'
                          : titleCase(r.status)}
                    </td>
                    <td>
                      {r.status === 'no_tiger_detected' ||
                      r.status === 'inference_unavailable' ||
                      r.status === 'rejected'
                        ? '—'
                        : titleCase(r.species ?? null)}
                    </td>
                    <td>
                      {r.tiger_code ? (
                        <Link to={`/tigers/${r.tiger_code}`} className="hover:text-tiger-700">
                          <Badge variant="tiger">{r.tiger_code}</Badge>
                        </Link>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td>{r.similarity != null ? r.similarity.toFixed(3) : '—'}</td>
                    <td>
                      {r.decision === 'high_confidence_match'
                        ? 'High confidence match'
                        : r.decision === 'review' || r.decision === 'human_review'
                          ? `Possible match${r.candidate_tiger ? `: ${r.candidate_tiger}` : ''}`
                          : r.decision === 'new_tiger' || r.decision === 'new_individual'
                            ? 'New tiger'
                            : titleCase(r.decision)}
                    </td>
                    <td>{r.alerts_created}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex gap-3 mt-5">
            <Link to="/observations" className="btn-secondary">
              View detections
            </Link>
            <Link to="/map" className="btn-secondary">
              Open map
            </Link>
            <Link to="/reviews" className="btn-secondary">
              Review queue
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
