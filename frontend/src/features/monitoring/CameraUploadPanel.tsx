import { useRef, useState } from 'react';
import { Upload, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';

import type { UploadResult } from './useMonitoring';

/**
 * Manual detection upload — the ONLY entry point for manual tiger images.
 * Rendered inside a camera detail panel, never as a global action.
 *
 * The panel is agnostic to how identification happens: it simply calls
 * `onUpload(cameraId, file)` and renders the result. Swapping the mock Re-ID
 * service for the real one requires no change here.
 */
export function CameraUploadPanel({
  cameraId,
  onUpload,
}: {
  cameraId: string;
  onUpload: (cameraId: string, image: File) => Promise<UploadResult>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<'idle' | 'processing' | 'done' | 'error'>('idle');
  const [message, setMessage] = useState<string>('');
  const [fileName, setFileName] = useState<string>('');

  const handleFile = async (file: File | undefined) => {
    if (!file) return;
    setFileName(file.name);
    setStatus('processing');
    setMessage('Running detection → identification → estimated location…');
    try {
      const result = await onUpload(cameraId, file);
      setStatus('done');
      setMessage(result.message);
    } catch (err) {
      setStatus('error');
      setMessage(err instanceof Error ? err.message : 'Upload failed.');
    } finally {
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  return (
    <div className="mt-2 border-t border-border/60 pt-2">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={status === 'processing'}
        className="btn-primary w-full flex items-center justify-center gap-2 !py-1.5 text-xs disabled:opacity-60"
      >
        {status === 'processing' ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          <Upload className="w-3.5 h-3.5" />
        )}
        Upload Detection
      </button>

      {status !== 'idle' && (
        <div
          className={`mt-2 text-[11px] leading-snug flex items-start gap-1.5 ${
            status === 'error'
              ? 'text-red-600'
              : status === 'done'
                ? 'text-green-700'
                : 'text-muted-foreground'
          }`}
        >
          {status === 'done' && <CheckCircle2 className="w-3.5 h-3.5 shrink-0 mt-px" />}
          {status === 'error' && <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-px" />}
          <span>
            {fileName && <span className="font-medium">{fileName}: </span>}
            {message}
          </span>
        </div>
      )}
    </div>
  );
}
