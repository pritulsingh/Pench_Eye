import { useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';

import { demoApi } from '@/api/client';
import { useApi } from '@/hooks/useApi';

export function DemoWarning() {
  const [dismissed, setDismissed] = useState(false);
  const { data } = useApi(() => demoApi.status());

  if (dismissed || !data?.demo_mode) return null;

  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-2 flex items-center justify-between gap-4">
      <div className="flex items-center gap-2 text-amber-800 text-sm font-medium">
        <AlertTriangle className="w-4 h-4 shrink-0" />
        <span>
          DEMO MODE ({data.ml_mode}, {data.model_version}) — detections, tiger identities and
          reserve boundaries are simulated. Not for conservation decisions.
        </span>
      </div>
      <button
        onClick={() => setDismissed(true)}
        aria-label="Dismiss demo mode notice"
        className="text-amber-700/70 hover:text-amber-900 transition-colors shrink-0"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
