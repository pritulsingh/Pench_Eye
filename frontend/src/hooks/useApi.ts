import { useCallback, useEffect, useState } from 'react';
import type { AxiosResponse } from 'axios';

interface State<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/**
 * Small fetch helper: keeps loading/error/data in sync and exposes `reload`.
 * `deps` behaves like a useEffect dependency list.
 */
export function useApi<T>(
  fetcher: () => Promise<AxiosResponse<T>>,
  deps: unknown[] = []
): State<T> & { reload: () => void } {
  const [state, setState] = useState<State<T>>({ data: null, loading: true, error: null });
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let cancelled = false;
    setState((s) => ({ ...s, loading: true, error: null }));

    fetcher()
      .then((res) => {
        if (!cancelled) setState({ data: res.data, loading: false, error: null });
      })
      .catch((err: { userMessage?: string; message?: string }) => {
        if (!cancelled)
          setState({
            data: null,
            loading: false,
            error: err.userMessage || err.message || 'Request failed',
          });
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { ...state, reload };
}
