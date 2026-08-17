/**
 * Shared monitoring store.
 *
 * `useMonitoring` owns the single source of truth for cameras, tigers,
 * territories, detections and the derived conflict/alert events. Historically
 * only the map consumed it, which meant the rest of the app (Dashboard, Tiger
 * Gallery, Detections, Alerts…) showed stale or duplicated data.
 *
 * This context lifts that one store to the app root so every view derives from
 * the SAME state. A camera upload now updates the map AND every page at once.
 */

import { createContext, useContext, type ReactNode } from 'react';

import { useMonitoring, type MonitoringStore } from './useMonitoring';

const MonitoringContext = createContext<MonitoringStore | null>(null);

export function MonitoringProvider({ children }: { children: ReactNode }) {
  const store = useMonitoring();
  return <MonitoringContext.Provider value={store}>{children}</MonitoringContext.Provider>;
}

/** Access the shared monitoring store. Must be used inside `MonitoringProvider`. */
export function useMonitoringStore(): MonitoringStore {
  const ctx = useContext(MonitoringContext);
  if (!ctx) {
    throw new Error('useMonitoringStore must be used within a MonitoringProvider');
  }
  return ctx;
}
