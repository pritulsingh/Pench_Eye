/**
 * Central configuration for the camera-trap monitoring system.
 *
 * Keep tunable values here (not scattered through components) so behaviour can
 * be adjusted in one place and the future real Re-ID pipeline can read the same
 * constants.
 */

import type { LatLng } from './geo';

/** Proximity radius between two tigers that raises a conflict event. */
export const TIGER_CONFLICT_RADIUS_KM = 5;

/** Default camera detection radius (distinct from the tiger conflict radius). */
export const CAMERA_DETECTION_RADIUS_KM = 2;

/** Time window in which two tigers at the same camera count as a co-detection. */
export const CO_DETECTION_WINDOW_HOURS = 12;

/** A tiger unseen for longer than this raises a "long detection gap" alert. */
export const LONG_DETECTION_GAP_DAYS = 21;

/** Reserve framing — mirrors the backend simulated geography (geo.py). */
export const RESERVE_CENTER: LatLng = [21.7, 79.26];
export const RESERVE_BOUNDS: [LatLng, LatLng] = [
  [21.56, 79.07],
  [21.86, 79.46],
];

/** Deterministic master seed so demo data is identical on every reload. */
export const DEMO_SEED = 20260817;

/** Visual tokens for the monitoring layers (reuse the app's palette where possible). */
export const MONITORING_COLORS = {
  camera: '#0ea5e9',
  cameraCoverage: '#0ea5e9',
  tigerMale: '#f59e0b',
  tigerFemale: '#a855f7',
  territoryFill: '#22c55e',
  territoryBorder: '#16a34a',
  overlap: '#eab308',
  conflict: '#ef4444',
  conflictZone: '#ef4444',
  movementPath: '#f59e0b',
} as const;
