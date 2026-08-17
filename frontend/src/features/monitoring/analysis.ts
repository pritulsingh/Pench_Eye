/**
 * Derived-event analysis for the monitoring system.
 *
 * These functions are pure: given the current cameras/tigers/territories/
 * detections they compute proximity conflicts, territory overlaps, camera
 * co-detections and alerts. Because they are derived (never hardcoded), the
 * results update automatically whenever a new detection changes tiger state.
 *
 * Territory overlap, tiger proximity and camera co-detection are kept as three
 * distinct concepts, as required.
 */

import {
  haversineKm,
  midpoint,
  pointInPolygon,
  polygonsIntersect,
  type LatLng,
} from './geo';
import {
  CO_DETECTION_WINDOW_HOURS,
  LONG_DETECTION_GAP_DAYS,
  TIGER_CONFLICT_RADIUS_KM,
} from './config';
import type {
  CameraCoDetection,
  CameraTrap,
  Detection,
  MonitoringAlert,
  ProximityConflict,
  Territory,
  TerritoryOverlap,
  TrackedTiger,
} from './types';

/** Tiger pairs whose current estimated positions are within the conflict radius. */
export function computeProximityConflicts(
  tigers: TrackedTiger[],
  cameras: CameraTrap[]
): ProximityConflict[] {
  const conflicts: ProximityConflict[] = [];
  for (let i = 0; i < tigers.length; i += 1) {
    for (let j = i + 1; j < tigers.length; j += 1) {
      const a = tigers[i];
      const b = tigers[j];
      const distanceKm = haversineKm(a.currentLocation, b.currentLocation);
      if (distanceKm <= TIGER_CONFLICT_RADIUS_KM) {
        const mid = midpoint(a.currentLocation, b.currentLocation);
        const nearbyCameraIds = cameras
          .filter((c) => haversineKm([c.latitude, c.longitude], mid) <= TIGER_CONFLICT_RADIUS_KM)
          .map((c) => c.id);
        conflicts.push({
          id: `CONF-${a.id}-${b.id}`,
          tigerA: a.id,
          tigerB: b.id,
          distanceKm: Number(distanceKm.toFixed(2)),
          midpoint: mid,
          positionA: a.currentLocation,
          positionB: b.currentLocation,
          nearbyCameraIds,
        });
      }
    }
  }
  return conflicts;
}

/** Territory polygon pairs that intersect (potential territorial overlap). */
export function computeTerritoryOverlaps(territories: Territory[]): TerritoryOverlap[] {
  const overlaps: TerritoryOverlap[] = [];
  for (let i = 0; i < territories.length; i += 1) {
    for (let j = i + 1; j < territories.length; j += 1) {
      const a = territories[i];
      const b = territories[j];
      if (polygonsIntersect(a.ring, b.ring)) {
        const center = midpoint(
          [a.centerLatitude, a.centerLongitude],
          [b.centerLatitude, b.centerLongitude]
        );
        overlaps.push({
          id: `OVL-${a.tigerId}-${b.tigerId}`,
          tigerA: a.tigerId,
          tigerB: b.tigerId,
          center,
        });
      }
    }
  }
  return overlaps;
}

/** Cameras that detected multiple distinct tigers within the co-detection window. */
export function computeCameraCoDetections(
  cameras: CameraTrap[],
  detections: Detection[]
): CameraCoDetection[] {
  const results: CameraCoDetection[] = [];
  const windowMs = CO_DETECTION_WINDOW_HOURS * 3600_000;

  for (const camera of cameras) {
    const camDetections = detections
      .filter((d) => d.cameraId === camera.id && d.tigerId)
      .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp));

    for (let i = 0; i < camDetections.length; i += 1) {
      const windowTigers = new Map<string, string>(); // tigerId -> detectionId
      const t0 = Date.parse(camDetections[i].timestamp);
      for (let j = i; j < camDetections.length; j += 1) {
        if (Date.parse(camDetections[j].timestamp) - t0 > windowMs) break;
        const tid = camDetections[j].tigerId!;
        if (!windowTigers.has(tid)) windowTigers.set(tid, camDetections[j].id);
      }
      if (windowTigers.size >= 2) {
        results.push({
          id: `CODET-${camera.id}-${camDetections[i].id}`,
          cameraId: camera.id,
          tigerIds: Array.from(windowTigers.keys()),
          detectionIds: Array.from(windowTigers.values()),
          windowHours: CO_DETECTION_WINDOW_HOURS,
        });
        break; // one co-detection event per camera is enough for the demo
      }
    }
  }
  return results;
}

function tigerName(tigers: TrackedTiger[], id: string): string {
  return tigers.find((t) => t.id === id)?.name ?? id;
}

/**
 * Build alerts from the derived events plus per-tiger checks (new-territory
 * movement, long detection gaps). All alerts share a common shape so they can
 * flow into any existing alert UI.
 */
export function computeAlerts(
  tigers: TrackedTiger[],
  territories: Territory[],
  conflicts: ProximityConflict[],
  overlaps: TerritoryOverlap[],
  coDetections: CameraCoDetection[],
  now: number
): MonitoringAlert[] {
  const alerts: MonitoringAlert[] = [];
  const territoryByTiger = new Map(territories.map((t) => [t.tigerId, t]));

  for (const c of conflicts) {
    alerts.push({
      id: `ALERT-${c.id}`,
      type: 'tiger_proximity',
      severity: c.distanceKm <= 2 ? 'high' : 'medium',
      tigerIds: [c.tigerA, c.tigerB],
      cameraId: c.nearbyCameraIds[0] ?? null,
      location: c.midpoint,
      distanceKm: c.distanceKm,
      timestamp: new Date(now).toISOString(),
      status: 'open',
      message: `${tigerName(tigers, c.tigerA)} and ${tigerName(tigers, c.tigerB)} are within ${c.distanceKm} km (< ${TIGER_CONFLICT_RADIUS_KM} km).`,
    });
  }

  for (const o of overlaps) {
    alerts.push({
      id: `ALERT-${o.id}`,
      type: 'territory_overlap',
      severity: 'low',
      tigerIds: [o.tigerA, o.tigerB],
      cameraId: null,
      location: o.center,
      distanceKm: null,
      timestamp: new Date(now).toISOString(),
      status: 'open',
      message: `Territories of ${tigerName(tigers, o.tigerA)} and ${tigerName(tigers, o.tigerB)} overlap.`,
    });
  }

  for (const cd of coDetections) {
    alerts.push({
      id: `ALERT-${cd.id}`,
      type: 'multiple_tiger_detection',
      severity: 'medium',
      tigerIds: cd.tigerIds,
      cameraId: cd.cameraId,
      location: null,
      distanceKm: null,
      timestamp: new Date(now).toISOString(),
      status: 'open',
      message: `Multiple tigers (${cd.tigerIds.join(', ')}) detected near ${cd.cameraId} within ${cd.windowHours} h.`,
    });
  }

  for (const tiger of tigers) {
    // New-territory movement: current estimate outside its own territory ring.
    const terr = territoryByTiger.get(tiger.id);
    if (terr && !pointInPolygon(tiger.currentLocation, terr.ring)) {
      alerts.push({
        id: `ALERT-NEWTERR-${tiger.id}`,
        type: 'new_territory_movement',
        severity: 'medium',
        tigerIds: [tiger.id],
        cameraId: tiger.lastDetectedCamera,
        location: tiger.currentLocation,
        distanceKm: null,
        timestamp: new Date(now).toISOString(),
        status: 'open',
        message: `${tiger.name} was detected outside its normal territory.`,
      });
    }

    // Long detection gap.
    if (tiger.lastDetectionTime) {
      const gapDays = (now - Date.parse(tiger.lastDetectionTime)) / 86_400_000;
      if (gapDays >= LONG_DETECTION_GAP_DAYS) {
        alerts.push({
          id: `ALERT-GAP-${tiger.id}`,
          type: 'long_detection_gap',
          severity: gapDays >= LONG_DETECTION_GAP_DAYS * 1.5 ? 'high' : 'medium',
          tigerIds: [tiger.id],
          cameraId: tiger.lastDetectedCamera,
          location: tiger.currentLocation,
          distanceKm: null,
          timestamp: new Date(now).toISOString(),
          status: 'open',
          message: `${tiger.name} has not been detected for ${Math.round(gapDays)} days.`,
        });
      }
    }
  }

  return alerts;
}

/** Nearby tigers within the conflict radius of a given tiger (for detail panels). */
export function nearbyTigers(
  tiger: TrackedTiger,
  tigers: TrackedTiger[]
): Array<{ id: string; name: string; distanceKm: number }> {
  return tigers
    .filter((t) => t.id !== tiger.id)
    .map((t) => ({
      id: t.id,
      name: t.name,
      distanceKm: Number(haversineKm(tiger.currentLocation, t.currentLocation).toFixed(2)),
    }))
    .filter((t) => t.distanceKm <= TIGER_CONFLICT_RADIUS_KM)
    .sort((a, b) => a.distanceKm - b.distanceKm);
}

export function findCameraById(cameras: CameraTrap[], id: string | null): CameraTrap | undefined {
  return id ? cameras.find((c) => c.id === id) : undefined;
}

export type { LatLng };
