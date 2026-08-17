/**
 * Monitoring store + detection pipeline hook.
 *
 * Owns the mutable snapshot (cameras, tigers, territories, detections) and
 * exposes the canonical detection pipeline:
 *
 *   Camera → Detection → Tiger → Estimated Location → Movement History
 *          → Conflict Analysis → Alerts → Map
 *
 * Derived events (conflicts, overlaps, co-detections, alerts) are recomputed
 * with `useMemo` from the current snapshot, so any new detection automatically
 * refreshes proximity zones, movement paths and alerts.
 */

import { useCallback, useMemo, useRef, useState } from 'react';

import {
  computeAlerts,
  computeCameraCoDetections,
  computeProximityConflicts,
  computeTerritoryOverlaps,
} from './analysis';
import { buildDemoData } from './demoData';
import { CAMERA_DETECTION_RADIUS_KM } from './config';
import { generateEstimatedDetectionLocation, hashString } from './geo';
import {
  MockTigerIdentificationService,
  type TigerIdentificationService,
} from './identification';
import type {
  CameraTrap,
  Detection,
  MonitoringSnapshot,
  TrackedTiger,
} from './types';

export interface UploadResult {
  detection: Detection;
  tigerId: string | null;
  confidence: number;
  message: string;
}

export interface MonitoringStore extends MonitoringSnapshot {
  uploadDetection: (cameraId: string, image: File) => Promise<UploadResult>;
  identificationKind: 'mock' | 'real';
}

export function useMonitoring(): MonitoringStore {
  // Base seeded dataset is built once; live uploads mutate copies in state.
  const base = useMemo(() => buildDemoData(), []);

  const [cameras, setCameras] = useState<CameraTrap[]>(base.cameras);
  const [tigers, setTigers] = useState<TrackedTiger[]>(base.tigers);
  const [detections, setDetections] = useState<Detection[]>(base.detections);

  // Camera → likely tigers hint (their home cameras) improves the mock's guess.
  const identifier = useRef<TigerIdentificationService>(
    new MockTigerIdentificationService(
      new Map(base.cameras.map((c) => [c.id, c.detectedTigerIds]))
    )
  );

  const knownTigerIds = useMemo(() => tigers.map((t) => t.id), [tigers]);

  const uploadDetection = useCallback(
    async (cameraId: string, image: File): Promise<UploadResult> => {
      const camera = base.cameras.find((c) => c.id === cameraId);
      if (!camera) throw new Error(`Unknown camera ${cameraId}`);

      // 1. Identify the tiger (mock today, real Re-ID later — same interface).
      const result = await identifier.current.identifyTiger(image, cameraId, knownTigerIds);

      // 2. Estimate a location near the camera (never exactly on it).
      const detSeed = hashString(`${image.name}:${Date.now()}:${cameraId}`);
      const { location, distanceKm } = generateEstimatedDetectionLocation(
        [camera.latitude, camera.longitude],
        camera.detectionRadiusKm || CAMERA_DETECTION_RADIUS_KM,
        detSeed
      );

      const timestamp = new Date().toISOString();
      const detection: Detection = {
        id: `DET-LIVE-${detSeed.toString(36)}`,
        tigerId: result.tigerId,
        cameraId,
        timestamp,
        latitude: location[0],
        longitude: location[1],
        confidence: result.confidence,
        imagePath: null,
        source: result.source, // 'manual' from mock; 'ai' from real Re-ID later
        estimatedDistanceFromCameraKm: Number(distanceKm.toFixed(3)),
      };

      // 3. Store detection.
      setDetections((prev) => [...prev, detection]);

      // 4. Update the identified tiger's current location + movement history.
      if (result.tigerId) {
        setTigers((prev) =>
          prev.map((t) =>
            t.id === result.tigerId
              ? {
                  ...t,
                  currentLocation: location,
                  lastDetectedCamera: cameraId,
                  lastDetectionTime: timestamp,
                  confidence: result.confidence,
                  detectionIds: [...t.detectionIds, detection.id],
                  movementHistory: [
                    ...t.movementHistory,
                    {
                      detectionId: detection.id,
                      cameraId,
                      latitude: location[0],
                      longitude: location[1],
                      timestamp,
                    },
                  ],
                }
              : t
          )
        );
      }

      // 5. Update the camera's detection summary.
      setCameras((prev) =>
        prev.map((c) =>
          c.id === cameraId
            ? {
                ...c,
                lastDetection: timestamp,
                detectedTigerIds: result.tigerId
                  ? Array.from(new Set([...c.detectedTigerIds, result.tigerId]))
                  : c.detectedTigerIds,
              }
            : c
        )
      );

      const message = result.tigerId
        ? `Identified ${result.tigerId} at ${(result.confidence * 100).toFixed(0)}% confidence, ~${distanceKm.toFixed(2)} km from ${cameraId}.`
        : `No confident match — logged as an unidentified individual near ${cameraId}.`;

      return { detection, tigerId: result.tigerId, confidence: result.confidence, message };
    },
    [base.cameras, knownTigerIds]
  );

  const { territories } = base;

  // Derived events — recomputed whenever tigers/detections change.
  const conflicts = useMemo(
    () => computeProximityConflicts(tigers, cameras),
    [tigers, cameras]
  );
  const overlaps = useMemo(() => computeTerritoryOverlaps(territories), [territories]);
  const coDetections = useMemo(
    () => computeCameraCoDetections(cameras, detections),
    [cameras, detections]
  );
  const alerts = useMemo(
    () => computeAlerts(tigers, territories, conflicts, overlaps, coDetections, Date.now()),
    [tigers, territories, conflicts, overlaps, coDetections]
  );

  return {
    cameras,
    tigers,
    territories,
    detections,
    conflicts,
    overlaps,
    coDetections,
    alerts,
    uploadDetection,
    identificationKind: identifier.current.kind,
  };
}
