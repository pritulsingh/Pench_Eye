import type { LatLng } from './geo';

/** Fixed camera-trap station. Positions never move. */
export interface CameraTrap {
  id: string;                 // e.g. "C01"
  name: string;
  latitude: number;
  longitude: number;
  status: 'active' | 'maintenance' | 'offline';
  detectionRadiusKm: number;
  installedAt: string;        // ISO
  territoryId?: string;
  lastDetection: string | null;
  detectedTigerIds: string[];
}

export type TigerSex = 'male' | 'female' | 'unknown';
export type TigerAgeClass = 'cub' | 'sub_adult' | 'adult';
export type TigerStatus = 'active' | 'inactive' | 'unknown';

/** A single point in a tiger's movement history. */
export interface MovementPoint {
  detectionId: string;
  cameraId: string;
  latitude: number;
  longitude: number;
  timestamp: string;
}

export interface TrackedTiger {
  id: string;                 // e.g. "T-01"
  name: string;
  sex: TigerSex;
  ageClass: TigerAgeClass;
  status: TigerStatus;
  territoryId: string;
  currentLocation: LatLng;    // estimated current position (never a camera coord)
  lastDetectedCamera: string | null;
  lastDetectionTime: string | null;
  confidence: number | null;  // last detection confidence
  referenceImage?: string;
  movementHistory: MovementPoint[];
  detectionIds: string[];
}

export interface Territory {
  id: string;                 // e.g. "TR-01"
  tigerId: string;
  centerLatitude: number;
  centerLongitude: number;
  ring: LatLng[];             // closed polygon of [lat, lng]
  areaLabelKm2: number;
}

export type DetectionSource = 'manual' | 'simulated' | 'ai';

/**
 * A detection links a Camera → Tiger → estimated location. `source` lets the
 * same pipeline serve seeded demo data, manual uploads, and (later) real
 * Tiger Re-ID output without the UI caring which produced it.
 */
export interface Detection {
  id: string;
  tigerId: string | null;     // null when identification failed / new individual
  cameraId: string;
  timestamp: string;
  latitude: number;
  longitude: number;
  confidence: number;
  imagePath: string | null;
  source: DetectionSource;
  estimatedDistanceFromCameraKm: number;
}

export type AlertType =
  | 'tiger_proximity'
  | 'multiple_tiger_detection'
  | 'territory_overlap'
  | 'new_territory_movement'
  | 'long_detection_gap'
  | 'high_risk_conflict_zone';

export type AlertSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';
export type AlertStatus = 'open' | 'acknowledged' | 'resolved';

export interface MonitoringAlert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  tigerIds: string[];
  cameraId: string | null;
  location: LatLng | null;
  distanceKm: number | null;
  timestamp: string;
  status: AlertStatus;
  message: string;
}

/** Two tigers whose current estimated positions are within the conflict radius. */
export interface ProximityConflict {
  id: string;
  tigerA: string;
  tigerB: string;
  distanceKm: number;
  midpoint: LatLng;
  positionA: LatLng;
  positionB: LatLng;
  nearbyCameraIds: string[];
}

/** Two territory polygons that intersect (potential territorial overlap). */
export interface TerritoryOverlap {
  id: string;
  tigerA: string;
  tigerB: string;
  center: LatLng;
}

/** Same camera detecting multiple tigers within a time window. */
export interface CameraCoDetection {
  id: string;
  cameraId: string;
  tigerIds: string[];
  detectionIds: string[];
  windowHours: number;
}

export interface MonitoringSnapshot {
  cameras: CameraTrap[];
  tigers: TrackedTiger[];
  territories: Territory[];
  detections: Detection[];
  conflicts: ProximityConflict[];
  overlaps: TerritoryOverlap[];
  coDetections: CameraCoDetection[];
  alerts: MonitoringAlert[];
}
