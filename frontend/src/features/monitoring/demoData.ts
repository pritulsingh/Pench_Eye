/**
 * Deterministic demo dataset for the camera-trap monitoring system.
 *
 * Everything here is SIMULATED. The same cameras, tigers, territories and
 * seeded detections appear on every reload (seeded PRNG), so the map is stable
 * and screenshot-friendly. The detection generation path is the same one a
 * real Tiger Re-ID pipeline would feed, so this module can later be replaced by
 * a backend service without touching the map.
 */

import {
  destinationPoint,
  generateEstimatedDetectionLocation,
  generateTerritoryPolygon,
  hashString,
  mulberry32,
  polygonCentroid,
  type LatLng,
} from './geo';
import {
  CAMERA_DETECTION_RADIUS_KM,
  DEMO_SEED,
} from './config';
import type {
  CameraTrap,
  Detection,
  MovementPoint,
  Territory,
  TrackedTiger,
  TigerSex,
} from './types';

// ── Camera network ────────────────────────────────────────────────────────
// 24 fixed camera traps distributed across the reserve footprint. Positions are
// hand-placed inside the reserve boundary (see backend geo.py) so nothing lands
// outside the map. Coordinates are [lat, lng].
interface RawCamera {
  id: string;
  name: string;
  lat: number;
  lng: number;
  status: CameraTrap['status'];
}

const RAW_CAMERAS: RawCamera[] = [
  { id: 'C01', name: 'Totladoh Reservoir North', lat: 21.762, lng: 79.288, status: 'active' },
  { id: 'C02', name: 'Totladoh Shoreline', lat: 21.748, lng: 79.305, status: 'active' },
  { id: 'C03', name: 'Karmajhiri Junction', lat: 21.736, lng: 79.322, status: 'active' },
  { id: 'C04', name: 'Alikatta Grassland', lat: 21.718, lng: 79.276, status: 'active' },
  { id: 'C05', name: 'Bodhanala Waterhole', lat: 21.702, lng: 79.254, status: 'active' },
  { id: 'C06', name: 'Piyorthadi Nala', lat: 21.688, lng: 79.308, status: 'active' },
  { id: 'C07', name: 'Chhindimatta Riverbed', lat: 21.674, lng: 79.232, status: 'active' },
  { id: 'C08', name: 'Jamtara Bamboo Trail', lat: 21.774, lng: 79.246, status: 'active' },
  { id: 'C09', name: 'Rukhad Buffer Ridge', lat: 21.822, lng: 79.348, status: 'active' },
  { id: 'C10', name: 'Kurai Corridor North', lat: 21.834, lng: 79.288, status: 'maintenance' },
  { id: 'C11', name: 'Turia Waterhole 4', lat: 21.618, lng: 79.268, status: 'active' },
  { id: 'C12', name: 'Teliya Buffer Corridor', lat: 21.606, lng: 79.188, status: 'active' },
  { id: 'C13', name: 'Eastern Corridor Neck', lat: 21.702, lng: 79.412, status: 'active' },
  { id: 'C14', name: 'Sillari Eco-zone Border', lat: 21.602, lng: 79.118, status: 'offline' },
  { id: 'C15', name: 'Khawasa Village Edge', lat: 21.612, lng: 79.134, status: 'active' },
  { id: 'C16', name: 'Sitaghat Ridge Pass', lat: 21.728, lng: 79.356, status: 'active' },
  { id: 'C17', name: 'Pyorthadi East Nala', lat: 21.694, lng: 79.352, status: 'active' },
  { id: 'C18', name: 'Kumbhapani Meadow', lat: 21.756, lng: 79.324, status: 'active' },
  { id: 'C19', name: 'Rukhad West Track', lat: 21.808, lng: 79.256, status: 'active' },
  { id: 'C20', name: 'Chhedia Plateau', lat: 21.664, lng: 79.298, status: 'active' },
  { id: 'C21', name: 'Gumtara Stream', lat: 21.652, lng: 79.212, status: 'active' },
  { id: 'C22', name: 'Wagholi Corridor', lat: 21.636, lng: 79.336, status: 'active' },
  { id: 'C23', name: 'Silari South Ridge', lat: 21.588, lng: 79.164, status: 'active' },
  { id: 'C24', name: 'Naharpani Waterhole', lat: 21.746, lng: 79.208, status: 'active' },
];

// ── Tiger roster ──────────────────────────────────────────────────────────
// Exactly 12 tigers. `homeCameras` seeds where each tiger is usually detected;
// territory centres are derived from the mean of those cameras. Pairs are
// arranged so some produce genuine 5 km proximity and some overlap territories.
interface RawTiger {
  id: string;
  name: string;
  sex: TigerSex;
  ageClass: TrackedTiger['ageClass'];
  status: TrackedTiger['status'];
  homeCameras: string[];
  territoryRadiusKm: number;
}

const RAW_TIGERS: RawTiger[] = [
  { id: 'T-01', name: 'Collarwali Legacy', sex: 'female', ageClass: 'adult', status: 'active', homeCameras: ['C01', 'C02', 'C18'], territoryRadiusKm: 3.2 },
  { id: 'T-02', name: 'Totladoh Male', sex: 'male', ageClass: 'adult', status: 'active', homeCameras: ['C02', 'C03', 'C16'], territoryRadiusKm: 4.0 },
  { id: 'T-03', name: 'Rukhad Tigress', sex: 'female', ageClass: 'adult', status: 'active', homeCameras: ['C09', 'C10', 'C19'], territoryRadiusKm: 3.6 },
  { id: 'T-04', name: 'Alikatta Male', sex: 'male', ageClass: 'adult', status: 'active', homeCameras: ['C04', 'C05', 'C20'], territoryRadiusKm: 3.8 },
  { id: 'T-05', name: 'Bodhanala Female', sex: 'female', ageClass: 'adult', status: 'active', homeCameras: ['C05', 'C07', 'C24'], territoryRadiusKm: 3.2 },
  { id: 'T-06', name: 'Chhindimatta Male', sex: 'male', ageClass: 'sub_adult', status: 'active', homeCameras: ['C07', 'C21', 'C24'], territoryRadiusKm: 3.0 },
  { id: 'T-07', name: 'Piyorthadi Male', sex: 'male', ageClass: 'adult', status: 'active', homeCameras: ['C06', 'C05', 'C20'], territoryRadiusKm: 3.6 },
  { id: 'T-08', name: 'Turia Female', sex: 'female', ageClass: 'adult', status: 'active', homeCameras: ['C11', 'C12', 'C21'], territoryRadiusKm: 3.4 },
  { id: 'T-09', name: 'Corridor Male', sex: 'male', ageClass: 'sub_adult', status: 'active', homeCameras: ['C13', 'C17', 'C16'], territoryRadiusKm: 3.4 },
  { id: 'T-10', name: 'Sitaghat Female', sex: 'female', ageClass: 'adult', status: 'active', homeCameras: ['C16', 'C17', 'C18'], territoryRadiusKm: 3.2 },
  { id: 'T-11', name: 'Sillari Tigress', sex: 'female', ageClass: 'adult', status: 'active', homeCameras: ['C14', 'C15', 'C23'], territoryRadiusKm: 3.0 },
  { id: 'T-12', name: 'Wagholi Male', sex: 'male', ageClass: 'sub_adult', status: 'inactive', homeCameras: ['C22', 'C06', 'C20'], territoryRadiusKm: 3.4 },
];

const cameraById = new Map(RAW_CAMERAS.map((c) => [c.id, c]));

// ── Tiger image dataset ───────────────────────────────────────────────────
// Reference + gallery images live in `public/tigers/tiger_XX/`, sourced from
// the project's `implement/` folder. Tiger "T-04" maps to folder "tiger_04".
// The same identity always reuses the same images (never mixed between tigers).
const TIGER_IMAGE_VARIANTS = [
  'original.png',
  'crop90.png',
  'crop80.png',
  'crop70.png',
  'bright_up.png',
  'contrast_up.png',
];

function tigerImageDir(tigerId: string): string {
  const num = tigerId.slice(2); // "T-04" → "04"
  return `/tigers/tiger_${num}`;
}

function tigerGallery(tigerId: string): string[] {
  const dir = tigerImageDir(tigerId);
  return TIGER_IMAGE_VARIANTS.map((v) => `${dir}/${v}`);
}

/** Reference (identity) image for a tiger — the first gallery entry. */
export function tigerReferenceImage(tigerId: string): string {
  return `${tigerImageDir(tigerId)}/original.png`;
}

/** A deterministic "detection" image for a tiger, rotated by detection index. */
export function tigerDetectionImage(tigerId: string, index: number): string {
  const gallery = tigerGallery(tigerId);
  return gallery[index % gallery.length];
}


function meanCenter(cameraIds: string[]): LatLng {
  const pts = cameraIds.map((id) => cameraById.get(id)!).filter(Boolean);
  const lat = pts.reduce((s, c) => s + c.lat, 0) / pts.length;
  const lng = pts.reduce((s, c) => s + c.lng, 0) / pts.length;
  return [lat, lng];
}

const MS_PER_HOUR = 3600_000;

/**
 * Build the full deterministic snapshot: cameras, tigers, territories and a
 * seeded detection history that drives movement paths and estimated positions.
 */
export function buildDemoData(now: number = Date.UTC(2026, 7, 17, 12, 0, 0)): {
  cameras: CameraTrap[];
  tigers: TrackedTiger[];
  territories: Territory[];
  detections: Detection[];
} {
  const territories: Territory[] = [];
  const tigers: TrackedTiger[] = [];
  const detections: Detection[] = [];
  const cameraDetectedTigers = new Map<string, Set<string>>();
  const cameraLastDetection = new Map<string, number>();

  RAW_TIGERS.forEach((raw, tigerIndex) => {
    const center = meanCenter(raw.homeCameras);
    const seed = hashString(raw.id) ^ DEMO_SEED;
    const ring = generateTerritoryPolygon(center, raw.territoryRadiusKm, seed);
    const centroid = polygonCentroid(ring);
    const territoryId = `TR-${raw.id.slice(2)}`;

    territories.push({
      id: territoryId,
      tigerId: raw.id,
      centerLatitude: centroid[0],
      centerLongitude: centroid[1],
      ring,
      areaLabelKm2: Math.round(Math.PI * raw.territoryRadiusKm ** 2),
    });

    // Seeded detection history: 6–9 detections walking through home cameras.
    const rand = mulberry32(seed);
    const detectionCount = 6 + Math.floor(rand() * 4);
    const homeCams = raw.homeCameras;
    const movementHistory: MovementPoint[] = [];
    const detectionIds: string[] = [];
    // Fixed realistic offset distances (km) rotated per detection.
    const offsetChoices = [0.2, 0.7, 1.1, 1.5, 0.5, 0.9];

    // Detections spread over the last ~30 days, oldest first.
    let cursor = now - (detectionCount + 1) * 30 * MS_PER_HOUR;
    for (let d = 0; d < detectionCount; d += 1) {
      const cam = cameraById.get(homeCams[d % homeCams.length])!;
      cursor += (18 + rand() * 40) * MS_PER_HOUR;
      const detSeed = (seed + d * 7919) >>> 0;
      const fixedDist = offsetChoices[d % offsetChoices.length];
      const { location, distanceKm } = generateEstimatedDetectionLocation(
        [cam.lat, cam.lng],
        CAMERA_DETECTION_RADIUS_KM,
        detSeed,
        fixedDist
      );
      const id = `DET-${raw.id}-${String(d + 1).padStart(2, '0')}`;
      const timestamp = new Date(cursor).toISOString();
      const confidence = Number((0.72 + rand() * 0.26).toFixed(3));

      detections.push({
        id,
        tigerId: raw.id,
        cameraId: cam.id,
        timestamp,
        latitude: location[0],
        longitude: location[1],
        confidence,
        imagePath: tigerDetectionImage(raw.id, d),
        source: 'simulated',
        estimatedDistanceFromCameraKm: Number(distanceKm.toFixed(3)),
      });
      detectionIds.push(id);
      movementHistory.push({
        detectionId: id,
        cameraId: cam.id,
        latitude: location[0],
        longitude: location[1],
        timestamp,
      });

      if (!cameraDetectedTigers.has(cam.id)) cameraDetectedTigers.set(cam.id, new Set());
      cameraDetectedTigers.get(cam.id)!.add(raw.id);
      const prev = cameraLastDetection.get(cam.id) ?? 0;
      if (cursor > prev) cameraLastDetection.set(cam.id, cursor);
    }

    const last = movementHistory[movementHistory.length - 1];
    // Some inactive tigers are deliberately stale to exercise gap alerts.
    const currentLocation: LatLng = [last.latitude, last.longitude];

    tigers.push({
      id: raw.id,
      name: raw.name,
      sex: raw.sex,
      ageClass: raw.ageClass,
      status: raw.status,
      territoryId,
      currentLocation,
      lastDetectedCamera: last.cameraId,
      lastDetectionTime: last.timestamp,
      confidence: detections.find((x) => x.id === last.detectionId)?.confidence ?? null,
      referenceImage: tigerReferenceImage(raw.id),
      gallery: tigerGallery(raw.id),
      movementHistory,
      detectionIds,
    });

    void tigerIndex;
  });

  // Nudge a few tiger pairs closer together so proximity conflicts are
  // guaranteed regardless of the seeded walk. This keeps the demo interesting
  // while remaining deterministic. Distances stay under the 5 km radius.
  forceProximity(tigers, 'T-04', 'T-07', 3.2);
  forceProximity(tigers, 'T-05', 'T-06', 2.6);

  const cameras: CameraTrap[] = RAW_CAMERAS.map((c) => {
    const detected = Array.from(cameraDetectedTigers.get(c.id) ?? []);
    const lastMs = cameraLastDetection.get(c.id);
    // Attach each camera to the territory of its most-frequent tiger, if any.
    const owningTiger = tigers.find((t) => t.movementHistory.some((m) => m.cameraId === c.id));
    return {
      id: c.id,
      name: c.name,
      latitude: c.lat,
      longitude: c.lng,
      status: c.status,
      detectionRadiusKm: CAMERA_DETECTION_RADIUS_KM,
      installedAt: new Date(now - (400 + hashString(c.id) % 300) * 24 * MS_PER_HOUR).toISOString(),
      territoryId: owningTiger?.territoryId,
      lastDetection: lastMs ? new Date(lastMs).toISOString() : null,
      detectedTigerIds: detected,
    };
  });

  return { cameras, tigers, territories, detections };
}

/**
 * Move tiger B so it sits `targetKm` from tiger A along the bearing between
 * their current positions. Deterministic and always < conflict radius.
 */
function forceProximity(tigers: TrackedTiger[], idA: string, idB: string, targetKm: number): void {
  const a = tigers.find((t) => t.id === idA);
  const b = tigers.find((t) => t.id === idB);
  if (!a || !b) return;
  const [latA, lngA] = a.currentLocation;
  const [latB, lngB] = b.currentLocation;
  const bearing = (Math.atan2(lngB - lngA, latB - latA) * 180) / Math.PI;
  const moved = destinationPoint([latA, lngA], targetKm, bearing);
  b.currentLocation = moved;
  // Reflect the adjusted position in B's latest movement point too.
  const lastPoint = b.movementHistory[b.movementHistory.length - 1];
  if (lastPoint) {
    lastPoint.latitude = moved[0];
    lastPoint.longitude = moved[1];
  }
}
