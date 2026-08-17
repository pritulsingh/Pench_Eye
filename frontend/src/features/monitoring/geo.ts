/**
 * Geographic utilities for the camera-trap monitoring system.
 *
 * All distances use proper geodesic maths (Haversine + destination-point on a
 * spherical earth) rather than raw lat/lng differences, so 1 km east and 1 km
 * north describe the same real-world distance regardless of latitude.
 */

export type LatLng = [number, number];

const EARTH_RADIUS_KM = 6371.0088;

const toRad = (deg: number): number => (deg * Math.PI) / 180;
const toDeg = (rad: number): number => (rad * 180) / Math.PI;

/** Great-circle distance between two points in kilometres. */
export function haversineKm(a: LatLng, b: LatLng): number {
  const [lat1, lon1] = a;
  const [lat2, lon2] = b;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(s));
}

/**
 * Point reached by travelling `distanceKm` from `origin` along `bearingDeg`
 * (0 = north, 90 = east). Geodesically correct — used to offset estimated
 * tiger positions from a camera without the lat/lng distortion of naive
 * `lat + random` maths.
 */
export function destinationPoint(origin: LatLng, distanceKm: number, bearingDeg: number): LatLng {
  const angular = distanceKm / EARTH_RADIUS_KM;
  const bearing = toRad(bearingDeg);
  const lat1 = toRad(origin[0]);
  const lon1 = toRad(origin[1]);

  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(angular) + Math.cos(lat1) * Math.sin(angular) * Math.cos(bearing)
  );
  const lon2 =
    lon1 +
    Math.atan2(
      Math.sin(bearing) * Math.sin(angular) * Math.cos(lat1),
      Math.cos(angular) - Math.sin(lat1) * Math.sin(lat2)
    );
  return [toDeg(lat2), toDeg(lon2)];
}

/** Deterministic seedable PRNG (mulberry32). Same seed → same sequence. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Stable integer hash of a string, for deriving per-entity seeds. */
export function hashString(value: string): number {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Estimated wildlife location near a camera.
 *
 * Camera traps only prove an animal passed the lens, so we place the estimate a
 * realistic, seeded distance/bearing away from the camera — inside the camera's
 * detection radius and never exactly on the camera coordinate. Deterministic
 * for a given seed so demo data is stable across reloads, and reusable by a
 * future real detection pipeline.
 */
export function generateEstimatedDetectionLocation(
  cameraLocation: LatLng,
  radiusKm: number,
  seed: number,
  fixedDistanceKm?: number
): { location: LatLng; distanceKm: number } {
  const rand = mulberry32(seed);
  // Bias distance away from 0 so the tiger is never sitting on the camera.
  const minKm = Math.min(0.15, radiusKm * 0.1);
  const distanceKm =
    fixedDistanceKm !== undefined
      ? Math.min(Math.max(fixedDistanceKm, minKm), radiusKm)
      : minKm + rand() * (radiusKm - minKm);
  const bearing = rand() * 360;
  return { location: destinationPoint(cameraLocation, distanceKm, bearing), distanceKm };
}

/**
 * Irregular territory polygon around a home-range centre. Vertices sit at a
 * seeded, wobbling radius so ranges look like natural wildlife territories
 * rather than perfect circles. Returns a closed ring of [lat, lng] points.
 */
export function generateTerritoryPolygon(
  center: LatLng,
  baseRadiusKm: number,
  seed: number,
  vertices = 11,
  irregularity = 0.42
): LatLng[] {
  const rand = mulberry32(seed);
  const ring: LatLng[] = [];
  for (let i = 0; i < vertices; i += 1) {
    const angle = (360 / vertices) * i + (rand() - 0.5) * (360 / vertices) * 0.6;
    const radius = baseRadiusKm * (1 - irregularity / 2 + rand() * irregularity);
    ring.push(destinationPoint(center, radius, angle));
  }
  ring.push(ring[0]);
  return ring;
}

/** Ray-casting point-in-polygon test. Ring is [lat, lng] pairs. */
export function pointInPolygon(point: LatLng, ring: LatLng[]): boolean {
  const [lat, lng] = point;
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [latI, lngI] = ring[i];
    const [latJ, lngJ] = ring[j];
    const intersects =
      lngI > lng !== lngJ > lng &&
      lat < ((latJ - latI) * (lng - lngI)) / (lngJ - lngI) + latI;
    if (intersects) inside = !inside;
  }
  return inside;
}

function segmentsIntersect(p1: LatLng, p2: LatLng, p3: LatLng, p4: LatLng): boolean {
  const d = (a: LatLng, b: LatLng, c: LatLng) =>
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const d1 = d(p3, p4, p1);
  const d2 = d(p3, p4, p2);
  const d3 = d(p1, p2, p3);
  const d4 = d(p1, p2, p4);
  return ((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0));
}

/** Whether two polygon rings overlap (edge crossing or full containment). */
export function polygonsIntersect(ringA: LatLng[], ringB: LatLng[]): boolean {
  for (let i = 0; i < ringA.length - 1; i += 1) {
    for (let j = 0; j < ringB.length - 1; j += 1) {
      if (segmentsIntersect(ringA[i], ringA[i + 1], ringB[j], ringB[j + 1])) return true;
    }
  }
  return pointInPolygon(ringA[0], ringB) || pointInPolygon(ringB[0], ringA);
}

/** Centroid of a ring, used to anchor overlap labels/zones. */
export function polygonCentroid(ring: LatLng[]): LatLng {
  let lat = 0;
  let lng = 0;
  const count = ring.length - 1; // last point repeats the first
  for (let i = 0; i < count; i += 1) {
    lat += ring[i][0];
    lng += ring[i][1];
  }
  return [lat / count, lng / count];
}

/** Midpoint between two coordinates (planar approximation, fine at reserve scale). */
export function midpoint(a: LatLng, b: LatLng): LatLng {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}
