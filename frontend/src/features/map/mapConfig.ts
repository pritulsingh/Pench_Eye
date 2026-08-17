import L from 'leaflet';

// Leaflet's default icon URLs break under bundlers; point them at the CDN copies.
delete (L.Icon.Default.prototype as unknown as { _getIconUrl?: unknown })._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

export const SATELLITE_TILES =
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
export const SATELLITE_ATTRIBUTION =
  'Tiles &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community';

// Transparent OSM labels/roads/places over satellite imagery. This layer is
// what makes real villages, hotels, gates, roads and nearby services visible.
export const PLACES_TILES =
  'https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png';
export const PLACES_ATTRIBUTION = '&copy; OpenStreetMap contributors &copy; CARTO';

export const MARKER_COLORS: Record<string, string> = {
  active: '#22c55e',
  recent_detection: '#f97316',
  warning: '#eab308',
  offline: '#ef4444',
  maintenance: '#38bdf8',
};

export const MARKER_LABELS: Record<string, string> = {
  active: 'Active',
  recent_detection: 'Detection <24 h',
  warning: 'Warning',
  offline: 'Offline',
  maintenance: 'Maintenance',
};

/** Camera marker: coloured ring, pulsing when a detection is fresh. */
export function cameraIcon(state: string): L.DivIcon {
  const color = MARKER_COLORS[state] ?? MARKER_COLORS.active;
  const pulse = state === 'recent_detection';
  return L.divIcon({
    className: 'pench-marker',
    html: `<span class="pench-camera-dot${pulse ? ' pench-pulse' : ''}" style="--marker-color:${color}"></span>`,
    iconSize: [11, 11],
    iconAnchor: [5.5, 5.5],
    popupAnchor: [0, -7],
  });
}

/** Sighting marker: diamond for tigers, small dot for other wildlife. */
export function sightingIcon(isTiger: boolean, highlighted = false): L.DivIcon {
  const color = isTiger ? '#f59e0b' : '#94a3b8';
  const size = isTiger ? 14 : 10;
  return L.divIcon({
    className: 'pench-marker',
    html: `<span class="pench-sighting${isTiger ? ' pench-sighting-tiger' : ''}${
      highlighted ? ' pench-sighting-active' : ''
    }" style="--marker-color:${color};width:${size}px;height:${size}px"></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2],
  });
}

export function gateIcon(): L.DivIcon {
  return L.divIcon({
    className: 'pench-marker',
    html: '<span class="pench-gate"></span>',
    iconSize: [12, 12],
    iconAnchor: [6, 6],
    popupAnchor: [0, -6],
  });
}

/** GeoJSON polygons use [lon, lat]; Leaflet wants [lat, lon]. */
export function geometryToLatLngs(
  geometry: { type: string; coordinates: number[][][] } | null
): [number, number][][] {
  if (!geometry || geometry.type !== 'Polygon') return [];
  return geometry.coordinates.map((ring) =>
    ring.map(([lon, lat]) => [lat, lon] as [number, number])
  );
}

type LatLng = [number, number];

/**
 * Sample a quadratic Bézier curve between two points, bowing the midpoint
 * perpendicular to the leg so movement legs render as gentle curves rather
 * than dead-straight lines. `bend` sets how far (as a fraction of leg length)
 * the control point deviates; `sign` alternates the bow direction per leg.
 */
export function curvedPath(from: LatLng, to: LatLng, bend = 0.18, sign = 1, steps = 24): LatLng[] {
  const [lat1, lon1] = from;
  const [lat2, lon2] = to;
  const midLat = (lat1 + lat2) / 2;
  const midLon = (lon1 + lon2) / 2;
  const dLat = lat2 - lat1;
  const dLon = lon2 - lon1;
  // Perpendicular vector (rotate the leg 90°) offset from the midpoint.
  const ctrlLat = midLat + sign * bend * -dLon;
  const ctrlLon = midLon + sign * bend * dLat;
  const points: LatLng[] = [];
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const mt = 1 - t;
    const lat = mt * mt * lat1 + 2 * mt * t * ctrlLat + t * t * lat2;
    const lon = mt * mt * lon1 + 2 * mt * t * ctrlLon + t * t * lon2;
    points.push([lat, lon]);
  }
  return points;
}

/** Andrew's monotone-chain convex hull; returns a closed-able ring of points. */
export function convexHull(points: LatLng[]): LatLng[] {
  const uniq = Array.from(new Set(points.map((p) => `${p[0]},${p[1]}`))).map(
    (s) => s.split(',').map(Number) as LatLng
  );
  if (uniq.length < 3) return uniq;
  const sorted = uniq.slice().sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  const cross = (o: LatLng, a: LatLng, b: LatLng) =>
    (a[1] - o[1]) * (b[0] - o[0]) - (a[0] - o[0]) * (b[1] - o[1]);
  const lower: LatLng[] = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper: LatLng[] = [];
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}
