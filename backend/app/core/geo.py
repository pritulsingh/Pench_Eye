"""
Pench Tiger Reserve — SIMULATED reference geography.

IMPORTANT: These polygons and points are an approximate, hand-authored
DEMONSTRATION dataset. They are inspired by the general location of Pench
Tiger Reserve (Madhya Pradesh / Maharashtra, ~21.6-21.8 N, 79.1-79.4 E)
but are NOT surveyed boundaries and must not be used for any operational,
legal, or conservation purpose.

Replace `PENCH_ZONES` with authoritative GeoJSON (Forest Department / WII
shapefiles converted to GeoJSON) and set GEO_DATA_SOURCE=official to switch
the UI labelling.
"""
from typing import Any, Dict, List

GEO_DATA_DISCLAIMER = (
    "Simulated geography — approximate demo boundaries for Pench Tiger Reserve. "
    "Not surveyed data. Do not use for conservation or legal decisions."
)

RESERVE_CENTER = (21.7000, 79.2600)
RESERVE_BOUNDS = [[21.5600, 79.0700], [21.8600, 79.4600]]


def _ring(coords: List[List[float]]) -> Dict[str, Any]:
    """Build a GeoJSON Polygon geometry from [lon, lat] pairs."""
    closed = coords + [coords[0]] if coords[0] != coords[-1] else coords
    return {"type": "Polygon", "coordinates": [closed]}


# Outer reserve boundary (simulated)
RESERVE_BOUNDARY = _ring([
    [79.085, 21.585], [79.150, 21.565], [79.245, 21.570], [79.330, 21.590],
    [79.405, 21.625], [79.445, 21.680], [79.440, 21.750], [79.395, 21.805],
    [79.315, 21.845], [79.220, 21.850], [79.140, 21.825], [79.090, 21.770],
    [79.070, 21.690],
])

PENCH_ZONES: List[Dict[str, Any]] = [
    {
        "zone_code": "PTR-BOUNDARY",
        "name": "Pench Tiger Reserve (boundary)",
        "zone_type": "reserve_boundary",
        "description": "Simulated outer boundary of the reserve used for map framing.",
        "center_latitude": 21.700,
        "center_longitude": 79.260,
        "area_km2": 757.0,
        "style_color": "#f59e0b",
        "geometry_json": RESERVE_BOUNDARY,
    },
    {
        "zone_code": "PTR-CORE",
        "name": "Core Critical Tiger Habitat",
        "zone_type": "core",
        "description": "Simulated core zone — highest protection, no tourism outside designated routes.",
        "center_latitude": 21.715,
        "center_longitude": 79.265,
        "area_km2": 411.0,
        "style_color": "#22c55e",
        "geometry_json": _ring([
            [79.175, 21.640], [79.265, 21.625], [79.350, 21.655], [79.385, 21.715],
            [79.355, 21.785], [79.270, 21.805], [79.190, 21.780], [79.155, 21.710],
        ]),
    },
    {
        "zone_code": "PTR-BUFFER-N",
        "name": "Northern Buffer (Rukhad side)",
        "zone_type": "buffer",
        "description": "Simulated northern buffer zone with corridor connectivity.",
        "center_latitude": 21.805,
        "center_longitude": 79.330,
        "area_km2": 168.0,
        "style_color": "#38bdf8",
        "geometry_json": _ring([
            [79.230, 21.800], [79.320, 21.795], [79.400, 21.775], [79.415, 21.825],
            [79.330, 21.855], [79.235, 21.850],
        ]),
    },
    {
        "zone_code": "PTR-BUFFER-S",
        "name": "Southern Buffer (Turia–Sillari side)",
        "zone_type": "buffer",
        "description": "Simulated southern buffer zone adjacent to tourism gates.",
        "center_latitude": 21.615,
        "center_longitude": 79.215,
        "area_km2": 178.0,
        "style_color": "#38bdf8",
        "geometry_json": _ring([
            [79.100, 21.590], [79.200, 21.575], [79.300, 21.588], [79.330, 21.630],
            [79.230, 21.645], [79.140, 21.640], [79.095, 21.620],
        ]),
    },
    {
        "zone_code": "PTR-CORRIDOR-E",
        "name": "Eastern Wildlife Corridor",
        "zone_type": "corridor",
        "description": "Simulated corridor used to illustrate inter-zone tiger movement.",
        "center_latitude": 21.700,
        "center_longitude": 79.410,
        "area_km2": 62.0,
        "style_color": "#a855f7",
        "geometry_json": _ring([
            [79.380, 21.660], [79.440, 21.670], [79.448, 21.735], [79.392, 21.742],
        ]),
    },
    {
        "zone_code": "PTR-VILLAGE-SW",
        "name": "South-West Village Interface",
        "zone_type": "village_adjacent",
        "description": "Simulated village-adjacent belt where human-wildlife conflict alerts matter most.",
        "center_latitude": 21.600,
        "center_longitude": 79.120,
        "area_km2": 44.0,
        "style_color": "#ef4444",
        "geometry_json": _ring([
            [79.078, 21.585], [79.150, 21.578], [79.160, 21.618], [79.088, 21.626],
        ]),
    },
]

# Simulated tourism / entry gates
PENCH_GATES: List[Dict[str, Any]] = [
    {"code": "GATE-TURIA", "name": "Turia Gate", "latitude": 21.6480, "longitude": 79.2960, "gate_type": "tourism"},
    {"code": "GATE-KARMAJHIRI", "name": "Karmajhiri Gate", "latitude": 21.7420, "longitude": 79.3120, "gate_type": "tourism"},
    {"code": "GATE-JAMTARA", "name": "Jamtara Gate", "latitude": 21.7850, "longitude": 79.2380, "gate_type": "tourism"},
    {"code": "GATE-SILLARI", "name": "Sillari Gate", "latitude": 21.5980, "longitude": 79.1520, "gate_type": "tourism"},
    {"code": "GATE-RUKHAD", "name": "Rukhad Buffer Gate", "latitude": 21.8280, "longitude": 79.3660, "gate_type": "buffer"},
]

# Camera network layout: (camera_id, name, zone enum value, zone_code, lat, lon, altitude_m)
PENCH_CAMERAS: List[Dict[str, Any]] = [
    {"camera_id": "CAM-001", "name": "Totladoh Reservoir North", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.7620, "longitude": 79.2880, "altitude_m": 340.0},
    {"camera_id": "CAM-002", "name": "Totladoh Shoreline", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.7480, "longitude": 79.3050, "altitude_m": 335.0},
    {"camera_id": "CAM-003", "name": "Karmajhiri Route Junction", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.7360, "longitude": 79.3220, "altitude_m": 360.0},
    {"camera_id": "CAM-004", "name": "Alikatta Grassland", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.7180, "longitude": 79.2760, "altitude_m": 320.0},
    {"camera_id": "CAM-005", "name": "Bodhanala Waterhole", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.7020, "longitude": 79.2540, "altitude_m": 315.0},
    {"camera_id": "CAM-006", "name": "Piyorthadi Nala Crossing", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.6880, "longitude": 79.3080, "altitude_m": 330.0},
    {"camera_id": "CAM-007", "name": "Chhindimatta Riverbed", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.6740, "longitude": 79.2320, "altitude_m": 310.0},
    {"camera_id": "CAM-008", "name": "Jamtara Bamboo Trail", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.7740, "longitude": 79.2460, "altitude_m": 352.0},
    {"camera_id": "CAM-009", "name": "Rukhad Buffer Ridge", "zone": "buffer", "zone_code": "PTR-BUFFER-N", "latitude": 21.8220, "longitude": 79.3480, "altitude_m": 390.0},
    {"camera_id": "CAM-010", "name": "Kurai Corridor North", "zone": "buffer", "zone_code": "PTR-BUFFER-N", "latitude": 21.8340, "longitude": 79.2880, "altitude_m": 410.0},
    {"camera_id": "CAM-011", "name": "Turia Waterhole #4", "zone": "buffer", "zone_code": "PTR-BUFFER-S", "latitude": 21.6180, "longitude": 79.2680, "altitude_m": 300.0},
    {"camera_id": "CAM-012", "name": "Teliya Buffer Corridor", "zone": "buffer", "zone_code": "PTR-BUFFER-S", "latitude": 21.6060, "longitude": 79.1880, "altitude_m": 305.0},
    {"camera_id": "CAM-013", "name": "Eastern Corridor Neck", "zone": "buffer", "zone_code": "PTR-CORRIDOR-E", "latitude": 21.7020, "longitude": 79.4120, "altitude_m": 372.0},
    {"camera_id": "CAM-014", "name": "Sillari Eco-zone Border", "zone": "village_adjacent", "zone_code": "PTR-VILLAGE-SW", "latitude": 21.6020, "longitude": 79.1180, "altitude_m": 295.0},
    {"camera_id": "CAM-015", "name": "Khawasa Village Edge", "zone": "village_adjacent", "zone_code": "PTR-VILLAGE-SW", "latitude": 21.6120, "longitude": 79.1340, "altitude_m": 292.0},
    {"camera_id": "CAM-016", "name": "Sitaghat Ridge Pass", "zone": "core", "zone_code": "PTR-CORE", "latitude": 21.7280, "longitude": 79.3560, "altitude_m": 368.0},
]


def zones_geojson() -> Dict[str, Any]:
    """Zones as a GeoJSON FeatureCollection (used by the map layer API)."""
    return {
        "type": "FeatureCollection",
        "disclaimer": GEO_DATA_DISCLAIMER,
        "features": [
            {
                "type": "Feature",
                "geometry": z["geometry_json"],
                "properties": {
                    "zone_code": z["zone_code"],
                    "name": z["name"],
                    "zone_type": z["zone_type"],
                    "area_km2": z["area_km2"],
                    "style_color": z["style_color"],
                    "is_demo": True,
                },
            }
            for z in PENCH_ZONES
        ],
    }
