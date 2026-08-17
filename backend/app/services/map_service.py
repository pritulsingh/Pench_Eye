"""
Geospatial / map service.

Serves the reserve boundary, zones, gates, camera markers, sightings and
tiger movement tracks from the database so the map is data-driven rather
than a static frontend decoration.
"""
from datetime import datetime, timedelta, timezone
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.geo import PENCH_ZONES
from app.models.alert import Alert, AlertStatus
from app.models.camera_station import CameraStation, CameraStatus
from app.models.observation import Observation
from app.models.tiger import Tiger
from app.models.zone import Zone


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def camera_marker_state(
    camera: CameraStation,
    last_detection_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> str:
    """
    Visual state for a camera marker:
      recent_detection > offline > warning > maintenance > active
    """
    now = now or datetime.now(timezone.utc)
    if camera.status == CameraStatus.MAINTENANCE:
        return "maintenance"
    if camera.status == CameraStatus.INACTIVE:
        return "offline"

    last_active = _as_utc(camera.last_active_at)
    if last_active is None or (now - last_active) > timedelta(hours=settings.CAMERA_OFFLINE_HOURS):
        return "offline"

    detection = _as_utc(last_detection_at) or _as_utc(camera.last_detection_at)
    if detection is not None and (now - detection) <= timedelta(hours=24):
        return "recent_detection"

    if camera.battery_percent is not None and camera.battery_percent < 25:
        return "warning"
    if (now - last_active) > timedelta(hours=settings.CAMERA_OFFLINE_HOURS / 2):
        return "warning"
    return "active"


class MapService:
    @staticmethod
    def reserve_bounds() -> List[List[float]]:
        return [
            [settings.RESERVE_SOUTH_LAT, settings.RESERVE_WEST_LON],
            [settings.RESERVE_NORTH_LAT, settings.RESERVE_EAST_LON],
        ]

    @staticmethod
    async def get_zones(db: AsyncSession) -> List[Dict[str, Any]]:
        zones = list(
            (await db.execute(select(Zone).where(Zone.is_demo.is_(False)))).scalars().all()
        )

        cam_counts = dict(
            (row[0], row[1])
            for row in (
                await db.execute(
                    select(CameraStation.zone_code, func.count(CameraStation.id)).group_by(
                        CameraStation.zone_code
                    )
                )
            ).all()
        )
        obs_counts = dict(
            (row[0], row[1])
            for row in (
                await db.execute(
                    select(CameraStation.zone_code, func.count(Observation.id))
                    .join(Observation, Observation.camera_id == CameraStation.camera_id)
                    .group_by(CameraStation.zone_code)
                )
            ).all()
        )

        # The reserve boundary and zone polygons are static reference geography
        # used purely for map framing, not simulated observations. If the DB has
        # no operational zones yet, fall back to the geo fixtures so the reserve
        # territory border always renders.
        if not zones:
            result = [
                {
                    "zone_code": z["zone_code"],
                    "name": z["name"],
                    "zone_type": z["zone_type"],
                    "description": z.get("description"),
                    "center_latitude": z.get("center_latitude"),
                    "center_longitude": z.get("center_longitude"),
                    "area_km2": z.get("area_km2"),
                    "style_color": z.get("style_color"),
                    "geometry_json": z["geometry_json"],
                    "camera_count": int(cam_counts.get(z["zone_code"], 0)),
                    "observation_count": int(obs_counts.get(z["zone_code"], 0)),
                    "is_demo": False,
                }
                for z in PENCH_ZONES
            ]
            order = {"reserve_boundary": 0, "core": 1, "buffer": 2, "corridor": 3, "village_adjacent": 4}
            result.sort(key=lambda z: order.get(z["zone_type"], 9))
            return result

        result = []
        for z in zones:
            result.append(
                {
                    "zone_code": z.zone_code,
                    "name": z.name,
                    "zone_type": z.zone_type,
                    "description": z.description,
                    "center_latitude": z.center_latitude,
                    "center_longitude": z.center_longitude,
                    "area_km2": z.area_km2,
                    "style_color": z.style_color,
                    "geometry_json": z.geometry_json,
                    "camera_count": int(cam_counts.get(z.zone_code, 0)),
                    "observation_count": int(obs_counts.get(z.zone_code, 0)),
                    "is_demo": bool(z.is_demo),
                }
            )
        # Boundary first, then core, then the rest — keeps map layer order sane.
        order = {"reserve_boundary": 0, "core": 1, "buffer": 2, "corridor": 3, "village_adjacent": 4}
        result.sort(key=lambda z: order.get(z["zone_type"], 9))
        return result

    @staticmethod
    async def get_map_cameras(db: AsyncSession) -> List[Dict[str, Any]]:
        cameras = list(
            (await db.execute(select(CameraStation).where(CameraStation.is_demo.is_(False))))
            .scalars()
            .all()
        )

        obs_rows = (
            await db.execute(
                select(
                    Observation.camera_id,
                    func.count(Observation.id),
                    func.max(Observation.timestamp),
                ).where(Observation.is_demo.is_(False)).group_by(Observation.camera_id)
            )
        ).all()
        obs_map = {r[0]: (int(r[1]), r[2]) for r in obs_rows}

        alert_rows = (
            await db.execute(
                select(Alert.camera_id, func.count(Alert.id))
                .where(Alert.status != AlertStatus.RESOLVED, Alert.is_demo.is_(False))
                .group_by(Alert.camera_id)
            )
        ).all()
        alert_map = {r[0]: int(r[1]) for r in alert_rows}

        now = datetime.now(timezone.utc)
        payload = []
        for cam in cameras:
            count, last_detection = obs_map.get(cam.camera_id, (0, None))
            payload.append(
                {
                    "camera_id": cam.camera_id,
                    "name": cam.name,
                    "zone": cam.zone.value if cam.zone else None,
                    "zone_code": cam.zone_code,
                    "latitude": cam.latitude,
                    "longitude": cam.longitude,
                    "status": cam.status.value if cam.status else None,
                    "marker_state": camera_marker_state(cam, last_detection, now),
                    "battery_percent": cam.battery_percent,
                    "last_active_at": cam.last_active_at,
                    "last_detection_at": last_detection or cam.last_detection_at,
                    "observation_count": count,
                    "open_alert_count": alert_map.get(cam.camera_id, 0),
                    "is_demo": bool(cam.is_demo),
                }
            )
        return payload

    @staticmethod
    async def get_map_sightings(
        db: AsyncSession,
        limit: int = 400,
        tiger_code: Optional[str] = None,
        days: Optional[int] = None,
        species: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from app.models.image import Image

        query = (
            select(Observation, Tiger, CameraStation, Image)
            .outerjoin(Tiger, Tiger.id == Observation.tiger_id)
            .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
            .outerjoin(Image, Image.id == Observation.image_id)
            .where(Observation.is_demo.is_(False))
            .order_by(Observation.timestamp.desc())
        )
        if tiger_code:
            query = query.where(Tiger.tiger_id == tiger_code)
        if species:
            query = query.where(Observation.species == species)
        if days:
            query = query.where(
                Observation.timestamp >= datetime.now(timezone.utc) - timedelta(days=days)
            )

        rows = (await db.execute(query.limit(limit))).all()
        sightings = []
        for obs, tiger, cam, image in rows:
            lat = obs.latitude if obs.latitude is not None else (cam.latitude if cam else None)
            lon = obs.longitude if obs.longitude is not None else (cam.longitude if cam else None)
            if lat is None or lon is None:
                continue
            sightings.append(
                {
                    "observation_id": obs.observation_id,
                    "tiger_code": tiger.tiger_id if tiger else None,
                    "tiger_name": tiger.name if tiger else None,
                    "camera_id": obs.camera_id,
                    "camera_name": cam.name if cam else None,
                    "timestamp": obs.timestamp,
                    "latitude": lat,
                    "longitude": lon,
                    "zone": obs.zone.value if obs.zone else None,
                    "species": obs.species,
                    "detection_type": obs.detection_type.value if obs.detection_type else None,
                    "identity_confidence": obs.identity_confidence,
                    "detection_confidence": obs.detection_confidence,
                    "image_id": image.image_id if image else None,
                    "image_url": f"/api/v1/images/{image.image_id}/file" if image else None,
                    "is_demo": bool(obs.is_demo),
                }
            )
        return sightings

    @staticmethod
    async def get_movement_tracks(
        db: AsyncSession,
        tiger_code: Optional[str] = None,
        max_tigers: int = 6,
        max_legs: int = 12,
    ) -> List[Dict[str, Any]]:
        """Chronological observed camera points and straight-line legs per tiger."""
        query = (
            select(Observation, Tiger, CameraStation)
            .join(Tiger, Tiger.id == Observation.tiger_id)
            .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
            .where(Observation.timestamp.isnot(None), Observation.is_demo.is_(False))
            .order_by(Tiger.tiger_id, Observation.timestamp)
        )
        if tiger_code:
            query = query.where(Tiger.tiger_id == tiger_code)

        rows = (await db.execute(query)).all()

        per_tiger: Dict[str, Dict[str, Any]] = {}
        for obs, tiger, cam in rows:
            lat = obs.latitude if obs.latitude is not None else (cam.latitude if cam else None)
            lon = obs.longitude if obs.longitude is not None else (cam.longitude if cam else None)
            if lat is None or lon is None:
                continue
            entry = per_tiger.setdefault(
                tiger.tiger_id, {"tiger_name": tiger.name, "points": []}
            )
            entry["points"].append(
                {
                    "camera_id": obs.camera_id,
                    "camera_name": cam.name if cam else None,
                    "lat": lat,
                    "lon": lon,
                    "ts": obs.timestamp,
                }
            )

        tracks: List[Dict[str, Any]] = []
        for code, data in per_tiger.items():
            points = data["points"]
            legs = []
            total = 0.0
            for a, b in zip(points, points[1:]):
                if a["camera_id"] == b["camera_id"]:
                    continue
                distance = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
                ts_a, ts_b = _as_utc(a["ts"]), _as_utc(b["ts"])
                hours = abs((ts_b - ts_a).total_seconds()) / 3600 if ts_a and ts_b else 0.0
                total += distance
                legs.append(
                    {
                        "from_camera_id": a["camera_id"],
                        "from_camera_name": a["camera_name"],
                        "from_latitude": a["lat"],
                        "from_longitude": a["lon"],
                        "from_timestamp": a["ts"],
                        "to_camera_id": b["camera_id"],
                        "to_camera_name": b["camera_name"],
                        "to_latitude": b["lat"],
                        "to_longitude": b["lon"],
                        "to_timestamp": b["ts"],
                        "distance_km": round(distance, 2),
                        "hours_elapsed": round(hours, 1),
                    }
                )
            legs = legs[-max_legs:]
            tracks.append(
                {
                    "tiger_code": code,
                    "tiger_name": data["tiger_name"],
                    "observations": [
                        {
                            "camera_id": point["camera_id"],
                            "camera_name": point["camera_name"],
                            "latitude": point["lat"],
                            "longitude": point["lon"],
                            "timestamp": point["ts"],
                        }
                        for point in points
                    ],
                    "legs": legs,
                    "total_distance_km": round(sum(l["distance_km"] for l in legs), 2),
                    "distance_label": "Observed camera-to-camera distance",
                    "sighting_count": len(points),
                }
            )

        tracks.sort(key=lambda t: t["sighting_count"], reverse=True)
        return tracks[:max_tigers]

    @staticmethod
    async def get_overview(
        db: AsyncSession,
        sighting_limit: int = 300,
        tiger_code: Optional[str] = None,
        days: Optional[int] = None,
    ) -> Dict[str, Any]:
        zones = await MapService.get_zones(db)
        cameras = await MapService.get_map_cameras(db)
        sightings = await MapService.get_map_sightings(
                db, limit=sighting_limit, tiger_code=tiger_code, days=days
            )
        coordinates = [
            (float(item["latitude"]), float(item["longitude"]))
            for item in [*cameras, *sightings]
            if item.get("latitude") is not None and item.get("longitude") is not None
        ]
        if not coordinates:
            coordinates = [
                (float(zone["center_latitude"]), float(zone["center_longitude"]))
                for zone in zones
                if zone.get("center_latitude") is not None
                and zone.get("center_longitude") is not None
            ]
        center = [settings.RESERVE_CENTER_LAT, settings.RESERVE_CENTER_LON]
        bounds = MapService.reserve_bounds()
        if coordinates:
            lats = [point[0] for point in coordinates]
            lons = [point[1] for point in coordinates]
            # Keep the real Pench landscape in view while expanding for any
            # legitimate station just outside the configured reserve extent.
            bounds = [
                [min(settings.RESERVE_SOUTH_LAT, min(lats)), min(settings.RESERVE_WEST_LON, min(lons))],
                [max(settings.RESERVE_NORTH_LAT, max(lats)), max(settings.RESERVE_EAST_LON, max(lons))],
            ]

        return {
            "center": center,
            "bounds": bounds,
            "data_source": "live",
            "disclaimer": (
                "Satellite imagery and community places come from external map providers; "
                "camera, sighting and zone overlays contain only non-demo database records."
            ),
            "zones": zones,
            "gates": [],
            "cameras": cameras,
            "sightings": sightings,
            "tracks": await MapService.get_movement_tracks(db, tiger_code=tiger_code),
        }


map_service = MapService()
