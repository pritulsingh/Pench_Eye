"""Camera station queries with activity metrics used by list/detail views."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert, AlertStatus
from app.models.camera_station import CameraStation
from app.models.image import Image
from app.models.observation import Observation
from app.models.tiger import Tiger
from app.services.map_service import camera_marker_state


class CameraService:
    @staticmethod
    async def list_cameras(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        zone: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        query = select(CameraStation).where(CameraStation.is_demo.is_(False))
        count_query = select(func.count(CameraStation.id)).where(CameraStation.is_demo.is_(False))

        conditions = []
        if zone:
            conditions.append(CameraStation.zone == zone)
        if status:
            conditions.append(CameraStation.status == status)
        if search:
            like = f"%{search.lower()}%"
            conditions.append(
                func.lower(CameraStation.name).like(like)
                | func.lower(CameraStation.camera_id).like(like)
            )
        for cond in conditions:
            query = query.where(cond)
            count_query = count_query.where(cond)

        total = (await db.execute(count_query)).scalar_one_or_none() or 0
        cameras = list(
            (await db.execute(query.order_by(CameraStation.camera_id).offset(skip).limit(limit)))
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

        img_rows = (
            await db.execute(
                select(Image.camera_id, func.count(Image.id))
                .where(Image.is_demo.is_(False))
                .group_by(Image.camera_id)
            )
        ).all()
        img_map = {r[0]: int(r[1]) for r in img_rows}

        alert_rows = (
            await db.execute(
                select(Alert.camera_id, func.count(Alert.id))
                .where(Alert.status != AlertStatus.RESOLVED, Alert.is_demo.is_(False))
                .group_by(Alert.camera_id)
            )
        ).all()
        alert_map = {r[0]: int(r[1]) for r in alert_rows}

        now = datetime.now(timezone.utc)
        items = []
        for cam in cameras:
            count, last_detection = obs_map.get(cam.camera_id, (0, None))
            items.append(
                {
                    "id": cam.id,
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
                    "image_count": img_map.get(cam.camera_id, 0),
                    "open_alert_count": alert_map.get(cam.camera_id, 0),
                    "is_demo": bool(cam.is_demo),
                }
            )
        return items, int(total)

    @staticmethod
    async def get_camera_detail(db: AsyncSession, camera_id: str) -> Optional[Dict[str, Any]]:
        cam = (
            await db.execute(select(CameraStation).where(CameraStation.camera_id == camera_id))
        ).scalar_one_or_none()
        if not cam:
            return None

        obs_rows = (
            await db.execute(
                select(Observation, Tiger, Image)
                .outerjoin(Tiger, Tiger.id == Observation.tiger_id)
                .outerjoin(Image, Image.id == Observation.image_id)
                .where(Observation.camera_id == camera_id, Observation.is_demo.is_(False))
                .order_by(Observation.timestamp.desc())
            )
        ).all()

        recent_detections = []
        timeline: Dict[str, int] = {}
        unique_tigers = set()
        for obs, tiger, image in obs_rows:
            if tiger:
                unique_tigers.add(tiger.tiger_id)
            if obs.timestamp:
                key = obs.timestamp.strftime("%Y-%m-%d")
                timeline[key] = timeline.get(key, 0) + 1
            if len(recent_detections) < 12:
                recent_detections.append(
                    {
                        "observation_id": obs.observation_id,
                        "timestamp": obs.timestamp,
                        "tiger_code": tiger.tiger_id if tiger else None,
                        "tiger_name": tiger.name if tiger else None,
                        "species": obs.species,
                        "identity_confidence": obs.identity_confidence,
                        "detection_confidence": obs.detection_confidence,
                        "image_id": image.image_id if image else None,
                    }
                )

        image_rows = (
            await db.execute(
                select(Image)
                .where(Image.camera_id == camera_id, Image.is_demo.is_(False))
                .order_by(Image.created_at.desc())
                .limit(12)
            )
        ).scalars().all()

        image_count = (
            await db.execute(
                select(func.count(Image.id)).where(
                    Image.camera_id == camera_id, Image.is_demo.is_(False)
                )
            )
        ).scalar_one_or_none() or 0

        open_alerts = (
            await db.execute(
                select(func.count(Alert.id)).where(
                    Alert.camera_id == camera_id,
                    Alert.status != AlertStatus.RESOLVED,
                    Alert.is_demo.is_(False),
                )
            )
        ).scalar_one_or_none() or 0

        last_detection = obs_rows[0][0].timestamp if obs_rows else None

        return {
            "id": cam.id,
            "camera_id": cam.camera_id,
            "name": cam.name,
            "zone": cam.zone.value if cam.zone else None,
            "zone_code": cam.zone_code,
            "latitude": cam.latitude,
            "longitude": cam.longitude,
            "altitude_m": cam.altitude_m,
            "status": cam.status.value if cam.status else None,
            "description": cam.description,
            "battery_percent": cam.battery_percent,
            "installed_at": cam.installed_at,
            "last_active_at": cam.last_active_at,
            "last_detection_at": last_detection or cam.last_detection_at,
            "created_at": cam.created_at,
            "updated_at": cam.updated_at,
            "is_demo": bool(cam.is_demo),
            "marker_state": camera_marker_state(cam, last_detection),
            "observation_count": len(obs_rows),
            "image_count": int(image_count),
            "unique_tigers": len(unique_tigers),
            "open_alert_count": int(open_alerts),
            "recent_detections": recent_detections,
            "recent_images": [
                {
                    "image_id": img.image_id,
                    "url": f"/api/v1/images/{img.image_id}/file",
                    "timestamp": img.timestamp or img.created_at,
                    "status": img.status.value if img.status else None,
                    "blank_probability": img.blank_probability,
                }
                for img in image_rows
            ],
            "detection_timeline": [
                {"date": d, "detections": c} for d, c in sorted(timeline.items())
            ],
        }


camera_service = CameraService()
