"""
Alert service — rule-based wildlife monitoring alerts.

Rules are evaluated on demand (dashboard/alerts fetch, image ingestion,
simulation tick) and are idempotent thanks to `Alert.dedupe_key`.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.alert import Alert, AlertSeverity, AlertStatus, AlertType
from app.models.camera_station import CameraStation, CameraStatus, CameraZone
from app.models.observation import Observation
from app.models.tiger import Tiger
from app.services.map_service import haversine_km

# Movement faster than this between two cameras is flagged as unusual.
UNUSUAL_SPEED_KMH = 8.0


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class AlertService:
    @staticmethod
    async def _create_if_new(db: AsyncSession, dedupe_key: str, **fields: Any) -> Optional[Alert]:
        existing = (
            await db.execute(select(Alert).where(Alert.dedupe_key == dedupe_key))
        ).scalar_one_or_none()
        if existing:
            return None
        alert = Alert(
            alert_id=f"ALT-{uuid.uuid4().hex[:10].upper()}",
            dedupe_key=dedupe_key,
            **fields,
        )
        db.add(alert)
        await db.flush()
        return alert

    @staticmethod
    async def evaluate_detection(
        db: AsyncSession,
        observation: Observation,
        camera: Optional[CameraStation],
        tiger: Optional[Tiger],
    ) -> List[Alert]:
        """Alerts triggered by a single new detection."""
        created: List[Alert] = []
        zone = camera.zone if camera else observation.zone
        zone_code = camera.zone_code if camera else None
        lat = observation.latitude if observation.latitude is not None else (camera.latitude if camera else None)
        lon = observation.longitude if observation.longitude is not None else (camera.longitude if camera else None)
        tiger_label = (tiger.name or tiger.tiger_id) if tiger else "Unidentified tiger"

        if zone == CameraZone.VILLAGE_ADJACENT:
            alert = await AlertService._create_if_new(
                db,
                dedupe_key=f"village-detection:{observation.observation_id}",
                alert_type=AlertType.HIGH_PRIORITY_DETECTION,
                severity=AlertSeverity.CRITICAL,
                status=AlertStatus.OPEN,
                title=f"Tiger near village interface — {observation.camera_id}",
                message=(
                    f"{tiger_label} detected at {camera.name if camera else observation.camera_id} "
                    "in a village-adjacent zone. Notify the rapid response team."
                ),
                camera_id=observation.camera_id,
                tiger_id=observation.tiger_id,
                observation_id=observation.id,
                zone_code=zone_code,
                latitude=lat,
                longitude=lon,
                detail_json={"zone": zone.value if zone else None},
                is_demo=bool(observation.is_demo),
            )
            if alert:
                created.append(alert)

        if (
            observation.identity_confidence is not None
            and observation.identity_confidence < settings.LOW_CONFIDENCE_THRESHOLD
        ):
            alert = await AlertService._create_if_new(
                db,
                dedupe_key=f"low-confidence:{observation.observation_id}",
                alert_type=AlertType.LOW_CONFIDENCE,
                severity=AlertSeverity.LOW,
                status=AlertStatus.OPEN,
                title=f"Low identity confidence on {observation.observation_id}",
                message=(
                    f"Identity confidence {observation.identity_confidence:.2f} is below the "
                    f"{settings.LOW_CONFIDENCE_THRESHOLD:.2f} threshold — human review recommended."
                ),
                camera_id=observation.camera_id,
                tiger_id=observation.tiger_id,
                observation_id=observation.id,
                zone_code=zone_code,
                latitude=lat,
                longitude=lon,
                detail_json={"identity_confidence": observation.identity_confidence},
                is_demo=bool(observation.is_demo),
            )
            if alert:
                created.append(alert)

        if tiger is not None:
            unusual = await AlertService._check_unusual_movement(db, observation, tiger)
            created.extend(unusual)

        await db.commit()
        return created

    @staticmethod
    async def _check_unusual_movement(
        db: AsyncSession, observation: Observation, tiger: Tiger
    ) -> List[Alert]:
        previous = (
            await db.execute(
                select(Observation, CameraStation)
                .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
                .where(
                    Observation.tiger_id == tiger.id,
                    Observation.id != observation.id,
                    Observation.timestamp.isnot(None),
                )
                .order_by(Observation.timestamp.desc())
                .limit(1)
            )
        ).first()
        if not previous:
            return []

        prev_obs, prev_cam = previous
        cur_ts, prev_ts = _as_utc(observation.timestamp), _as_utc(prev_obs.timestamp)
        if not cur_ts or not prev_ts or prev_obs.camera_id == observation.camera_id:
            return []

        lat1 = prev_obs.latitude if prev_obs.latitude is not None else (prev_cam.latitude if prev_cam else None)
        lon1 = prev_obs.longitude if prev_obs.longitude is not None else (prev_cam.longitude if prev_cam else None)
        if lat1 is None or lon1 is None or observation.latitude is None or observation.longitude is None:
            return []

        hours = abs((cur_ts - prev_ts).total_seconds()) / 3600
        if hours <= 0:
            return []
        distance = haversine_km(lat1, lon1, observation.latitude, observation.longitude)
        speed = distance / hours
        if speed < UNUSUAL_SPEED_KMH:
            return []

        alert = await AlertService._create_if_new(
            db,
            dedupe_key=f"unusual-movement:{observation.observation_id}",
            alert_type=AlertType.UNUSUAL_MOVEMENT,
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.OPEN,
            title=f"Unusual movement rate — {tiger.tiger_id}",
            message=(
                f"{tiger.tiger_id} moved {distance:.1f} km from {prev_obs.camera_id} to "
                f"{observation.camera_id} in {hours:.1f} h (~{speed:.1f} km/h). "
                "Verify identity match or check camera clocks."
            ),
            camera_id=observation.camera_id,
            tiger_id=tiger.id,
            observation_id=observation.id,
            latitude=observation.latitude,
            longitude=observation.longitude,
            detail_json={
                "from_camera": prev_obs.camera_id,
                "to_camera": observation.camera_id,
                "distance_km": round(distance, 2),
                "hours": round(hours, 2),
                "speed_kmh": round(speed, 2),
            },
            is_demo=bool(observation.is_demo),
        )
        return [alert] if alert else []

    @staticmethod
    async def evaluate_system_rules(db: AsyncSession) -> int:
        """Camera-offline and high-activity rules. Returns number created."""
        now = datetime.now(timezone.utc)
        created = 0

        cameras = list((await db.execute(select(CameraStation))).scalars().all())
        for cam in cameras:
            last_active = _as_utc(cam.last_active_at)
            offline_for = None
            if last_active is None:
                offline_for = None
            else:
                offline_for = (now - last_active).total_seconds() / 3600

            is_offline = cam.status == CameraStatus.INACTIVE or (
                offline_for is not None and offline_for > settings.CAMERA_OFFLINE_HOURS
            )
            if not is_offline:
                continue

            bucket = last_active.strftime("%Y-%m-%d") if last_active else "never"
            alert = await AlertService._create_if_new(
                db,
                dedupe_key=f"camera-offline:{cam.camera_id}:{bucket}",
                alert_type=AlertType.CAMERA_OFFLINE,
                severity=AlertSeverity.HIGH,
                status=AlertStatus.OPEN,
                title=f"Camera offline — {cam.camera_id}",
                message=(
                    f"{cam.name} has not reported for "
                    f"{f'{offline_for:.0f} h' if offline_for else 'an unknown period'} "
                    f"(threshold {settings.CAMERA_OFFLINE_HOURS} h)."
                ),
                camera_id=cam.camera_id,
                zone_code=cam.zone_code,
                latitude=cam.latitude,
                longitude=cam.longitude,
                detail_json={"hours_offline": round(offline_for, 1) if offline_for else None},
            )
            if alert:
                created += 1

        window_start = now - timedelta(days=1)
        rows = (
            await db.execute(
                select(Observation.camera_id, func.count(Observation.id))
                .where(Observation.timestamp >= window_start)
                .group_by(Observation.camera_id)
                .having(func.count(Observation.id) >= settings.HIGH_ACTIVITY_DETECTIONS_PER_DAY)
            )
        ).all()
        cam_map = {c.camera_id: c for c in cameras}
        for camera_id, count in rows:
            cam = cam_map.get(camera_id)
            alert = await AlertService._create_if_new(
                db,
                dedupe_key=f"high-activity:{camera_id}:{now.strftime('%Y-%m-%d')}",
                alert_type=AlertType.HIGH_ACTIVITY,
                severity=AlertSeverity.MEDIUM,
                status=AlertStatus.OPEN,
                title=f"High detection activity — {camera_id}",
                message=(
                    f"{int(count)} detections in the last 24 h at "
                    f"{cam.name if cam else camera_id} "
                    f"(threshold {settings.HIGH_ACTIVITY_DETECTIONS_PER_DAY})."
                ),
                camera_id=camera_id,
                zone_code=cam.zone_code if cam else None,
                latitude=cam.latitude if cam else None,
                longitude=cam.longitude if cam else None,
                detail_json={"detections_24h": int(count)},
            )
            if alert:
                created += 1

        await db.commit()
        return created

    @staticmethod
    async def list_alerts(
        db: AsyncSession,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        camera_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[Alert], int]:
        query = select(Alert)
        count_query = select(func.count(Alert.id))

        conditions = []
        if status:
            conditions.append(Alert.status == AlertStatus(status))
        if severity:
            conditions.append(Alert.severity == AlertSeverity(severity))
        if alert_type:
            conditions.append(Alert.alert_type == AlertType(alert_type))
        if camera_id:
            conditions.append(Alert.camera_id == camera_id)
        for cond in conditions:
            query = query.where(cond)
            count_query = count_query.where(cond)

        total = (await db.execute(count_query)).scalar_one_or_none() or 0
        rows = (
            await db.execute(
                query.order_by(Alert.created_at.desc()).offset(skip).limit(limit)
            )
        ).scalars().all()
        return list(rows), int(total)

    @staticmethod
    async def summary(db: AsyncSession) -> Dict[str, int]:
        status_rows = (
            await db.execute(select(Alert.status, func.count(Alert.id)).group_by(Alert.status))
        ).all()
        sev_rows = (
            await db.execute(
                select(Alert.severity, func.count(Alert.id))
                .where(Alert.status != AlertStatus.RESOLVED)
                .group_by(Alert.severity)
            )
        ).all()

        out = {"open": 0, "acknowledged": 0, "resolved": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}
        for status, count in status_rows:
            key = status.value if hasattr(status, "value") else str(status)
            if key in out:
                out[key] = int(count)
        for severity, count in sev_rows:
            key = severity.value if hasattr(severity, "value") else str(severity)
            if key in out:
                out[key] = int(count)
        return out

    @staticmethod
    async def update_status(
        db: AsyncSession, alert_id: str, status: str, actor: Optional[str] = None
    ) -> Optional[Alert]:
        alert = (
            await db.execute(select(Alert).where(Alert.alert_id == alert_id))
        ).scalar_one_or_none()
        if not alert:
            return None

        new_status = AlertStatus(status)
        alert.status = new_status
        if new_status == AlertStatus.RESOLVED:
            alert.resolved_at = datetime.now(timezone.utc)
        if actor:
            alert.acknowledged_by = actor
        await db.commit()
        await db.refresh(alert)
        return alert


alert_service = AlertService()
