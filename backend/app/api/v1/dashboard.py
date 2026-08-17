"""
Dashboard API — command-center KPIs, all derived from database rows.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.alert import Alert, AlertStatus
from app.models.camera_station import CameraStation, CameraStatus
from app.models.image import Image, ImageStatus
from app.models.observation import Observation
from app.models.review_queue import QueueStatus, ReviewQueue
from app.models.tiger import Tiger, TigerStatus
from app.schemas.dashboard import CameraHealth, DashboardStats
from app.services.alert_service import AlertService
from app.services.map_service import camera_marker_state

router = APIRouter()


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: AsyncSession = Depends(get_db)):
    now = datetime.now(timezone.utc)

    img_row = (
        await db.execute(
            select(
                func.count(Image.id).label("total"),
                func.sum(case((Image.status == ImageStatus.QUARANTINED, 1), else_=0)).label("quarantined"),
                func.sum(case((Image.blank_probability >= settings.BLANK_THRESHOLD, 1), else_=0)).label("blank"),
                func.sum(func.coalesce(Image.file_size_bytes, 0)).label("total_bytes"),
                func.sum(
                    case(
                        (
                            Image.status == ImageStatus.QUARANTINED,
                            func.coalesce(Image.file_size_bytes, 0),
                        ),
                        else_=0,
                    )
                ).label("quarantined_bytes"),
            )
        )
    ).one_or_none()

    total_images = int(getattr(img_row, "total", 0) or 0)
    quarantined_images = int(getattr(img_row, "quarantined", 0) or 0)
    blank_images = int(getattr(img_row, "blank", 0) or 0)
    total_storage_bytes = int(getattr(img_row, "total_bytes", 0) or 0)
    storage_saved_bytes = int(getattr(img_row, "quarantined_bytes", 0) or 0)

    tiger_count = (await db.execute(select(func.count(Tiger.id)))).scalar_one_or_none() or 0
    active_tigers = (
        await db.execute(select(func.count(Tiger.id)).where(Tiger.status == TigerStatus.ACTIVE))
    ).scalar_one_or_none() or 0
    obs_count = (await db.execute(select(func.count(Observation.id)))).scalar_one_or_none() or 0
    recent_obs_count = (
        await db.execute(
            select(func.count(Observation.id)).where(
                Observation.timestamp >= now - timedelta(days=7)
            )
        )
    ).scalar_one_or_none() or 0
    pending_reviews = (
        await db.execute(
            select(func.count(ReviewQueue.id)).where(ReviewQueue.status == QueueStatus.PENDING)
        )
    ).scalar_one_or_none() or 0
    open_alerts = (
        await db.execute(
            select(func.count(Alert.id)).where(Alert.status != AlertStatus.RESOLVED)
        )
    ).scalar_one_or_none() or 0
    mean_conf = (
        await db.execute(select(func.avg(Observation.identity_confidence)))
    ).scalar_one_or_none()

    # ── Camera health ────────────────────────────────────────────────────
    cameras = list((await db.execute(select(CameraStation))).scalars().all())
    last_detect_rows = (
        await db.execute(
            select(Observation.camera_id, func.max(Observation.timestamp)).group_by(
                Observation.camera_id
            )
        )
    ).all()
    last_detect_map = {r[0]: r[1] for r in last_detect_rows}

    health = CameraHealth(total=len(cameras))
    for cam in cameras:
        state = camera_marker_state(cam, last_detect_map.get(cam.camera_id), now)
        if state == "offline":
            health.offline += 1
        elif state == "maintenance":
            health.maintenance += 1
        elif state == "warning":
            health.warning += 1
        else:
            health.active += 1

    # ── Recent identifications ───────────────────────────────────────────
    recent_rows = (
        await db.execute(
            select(Observation, Tiger, CameraStation, Image)
            .outerjoin(Tiger, Tiger.id == Observation.tiger_id)
            .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
            .outerjoin(Image, Image.id == Observation.image_id)
            .order_by(Observation.timestamp.desc())
            .limit(10)
        )
    ).all()
    recent_identifications = [
        {
            "observation_id": obs.observation_id,
            "tiger_id": str(obs.tiger_id) if obs.tiger_id else None,
            "tiger_code": tiger.tiger_id if tiger else None,
            "tiger_name": tiger.name if tiger else None,
            "camera_id": obs.camera_id,
            "camera_name": cam.name if cam else None,
            "timestamp": obs.timestamp.isoformat() if obs.timestamp else None,
            "identity_confidence": obs.identity_confidence,
            "match_type": obs.match_type.value if obs.match_type else None,
            "image_id": img.image_id if img else None,
            "image_url": f"/api/v1/images/{img.image_id}/file" if img else None,
        }
        for obs, tiger, cam, img in recent_rows
    ]

    # ── Recent alerts (run system rules first so the panel is current) ────
    await AlertService.evaluate_system_rules(db)
    alert_rows = (
        await db.execute(
            select(Alert)
            .where(Alert.status != AlertStatus.RESOLVED)
            .order_by(Alert.created_at.desc())
            .limit(8)
        )
    ).scalars().all()
    recent_alerts = [
        {
            "alert_id": a.alert_id,
            "alert_type": a.alert_type.value,
            "severity": a.severity.value,
            "status": a.status.value,
            "title": a.title,
            "message": a.message,
            "camera_id": a.camera_id,
            "zone_code": a.zone_code,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in alert_rows
    ]

    # ── Recent images ────────────────────────────────────────────────────
    image_rows = (
        await db.execute(select(Image).order_by(Image.created_at.desc()).limit(8))
    ).scalars().all()
    recent_images = [
        {
            "image_id": img.image_id,
            "camera_id": img.camera_id,
            "status": img.status.value if img.status else None,
            "blank_probability": img.blank_probability,
            "timestamp": (img.timestamp or img.created_at).isoformat()
            if (img.timestamp or img.created_at)
            else None,
            "url": f"/api/v1/images/{img.image_id}/file",
        }
        for img in image_rows
    ]

    cam_img_rows = (
        await db.execute(
            select(Image.camera_id, func.count(Image.id))
            .where(Image.camera_id.isnot(None))
            .group_by(Image.camera_id)
            .order_by(func.count(Image.id).desc())
            .limit(12)
        )
    ).all()

    zone_rows = (
        await db.execute(
            select(CameraStation.zone, func.count(Observation.id))
            .join(Observation, Observation.camera_id == CameraStation.camera_id)
            .group_by(CameraStation.zone)
        )
    ).all()

    tiger_rows = (
        await db.execute(
            select(Tiger.tiger_id, func.count(Observation.id))
            .join(Observation, Observation.tiger_id == Tiger.id)
            .group_by(Tiger.tiger_id)
            .order_by(func.count(Observation.id).desc())
            .limit(8)
        )
    ).all()

    # ── 14-day detection trend ───────────────────────────────────────────
    trend_start = now - timedelta(days=13)
    trend: Dict[str, Dict[str, int]] = {
        (trend_start + timedelta(days=i)).strftime("%Y-%m-%d"): {"detections": 0, "blanks": 0}
        for i in range(14)
    }
    for ts, in (
        await db.execute(
            select(Observation.timestamp).where(Observation.timestamp >= trend_start)
        )
    ).all():
        if ts:
            key = ts.strftime("%Y-%m-%d")
            if key in trend:
                trend[key]["detections"] += 1
    for ts, blank_prob in (
        await db.execute(
            select(Image.timestamp, Image.blank_probability).where(Image.timestamp >= trend_start)
        )
    ).all():
        if ts and blank_prob is not None and blank_prob >= settings.BLANK_THRESHOLD:
            key = ts.strftime("%Y-%m-%d")
            if key in trend:
                trend[key]["blanks"] += 1

    demo_obs = (
        await db.execute(select(func.count(Observation.id)).where(Observation.is_demo.is_(True)))
    ).scalar_one_or_none() or 0

    return DashboardStats(
        total_images=total_images,
        blank_images=blank_images,
        subject_images=max(0, total_images - blank_images),
        quarantined_images=quarantined_images,
        total_tigers=int(tiger_count),
        active_tigers=int(active_tigers),
        total_observations=int(obs_count),
        detections_last_7_days=int(recent_obs_count),
        total_cameras=len(cameras),
        active_cameras=health.active,
        pending_reviews=int(pending_reviews),
        open_alerts=int(open_alerts),
        storage_saved_bytes=storage_saved_bytes,
        total_storage_bytes=total_storage_bytes,
        mean_identity_confidence=round(float(mean_conf), 3) if mean_conf is not None else None,
        demo_mode=settings.is_demo_mode,
        data_source="demo" if (obs_count and demo_obs >= obs_count / 2) else "mixed",
        camera_health=health,
        recent_identifications=recent_identifications,
        recent_alerts=recent_alerts,
        recent_images=recent_images,
        images_by_camera=[{"label": r[0], "count": int(r[1])} for r in cam_img_rows],
        detections_by_zone=[
            {"label": (r[0].value if hasattr(r[0], "value") else str(r[0])), "count": int(r[1])}
            for r in zone_rows
        ],
        detection_trend=[
            {"date": d, "detections": v["detections"], "blanks": v["blanks"]}
            for d, v in sorted(trend.items())
        ],
        most_active_tigers=[{"label": r[0], "count": int(r[1])} for r in tiger_rows],
    )