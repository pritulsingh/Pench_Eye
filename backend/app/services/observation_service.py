from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, desc
from app.models.camera_station import CameraStation
from app.models.image import Image
from app.models.observation import Observation, MatchType, DetectionType
from app.models.tiger import Tiger
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone
import uuid


def serialize_observation(
    obs: Observation,
    tiger: Optional[Tiger] = None,
    camera: Optional[CameraStation] = None,
    image: Optional[Image] = None,
) -> Dict[str, Any]:
    return {
        "id": obs.id,
        "observation_id": obs.observation_id,
        "tiger_id": obs.tiger_id,
        "tiger_code": tiger.tiger_id if tiger else None,
        "tiger_name": tiger.name if tiger else None,
        "image_id": obs.image_id,
        "image_code": image.image_id if image else None,
        "image_url": f"/api/v1/images/{image.image_id}/file" if image else None,
        "camera_id": obs.camera_id,
        "camera_name": camera.name if camera else None,
        "timestamp": obs.timestamp,
        "latitude": obs.latitude if obs.latitude is not None else (camera.latitude if camera else None),
        "longitude": obs.longitude if obs.longitude is not None else (camera.longitude if camera else None),
        "zone": obs.zone.value if obs.zone else (camera.zone.value if camera and camera.zone else None),
        "species": obs.species,
        "detection_type": obs.detection_type.value if obs.detection_type else None,
        "detection_confidence": obs.detection_confidence,
        "identity_confidence": obs.identity_confidence,
        "match_type": obs.match_type.value if obs.match_type else None,
        "review_status": obs.review_status.value if obs.review_status else None,
        "flank_side": obs.flank_side.value if obs.flank_side else None,
        "bounding_box_json": obs.bounding_box_json,
        "model_version": obs.model_version,
        "is_demo": bool(obs.is_demo),
    }


class ObservationService:
    @staticmethod
    async def create_observation(
        db: AsyncSession,
        image_id,
        camera_id: Optional[str],
        tiger_id=None,
        detection_data: Optional[Dict[str, Any]] = None,
        identity_data: Optional[Dict[str, Any]] = None,
        match_type: MatchType = MatchType.NEW_INDIVIDUAL,
        timestamp: Optional[datetime] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        zone=None,
        species: str = "tiger",
        detection_type: DetectionType = DetectionType.TIGER,
        is_demo: bool = False,
        model_version: Optional[str] = None,
    ) -> Observation:
        obs = Observation(
            observation_id=f"OBS-{uuid.uuid4().hex[:10].upper()}",
            tiger_id=tiger_id,
            image_id=image_id,
            camera_id=camera_id,
            timestamp=timestamp or datetime.now(timezone.utc),
            latitude=lat,
            longitude=lon,
            zone=zone,
            species=species,
            detection_type=detection_type,
            match_type=match_type,
            is_demo=is_demo,
            model_version=model_version,
        )

        if detection_data:
            obs.bounding_box_json = detection_data.get("bbox")
            obs.detection_confidence = detection_data.get("confidence")

        if identity_data:
            obs.identity_confidence = identity_data.get("confidence")
            flank = identity_data.get("flank_side")
            if flank:
                obs.flank_side = flank

        db.add(obs)
        await db.commit()
        await db.refresh(obs)
        return obs

    @staticmethod
    async def query_observations(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        tiger_code: Optional[str] = None,
        camera_id: Optional[str] = None,
        zone: Optional[str] = None,
        species: Optional[str] = None,
        min_confidence: Optional[float] = None,
        days: Optional[int] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        base = (
            select(Observation, Tiger, CameraStation, Image)
            .outerjoin(Tiger, Tiger.id == Observation.tiger_id)
            .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
            .outerjoin(Image, Image.id == Observation.image_id)
        )
        count_query = select(func.count(Observation.id)).outerjoin(
            Tiger, Tiger.id == Observation.tiger_id
        )

        conditions = []
        if tiger_code:
            conditions.append(Tiger.tiger_id == tiger_code)
        if camera_id:
            conditions.append(Observation.camera_id == camera_id)
        if zone:
            conditions.append(Observation.zone == zone)
        if species:
            conditions.append(Observation.species == species)
        if min_confidence is not None:
            conditions.append(Observation.identity_confidence >= min_confidence)
        if days:
            conditions.append(
                Observation.timestamp >= datetime.now(timezone.utc) - timedelta(days=days)
            )
        if date_from:
            conditions.append(Observation.timestamp >= date_from)
        if date_to:
            conditions.append(Observation.timestamp <= date_to)

        for cond in conditions:
            base = base.where(cond)
            count_query = count_query.where(cond)

        total = (await db.execute(count_query)).scalar_one_or_none() or 0
        rows = (
            await db.execute(
                base.order_by(desc(Observation.timestamp)).offset(skip).limit(limit)
            )
        ).all()

        items = [serialize_observation(obs, tiger, cam, img) for obs, tiger, cam, img in rows]
        return items, int(total)

    @staticmethod
    async def get_observations(
        db: AsyncSession, skip: int = 0, limit: int = 50, tiger_id: Optional[str] = None
    ) -> List[Observation]:
        query = select(Observation).order_by(desc(Observation.timestamp))
        if tiger_id:
            query = query.join(Tiger, Tiger.id == Observation.tiger_id).where(
                Tiger.tiger_id == tiger_id
            )
        result = await db.execute(query.offset(skip).limit(limit))
        return list(result.scalars().all())

    @staticmethod
    async def get_observation_detail(
        db: AsyncSession, observation_id: str
    ) -> Optional[Dict[str, Any]]:
        row = (
            await db.execute(
                select(Observation, Tiger, CameraStation, Image)
                .outerjoin(Tiger, Tiger.id == Observation.tiger_id)
                .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
                .outerjoin(Image, Image.id == Observation.image_id)
                .where(Observation.observation_id == observation_id)
            )
        ).first()
        if not row:
            return None
        obs, tiger, cam, img = row
        return serialize_observation(obs, tiger, cam, img)

    @staticmethod
    async def get_observation(db: AsyncSession, observation_id: str) -> Optional[Observation]:
        result = await db.execute(
            select(Observation).where(Observation.observation_id == observation_id)
        )
        return result.scalar_one_or_none()
