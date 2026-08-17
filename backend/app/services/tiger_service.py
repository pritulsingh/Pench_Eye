from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
import re
from app.models.camera_station import CameraStation
from app.models.image import Image
from app.models.observation import Observation
from app.models.tiger import Tiger, TigerStatus
from typing import Any, Dict, List, Optional

class TigerService:
    @staticmethod
    async def get_next_tiger_id(db: AsyncSession) -> str:
        """Generate the next code from persisted IDs, without count/delete collisions."""
        rows = (await db.execute(select(Tiger.tiger_id))).scalars().all()
        used = {
            int(match.group(1))
            for code in rows
            if (match := re.fullmatch(r"TIGER-(\d+)", code or ""))
        }
        number = max(used, default=0) + 1
        while number in used:
            number += 1
        return f"TIGER-{number:03d}"
        
    @staticmethod
    async def get_all_tigers(db: AsyncSession, skip: int = 0, limit: int = 50) -> List[Tiger]:
        result = await db.execute(select(Tiger).offset(skip).limit(limit))
        return list(result.scalars().all())

    @staticmethod
    async def list_tigers(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        query = select(Tiger).where(Tiger.is_demo.is_(False))
        count_query = select(func.count(Tiger.id)).where(Tiger.is_demo.is_(False))

        conditions = []
        if status:
            conditions.append(Tiger.status == status)
        if search:
            like = f"%{search.lower()}%"
            conditions.append(
                func.lower(Tiger.tiger_id).like(like) | func.lower(Tiger.name).like(like)
            )
        for cond in conditions:
            query = query.where(cond)
            count_query = count_query.where(cond)

        total = (await db.execute(count_query)).scalar_one_or_none() or 0
        tigers = list(
            (await db.execute(query.order_by(Tiger.tiger_id).offset(skip).limit(limit)))
            .scalars()
            .all()
        )

        stats_rows = (
            await db.execute(
                select(
                    Observation.tiger_id,
                    func.count(Observation.id),
                    func.avg(Observation.identity_confidence),
                    func.count(func.distinct(Observation.camera_id)),
                    func.min(Observation.timestamp),
                    func.max(Observation.timestamp),
                )
                .where(Observation.tiger_id.isnot(None), Observation.is_demo.is_(False))
                .group_by(Observation.tiger_id)
            )
        ).all()
        stats = {r[0]: r for r in stats_rows}

        items = []
        for t in tigers:
            row = stats.get(t.id)
            items.append(
                {
                    "id": t.id,
                    "tiger_id": t.tiger_id,
                    "name": t.name,
                    "sex": t.sex.value if t.sex else None,
                    "status": t.status.value if t.status else None,
                    "total_observations": int(row[1]) if row else int(t.total_observations or 0),
                    "first_seen": (row[4] if row and row[4] else t.first_seen),
                    "last_seen": (row[5] if row and row[5] else t.last_seen),
                    "mean_confidence": round(float(row[2]), 3) if row and row[2] is not None else None,
                    "camera_count": int(row[3]) if row else 0,
                    "is_demo": bool(t.is_demo),
                }
            )
        return items, int(total)

    @staticmethod
    async def get_tiger_profile(db: AsyncSession, tiger_code: str) -> Optional[Dict[str, Any]]:
        tiger = await TigerService.get_tiger(db, tiger_code)
        if not tiger:
            return None

        rows = (
            await db.execute(
                select(Observation, CameraStation, Image)
                .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
                .outerjoin(Image, Image.id == Observation.image_id)
                .where(Observation.tiger_id == tiger.id, Observation.is_demo.is_(False))
                .order_by(Observation.timestamp.desc())
            )
        ).all()

        by_camera: Dict[str, Dict[str, Any]] = {}
        by_zone: Dict[str, int] = {}
        by_month: Dict[str, int] = {}
        confidences: List[float] = []
        recent: List[Dict[str, Any]] = []

        for obs, cam, img in rows:
            if obs.camera_id:
                entry = by_camera.setdefault(
                    obs.camera_id,
                    {
                        "camera_id": obs.camera_id,
                        "camera_name": cam.name if cam else None,
                        "detections": 0,
                        "latitude": cam.latitude if cam else obs.latitude,
                        "longitude": cam.longitude if cam else obs.longitude,
                    },
                )
                entry["detections"] += 1

            zone = obs.zone.value if obs.zone else (cam.zone.value if cam and cam.zone else "unknown")
            by_zone[zone] = by_zone.get(zone, 0) + 1

            if obs.timestamp:
                key = obs.timestamp.strftime("%Y-%m")
                by_month[key] = by_month.get(key, 0) + 1

            if obs.identity_confidence is not None:
                confidences.append(float(obs.identity_confidence))

            if len(recent) < 12:
                recent.append(
                    {
                        "observation_id": obs.observation_id,
                        "timestamp": obs.timestamp,
                        "camera_id": obs.camera_id,
                        "camera_name": cam.name if cam else None,
                        "zone": zone,
                        "identity_confidence": obs.identity_confidence,
                        "latitude": obs.latitude if obs.latitude is not None else (cam.latitude if cam else None),
                        "longitude": obs.longitude if obs.longitude is not None else (cam.longitude if cam else None),
                        "image_id": img.image_id if img else None,
                        "image_url": f"/api/v1/images/{img.image_id}/file" if img else None,
                    }
                )

        timestamps = [obs.timestamp for obs, _, _ in rows if obs.timestamp]

        return {
            "id": tiger.id,
            "tiger_id": tiger.tiger_id,
            "name": tiger.name,
            "sex": tiger.sex.value if tiger.sex else None,
            "status": tiger.status.value if tiger.status else None,
            "estimated_age_years": tiger.estimated_age_years,
            "total_observations": len(rows) or int(tiger.total_observations or 0),
            "first_seen": min(timestamps) if timestamps else tiger.first_seen,
            "last_seen": max(timestamps) if timestamps else tiger.last_seen,
            "notes": tiger.notes,
            "is_demo": bool(tiger.is_demo),
            "created_at": tiger.created_at,
            "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "camera_count": len(by_camera),
            "zone_distribution": [
                {"label": k, "count": v}
                for k, v in sorted(by_zone.items(), key=lambda kv: kv[1], reverse=True)
            ],
            "frequent_cameras": sorted(
                by_camera.values(), key=lambda c: c["detections"], reverse=True
            )[:6],
            "recent_observations": recent,
            "detections_by_month": [
                {"label": k, "count": v} for k, v in sorted(by_month.items())
            ],
        }
        
    @staticmethod
    async def get_tiger(db: AsyncSession, tiger_id: str) -> Optional[Tiger]:
        result = await db.execute(select(Tiger).where(Tiger.tiger_id == tiger_id))
        return result.scalar_one_or_none()
        
    @staticmethod
    async def create_tiger(
        db: AsyncSession,
        name: str = None,
        sex: str = "unknown",
        notes: str = None,
        tiger_id: Optional[str] = None,
    ) -> Tiger:
        tiger_str_id = tiger_id or await TigerService.get_next_tiger_id(db)
        tiger = Tiger(
            tiger_id=tiger_str_id,
            name=name,
            sex=sex,
            notes=notes,
            status=TigerStatus.ACTIVE,
        )
        db.add(tiger)
        await db.commit()
        await db.refresh(tiger)
        return tiger
        
    @staticmethod
    async def update_tiger_stats(db: AsyncSession, tiger_db_id):
        obs_result = await db.execute(
            select(
                func.count(Observation.id),
                func.min(Observation.timestamp),
                func.max(Observation.timestamp),
            ).where(Observation.tiger_id == tiger_db_id)
        )
        obs_count, first_seen, last_seen = obs_result.one()

        tiger = await db.get(Tiger, tiger_db_id)
        if tiger:
            tiger.total_observations = int(obs_count or 0)
            if first_seen:
                tiger.first_seen = first_seen
            if last_seen:
                tiger.last_seen = last_seen
            await db.commit()
