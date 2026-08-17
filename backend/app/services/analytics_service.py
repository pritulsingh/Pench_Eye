"""
Analytics service — all figures derive from rows in the database.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.camera_station import CameraStation
from app.models.image import Image
from app.models.observation import Observation
from app.models.tiger import Tiger

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CONFIDENCE_BUCKETS = [
    ("<0.60", 0.0, 0.60),
    ("0.60–0.75", 0.60, 0.75),
    ("0.75–0.85", 0.75, 0.85),
    ("0.85–0.95", 0.85, 0.95),
    ("≥0.95", 0.95, 1.01),
]


class AnalyticsService:
    @staticmethod
    async def overview(db: AsyncSession, days: int = 90) -> Dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(days=days)

        rows = (
            await db.execute(
                select(Observation, CameraStation, Tiger)
                .outerjoin(CameraStation, CameraStation.camera_id == Observation.camera_id)
                .outerjoin(Tiger, Tiger.id == Observation.tiger_id)
                .where(Observation.timestamp >= since, Observation.is_demo.is_(False))
            )
        ).all()

        by_day: Dict[str, Dict[str, int]] = {}
        by_camera: Dict[str, int] = {}
        by_zone: Dict[str, int] = {}
        by_hour: Dict[int, int] = {h: 0 for h in range(24)}
        by_weekday: Dict[int, int] = {i: 0 for i in range(7)}
        by_species: Dict[str, int] = {}
        by_tiger: Dict[str, int] = {}
        confidences: List[float] = []
        tiger_points: Dict[str, List[tuple]] = {}

        for obs, cam, tiger in rows:
            ts = obs.timestamp
            if ts is not None:
                key = ts.strftime("%Y-%m-%d")
                bucket = by_day.setdefault(key, {"detections": 0, "tigers": 0, "blanks": 0})
                bucket["detections"] += 1
                if obs.tiger_id:
                    bucket["tigers"] += 1
                by_hour[ts.hour] = by_hour.get(ts.hour, 0) + 1
                by_weekday[ts.weekday()] = by_weekday.get(ts.weekday(), 0) + 1

            if obs.camera_id:
                by_camera[obs.camera_id] = by_camera.get(obs.camera_id, 0) + 1

            zone = (cam.zone.value if cam and cam.zone else (obs.zone.value if obs.zone else "unknown"))
            by_zone[zone] = by_zone.get(zone, 0) + 1

            species = obs.species or "unknown"
            by_species[species] = by_species.get(species, 0) + 1

            if tiger:
                label = tiger.tiger_id
                by_tiger[label] = by_tiger.get(label, 0) + 1
                if obs.camera_id and ts:
                    tiger_points.setdefault(label, []).append((ts, obs.camera_id))

            if obs.identity_confidence is not None:
                confidences.append(float(obs.identity_confidence))

        blank_rows = (
            await db.execute(
                select(Image.timestamp, Image.blank_probability).where(
                    Image.timestamp >= since, Image.is_demo.is_(False)
                )
            )
        ).all()
        for ts, blank_prob in blank_rows:
            if ts is None or blank_prob is None:
                continue
            key = ts.strftime("%Y-%m-%d")
            bucket = by_day.setdefault(key, {"detections": 0, "tigers": 0, "blanks": 0})
            if blank_prob >= 0.95:
                bucket["blanks"] += 1

        movement: Dict[tuple, int] = {}
        for label, points in tiger_points.items():
            points.sort()
            for (_, cam_a), (_, cam_b) in zip(points, points[1:]):
                if cam_a == cam_b:
                    continue
                movement[(cam_a, cam_b)] = movement.get((cam_a, cam_b), 0) + 1

        conf_dist = []
        for label, low, high in CONFIDENCE_BUCKETS:
            conf_dist.append(
                {"range": label, "count": sum(1 for c in confidences if low <= c < high)}
            )

        cam_activity_rows = (
            await db.execute(
                select(Image.camera_id, func.count(Image.id))
                .where(Image.camera_id.isnot(None), Image.is_demo.is_(False))
                .group_by(Image.camera_id)
            )
        ).all()

        return {
            "range_days": days,
            "detections_over_time": [
                {"date": d, "detections": v["detections"], "tigers": v["tigers"], "blanks": v["blanks"]}
                for d, v in sorted(by_day.items())
            ],
            "detections_by_camera": _top_labels(by_camera, 20),
            "detections_by_zone": _top_labels(by_zone, 10),
            "detections_by_hour": [{"label": f"{h:02d}:00", "count": by_hour[h]} for h in range(24)],
            "detections_by_weekday": [
                {"label": WEEKDAYS[i], "count": by_weekday[i]} for i in range(7)
            ],
            "species_distribution": _top_labels(by_species, 10),
            "top_tigers": _top_labels(by_tiger, 10),
            "confidence_distribution": conf_dist,
            "movement_frequency": [
                {"from_camera": a, "to_camera": b, "transitions": n}
                for (a, b), n in sorted(movement.items(), key=lambda kv: kv[1], reverse=True)[:15]
            ],
            "camera_activity": _top_labels({r[0]: int(r[1]) for r in cam_activity_rows}, 20),
            "mean_identity_confidence": round(sum(confidences) / len(confidences), 3)
            if confidences
            else None,
            "is_demo_data": False,
        }


def _top_labels(mapping: Dict[Any, int], limit: int) -> List[Dict[str, Any]]:
    return [
        {"label": str(k), "count": int(v)}
        for k, v in sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    ]


analytics_service = AnalyticsService()
