"""
DEMO data seeding — Pench Eye.

Everything created here is SIMULATED and flagged `is_demo=True`:
zones (approximate boundaries), camera stations, tiger profiles, historical
observations with synthetic frames, review-queue entries and alerts.

Run with `make seed`, `python -m scripts.seed_demo_data`, or POST
/api/v1/demo/simulate for live single events.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.geo import PENCH_CAMERAS, PENCH_ZONES
from app.models.alert import Alert
from app.models.camera_station import CameraStation, CameraStatus, CameraZone
from app.models.embedding import Embedding
from app.models.image import Image, ImageStatus, ProcessingStatus, SourceType
from app.models.observation import (
    DetectionType,
    FlankSide,
    MatchType,
    Observation,
    ReviewStatus,
)
from app.models.review_queue import QueueStatus, ReviewQueue
from app.models.tiger import Tiger, TigerSex, TigerStatus
from app.models.zone import Zone
from app.services.alert_service import AlertService
from app.services.simulation_service import synth_frame
from app.services.storage_service import storage_service

# Tiger names in Pench are widely reported in the press; these profiles are
# illustrative placeholders for the demo catalogue, not verified individuals.
DEMO_TIGERS = [
    ("TIGER-001", "Core Female A", TigerSex.FEMALE, TigerStatus.ACTIVE, ["CAM-001", "CAM-002", "CAM-004"]),
    ("TIGER-002", "Totladoh Male", TigerSex.MALE, TigerStatus.ACTIVE, ["CAM-002", "CAM-003", "CAM-016"]),
    ("TIGER-003", "Rukhad Tigress", TigerSex.FEMALE, TigerStatus.ACTIVE, ["CAM-009", "CAM-010"]),
    ("TIGER-004", "Alikatta Sub-adult", TigerSex.MALE, TigerStatus.ACTIVE, ["CAM-004", "CAM-005", "CAM-006"]),
    ("TIGER-005", "Bodhanala Female", TigerSex.FEMALE, TigerStatus.ACTIVE, ["CAM-005", "CAM-007"]),
    ("TIGER-006", "Karmajhiri Dominant Male", TigerSex.MALE, TigerStatus.ACTIVE, ["CAM-003", "CAM-016", "CAM-013"]),
    ("TIGER-007", "Corridor Sub-adult", TigerSex.MALE, TigerStatus.INACTIVE, ["CAM-013", "CAM-010"]),
    ("TIGER-008", "Turia Female", TigerSex.FEMALE, TigerStatus.ACTIVE, ["CAM-011", "CAM-012", "CAM-014"]),
]

OTHER_SPECIES = ["leopard", "sambar", "chital", "wild_dog", "gaur", "sloth_bear"]


async def clear_demo_data(db: AsyncSession) -> None:
    """Remove previously seeded demo rows so re-seeding stays idempotent."""
    demo_keys = [
        key
        for (key,) in (
            await db.execute(
                select(Image.storage_key).where(
                    Image.is_demo.is_(True), Image.storage_key.isnot(None)
                )
            )
        ).all()
    ] + [
        key
        for (key,) in (
            await db.execute(
                select(Image.quarantine_key).where(
                    Image.is_demo.is_(True), Image.quarantine_key.isnot(None)
                )
            )
        ).all()
    ]

    await db.execute(delete(Alert).where(Alert.is_demo.is_(True)))
    await db.execute(
        delete(ReviewQueue).where(
            ReviewQueue.observation_id.in_(
                select(Observation.id).where(Observation.is_demo.is_(True))
            )
        )
    )
    await db.execute(
        delete(Embedding).where(
            Embedding.observation_id.in_(
                select(Observation.id).where(Observation.is_demo.is_(True))
            )
        )
    )
    await db.execute(delete(Observation).where(Observation.is_demo.is_(True)))
    await db.execute(delete(Image).where(Image.is_demo.is_(True)))
    await db.execute(delete(Tiger).where(Tiger.is_demo.is_(True)))
    await db.execute(
        delete(CameraStation).where(
            (
                CameraStation.is_demo.is_(True)
                | (
                    CameraStation.description
                    == "Simulated camera station for the Pench Eye demo dataset."
                )
            )
            & ~CameraStation.camera_id.in_(
                select(Observation.camera_id).where(
                    Observation.is_demo.is_(False),
                    Observation.camera_id.isnot(None),
                )
            )
        )
    )
    await db.execute(delete(Zone).where(Zone.is_demo.is_(True)))
    await db.commit()

    for key in demo_keys:
        try:
            await storage_service.delete_object(key)
        except Exception:
            pass


async def seed_zones(db: AsyncSession) -> int:
    existing = {
        code for (code,) in (await db.execute(select(Zone.zone_code))).all()
    }
    created = 0
    for z in PENCH_ZONES:
        if z["zone_code"] in existing:
            continue
        db.add(
            Zone(
                zone_code=z["zone_code"],
                name=z["name"],
                zone_type=z["zone_type"],
                description=z["description"],
                center_latitude=z["center_latitude"],
                center_longitude=z["center_longitude"],
                area_km2=z["area_km2"],
                geometry_json=z["geometry_json"],
                style_color=z["style_color"],
                is_demo=True,
            )
        )
        created += 1
    await db.commit()
    return created


async def seed_cameras(db: AsyncSession) -> int:
    existing = {
        code for (code,) in (await db.execute(select(CameraStation.camera_id))).all()
    }
    now = datetime.now(timezone.utc)
    rng = random.Random(42)
    created = 0
    for idx, cam in enumerate(PENCH_CAMERAS):
        if cam["camera_id"] in existing:
            continue
        # A couple of cameras are deliberately stale/inactive so camera-health
        # and offline alerts have something real to report.
        if idx == 6:
            status = CameraStatus.INACTIVE
            last_active = now - timedelta(days=6)
        elif idx == 11:
            status = CameraStatus.MAINTENANCE
            last_active = now - timedelta(days=2)
        else:
            status = CameraStatus.ACTIVE
            last_active = now - timedelta(hours=rng.randint(1, 20))

        db.add(
            CameraStation(
                camera_id=cam["camera_id"],
                name=cam["name"],
                zone=CameraZone(cam["zone"]),
                zone_code=cam["zone_code"],
                latitude=cam["latitude"],
                longitude=cam["longitude"],
                altitude_m=cam["altitude_m"],
                status=status,
                description="Simulated camera station for the Pench Eye demo dataset.",
                battery_percent=rng.choice([18, 34, 52, 67, 81, 94]),
                installed_at=now - timedelta(days=rng.randint(120, 700)),
                last_active_at=last_active,
                is_demo=True,
            )
        )
        created += 1
    await db.commit()
    return created


async def seed_tigers(db: AsyncSession) -> Dict[str, Tiger]:
    tigers: Dict[str, Tiger] = {}
    for code, name, sex, status, _ in DEMO_TIGERS:
        existing = (
            await db.execute(select(Tiger).where(Tiger.tiger_id == code))
        ).scalar_one_or_none()
        if existing:
            tigers[code] = existing
            continue
        tiger = Tiger(
            tiger_id=code,
            name=name,
            sex=sex,
            status=status,
            notes="DEMO profile — simulated individual for demonstration only.",
            is_demo=True,
        )
        db.add(tiger)
        tigers[code] = tiger
    await db.commit()
    for t in tigers.values():
        await db.refresh(t)
    return tigers


async def seed_observations(
    db: AsyncSession,
    tigers: Dict[str, Tiger],
    *,
    days_back: int = 150,
    per_tiger: int = 14,
    store_frames: bool = True,
) -> int:
    cameras = {
        c.camera_id: c
        for c in (await db.execute(select(CameraStation))).scalars().all()
    }
    if not cameras:
        return 0

    rng = random.Random(2026)
    now = datetime.now(timezone.utc)
    created = 0

    for code, name, _, _, home_cams in DEMO_TIGERS:
        tiger = tigers.get(code)
        if tiger is None:
            continue
        available = [cid for cid in home_cams if cid in cameras] or list(cameras.keys())
        timestamps = sorted(
            now - timedelta(days=rng.randint(1, days_back), hours=rng.randint(0, 23))
            for _ in range(per_tiger)
        )

        for i, ts in enumerate(timestamps):
            camera = cameras[available[i % len(available)]]
            confidence = round(rng.uniform(0.68, 0.99), 3)
            if confidence >= settings.AUTO_MATCH_THRESHOLD:
                match_type, review_status = MatchType.AUTO_MATCH, ReviewStatus.APPROVED
            elif confidence >= settings.REVIEW_THRESHOLD:
                match_type, review_status = MatchType.HUMAN_VERIFIED, ReviewStatus.APPROVED
            else:
                match_type, review_status = MatchType.DEMO, ReviewStatus.PENDING_REVIEW

            image = await _create_demo_image(
                db,
                camera=camera,
                timestamp=ts,
                label=f"DEMO {camera.camera_id} {code}",
                blank=False,
                store_frames=store_frames,
                rng=rng,
            )

            observation = Observation(
                observation_id=f"OBS-{uuid.uuid4().hex[:10].upper()}",
                tiger_id=tiger.id,
                image_id=image.id,
                camera_id=camera.camera_id,
                timestamp=ts,
                latitude=camera.latitude + rng.uniform(-0.004, 0.004),
                longitude=camera.longitude + rng.uniform(-0.004, 0.004),
                zone=camera.zone,
                species="tiger",
                detection_type=DetectionType.TIGER,
                detection_confidence=round(rng.uniform(0.85, 0.99), 3),
                identity_confidence=confidence,
                match_type=match_type,
                review_status=review_status,
                flank_side=rng.choice(list(FlankSide)),
                bounding_box_json={"bbox": [96, 84, 486, 322]},
                model_version="demo-inference-v1",
                is_demo=True,
            )
            db.add(observation)
            await db.commit()
            await db.refresh(observation)
            created += 1

            if review_status == ReviewStatus.PENDING_REVIEW:
                others = [c for c in tigers if c != code][:2]
                candidates = [
                    {"tiger_code": code, "score": confidence, "rank": 1},
                    *[
                        {
                            "tiger_code": other,
                            "score": round(max(0.0, confidence - 0.03 * (r + 1)), 3),
                            "rank": r + 2,
                        }
                        for r, other in enumerate(others)
                    ],
                ]
                db.add(
                    ReviewQueue(
                        review_id=f"REV-{uuid.uuid4().hex[:10].upper()}",
                        observation_id=observation.id,
                        candidate_tiger_ids=[c["tiger_code"] for c in candidates],
                        candidate_scores={c["tiger_code"]: c["score"] for c in candidates},
                        alternative_candidates_json=candidates,
                        status=QueueStatus.PENDING,
                    )
                )
                await db.commit()

            # SQLite returns naive datetimes, so normalise before comparing.
            previous = camera.last_detection_at
            if previous is not None and previous.tzinfo is None:
                previous = previous.replace(tzinfo=timezone.utc)
            camera.last_detection_at = max(previous, ts) if previous else ts
            await db.commit()

        stats = (
            await db.execute(
                select(
                    func.count(Observation.id),
                    func.min(Observation.timestamp),
                    func.max(Observation.timestamp),
                ).where(Observation.tiger_id == tiger.id)
            )
        ).one()
        tiger.total_observations = int(stats[0] or 0)
        tiger.first_seen = stats[1]
        tiger.last_seen = stats[2]
        await db.commit()

    # Non-tiger wildlife + blank frames so triage/species analytics are not empty.
    for cid, camera in list(cameras.items())[:10]:
        for _ in range(2):
            ts = now - timedelta(days=rng.randint(1, days_back), hours=rng.randint(0, 23))
            species = rng.choice(OTHER_SPECIES)
            image = await _create_demo_image(
                db,
                camera=camera,
                timestamp=ts,
                label=f"DEMO {cid} {species}",
                blank=False,
                store_frames=store_frames,
                rng=rng,
            )
            db.add(
                Observation(
                    observation_id=f"OBS-{uuid.uuid4().hex[:10].upper()}",
                    image_id=image.id,
                    camera_id=cid,
                    timestamp=ts,
                    latitude=camera.latitude,
                    longitude=camera.longitude,
                    zone=camera.zone,
                    species=species,
                    detection_type=DetectionType.OTHER_WILDLIFE,
                    detection_confidence=round(rng.uniform(0.7, 0.96), 3),
                    model_version="demo-inference-v1",
                    is_demo=True,
                )
            )
            created += 1
        for _ in range(2):
            ts = now - timedelta(days=rng.randint(1, days_back), hours=rng.randint(0, 23))
            await _create_demo_image(
                db,
                camera=camera,
                timestamp=ts,
                label=f"DEMO {cid} blank",
                blank=True,
                store_frames=store_frames,
                rng=rng,
            )
    await db.commit()
    return created


async def _create_demo_image(
    db: AsyncSession,
    *,
    camera: CameraStation,
    timestamp: datetime,
    label: str,
    blank: bool,
    store_frames: bool,
    rng: random.Random,
) -> Image:
    image_id = f"IMG-{uuid.uuid4().hex[:12].upper()}"
    content = synth_frame(rng.randint(0, 2**31), f"{label} {timestamp:%Y-%m-%d %H:%M}") if store_frames else b""
    blank_prob = round(rng.uniform(0.96, 0.999), 3) if blank else round(rng.uniform(0.02, 0.35), 3)
    status = ImageStatus.QUARANTINED if blank else ImageStatus.PROCESSED

    image = Image(
        image_id=image_id,
        original_filename=f"{image_id.lower()}.jpg",
        source_filename=f"{image_id.lower()}.jpg",
        camera_id=camera.camera_id,
        timestamp=timestamp,
        latitude=camera.latitude,
        longitude=camera.longitude,
        file_size_bytes=len(content) or rng.randint(180_000, 620_000),
        width_px=640,
        height_px=400,
        sha256_hash=uuid.uuid4().hex + uuid.uuid4().hex[:0],
        quality_score=round(rng.uniform(0.4, 0.95), 3),
        blank_probability=blank_prob,
        blank_threshold_used=settings.BLANK_THRESHOLD,
        triage_reason="blank_frame_simulated" if blank else "subject_detected_simulated",
        status=status,
        source_type=SourceType.IMAGE,
        processing_status=ProcessingStatus.COMPLETED,
        is_demo=True,
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)

    if store_frames and content:
        key = f"{'quarantine' if blank else 'active'}/{image_id}.jpg"
        try:
            await storage_service.upload_image(content, key)
            if blank:
                image.quarantine_key = key
            else:
                image.storage_key = key
            await db.commit()
        except Exception:
            pass
    return image


async def seed_all(db: AsyncSession, *, reset: bool = False, store_frames: bool = True) -> Dict[str, int]:
    if reset:
        await clear_demo_data(db)

    zones = await seed_zones(db)
    cameras = await seed_cameras(db)
    tigers = await seed_tigers(db)

    existing_obs = (
        await db.execute(select(func.count(Observation.id)).where(Observation.is_demo.is_(True)))
    ).scalar_one_or_none() or 0
    observations = 0
    if existing_obs == 0:
        observations = await seed_observations(db, tigers, store_frames=store_frames)

    alerts = await AlertService.evaluate_system_rules(db)

    return {
        "zones": zones,
        "cameras": cameras,
        "tigers": len(tigers),
        "observations": observations,
        "alerts": alerts,
    }
